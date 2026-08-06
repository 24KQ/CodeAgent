"""Durable memory store (fusion P2, M3).

Ported from pico `features/memory.py:760-1020` with the M3 write-upgrade
mandated by the Codex review: pico's plain `write_text` and PID lock are
replaced by temp+rename atomic publication and a store-level portalocker
cross-process lock, so index/topic/metadata transactions cannot be torn
or interleaved across processes.

Layout: `MEMORY.md` index + `topics/<topic>.md` + `<topic>.metadata.jsonl`
sidecar, note_id = sha256(topic + text)[:12], evidence{source_path,
session_id, anchor_hash, scope}. Implements the P0 `MemoryStorePort`
shape. The daily-log evidence sidecar (per-entry provenance) is the
FirstCoder extension that lets the P0 port's `source` argument survive
capture until promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from firstcoder.memory.logs import ENTRYPOINT_NAME, append_to_daily_log, daily_lock_path, ensure_memory_dir
from firstcoder.memory.models import MemoryEvidence, MemoryNote
from firstcoder.memory.paths import ensure_no_link_or_junction, validate_memory_root
from firstcoder.memory.provenance import (
    apply_evidence_staleness,
    compute_anchor_hash,
    source_path_for_evidence,
    workspace_fingerprint,
)
from firstcoder.memory.security import should_quarantine
from firstcoder.memory.write import atomic_write_bytes, cross_process_lock

DURABLE_TOPIC_DEFAULTS = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}

#: topic slug 必须是安全目录名：字母数字 + 下划线/连字符，拒绝路径分隔符、
#: 点、空格与 Windows 保留名（Codex P2 review #1：topic 路径无约束会逃逸
#: topics 目录）。
_TOPIC_SLUG_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def note_id_for(topic_slug: str, note_text: str) -> str:
    return hashlib.sha256(f"{topic_slug}\n{note_text}".encode("utf-8")).hexdigest()[:12]


def _fold_note_text(note_text: object) -> str:
    """多行 note 折叠为单行：topic 文件的 `- ` 行格式不允许内嵌换行。

    折叠必须在 note_id 计算前统一执行——promote 与 upsert 共用
    （Codex P2 review #3：原实现两处不一致导致 upsert 的 evidence 丢失）。
    """
    return " ".join(str(note_text or "").split()).strip()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _tokenize(text: str) -> set[str]:
    """分词：ASCII 词 + 中文连续块 bigram（与 retrieval 一致，Codex P2 review #6）。

    中文 subject（如"单元测试"）经 2-gram 切分后可以稳定比较，supersession
    对中文主题生效。
    """
    raw = str(text)
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", raw)}
    for block in re.findall(r"[一-鿿]+", raw):
        if len(block) == 1:
            tokens.add(block)
        else:
            for i in range(len(block) - 1):
                tokens.add(block[i : i + 2])
    return tokens


class DurableMemoryStore:
    def __init__(self, root: str | Path, workspace_root: str | Path | None = None) -> None:
        if workspace_root is not None:
            # P1（Codex P2 review #1）：root 必须在 workspace 内，越界立即拒绝——
            # 否则 memory 写入可被导向任意目录。
            self.root = validate_memory_root(Path(root), workspace_root)
            self.workspace_root = str(workspace_root)
        else:
            self.root = Path(root)
            self.workspace_root = None
        self.index_path = self.root / ENTRYPOINT_NAME
        self.topics_dir = self.root / "topics"
        self.lock_path = self.root / ".store.lock"

    @staticmethod
    def _check_topic_slug(topic: object) -> str:
        """topic slug 规范化 + 校验（Codex P2 review #1，P1 项）。

        契约：strip 首尾空白后必须是安全目录名（`[A-Za-z0-9_-]+`）且不是
        Windows 保留名——topic 会拼进 `topics/<topic>.md` 路径，不校验就
        是目录逃逸口。首尾空白属于规范化（返回 strip 后的值），内容里的
        空格/分隔符/点则是拒绝。
        """
        value = str(topic or "").strip()
        if not _TOPIC_SLUG_PATTERN.fullmatch(value):
            raise ValueError(f"topic slug {value!r} is not a safe directory name")
        if value.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"topic slug {value!r} is a Windows reserved name")
        return value

    # --- 事务边界 -------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """跨进程互斥 + 目录就绪：promote 等复合写入的锁边界。

        锁不可重入（portalocker LOCK_EX）：事务内一律使用 *_unlocked
        内部变体读，公共读取 API 各自持锁调用这些变体。

        写前守卫：root/topics/logs 目录若被预置为 symlink/junction，原子
        写会被导向 workspace 外（Codex P2 review #1，P1 项）——拒绝写入。
        """
        ensure_memory_dir(self.root)
        ensure_no_link_or_junction(self.root / "topics")
        ensure_no_link_or_junction(self.root / "logs")
        with cross_process_lock(self.lock_path):
            yield

    # --- 路径与原始读写 ---------------------------------------------------

    def _topic_path(self, topic: str) -> Path:
        return self.topics_dir / f"{self._check_topic_slug(topic)}.md"

    def _metadata_path(self, topic: str) -> Path:
        return self.topics_dir / f"{self._check_topic_slug(topic)}.metadata.jsonl"

    # --- 索引 -------------------------------------------------------------

    def topic_slugs(self) -> list[str]:
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self) -> list[dict]:
        with cross_process_lock(self.lock_path):
            return self._load_index_unlocked()

    def _load_index_unlocked(self) -> list[dict]:
        """持锁读取的 index 解析（事务内调用，外部用 `load_index`）。"""
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics = []
        current = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def index_version(self) -> int:
        """当前 index 版本号（M3 版本号/CAS 升级的版本戳，0 表示无 index）。"""
        with cross_process_lock(self.lock_path):
            return self._index_version_unlocked()

    def _index_version_unlocked(self) -> int:
        if not self.index_path.exists():
            return 0
        for raw in self.index_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"- version:\s*(\d+)", raw.strip())
            if match:
                return int(match.group(1))
        return 0

    def _write_index(self, topics: list[dict], version: int) -> None:
        ensure_memory_dir(self.root)
        lines = ["# Durable Memory Index", f"- version: {version}", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        atomic_write_bytes(self.index_path, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))

    # --- topic 与 metadata --------------------------------------------------

    def _load_topic_metadata(self, topic: str) -> dict[str, dict]:
        path = self._metadata_path(topic)
        if not path.exists():
            return {}
        rows: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            note_id = str(row.get("note_id", "")).strip()
            if note_id:
                rows[note_id] = row
        return rows

    def _write_topic_metadata(self, topic: str, rows: dict[str, dict]) -> None:
        ensure_memory_dir(self.root)
        ordered = sorted(rows.values(), key=lambda row: str(row.get("note_id", "")))
        lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in ordered]
        text = "\n".join(lines).rstrip() + ("\n" if lines else "")
        atomic_write_bytes(self._metadata_path(topic), text.encode("utf-8"))

    def _scope(self) -> str:
        """scope 值：提供 workspace_root 时用真实 fingerprint（跨 workspace
        隔离由检索比较 fingerprint 实现）；否则用字面量标记"按 workspace
        限定"（旧数据/无 workspace 上下文的兼容值）。"""
        if self.workspace_root is not None:
            return workspace_fingerprint(self.workspace_root)
        return "workspace_fingerprint"

    def _default_note_metadata(self, topic: str, note_text: str, topic_path: Path | None = None) -> dict:
        topic_path = Path(topic_path) if topic_path is not None else self._topic_path(topic)
        created_at = (
            datetime.fromtimestamp(topic_path.stat().st_mtime).astimezone().isoformat()
            if topic_path.exists()
            else now_iso()
        )
        return {
            "note_id": note_id_for(topic, note_text),
            "status": "active",
            "supersedes": None,
            "evidence": {
                "session_id": "legacy",
                "source_path": None,
                "created_at": created_at,
                "evidence_anchor_hash": None,
            },
            "scope": self._scope(),
        }

    def _metadata_for_note(self, topic: str, note_text: str, metadata: dict, topic_path: Path | None = None) -> dict:
        note_id = note_id_for(topic, note_text)
        row = dict(metadata.get(note_id) or self._default_note_metadata(topic, note_text, topic_path=topic_path))
        row["note_id"] = note_id
        row.setdefault("status", "active")
        row.setdefault("supersedes", None)
        default_evidence = self._default_note_metadata(topic, note_text, topic_path=topic_path)["evidence"]
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        default_evidence.update(evidence)
        # 证据锚点默认生成（pico memory.py:853-857，Codex P2 review #8）：
        # source_path 存在且 anchor 缺失时按当前文件内容自动计算——内容在
        # 捕获后变更会立即表现为 stale_evidence。
        if not default_evidence.get("evidence_anchor_hash") and default_evidence.get("source_path"):
            anchor = compute_anchor_hash(
                source_path_for_evidence(self.workspace_root, default_evidence.get("source_path"))
            )
            if anchor:
                default_evidence["evidence_anchor_hash"] = anchor
        row["evidence"] = default_evidence
        row.setdefault("scope", self._scope())
        return row

    def load_topic_notes(self, topic: str) -> list[dict]:
        with cross_process_lock(self.lock_path):
            return self._load_topic_notes_unlocked(topic)

    def _load_topic_notes_unlocked(self, topic: str) -> list[dict]:
        """持锁读取 topic 笔记（事务内调用，外部用 `load_topic_notes`）。

        惰性 metadata 回填与 `promote`/`upsert_topic` 的写互斥（同一把
        store 锁），不会覆盖并发写入的 evidence（Codex P2 review #3）。
        """
        path = self._topic_path(topic)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = self._load_topic_metadata(topic)
        metadata_exists = self._metadata_path(topic).exists()
        metadata_changed = False
        notes = []
        capture = False
        updated_at = ""
        tags = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                notes.append(
                    {
                        "text": line[2:].strip(),
                        "tags": tags,
                        "source": topic,
                        "created_at": updated_at or now_iso(),
                        "kind": "durable",
                    }
                )
        for note in notes:
            row = self._metadata_for_note(topic, note["text"], metadata, topic_path=path)
            note.update(row)
            if row["note_id"] not in metadata or not metadata_exists:
                metadata_changed = True
            metadata[row["note_id"]] = row
        if metadata_changed:
            self._write_topic_metadata(topic, metadata)
        return notes

    @staticmethod
    def _subject_key(text: str) -> str | None:
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(_tokenize(match.group(1)))
                return subject or None
        return None

    # --- 写操作 --------------------------------------------------------------

    def promote(self, promotions: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        """把 (topic, note_text) 提升为 durable 笔记，返回 (results, superseded)。

        一致性模型（M3 版本号/CAS 升级，Codex P2 review #1）：
        - 整个提升在同一把 store 锁（`_transaction`）内完成，读写互斥；
        - 对每个 topic：topic 文件先落盘、metadata 后落盘（`_write_topic`
          内部顺序），两者之间的崩溃窗口由持锁读取的惰性 metadata 回填
          自愈——读取永远能得到与 topic 文件一致的新 metadata；
        - index 最后发布为全局提交点：读者持锁要么看到旧 index（新 topic
          未注册，等价旧版本），要么看到含新 topic 的完整新 index；
        - index 带递增版本号（`- version: N`），供外部检测代次；
        - 崩溃（进程被杀）可能留下已写但未注册的 topic/metadata，不会
          产生撕裂可见性；index 未推进时，下一次 promote（含重复 note）
          会把 index 补发到最新版本（自愈）。
        """
        if not promotions:
            return [], []
        with self._transaction():
            topics = {topic["topic"]: topic for topic in self._load_index_unlocked()}
            topic_notes = {
                slug: [note["text"] for note in self._load_topic_notes_unlocked(slug)] for slug in topics
            }
            topic_metadata = {slug: self._load_topic_metadata(slug) for slug in topics}
            results = []
            superseded = []
            for topic, note_text in promotions:
                topic = self._check_topic_slug(topic)
                # 多行 note 折叠为单行：note_id 必须基于折叠后的文本
                # （与 `_apply_note_metadata` 共用同一 helper，见该函数）。
                note_text = _fold_note_text(note_text)
                if not note_text:
                    continue
                meta = DURABLE_TOPIC_DEFAULTS[topic]
                topics.setdefault(
                    topic,
                    {
                        "topic": topic,
                        "title": meta["title"],
                        "summary": meta["summary"],
                        "tags": list(meta["tags"]),
                    },
                )
                existing = topic_notes.setdefault(topic, [])
                metadata = topic_metadata.setdefault(topic, {})
                if note_text in existing:
                    continue
                # quarantine 判定先于 supersession（Codex P2 review #7）：
                # 恶意/secret-shaped 新笔记不得先把有效旧笔记标记 superseded
                # 再把自己隔离——那会让旧记忆不可检索。
                quarantined = should_quarantine(note_text)
                new_subject = None if quarantined else self._subject_key(note_text)
                replaced = False
                supersedes = None
                if new_subject:
                    for index, old_text in enumerate(list(existing)):
                        if self._subject_key(old_text) == new_subject:
                            superseded.append(f"{topic}: {old_text} -> {note_text}")
                            old_id = note_id_for(topic, old_text)
                            old_meta = self._metadata_for_note(topic, old_text, metadata)
                            old_meta["status"] = "superseded"
                            metadata[old_id] = old_meta
                            supersedes = old_id
                            existing[index] = note_text
                            replaced = True
                            break
                if not replaced:
                    existing.append(note_text)
                new_meta = self._metadata_for_note(topic, note_text, metadata)
                new_meta["status"] = "quarantined" if quarantined else "active"
                new_meta["supersedes"] = None if quarantined else supersedes
                metadata[new_meta["note_id"]] = new_meta
                results.append(f"{topic}: {note_text}")
            # 写顺序：metadata/topic 先落盘，index 最后发布为提交点。
            for topic, notes in topic_notes.items():
                self._write_topic(topic, notes, metadata=topic_metadata.get(topic, {}))
            self._write_index([topics[slug] for slug in sorted(topics)], self._index_version_unlocked() + 1)
            return results, superseded

    def _write_topic(self, topic: str, notes: list[str], metadata: dict | None = None) -> None:
        ensure_memory_dir(self.root)
        meta = DURABLE_TOPIC_DEFAULTS[topic]
        lines = [
            f"# {meta['title']}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta['summary']}",
            f"- tags: {', '.join(meta['tags'])}",
            f"- updated_at: {now_iso()}",
            "",
            "## Notes",
        ]
        for note in notes:
            lines.append(f"- {note}")
        path = self._topic_path(topic)
        atomic_write_bytes(path, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))
        metadata = dict(metadata or self._load_topic_metadata(topic))
        for note in notes:
            row = self._metadata_for_note(topic, note, metadata, topic_path=path)
            metadata[row["note_id"]] = row
        self._write_topic_metadata(topic, metadata)

    # --- P0 MemoryStorePort 形状 ----------------------------------------------

    def append_daily_log(self, text: str, *, source: MemoryEvidence | None = None) -> Path | None:
        """写每日日志，并把捕获时刻的 provenance 记到当天的 evidence 侧车。

        evidence 侧车是 FirstCoder 对 pico 格式的扩展：pico 的 daily log
        不保留捕获证据，promote 时无从追溯。P3 的 /remember 从
        `load_daily_log_evidence` 取回 session/source/anchor，填进 durable
        metadata 的 evidence。

        委托 logs.append_to_daily_log：日志行与侧车行在同一把 `.daily.lock`
        内读改写（Codex P2 review #4：与 logs 层 API 共用同一把锁，两种
        入口混用时行序一致）；进程崩溃可能留下"日志已写、侧车未写"的窗口
        （调用方会收到异常），但不会有并发撕裂。
        """
        return append_to_daily_log(self.root, text, source=source)

    def load_daily_log_evidence(self, today: "date | None" = None) -> list[dict]:
        """读取当天 evidence 侧车（P3 /remember 的证据来源）。

        返回侧车行列表；无侧车文件返回空列表。持 `.daily.lock` 读取，
        与写入互斥（Codex P2 review #4）。
        """
        from firstcoder.memory.logs import daily_log_path

        path = daily_log_path(self.root, today=today)
        evidence_path = path.with_name(path.stem + ".evidence.jsonl")
        if not evidence_path.exists():
            return []
        rows = []
        with cross_process_lock(daily_lock_path(self.root)):
            for line in evidence_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def upsert_topic(self, note: MemoryNote) -> None:
        """按契约形状的单一笔记写入（promote 的面向契约入口）。

        quarantine 判定由 promote 统一负责（与 pico 语义一致）：
        这里只补 evidence / supersedes，不覆盖 status。
        """
        self.promote([(note.topic, note.text)])
        self._apply_note_metadata(note)

    def _apply_note_metadata(self, note: MemoryNote) -> None:
        """把契约笔记里的 evidence/supersedes/scope 覆盖到 metadata 行。

        note_id 基于折叠后的文本查找（与 `promote` 的落盘文本一致，
        Codex P2 review #3：原实现用原始文本找行，多行 note 的 evidence
        静默丢失）。
        """
        with self._transaction():
            metadata = self._load_topic_metadata(note.topic)
            row = metadata.get(note_id_for(note.topic, _fold_note_text(note.text)))
            if row is None:
                return
            if note.evidence.session_id or note.evidence.source_path or note.evidence.scope:
                evidence = dict(row.get("evidence") or {})
                if note.evidence.session_id:
                    evidence["session_id"] = note.evidence.session_id
                if note.evidence.source_path:
                    evidence["source_path"] = note.evidence.source_path
                if note.evidence.anchor_hash:
                    evidence["evidence_anchor_hash"] = note.evidence.anchor_hash
                # 锚点缺失时按 source 文件当前内容自动生成（同
                # `_metadata_for_note`，Codex P2 review #8）。
                if not evidence.get("evidence_anchor_hash") and evidence.get("source_path"):
                    anchor = compute_anchor_hash(
                        source_path_for_evidence(self.workspace_root, evidence.get("source_path"))
                    )
                    if anchor:
                        evidence["evidence_anchor_hash"] = anchor
                row["evidence"] = evidence
            # scope 落在 row 顶层（`_default_note_metadata` 约定）；
            # 契约里的 scope（如 "global"）必须持久化（Codex P2 review #3）。
            if note.evidence.scope:
                row["scope"] = note.evidence.scope
            if note.supersedes:
                row["supersedes"] = note.supersedes
            self._write_topic_metadata(note.topic, metadata)

    def read_index(self) -> list[MemoryNote]:
        """契约形状的索引读取：topic -> 活跃/全部笔记（含 metadata）。

        单锁内完成（index + 各 topic 的读取），避免两次独立锁之间插入写。
        """
        with cross_process_lock(self.lock_path):
            notes: list[MemoryNote] = []
            for topic in self._load_index_unlocked():
                for note in self._load_topic_notes_unlocked(topic["topic"]):
                    evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
                    notes.append(
                        MemoryNote(
                            topic=topic["topic"],
                            text=note.get("text", ""),
                            note_id=str(note.get("note_id", "")),
                            status=note.get("status", "active"),
                            supersedes=str(note.get("supersedes") or ""),
                            evidence=MemoryEvidence(
                                source_path=str(evidence.get("source_path") or ""),
                                session_id=str(evidence.get("session_id") or ""),
                                anchor_hash=str(evidence.get("evidence_anchor_hash") or ""),
                                # scope 落在 metadata row 顶层（`_default_note_metadata`），
                                # 不在 evidence dict 里——契约映射从这里回填。
                                scope=str(note.get("scope") or evidence.get("scope") or "workspace"),
                            ),
                            created_at=str(note.get("created_at", "")),
                        )
                    )
        return notes

    def snapshot(self, workspace_root: str | Path | None = None) -> list[dict]:
        """单一快照读：单锁内读 index + 全部 topic 笔记（含 metadata 回填）。

        Codex P2 review #5：一次 retrieval 若分多次锁读（先 index 再逐
        topic），锁间可插入一次 promote，导致同一查询看到不同代次的数据；
        retrieval 应使用本方法代替逐锁读取。返回原始 dict 形状并应用
        evidence staleness 判定。
        """
        with cross_process_lock(self.lock_path):
            notes: list[dict] = []
            for topic in self._load_index_unlocked():
                for note in self._load_topic_notes_unlocked(topic["topic"]):
                    notes.append(apply_evidence_staleness(dict(note), workspace_root))
            return notes
