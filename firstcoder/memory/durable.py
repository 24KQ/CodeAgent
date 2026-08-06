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
from datetime import datetime
from pathlib import Path
from typing import Iterator

from firstcoder.memory.logs import ENTRYPOINT_NAME, ensure_memory_dir
from firstcoder.memory.models import MemoryEvidence, MemoryNote
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


def note_id_for(topic_slug: str, note_text: str) -> str:
    return hashlib.sha256(f"{topic_slug}\n{note_text}".encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


class DurableMemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / ENTRYPOINT_NAME
        self.topics_dir = self.root / "topics"
        self.lock_path = self.root / ".store.lock"

    # --- 事务边界 -------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """跨进程互斥 + 目录就绪：promote 等复合写入的锁边界。"""
        ensure_memory_dir(self.root)
        with cross_process_lock(self.lock_path):
            yield

    # --- 路径与原始读写 ---------------------------------------------------

    def _topic_path(self, topic: str) -> Path:
        return self.topics_dir / f"{topic}.md"

    def _metadata_path(self, topic: str) -> Path:
        return self.topics_dir / f"{topic}.metadata.jsonl"

    # --- 索引 -------------------------------------------------------------

    def topic_slugs(self) -> list[str]:
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self) -> list[dict]:
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

    def _write_index(self, topics: list[dict]) -> None:
        ensure_memory_dir(self.root)
        lines = ["# Durable Memory Index", ""]
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
            "scope": "workspace_fingerprint",
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
        row["evidence"] = default_evidence
        row.setdefault("scope", "workspace_fingerprint")
        return row

    def load_topic_notes(self, topic: str) -> list[dict]:
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
            # 惰性回填是确定性的（同样的 note_id/row），并发写只会写出等价内容。
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

        整个提升在同一事务里完成：index + topic + metadata 要么全部可见，要么
        保持旧版本（每个文件本身是原子替换，跨文件一致性由锁保证）。
        """
        if not promotions:
            return [], []
        with self._transaction():
            topics = {topic["topic"]: topic for topic in self.load_index()}
            topic_notes = {slug: [note["text"] for note in self.load_topic_notes(slug)] for slug in topics}
            topic_metadata = {slug: self._load_topic_metadata(slug) for slug in topics}
            results = []
            superseded = []
            for topic, note_text in promotions:
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
                new_subject = self._subject_key(note_text)
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
                new_meta["status"] = "active"
                if should_quarantine(note_text):
                    new_meta["status"] = "quarantined"
                new_meta["supersedes"] = supersedes
                metadata[new_meta["note_id"]] = new_meta
                results.append(f"{topic}: {note_text}")
            self._write_index([topics[slug] for slug in sorted(topics)])
            for topic, notes in topic_notes.items():
                self._write_topic(topic, notes, metadata=topic_metadata.get(topic, {}))
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
        不保留捕获证据，promote 时无从追溯。P3 的 /remember 从这里取回
        session/source/anchor，填进 durable metadata 的 evidence。
        """
        from firstcoder.memory.logs import append_to_daily_log as _append

        path = _append(self.root, text)
        if path is None or source is None:
            return path
        row = {
            "text": str(text).strip(),
            "session_id": source.session_id,
            "source_path": source.source_path,
            "anchor_hash": source.anchor_hash,
            "scope": source.scope,
            "at": now_iso(),
        }
        evidence_path = path.with_name(path.stem + ".evidence.jsonl")
        with cross_process_lock(self.root / ".evidence.lock"):
            existing = evidence_path.read_text(encoding="utf-8") if evidence_path.exists() else ""
            lines = [line for line in existing.splitlines() if line.strip()]
            lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            atomic_write_bytes(evidence_path, ("\n".join(lines) + "\n").encode("utf-8"))
        return path

    def upsert_topic(self, note: MemoryNote) -> None:
        """按契约形状的单一笔记写入（promote 的面向契约入口）。

        quarantine 判定由 promote 统一负责（与 pico 语义一致）：
        这里只补 evidence / supersedes，不覆盖 status。
        """
        self.promote([(note.topic, note.text)])
        self._apply_note_metadata(note)

    def _apply_note_metadata(self, note: MemoryNote) -> None:
        """把契约笔记里的 evidence/supersedes 覆盖到 metadata 行。"""
        with self._transaction():
            metadata = self._load_topic_metadata(note.topic)
            row = metadata.get(note_id_for(note.topic, note.text))
            if row is None:
                return
            if note.evidence.session_id or note.evidence.source_path:
                evidence = dict(row.get("evidence") or {})
                if note.evidence.session_id:
                    evidence["session_id"] = note.evidence.session_id
                if note.evidence.source_path:
                    evidence["source_path"] = note.evidence.source_path
                if note.evidence.anchor_hash:
                    evidence["evidence_anchor_hash"] = note.evidence.anchor_hash
                row["evidence"] = evidence
            if note.supersedes:
                row["supersedes"] = note.supersedes
            self._write_topic_metadata(note.topic, metadata)

    def read_index(self) -> list[MemoryNote]:
        """契约形状的索引读取：topic -> 活跃/全部笔记（含 metadata）。"""
        notes: list[MemoryNote] = []
        for topic in self.load_index():
            for note in self.load_topic_notes(topic["topic"]):
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
