"""Memory retrieval: query -> ranked notes with an audit trail (fusion P2, M4).

Ported from pico `features/memory.py:1421-1510`. The ranking stays simple
and transparent: exact tag hit *1000, keyword overlap *10, recency, then
index order — no embeddings (M4). The durable store is injected rather
than hardcoded to a pico path, and output follows the P0 `RetrievalResult`
contract (selected/rejected with reject_reason + score).

审计一致性（Codex P2 review #6/#8）：score 是与排序 tuple 键严格同向的
审计值——exact tag 恒高于 keyword 重叠（keyword 分量封顶），recency /
note_index 分量归一化后不跨界、保持单调；`selections` 全局按 score 降序，
高分 rejected（如 quarantine）排在低分 selected 之前，与排序序一致。

快照读（Codex P2 review #5）：durable 数据经 `store.snapshot()` 单锁
读入，一次查询不会看到不同代次的数据；episodic_notes 来自内存 state，
无锁问题。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from firstcoder.memory.durable import DurableMemoryStore, note_id_for
from firstcoder.memory.models import (
    MemoryEvidence,
    MemoryNote,
    MemoryQuery,
    MEMORY_VISIBILITIES,
    RetrievalResult,
    RetrievalSelection,
)
from firstcoder.memory.provenance import workspace_fingerprint


def _tokenize(text: str) -> set[str]:
    """分词：ASCII 词 + 中文连续块按 bigram 切分（Codex P2 review #6）。

    连续中文整段作一个 token 时，query 子串（如"测试" vs 笔记里的
    "单元测试"）无法重叠召回；按 2-gram 切分让子串匹配成为可能。
    单字块保留原字（"是"等虚词可能带来少量假重叠，可接受）。
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


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _query_hash(query: str) -> str:
    import hashlib

    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()[:12]


def _note_to_contract(note: dict) -> MemoryNote:
    evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
    return MemoryNote(
        topic=str(note.get("source", "")),
        text=str(note.get("text", "")),
        note_id=str(note.get("note_id", "") or note_id_for(str(note.get("source", "")), str(note.get("text", "")))),
        status=str(note.get("status", "active")),
        supersedes=str(note.get("supersedes") or ""),
        evidence=MemoryEvidence(
            source_path=str(evidence.get("source_path") or ""),
            session_id=str(evidence.get("session_id") or ""),
            anchor_hash=str(evidence.get("evidence_anchor_hash") or ""),
            # scope 在 note dict 顶层（metadata row），不在 evidence dict 里。
            scope=str(note.get("scope") or evidence.get("scope") or "workspace"),
            # 统一经过 legacy 推断，避免旧 metadata 或非法值在契约对象中
            # 伪装成一个调用方无法处理的 visibility。
            visibility=_legacy_visibility(note),
        ),
        created_at=str(note.get("created_at", "")),
    )


def _legacy_visibility(note: dict) -> str:
    """Infer visibility for P2 rows that predate the explicit field."""

    value = str(note.get("visibility") or "").strip()
    if value in MEMORY_VISIBILITIES:
        return value
    if str(note.get("scope") or "").strip() == "global":
        return "global"
    return "workspace"


def _retrieval_reject_reason(
    note: dict,
    workspace_root: str | None = None,
    *,
    session_id: str = "",
    include_global: bool = False,
) -> str:
    status = str(note.get("status", "active")).strip() or "active"
    if status == "quarantined":
        return "quarantined"
    if status == "superseded":
        return "superseded"
    if bool(note.get("stale_evidence")):
        return "stale_evidence"
    visibility = _legacy_visibility(note)
    if visibility == "global":
        if not include_global:
            return "global_disabled"
    elif visibility == "session":
        evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
        note_session_id = str(evidence.get("session_id") or note.get("session_id") or "")
        if not session_id or note_session_id != session_id:
            return "session_mismatch"
    elif visibility not in {"workspace"}:
        return "scope_mismatch"

    # workspace fingerprint remains a separate compatibility/access field;
    # visibility must not overwrite its meaning.
    scope = str(note.get("scope", "")).strip()
    if visibility in {"session", "workspace"} and scope and scope != "global":
        if scope == "workspace_fingerprint":
            pass  # 字面量标记（无 workspace 上下文写入的兼容值）：不比较
        elif workspace_root is not None and re.fullmatch(r"[0-9a-f]{12}", scope):
            # 真实 fingerprint：必须与当前 workspace 一致，否则跨 workspace
            # 记忆泄漏（Codex P2 review #5）。
            if scope != workspace_fingerprint(workspace_root):
                return "scope_mismatch"
        else:
            return "scope_mismatch"
    if bool(note.get("scope_mismatch")):
        return "scope_mismatch"
    return ""


class MemoryRetriever:
    """Ranked retrieval over durable notes (P0 `MemoryRetrievalPort` shape).

    `state` is the working-memory dict (M2); when present, its
    `episodic_notes` are folded into the candidate set like pico's
    state-based retrieval.

    用法：可只传 `state`（episodic 检索），也可传 `store`（durable 检索）；
    `workspace_root` 用于 fingerprint scope 比较与 evidence staleness
    判定（与 `promote` 写 scope 时的 workspace 一致）；`session_id` 用于
    session-only 记忆过滤；global store 只有 query 明确 include_global 时
    才会读取。
    """

    def __init__(
        self,
        store: DurableMemoryStore | None = None,
        state: dict | None = None,
        workspace_root: str | None = None,
        session_id: str = "",
        global_store: DurableMemoryStore | None = None,
    ) -> None:
        self.store = store
        self.global_store = global_store
        self.state = dict(state or {})
        self.session_id = str(session_id or "")
        # 未显式传 workspace_root 时继承 store 的上下文（Codex P2 review #3）：
        # 配置了 workspace_root 的 store 不应在检索时被"无上下文"重解析
        # （fingerprint 比较会误判 scope_mismatch）。
        self.workspace_root = (
            workspace_root if workspace_root is not None else getattr(store, "workspace_root", None)
        )

    def _iter_notes(self, *, include_global: bool = False) -> Any:
        for note in self.state.get("episodic_notes", []):
            yield dict(note)
        if self.store is not None:
            # 单一快照（Codex P2 review #5）：单锁内读完 index + 全部 topic，
            # 避免一次查询跨锁读到不同代次的数据。
            for note in self.store.snapshot(self.workspace_root):
                yield note
        if include_global and self.global_store is not None and self.global_store is not self.store:
            # Global notes have no workspace evidence root. They are read only
            # after the caller explicitly opts in through MemoryQuery.
            for note in self.global_store.snapshot(None):
                yield note

    def visible_notes(self, query: MemoryQuery | None = None) -> list[MemoryNote]:
        """返回当前上下文允许读取的全部 note，不执行关键词排名。

        ``/memory`` 需要列出可见内容，但不能把 ``MEMORY.md`` 当成权限边界：
        index 只知道 topic，不知道 session/global 过滤结果。这个公开方法复用
        与普通检索完全相同的拒绝规则，作为命令和其他展示层的统一读取入口。
        """

        context = query or MemoryQuery(text="")
        session_id = context.session_id or self.session_id
        visible: list[MemoryNote] = []
        for note in self._iter_notes(include_global=context.include_global):
            reject_reason = _retrieval_reject_reason(
                note,
                self.workspace_root,
                session_id=session_id,
                include_global=context.include_global,
            )
            if reject_reason == "quarantined" and context.include_quarantined:
                reject_reason = ""
            if not reject_reason:
                visible.append(_note_to_contract(note))
        return visible

    def _ranked(
        self,
        query: str,
        *,
        include_global: bool = False,
    ) -> list[tuple[tuple[int, int, float, int], float, dict]]:
        query_tokens = _tokenize(query)
        ranked = []
        for note in self._iter_notes(include_global=include_global):
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            if exact_tag_match == 0 and keyword_overlap == 0:
                continue
            recency = _parse_timestamp(note.get("created_at"))
            note_index = int(note.get("note_index", 0))
            # 排序以 tuple 键 (exact_tag, keyword_overlap, recency, note_index)
            # 为准；score 是与排序严格同向的可读审计值（Codex P2 review #8）：
            # - exact 1000 > keyword 封顶 99 个 = 990（重叠再多也压不过 exact）；
            # - keyword 单位 10 > recency 归一化全范围（~0.002，永不封顶，
            #   现代时间戳之间可区分）；
            # - 固定加权无法对 (recency, note_index) 保序嵌入（秒级 recency
            #   差异 1e-12 恒小于任意 note_index 权重），因此 note_index 不进
            #   score——它只作排序 tiebreak；同分条目的先后由排序键决定。
            score = (
                exact_tag_match * 1000
                + min(keyword_overlap, 99) * 10
                + min(recency / 1_000_000_000_000, 0.999)
            )
            ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        selected: list[RetrievalSelection] = []
        rejected: list[RetrievalSelection] = []
        for _, score, note in self._ranked(query.text, include_global=query.include_global):
            reject_reason = _retrieval_reject_reason(
                note,
                self.workspace_root,
                session_id=query.session_id or self.session_id,
                include_global=query.include_global,
            )
            if reject_reason == "quarantined" and query.include_quarantined:
                reject_reason = ""
            if reject_reason:
                rejected.append(
                    RetrievalSelection(
                        note=_note_to_contract(note),
                        selected=False,
                        reject_reason=reject_reason,
                        score=score,
                    )
                )
                continue
            if len(selected) < int(query.limit):
                selected.append(RetrievalSelection(note=_note_to_contract(note), selected=True, score=score))
            else:
                rejected.append(
                    RetrievalSelection(
                        note=_note_to_contract(note),
                        selected=False,
                        reject_reason="below_limit",
                        score=score,
                    )
                )
        # 审计 trail 全局按 score 降序（Codex P2 review #8）：高分 rejected
        # （quarantine/stale 等）排在低分 selected 之前，与排序序一致。
        selections = sorted(selected + rejected, key=lambda s: s.score, reverse=True)
        return RetrievalResult(
            query=query,
            selections=selections,
            query_hash=_query_hash(query.text),
        )
