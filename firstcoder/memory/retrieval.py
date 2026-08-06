"""Memory retrieval: query -> ranked notes with an audit trail (fusion P2, M4).

Ported from pico `features/memory.py:1421-1510`. The ranking stays simple
and transparent: exact tag hit *1000, keyword overlap *10, recency, then
index order — no embeddings (M4). The durable store is injected rather
than hardcoded to a pico path, and output follows the P0 `RetrievalResult`
contract (selected/rejected with reject_reason + score).
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
    RetrievalResult,
    RetrievalSelection,
)
from firstcoder.memory.provenance import apply_evidence_staleness, workspace_fingerprint


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
        ),
        created_at=str(note.get("created_at", "")),
    )


def _retrieval_reject_reason(note: dict, workspace_root: str | None = None) -> str:
    status = str(note.get("status", "active")).strip() or "active"
    if status == "quarantined":
        return "quarantined"
    if status == "superseded":
        return "superseded"
    if bool(note.get("stale_evidence")):
        return "stale_evidence"
    scope = str(note.get("scope", "")).strip()
    if scope and scope != "global":
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
    """

    def __init__(
        self,
        store: DurableMemoryStore | None = None,
        state: dict | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.store = store
        self.state = dict(state or {})
        self.workspace_root = workspace_root

    def _iter_notes(self) -> Any:
        for note in self.state.get("episodic_notes", []):
            yield dict(note)
        if self.store is not None:
            for topic in self.store.load_index():
                for note in self.store.load_topic_notes(topic["topic"]):
                    yield apply_evidence_staleness(dict(note), self.workspace_root)

    def _ranked(self, query: str) -> list[tuple[tuple[int, int, float, int], float, dict]]:
        query_tokens = _tokenize(query)
        ranked = []
        for note in self._iter_notes():
            note_tags = {tag.lower() for tag in note.get("tags", [])}
            note_tokens = _tokenize(note.get("text", "")) | _tokenize(note.get("source", "")) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            if exact_tag_match == 0 and keyword_overlap == 0:
                continue
            recency = _parse_timestamp(note.get("created_at"))
            note_index = int(note.get("note_index", 0))
            # 排序以 tuple 键 (exact_tag, keyword_overlap, recency, note_index)
            # 为准；score 是与排序方向一致的启发式审计值（Codex P2 review #8）：
            # recency 归一化到 <10，避免 epoch 秒数（~1.7e9）盖过 keyword 权重，
            # 让审计里"分数更高"始终对应"排序更靠前"。
            score = (
                exact_tag_match * 1000
                + keyword_overlap * 10
                + min(recency / 1_000_000_000, 0.999)
                + min(note_index / 1_000_000_000, 0.001)
            )
            ranked.append(((exact_tag_match, keyword_overlap, recency, note_index), score, note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        selected: list[RetrievalSelection] = []
        rejected: list[RetrievalSelection] = []
        for _, score, note in self._ranked(query.text):
            reject_reason = _retrieval_reject_reason(note, self.workspace_root)
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
        return RetrievalResult(
            query=query,
            selections=selected + rejected,
            query_hash=_query_hash(query.text),
        )
