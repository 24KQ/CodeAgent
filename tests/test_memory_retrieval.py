"""P2 tests: memory retrieval with audit trail (retrieval.py, fusion M4)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from firstcoder.memory.durable import DurableMemoryStore, note_id_for
from firstcoder.memory.models import MemoryEvidence, MemoryNote, MemoryQuery
from firstcoder.memory.retrieval import MemoryRetriever


def _state_note(text: str, **extra) -> dict:
    note = {
        "text": text,
        "tags": [],
        "source": "",
        "created_at": "2026-08-06T00:00:00+00:00",
        "note_index": 0,
        "kind": "episodic",
    }
    note.update(extra)
    return note


def _retriever(state: dict | None = None, **kwargs) -> MemoryRetriever:
    return MemoryRetriever(state=state or {}, **kwargs)


def test_no_match_returns_empty() -> None:
    result = _retriever({"episodic_notes": [_state_note("nothing in common")]}).retrieve(
        MemoryQuery(text="pytest")
    )
    assert result.selections == []
    assert result.selected_notes == []


def test_exact_tag_beats_keyword_overlap() -> None:
    state = {
        "episodic_notes": [
            _state_note("pytest is fast", note_index=0),
            _state_note("关于 python 的一切", tags=["pytest"], note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    notes = [selection.note.text for selection in result.selections]
    assert notes[0] == "关于 python 的一切"  # exact tag 优先


def test_recency_breaks_ties() -> None:
    state = {
        "episodic_notes": [
            _state_note("old pytest note", created_at="2026-01-01T00:00:00+00:00", note_index=0),
            _state_note("new pytest note", created_at="2026-08-01T00:00:00+00:00", note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert result.selected_notes[0].text == "new pytest note"


def test_query_hash_is_deterministic() -> None:
    first = _retriever({}).retrieve(MemoryQuery(text="pytest"))
    second = _retriever({}).retrieve(MemoryQuery(text="pytest"))
    assert first.query_hash == second.query_hash
    assert len(first.query_hash) == 12


def test_quarantined_rejected_unless_requested() -> None:
    state = {"episodic_notes": [_state_note("pytest note", status="quarantined")]}
    default = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert default.selected_notes == []
    assert default.selections[0].reject_reason == "quarantined"

    included = _retriever(state).retrieve(MemoryQuery(text="pytest", include_quarantined=True))
    assert [s.text for s in included.selected_notes] == ["pytest note"]


def test_superseded_rejected() -> None:
    state = {"episodic_notes": [_state_note("old pytest note", status="superseded")]}
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert result.selected_notes == []
    assert result.selections[0].reject_reason == "superseded"


def test_scope_mismatch_rejected() -> None:
    state = {"episodic_notes": [_state_note("pytest note", scope="other-project")]}
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert result.selected_notes == []
    assert result.selections[0].reject_reason == "scope_mismatch"


def test_limit_caps_selected_and_rejects_rest() -> None:
    state = {
        "episodic_notes": [
            _state_note(f"pytest note {i}", note_index=i) for i in range(3)
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest", limit=1))
    assert len(result.selected_notes) == 1
    reasons = [s.reject_reason for s in result.selections if not s.selected]
    assert reasons == ["below_limit", "below_limit"]


def test_stale_evidence_rejected(tmp_path: Path) -> None:
    """anchor 与当前文件 hash 不一致 → stale_evidence 拒绝。"""
    target = tmp_path / "src" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("v1", encoding="utf-8")

    store = DurableMemoryStore(tmp_path / "memory")
    store.promote([("key-decisions", "pytest is the runner")])
    store.upsert_topic(
        MemoryNote(
            topic="key-decisions",
            text="pytest is the runner",
            evidence=MemoryEvidence(
                source_path="src/a.py",
                session_id="s1",
                anchor_hash=hashlib.sha256(b"v1").hexdigest(),
            ),
        )
    )
    target.write_text("v2", encoding="utf-8")  # 文件已变

    result = MemoryRetriever(store=store, workspace_root=str(tmp_path)).retrieve(
        MemoryQuery(text="pytest")
    )
    assert result.selected_notes == []
    assert result.selections[0].reject_reason == "stale_evidence"


def test_retrieves_from_durable_store(tmp_path: Path) -> None:
    store = DurableMemoryStore(tmp_path / "memory")
    store.promote([("key-decisions", "pytest is the test runner")])
    store.promote([("user-preferences", "dark mode preferred")])

    result = MemoryRetriever(store=store).retrieve(MemoryQuery(text="pytest"))
    assert [s.text for s in result.selected_notes] == ["pytest is the test runner"]
    # 契约形状：note_id / evidence 从 metadata 映射
    note = result.selected_notes[0]
    assert note.note_id == note_id_for("key-decisions", "pytest is the test runner")
    assert note.evidence.scope == "workspace_fingerprint"


def test_folds_episodic_and_durable(tmp_path: Path) -> None:
    store = DurableMemoryStore(tmp_path / "memory")
    store.promote([("key-decisions", "pytest is the runner")])
    state = {"episodic_notes": [_state_note("episodic pytest thought")]}
    result = MemoryRetriever(store=store, state=state).retrieve(MemoryQuery(text="pytest"))
    texts = {s.text for s in result.selected_notes}
    assert texts == {"pytest is the runner", "episodic pytest thought"}


def test_zh_query_recalls_zh_note() -> None:
    """中文 query 子串可召回（Codex P2 review #6：bigram 分词）。"""
    state = {"episodic_notes": [_state_note("单元测试很快")]}
    result = _retriever(state).retrieve(MemoryQuery(text="测试"))
    assert [s.text for s in result.selected_notes] == ["单元测试很快"]


def test_zh_exact_tag_and_recall() -> None:
    state = {
        "episodic_notes": [
            _state_note("完全无关的英文", note_index=0),
            _state_note("关于 Python 的一切", tags=["pytest"], note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert [s.text for s in result.selected_notes] == ["关于 Python 的一切"]


def test_scope_fingerprint_isolation(tmp_path: Path) -> None:
    """workspace A 的记忆在 workspace B 被拒（Codex P2 review #5）。"""
    ws_a = tmp_path / "ws_a"
    ws_b = tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    store = DurableMemoryStore(ws_a / ".firstcoder" / "memory", workspace_root=ws_a)
    store.promote([("key-decisions", "pytest is the runner")])

    # 同 workspace 命中
    same = MemoryRetriever(store=store, workspace_root=str(ws_a)).retrieve(MemoryQuery(text="pytest"))
    assert [s.text for s in same.selected_notes] == ["pytest is the runner"]

    # 跨 workspace 拒绝（scope_mismatch）
    other = MemoryRetriever(store=store, workspace_root=str(ws_b)).retrieve(MemoryQuery(text="pytest"))
    assert other.selected_notes == []
    assert other.selections[0].reject_reason == "scope_mismatch"


def test_global_scope_always_selected(tmp_path: Path) -> None:
    state = {"episodic_notes": [_state_note("pytest note", scope="global")]}
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert [s.text for s in result.selected_notes] == ["pytest note"]


def test_score_monotone_with_ranking() -> None:
    """score 与排序方向一致：exact tag 的分数高于 keyword-only（Codex P2 review #8）。"""
    state = {
        "episodic_notes": [
            _state_note("pytest is fast", note_index=0),
            _state_note("关于 python 的一切", tags=["pytest"], note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    scores = [s.score for s in result.selections]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= 1000
    assert scores[1] < 1000


def test_score_keyword_cap_below_exact_tag() -> None:
    """keyword 分量封顶：重叠 100+ 个词也压不过 exact tag（Codex P2 review #8）。"""
    many_words = " ".join(f"w{i}" for i in range(120))
    state = {
        "episodic_notes": [
            _state_note(many_words, note_index=0),
            _state_note("关于 python 的一切", tags=["pytest"], note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest " + many_words))
    scores = [s.score for s in result.selections]
    # 两个候选都命中（词重叠），但 exact tag 必须排最前且分数封顶不越界
    assert result.selected_notes[0].text == "关于 python 的一切"
    assert scores[0] >= 1000  # exact tag
    assert scores[1] < 1000  # 99 个词封顶：990 + 归一化分量

def test_score_recency_monotonic_modern_timestamps() -> None:
    """现代时间戳（epoch 秒 >1e9）之间 score 仍随 recency 单调（Codex P2 review #8）。"""
    state = {
        "episodic_notes": [
            _state_note("old pytest note", created_at="2026-01-01T00:00:00+00:00", note_index=0),
            _state_note("new pytest note", created_at="2026-08-01T00:00:00+00:00", note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    scores = [s.score for s in result.selections]
    assert scores[0] > scores[1]
    assert result.selected_notes[0].text == "new pytest note"


def test_selections_globally_ranked() -> None:
    """selections 全局按 score 降序：高分 rejected（quarantine）排在低分 selected 前。"""
    state = {
        "episodic_notes": [
            _state_note("quarantined pytest note", tags=["pytest"], status="quarantined", note_index=0),
            _state_note("clean pytest note", note_index=1),
        ]
    }
    result = _retriever(state).retrieve(MemoryQuery(text="pytest"))
    assert [s.text for s in result.selected_notes] == ["clean pytest note"]
    assert result.selections[0].note.text == "quarantined pytest note"  # 分数最高，排最前
    assert result.selections[0].reject_reason == "quarantined"
    scores = [s.score for s in result.selections]
    assert scores == sorted(scores, reverse=True)
