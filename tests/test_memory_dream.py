"""P6 memory dream 的公开契约测试。

这些测试只通过 gate、报告、持久状态和 scheduler 的公共入口观察行为，避免把
实现细节固定成测试契约。P6 的真实 provider 仍由 fake runner 替代，测试本身
不会访问网络，也不会触碰用户级 global memory。
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest

from firstcoder.memory.dream.gate import evaluate_auto_dream_gate
from firstcoder.memory.dream.models import (
    DreamCandidate,
    DreamProposal,
    MemoryMaintenanceEntry,
    MemoryMaintenanceSnapshot,
)
from firstcoder.memory.dream.report import build_dream_report, write_dream_report
from firstcoder.memory.dream.scheduler import (
    MemoryMaintenanceConfig,
    MemoryMaintenanceScheduler,
    ProviderBoundedDreamRunner,
)
from firstcoder.memory.dream.state import DreamTaskState, MaintenanceStateStore
from firstcoder.memory.durable import DurableMemoryStore
from firstcoder.memory.models import MemoryEvidence, MemoryNote
from firstcoder.memory.paths import DefaultWorkspaceScope
from firstcoder.memory.provenance import workspace_fingerprint


def test_auto_dream_gate_requires_both_interval_and_session_thresholds(tmp_path: Path) -> None:
    """只有时间门槛和 session 门槛同时满足时才允许维护。"""

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    last_success = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    old_path = sessions_dir / "sess_old.jsonl"
    new_path = sessions_dir / "sess_new.jsonl"
    old_path.write_text("old\n", encoding="utf-8")
    new_path.write_text("new\n", encoding="utf-8")
    os.utime(old_path, (last_success + 1, last_success + 1))
    os.utime(new_path, (last_success + 2, last_success + 2))

    result = evaluate_auto_dream_gate(
        last_success_at=last_success,
        sessions_dir=sessions_dir,
        current_session_id="sess_current",
        min_interval_hours=24,
        min_sessions=3,
        now=last_success + 25 * 3600,
    )

    assert result.should_run is False
    assert result.skip_reason == "session_gate"
    assert result.session_ids == ("sess_old", "sess_new")


def test_auto_dream_gate_excludes_current_session_and_reports_interval_skip(tmp_path: Path) -> None:
    """当前 session 不应为自己触发维护，时间门槛优先返回稳定原因。"""

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sess_current.jsonl").write_text("current\n", encoding="utf-8")
    (sessions_dir / "sess_other.jsonl").write_text("other\n", encoding="utf-8")
    now = datetime(2026, 1, 2, tzinfo=UTC).timestamp()

    result = evaluate_auto_dream_gate(
        last_success_at=now - 1 * 3600,
        sessions_dir=sessions_dir,
        current_session_id="sess_current",
        min_interval_hours=24,
        min_sessions=1,
        now=now,
    )

    assert result.should_run is False
    assert result.skip_reason == "interval_gate"
    assert result.session_ids == ("sess_other",)


def test_auto_dream_gate_orders_sessions_by_mtime_for_max_session_selection(tmp_path: Path) -> None:
    """随机 session id 的字母序不能导致最近会话被 max_sessions 丢掉。"""

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    old = sessions_dir / "sess_z_old.jsonl"
    recent = sessions_dir / "sess_a_recent.jsonl"
    old.write_text("old\n", encoding="utf-8")
    recent.write_text("recent\n", encoding="utf-8")
    os.utime(old, (100.0, 100.0))
    os.utime(recent, (200.0, 200.0))

    result = evaluate_auto_dream_gate(
        last_success_at=0.0,
        sessions_dir=sessions_dir,
        min_interval_hours=0,
        min_sessions=1,
        now=300.0,
    )

    assert result.session_ids == ("sess_z_old", "sess_a_recent")


def test_maintenance_state_round_trips_running_task_without_raw_prompt(tmp_path: Path) -> None:
    """任务状态可跨进程恢复，且状态文件不保存 prompt 或 note 正文。"""

    store = MaintenanceStateStore(tmp_path / "dream")
    state = DreamTaskState(
        task_id="dream_20260102T000000Z_abcd1234",
        status="running",
        workspace_fingerprint="0123456789ab",
        trigger_session_id="sess_current",
        session_ids=("sess_old",),
        snapshot_id="snapshot_1234",
        attempt=1,
        recovery_count=0,
        created_at=1767312000.0,
        started_at=1767312001.0,
    )

    store.save(state)

    assert store.load() == state
    raw_state = store.state_path.read_text(encoding="utf-8")
    assert "prompt" not in raw_state
    assert "private note" not in raw_state


def test_scheduler_prompt_budget_preserves_valid_json_by_dropping_old_entries(tmp_path: Path) -> None:
    """大 workspace 只收缩输入条目，不把 JSON payload 截断成非法字符串。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(DreamProposal()),
        config=MemoryMaintenanceConfig(max_prompt_chars=2_000),
    )
    snapshot = MemoryMaintenanceSnapshot(
        snapshot_id="snapshot_budget",
        index_version=0,
        entries=tuple(
            MemoryMaintenanceEntry(text=f"entry-{index} " + "x" * 600, session_id=f"sess-{index}")
            for index in range(20)
        ),
    )

    prompt = scheduler._build_prompt(snapshot)
    payload = json.loads(prompt.split("Input snapshot: ", 1)[1])

    assert len(prompt) <= 2_000
    assert len(payload["captured_entries"]) < 20
    assert payload["captured_entries"][-1]["session_id"] == "sess-19"
    scheduler.close()


def test_dream_report_counts_quality_changes_and_uses_windows_safe_filename(tmp_path: Path) -> None:
    """报告只保存稳定计数，且报告文件名不含 Windows 非法冒号。"""

    before = [
        {"text": "pytest uses fixtures", "status": "active", "evidence": {"session_id": "sess_a"}},
        {"text": "pytest uses fixtures", "status": "active", "evidence": {"session_id": "sess_b"}},
        {"text": "assistant acknowledged", "status": "active", "evidence": {"session_id": "noise"}},
        {"text": "API key sk-12345678901234567890", "status": "quarantined", "evidence": {"session_id": "sess_a"}},
        {"text": "release tomorrow", "status": "active", "evidence": {"session_id": "sess_a"}},
    ]
    after = [
        {"text": "pytest uses fixtures", "status": "active", "evidence": {"session_id": "sess_a"}},
        {"text": "release 2026-01-03", "status": "active", "evidence": {"session_id": "sess_a"}},
    ]

    report = build_dream_report(
        before,
        after,
        rejected_reasons=("secret_shaped",),
        relative_dates_absolutized=1,
    )
    path = write_dream_report(
        tmp_path / "reports",
        report,
        timestamp="2026-01-02T03:04:05Z",
        task_id="dream_1234",
    )

    assert report["notes_in_before"] == 5
    assert report["notes_in_after"] == 2
    assert report["signal_retained"] == 2
    assert report["noise_dropped"] == 1
    assert report["secrets_rejected"] == 1
    assert report["duplicates_merged"] == 1
    assert report["relative_dates_absolutized"] == 1
    assert ":" not in path.name
    assert path.read_text(encoding="utf-8").count("sk-123") == 0


def test_maintenance_batch_commit_preserves_workspace_provenance(tmp_path: Path) -> None:
    """维护批量提交复用 durable store，并保留 workspace 内的 evidence。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "README.md"
    source.write_text("pytest uses deterministic fixtures\n", encoding="utf-8")
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    note = MemoryNote(
        topic="project-conventions",
        text="pytest uses deterministic fixtures",
        evidence=MemoryEvidence(
            source_path="README.md",
            session_id="sess_old",
            visibility="workspace",
        ),
    )

    results, superseded = store.promote_maintenance([note])

    assert results == ["project-conventions: pytest uses deterministic fixtures"]
    assert superseded == []
    stored = store.snapshot(workspace)[0]
    assert stored["visibility"] == "workspace"
    assert stored["evidence"]["session_id"] == "sess_old"
    assert stored["evidence"]["source_path"] == "README.md"
    assert stored["evidence"]["evidence_anchor_hash"]


def test_maintenance_batch_rejects_global_visibility_before_writing(tmp_path: Path) -> None:
    """auto-dream 不得借维护入口写入 global memory。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    note = MemoryNote(
        topic="key-decisions",
        text="global candidate must be rejected",
        evidence=MemoryEvidence(visibility="global"),
    )

    try:
        store.promote_maintenance([note])
    except ValueError as exc:
        assert "workspace" in str(exc)
    else:
        raise AssertionError("global maintenance candidate must be rejected")

    assert store.load_index() == []


def test_maintenance_batch_rejects_unknown_topic_with_value_error(tmp_path: Path) -> None:
    """公开维护入口也必须拒绝未知 topic，而不是泄露 KeyError。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    note = MemoryNote(
        topic="not-a-durable-topic",
        text="must be rejected",
        evidence=MemoryEvidence(visibility="workspace"),
    )

    with pytest.raises(ValueError, match="durable topic"):
        store.promote_maintenance([note])

    assert store.load_index() == []


def test_maintenance_batch_rejects_relative_provenance_escape(tmp_path: Path) -> None:
    """``../`` provenance 不能借相对路径越过 workspace 读取或哈希文件。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside evidence", encoding="utf-8")
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    note = MemoryNote(
        topic="key-decisions",
        text="the evidence must stay in workspace",
        evidence=MemoryEvidence(source_path="../outside.txt", visibility="workspace"),
    )

    with pytest.raises(ValueError, match="outside"):
        store.promote_maintenance([note])

    assert store.load_index() == []


class _FakeDreamRunner:
    """测试边界 fake：只返回结构化结果，不直接访问文件。"""

    def __init__(self, candidate: object) -> None:
        self.candidate = candidate
        self.calls = []

    def run_maintenance(self, *, prompt, snapshot, write_scope):
        self.calls.append((prompt, snapshot, write_scope))
        return self.candidate


class _FailingDreamRunner:
    """模拟 provider/runner 失败，验证失败不推进成功时间戳。"""

    def run_maintenance(self, *, prompt, snapshot, write_scope):
        raise RuntimeError("provider unavailable: sk-123456789012345678901234")


class _BlockingDreamRunner:
    """让第二次触发落在第一任务运行期间，验证单任务并发限制。"""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def run_maintenance(self, *, prompt, snapshot, write_scope):
        self.started.set()
        self.release.wait(timeout=30)
        return DreamProposal()


class _FakeMaintenanceProvider:
    """只实现 FirstCoder ChatProvider 所需的 complete 形状，不创建任何文件。"""

    name = "fake-maintenance"
    model = "fake-maintenance-model"

    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return type("Response", (), {"content": self.content})()


def test_manual_dream_runs_fake_runner_and_persists_success_audit(tmp_path: Path) -> None:
    """手动 dream 绕过自动 gate，但仍通过 scheduler 和 durable store。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_root = workspace / ".firstcoder"
    sessions_dir = data_root / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "sess_old.jsonl").write_text("old\n", encoding="utf-8")
    store = DurableMemoryStore(data_root / "memory", workspace_root=workspace)
    store.append_daily_log(
        "pytest uses deterministic fixtures",
        source=MemoryEvidence(session_id="sess_old", visibility="session"),
    )
    runner = _FakeDreamRunner(
        DreamProposal(
            candidates=(
                DreamCandidate(
                    topic="project-conventions",
                    text="pytest uses deterministic fixtures",
                    session_id="sess_old",
                    reason="stable project convention",
                ),
            )
        )
    )
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=sessions_dir,
        runner=runner,
        config=MemoryMaintenanceConfig(enabled=False),
    )

    scheduled = scheduler.request_run("sess_current")
    assert scheduled.status == "scheduled"
    scheduler.wait_for_idle(timeout=5)

    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "succeeded"
    assert runner.calls[0][1].entries[0].text == "pytest uses deterministic fixtures"
    assert scheduler.memory_store.snapshot(workspace)[0]["text"] == "pytest uses deterministic fixtures"
    audit = scheduler.audit_path.read_text(encoding="utf-8")
    assert '"event": "succeeded"' in audit
    assert "pytest uses deterministic fixtures" not in audit
    scheduler.close()


def test_auto_dream_disabled_skips_without_provider_call(tmp_path: Path) -> None:
    """默认关闭时普通自动检查不得调用 runner。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = workspace / ".firstcoder" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "sess_other.jsonl").write_text("other\n", encoding="utf-8")
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    runner = _FakeDreamRunner(DreamProposal())
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=sessions_dir,
        runner=runner,
        config=MemoryMaintenanceConfig(),
    )

    result = scheduler.maybe_schedule("sess_current")

    assert result.status == "disabled"
    assert runner.calls == []
    assert not scheduler.audit_path.exists()
    scheduler.close()


def test_scheduler_failure_persists_failed_without_last_success(tmp_path: Path) -> None:
    """runner 异常写入 failed，且不会把失败时刻当作成功时间。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FailingDreamRunner(),
    )

    result = scheduler.request_run("sess_current")
    assert result.status == "scheduled"
    scheduler.wait_for_idle(timeout=5)

    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "failed"
    assert state.last_success_at is None
    assert state.error_code == "runner_error"
    assert "sk-123456789012345678901234" not in state.error_message
    scheduler.close()


def test_scheduler_restarts_running_task_as_pending_and_retries_once(tmp_path: Path) -> None:
    """重启看到 running 时原子恢复为 pending，并自动重试一次。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    state_store = MaintenanceStateStore(store.root / "dream")
    state_store.save(
        DreamTaskState(
            task_id="dream_recover_once",
            status="running",
            workspace_fingerprint=workspace_fingerprint(workspace),
            session_ids=(),
            attempt=1,
            created_at=1.0,
        )
    )

    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(DreamProposal()),
    )
    recovered = scheduler.state_store.load()
    assert recovered is not None
    assert recovered.recovery_count == 1

    scheduler.wait_for_idle(timeout=5)
    completed = scheduler.state_store.load()
    assert completed is not None
    assert completed.status == "succeeded"
    scheduler.close()


def test_scheduler_marks_second_restart_recovery_failed(tmp_path: Path) -> None:
    """同一个 running 状态第二次恢复直接 failed，避免无限后台重试。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    MaintenanceStateStore(store.root / "dream").save(
        DreamTaskState(
            task_id="dream_recover_limit",
            status="running",
            workspace_fingerprint=workspace_fingerprint(workspace),
            recovery_count=1,
            created_at=1.0,
        )
    )

    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(DreamProposal()),
    )

    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "failed"
    assert state.error_code == "recovery_limit"
    scheduler.close()


def test_scheduler_rejects_invalid_runner_result_without_memory_write(tmp_path: Path) -> None:
    """非结构化 runner 输出必须整次拒绝，不能部分写入 durable store。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(object()),
    )

    assert scheduler.request_run("sess_current").status == "scheduled"
    scheduler.wait_for_idle(timeout=5)
    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "failed"
    assert state.error_code == "invalid_runner_result"
    assert store.load_index() == []
    scheduler.close()


def test_scheduler_rejects_unknown_topic_as_a_candidate_rejection(tmp_path: Path) -> None:
    """未知 topic 只进入维护拒绝计数，不应让 durable 事务抛出 KeyError。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(
            DreamProposal(
                candidates=(DreamCandidate(topic="unknown-topic", text="must be rejected"),),
            )
        ),
    )

    assert scheduler.request_run("sess_current").status == "scheduled"
    scheduler.wait_for_idle(timeout=5)
    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "succeeded"
    assert store.load_index() == []
    report_files = list((store.root / "dream" / "reports").glob("*.json"))
    assert len(report_files) == 1
    assert '"notes_in_after": 0' in report_files[0].read_text(encoding="utf-8")
    scheduler.close()


def test_scheduler_allows_only_one_running_task(tmp_path: Path) -> None:
    """同一 scheduler 的重复手动触发只返回 running，不创建第二次 provider 调用。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    runner = _BlockingDreamRunner()
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=runner,
    )

    first = scheduler.request_run("sess_current")
    assert first.status == "scheduled"
    assert runner.started.wait(timeout=5)
    second = scheduler.request_run("sess_current")
    assert second.status == "running"
    assert second.task_id == first.task_id

    runner.release.set()
    scheduler.wait_for_idle(timeout=5)
    scheduler.close()


def test_second_workspace_scheduler_returns_busy_without_waiting_for_provider(tmp_path: Path) -> None:
    """跨进程等价的第二个 scheduler 不能把主调用阻塞在 provider 上。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory_root = workspace / ".firstcoder" / "memory"
    store = DurableMemoryStore(memory_root, workspace_root=workspace)
    runner = _BlockingDreamRunner()
    first = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=runner,
    )
    assert first.request_run("sess_first").status == "scheduled"
    assert runner.started.wait(timeout=5)

    second = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=DurableMemoryStore(memory_root, workspace_root=workspace),
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=_FakeDreamRunner(DreamProposal()),
    )
    assert second.request_run("sess_second").status == "busy"

    runner.release.set()
    first.wait_for_idle(timeout=5)
    first.close()
    second.close()


def test_scheduler_does_not_resume_state_from_another_workspace(tmp_path: Path) -> None:
    """复制来的 dream 状态不能在另一 workspace 触发 runner。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = DurableMemoryStore(workspace / ".firstcoder" / "memory", workspace_root=workspace)
    MaintenanceStateStore(store.root / "dream").save(
        DreamTaskState(
            task_id="dream_wrong_workspace",
            status="running",
            workspace_fingerprint="000000000000",
            created_at=1.0,
        )
    )
    runner = _FakeDreamRunner(DreamProposal())
    scheduler = MemoryMaintenanceScheduler(
        workspace_root=workspace,
        memory_store=store,
        sessions_dir=workspace / ".firstcoder" / "sessions",
        runner=runner,
    )

    state = scheduler.state_store.load()
    assert state is not None
    assert state.status == "failed"
    assert state.error_code == "workspace_mismatch"
    assert runner.calls == []
    scheduler.close()


def test_provider_bounded_runner_uses_json_only_provider_contract_without_file_write(tmp_path: Path) -> None:
    """bounded runner 只调用现有 provider 协议，不携带工具或直接写盘。"""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _FakeMaintenanceProvider(
        '{"candidates": [{"topic": "key-decisions", "text": "use pytest", "visibility": "workspace"}], '
        '"rejections": [], "relative_dates_absolutized": 0}'
    )
    runner = ProviderBoundedDreamRunner(lambda: provider)
    snapshot = MemoryMaintenanceSnapshot(snapshot_id="snap_1", index_version=0)
    scope = DefaultWorkspaceScope(workspace, memory_root=workspace / ".firstcoder" / "memory")

    proposal = runner.run_maintenance(prompt="maintain", snapshot=snapshot, write_scope=scope)

    assert proposal.candidates[0].topic == "key-decisions"
    assert provider.requests[0].tools == []
    assert provider.requests[0].tool_choice == "none"
    assert not list(workspace.rglob("*"))


def test_provider_bounded_runner_rejects_non_json_result(tmp_path: Path) -> None:
    """provider 返回自然语言时不能被当作维护成功。"""

    provider = _FakeMaintenanceProvider("not json")
    runner = ProviderBoundedDreamRunner(lambda: provider)
    scope = DefaultWorkspaceScope(tmp_path, memory_root=tmp_path / ".firstcoder" / "memory")

    with pytest.raises(ValueError, match="JSON"):
        runner.run_maintenance(
            prompt="maintain",
            snapshot=MemoryMaintenanceSnapshot(snapshot_id="snap_1", index_version=0),
            write_scope=scope,
        )
