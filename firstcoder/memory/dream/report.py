"""P6 dream 质量报告和 Windows-safe 原子写入。"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from firstcoder.memory.paths import ensure_no_link_or_junction
from firstcoder.memory.redact import MemoryRedactor
from firstcoder.memory.security import RELATIVE_DATE_PATTERN, SECRET_PATTERNS
from firstcoder.memory.write import atomic_write_text

_DREAM_NOISE_PATTERN = re.compile(r"(?i)\b(user said hi|assistant acknowledged|acknowledged|hello|said hi)\b")


def build_dream_report(
    before_notes: Iterable[dict[str, Any]],
    after_notes: Iterable[dict[str, Any]],
    *,
    rejected_reasons: Iterable[str] = (),
    relative_dates_absolutized: int | None = None,
) -> dict[str, int]:
    """根据维护前后快照计算稳定计数，不把 note 正文写入报告。"""

    before = [dict(note) for note in before_notes]
    after = [dict(note) for note in after_notes]
    active_after = [note for note in after if str(note.get("status", "active")) == "active"]
    active_after_texts = [str(note.get("text", "")) for note in active_after]
    active_after_set = set(active_after_texts)
    after_counts = Counter(active_after_texts)
    before_counts = Counter(str(note.get("text", "")) for note in before)

    rejected = tuple(str(reason) for reason in rejected_reasons)
    secret_rejections = sum(reason in {"secret_shaped", "secret", "quarantined"} for reason in rejected)
    secrets_in_before = sum(
        1
        for note in before
        if _is_secret(str(note.get("text", "")))
        and (str(note.get("text", "")) not in active_after_set or str(note.get("status", "")) == "quarantined")
    )
    relative_count = (
        max(0, int(relative_dates_absolutized))
        if relative_dates_absolutized is not None
        else _count_relative_dates(before, active_after_set, active_after)
    )
    return {
        "notes_in_before": len(before),
        "notes_in_after": len(after),
        "signal_retained": sum(
            1
            for note in active_after
            if not _is_noise(note) and not _is_secret(str(note.get("text", ""))) and not _has_relative_date(note)
        ),
        "noise_dropped": sum(
            1 for note in before if _is_noise(note) and str(note.get("text", "")) not in active_after_set
        ),
        "secrets_rejected": max(secret_rejections, secrets_in_before),
        "duplicates_merged": sum(
            count - 1 for text, count in before_counts.items() if count > 1 and after_counts.get(text, 0) == 1
        ),
        "relative_dates_absolutized": relative_count,
    }


def write_dream_report(
    report_dir: str | Path,
    report: dict[str, Any],
    *,
    timestamp: str | datetime | None = None,
    task_id: str = "",
    security: MemoryRedactor | None = None,
) -> Path:
    """将脱敏报告以 Windows 合法文件名和原子替换写入磁盘。"""

    directory = Path(report_dir)
    if directory.exists():
        ensure_no_link_or_junction(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ensure_no_link_or_junction(directory)
    safe_timestamp = _safe_timestamp(timestamp)
    safe_task_id = _safe_component(task_id)
    filename = f"{safe_timestamp}_{safe_task_id}.json" if safe_task_id else f"{safe_timestamp}.json"
    path = directory / filename
    if path.exists():
        ensure_no_link_or_junction(path)
    redactor = security or MemoryRedactor()
    sanitized = redactor.redact_value(dict(report))
    payload = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    atomic_write_text(path, payload)
    return path


def _is_noise(note: dict[str, Any]) -> bool:
    evidence = note.get("evidence") if isinstance(note.get("evidence"), dict) else {}
    session_id = str(evidence.get("session_id", "")).strip().lower()
    return session_id == "noise" or bool(_DREAM_NOISE_PATTERN.search(str(note.get("text", ""))))


def _is_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _has_relative_date(note: dict[str, Any]) -> bool:
    return bool(RELATIVE_DATE_PATTERN.search(str(note.get("text", ""))))


def _count_relative_dates(
    before: list[dict[str, Any]],
    active_after_set: set[str],
    active_after: list[dict[str, Any]],
) -> int:
    relative_after_present = any(_has_relative_date(note) for note in active_after)
    return sum(
        1
        for note in before
        if _has_relative_date(note)
        and str(note.get("text", "")) not in active_after_set
        and not relative_after_present
    )


def _safe_timestamp(value: str | datetime | None) -> str:
    if value is None:
        value = datetime.now(UTC).replace(microsecond=0)
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    else:
        normalized = _safe_component(str(value))
    return normalized or "maintenance"


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value).strip())[:120]


__all__ = ["build_dream_report", "write_dream_report"]
