# FirstCoder × pico-v3 融合方案：三层引入清单与评审任务

> 版本：v2（评审后定稿：Codex 审查 + 人工自查 交叉验证，2026-08-06）
> 评审人：Codex（会话 019fd525-cebf-74a2-bc68-890467afebc8，源码级审查）+ 人工自查
> 评审结论：**有条件同意**（8 条条件见 §7.1；修订后的实施顺序 P0-P7 见 §7.4）
> 文档用途：交付给 Codex 逐层审查，验证路径、判断和引入方式，指出错误、遗漏与风险，最终定稿。

---

## 0. 总体结论（先读这个）

**基底：FirstCoder。** 它的核心资产是自研 agent 执行链（AgentLoop / Provider / Tools / Permissions / Worktree）+ 事件溯源事实层（JSONL Session / Replay / Projection）+ **L1-L4 安全上下文压缩（task-boundary compaction 是最大亮点）** + Harbor benchmark 集成。

**从 pico-v3 引入三层内容：**

1. **记忆层** —— FirstCoder **完全缺失**，pico 有成熟领域资产 → 全量引入（运行时接线重写）
2. **上下文编排层** —— FirstCoder **部分缺失**（已有更强的压缩内核，缺编排策略）→ 选择性引入 4 个模块
3. **harness 数据契约与实验方法** —— FirstCoder **已有 benchmark 基础**（缺 run-level 证据与专项实验）→ 引入数据契约 + 实验方法，不搬运行时

**一句话主线：FirstCoder 的 task-boundary compaction 是主角；pico 的记忆补最高频功能缺口，上下文编排让压缩可编排，harness 数据契约给卖点提供可审计证据。**

**铁律：不引入 pico 的运行时（ContextManager / run_dream / Engine）、手写 HTTP/SSE 客户端、evaluation 实验运行时、非原子写和 PID 锁。**

---

## 1. 第一层：记忆层（最高优先级，FirstCoder 完全缺失）

### 1.1 是 FirstCoder 缺失，还是 pico 亮点？

**两者都是，且偏向"缺失"——这是引入理由最强的一层。**

**FirstCoder 缺失（已用全目录 grep 证实）：**

- `firstcoder/` 全目录搜索 `memory / durable / remember / knowledge / RunStore / run_id / task_state / harness / verification`：
  - `knowledge`：**0 命中**
  - `RunStore / run_id / task_state`：**0 命中**
  - `memory`：仅 `InMemoryHistory`（CLI 终端历史）和 `ToolResultArchive` docstring 里的 "Durable..."——都不是记忆
- FirstCoder 现有全部持久化，均为 **per-session / per-project 作用域**：
  - `.firstcoder/sessions/{id}.jsonl` 事件日志（`firstcoder/context/store.py:35-68`）
  - `.firstcoder/archives/{session_id}/…` 工具结果归档（`firstcoder/context/archive.py`）——**session 作用域的内容寻址缓存，是压缩 backing，不是跨 session 知识**
  - `.firstcoder/session_index.json` 会话目录（仅标题/最后输入输出元数据）
  - `permissions.json` 权限 grant
- `ToolResultArchive` 硬约束 `part.kind == "tool_result"`（`archive.py:217-220`），且路径含 session_id（`archive.py:211-215`）——**不可能直接当记忆库**，只能复用其原子写 + 完整性 pattern
- `SystemPromptInputs` 有 fingerprint 缓存（`firstcoder/context/system_prompt.py:20-62`）——记忆不能进 system_prefix，否则破坏前缀缓存

**pico 是亮点：**

- 分层记忆设计是成熟产品级资产：working memory（每轮 state dict 纯函数变换）→ daily logs（append-only）→ durable topics（带 provenance）→ 检索召回 → dream 维护
- 核心算法层是纯函数或只依赖路径，可移植
- 配套安全规则（lint / quarantine / secret patterns）是纯静态规则，零运行时依赖

### 1.2 引入清单

| # | 引入内容 | pico 源路径 | 引入方式 | FirstCoder 落点 |
|---|---|---|---|---|
| M1 | 记忆目录布局 + 每日日志原语（`ensure_memory_dir` / `daily_log_path` / `append_to_daily_log`） | `pico/features/memory.py:78-110` | 原样迁移（append 改为原子追加） | 新建 `firstcoder/memory/logs.py` |
| M2 | **工作记忆 state schema + 纯函数变换**（`default_memory_state` / `set_task_summary` / `remember_file` / `append_note` / `set_file_summary` / `invalidate_stale_file_summaries` / `summarize_read_result`） | `pico/features/memory.py:744-1418` | 原样迁移（纯函数，零依赖） | `firstcoder/memory/working.py` |
| M3 | **durable 存储模型**（`DurableMemoryStore`：MEMORY.md 索引 / topics/*.md / metadata sidecar；promote / supersede；note_id = sha256(topic+text)[:12]；evidence{source_path, session_id, anchor_hash, scope}） | `pico/features/memory.py:760-1020, 828-842, 1121-1122` | 迁移数据模型；**写盘升级**：temp+rename 原子发布 + portalocker 跨进程锁 + 版本号/CAS（pico 现有 `write_text` 直写（`:826,:945,:963`）和 PID 锁（`:320-336`）是缺陷，不搬） | `firstcoder/memory/durable.py` |
| M4 | **检索排序 + 审计**（`_ranked_retrieval_notes`：tag 精确命中×1000 → 关键词重叠×10 → 新旧 → 索引序；`retrieval_view_structured` 的 selected/rejected + reject_reason） | `pico/features/memory.py:1421-1510` | 原样迁移（纯函数） | `firstcoder/memory/retrieval.py` |
| M5 | **durable 提取启发式**（`extract_durable_promotions` / `reject_durable_reason` / `DURABLE_MEMORY_LINE_PATTERNS`） | `pico/features/memory.py:562-613, 38-50` | 原样迁移 | `firstcoder/memory/promotion.py` |
| M6 | **证据失效追踪**（`canonicalize_path` / `file_freshness` / `compute_anchor_hash` / `workspace_fingerprint` / `_apply_evidence_staleness`） | `pico/features/memory.py:1046-1101, 1180-1190` | 原样迁移 | `firstcoder/memory/provenance.py` |
| M7 | **记忆安全策略包**：`SECRET_PATTERNS`（5 条）/ `RELATIVE_DATE_PATTERN`、lint 五规则（missing_evidence / relative_date / secret_shaped / duplicate_active_subject / orphan_supersede）、quarantine 注入特征 | `pico/features/memory_lint.py:13-23, 97-144`；`pico/features/memory_quarantine.py:7-15` | **几乎原样拷贝**（纯静态规则，共享正则源） | `firstcoder/memory/security/` |
| M8 | secret 检测/redaction 三层设计（env 值替换 + 静态正则 + 隔离门） | `pico/core/runtime_secrets.py:5-17, 37-54` | 抽成独立 mixin | `firstcoder/memory/redact.py` |
| M9 | 记忆注入 prompt 模板（`build_memory_system_section` / `load_memory_index_text`，索引上限 10000 chars） | `pico/features/memory.py:399-484, 290-306` | 原样迁移模板；**注入点用 FirstCoder 接缝**：`AgentLoop._request_messages`（`firstcoder/agent/loop.py:998-1021`，即 task_plan 快照块），在预算计算之前注入，不用 system_prefix（防 fingerprint 缓存破坏） | `firstcoder/memory/prompt.py` + `firstcoder/agent/loop.py` 窄 hook |
| M10 | **dream 质量报告度量**（`build_dream_report`：promotions/rejections/superseded/secrets_rejected/relative_dates 统计） | `pico/features/memory.py:175-259, 212-234` | 保留度量逻辑 | `firstcoder/memory/dream/report.py` |
| M11 | auto-dream 门控（interval + session 双条件，`evaluate_auto_dream_gate` / `list_sessions_since`） | `pico/features/memory.py:309-396, 358-371` | 保留门控策略 | `firstcoder/memory/dream/gate.py` |
| M12 | 记忆命令与工具（`/remember` `/memory` `/dream` 命令；`memory_note` / `memory_promote` 工具） | 参考 pico 命令入口（`pico/commands/`） | 新建，注册进 FirstCoder 的 `CompositeCommandHandler`（`firstcoder/app/factory.py:247`）和 `create_session_tool_registry`（`firstcoder/tools/session_registry.py:42-111`，模板见 `create_retrieve_archive_tool` 注册处 `:101-108`；**不进** builtin 默认集） | `firstcoder/app/memory_commands.py` + `firstcoder/tools/` |

### 1.3 必须重写（不搬实现）

| 内容 | 源路径 | 不搬原因 | 重写为 |
|---|---|---|---|
| `run_dream` | `pico/features/memory.py:626-673` | 内部**嵌套实例化一个完整 Pico 子运行时**（`:638-653`），依赖 model_client / session_store / feature_flags / write_scope / tool_profile | `BoundedDreamRunner` port：给定 dream prompt + 受限 write_scope 的受限 agent 执行器 |
| `maintain_memory_after_turn` | `pico/features/memory.py:676-741` | 依赖 `agent.auto_dream` 等配置 + session_event_bus + 自建后台线程 | 独立 `MemoryMaintenanceScheduler`（gate → snapshot → runner → 原子发布 → audit），**不套用** FirstCoder `BackgroundJobManager`（无持久 job 状态/无崩溃恢复，`firstcoder/agent/background.py:190-206`） |
| `promote_durable_memory` 的 agent 状态写入 | `pico/features/memory.py:616-623` | 直写 `agent.session["memory"]` / `agent.last_durable_*` | 改由显式工具/命令触发，写独立 memory store |

### 1.4 记忆事件与事实层的关系

- `SessionEvent` 是泛型 `type: str + payload: dict`（`firstcoder/context/events.py:11-23`），`SessionEventWriter.append_event(event_type, payload)`（`firstcoder/context/writer.py:42-50`）加 `memory_recorded` / `memory_retrieved` 事件零成本
- 重放时未知事件被静默忽略（`firstcoder/context/store.py:92-94`）——**第一版记忆事件只做审计，不做重放**；需要 resume 记忆状态时再补 replay 分支
- **不要把 durable memory 写成普通 user/assistant 消息**——会污染会话历史与 tool_call/tool_result 闭合关系（`context_builder.py:47-49` 校验序列）

### 1.5 第一阶段启用范围

显式记忆闭环（M1-M9 全部 + 命令/工具）→ 检索注入 → 安全策略全开。**auto-dream（M10-M12 的 dream 部分）后置**，等显式路径稳定后再接。

---

## 2. 第二层：上下文编排层（选择性引入 4 个模块）

### 2.1 是 FirstCoder 缺失，还是 pico 亮点？

**偏"亮点"——FirstCoder 不是缺失，而是已有更强的内核、缺编排策略。这是引入理由最需要谨慎的一层。**

**FirstCoder 已有（且部分更强）：**

- `ContextBudget`：context_window → 95% usable → output reserve → fixed(系统+tools)/history 两桶 + 高/低水位（`firstcoder/context/token_budget.py:16-75`）
- 高水位触发（`firstcoder/context/triggers.py:34-51`）
- **L1-L4 程序化压缩 pipeline**：L1 旧任务 trim（保护 tool_call 消息，`compaction.py:256-293`）、L2 内容路由压缩（JSON/GitDiff/HTML/SourceCode/SearchResults/BuildOutput 专用压缩器，`compaction.py:295-341`）、L3 archive-backed placeholder + 生命周期（STALE/SUPERSEDED/DUPLICATE/DERIVED，`compaction.py:343-408`）、L4 LLM checkpoint 摘要（候选生成→验证→commit→fallback，`manager.py:260-321`）
- **task-boundary compaction（最大亮点）**：`TASK_HASH_CHANGED` 触发时强制 L2/L3 + 旧任务清理（`manager.py:25-35, 148-165`）
- 韧性机制：熔断、no-effect 去重、fallback 策略（`manager.py:109-135, 323-479`）
- 工具结果保留保护：lifecycle + consumed + retrieval-protected（`compaction.py:526-554`）

**FirstCoder 缺失的部分（pico 亮点所在）：**

| 能力 | pico 证据 | FirstCoder 现状 |
|---|---|---|
| section 级预算（6 个 section 各有 budget/floor/reduction_rank/protected） | `pico/core/context_sections.py:8-27, 99-148` | 只有 fixed/history 两桶 |
| 削减优先级（REDUCTION_ORDER = relevant_memory→skills→history→memory→prefix） | `pico/core/context_sections.py:18-19` | 无 |
| 压力分档（tier0_observe / tier1_snip / tier2_prune / tier3_summary） | `pico/core/context_pressure.py:49-57` | 只有高水位单点触发 |
| actual usage 校准（identity 匹配时用 provider 实际 token 替代估算） | `pico/core/context_pressure.py:88-125`（IDENTITY_KEYS `:8-15`） | 纯估算（`estimate_text_tokens`，`token_budget.py:29-37`） |
| 削减量化度量（section saved_chars / compact_net_benefit_tokens / replacement_cache_hits） | `pico/core/context_budget_summary.py:5-50` | CompactionEvent 只有 level 级指标 |
| request 级渲染报告（每 section raw/rendered chars + reduction log + relevant_memory 审计） | `pico/core/context_report.py:15-55, 57-73` | 无 |

### 2.2 引入清单

| # | 引入内容 | pico 源路径 | 引入方式 | FirstCoder 落点 |
|---|---|---|---|---|
| C1 | **Section 预算策略数据模型**（`ContextSectionPolicy`：name/budget_chars/floor_chars/reduction_rank/sources/protected；SECTION_RATIOS / MIN_SECTION_BUDGETS / REDUCTION_ORDER） | `pico/core/context_sections.py:8-27, 99-148` | 原样迁移（纯数据） | 新建 `firstcoder/context/section_policy.py` |
| C2 | **压力分档模型**（tier 阈值 0.6/0.8/0.95；`ContextPressure.pressure_ratio/window_ratio/pressure_tier`） | `pico/core/context_pressure.py:27-57` | 迁移模型；**不建第二套预算**，接入现有 `ContextBudget` | 扩展 `firstcoder/context/token_budget.py` |
| C3 | **actual usage 校准**（`ContextPressureController.evaluate`：last_completion_metadata 的 actual input_tokens，IDENTITY_KEYS 匹配才用） | `pico/core/context_pressure.py:73-126` | 迁移逻辑，接 FirstCoder provider metadata | `firstcoder/context/usage_calibration.py` |
| C4 | **削减量化度量**（`context_budget_summary`：budget_unit/reductions/saved_chars/pressure_tier/compact_net_benefit_tokens/usage_source） | `pico/core/context_budget_summary.py:5-50` | 迁移为数据契约 | `firstcoder/context/budget_summary.py` |
| C5 | **request 级渲染报告**（每 section raw/rendered/budget + reduction_log + relevant_memory 审计：selected_notes/selected_sources/reject 信息） | `pico/core/context_report.py:15-55, 57-73` | 迁移数据契约 | `firstcoder/context/request_report.py` |
| C6 | 可复用参考（**不平行实现，合并进现有模块**）：history 保留策略（protected tools / bulk tools / changed_paths / failed-tool 保护）、替换账本（event_id+sha256 跨轮复用）、handoff 摘要 schema（Goal/Constraints/Files/Key Decisions/Blockers/Next Steps） | `pico/core/context_retention.py:32-57`；`pico/core/context_replacements.py:8-37, 48-98`；`pico/core/context_handoff.py:10-50, 53-65` | 取思想，与 FirstCoder 的 lifecycle/archive/checkpoint 合并 | 合并进 `firstcoder/context/` |

### 2.3 明确不引入（防误搬）

| 内容 | 源路径 | 不搬原因 |
|---|---|---|
| `ContextManager` 组装器 | `pico/core/context_manager.py:51-188` | 直接持有 agent（`self.agent.memory/session/skills/prefix/model_client`），输出纯字符串 prompt；与 FirstCoder 的 ChatMessage + tool-call 序列校验不兼容 |
| `ContextOrchestrator` | `pico/core/context_orchestrator.py:30-96` | 编排逻辑重写为 FirstCoder 风格的 `ContextWindowManager` 扩展（`firstcoder/context/manager.py:89+`），决策数据进 trace |
| 通用 `tail_clip()` 作为安全压缩手段 | `pico/core/turn_history.py` | 可作最后的软裁剪，但不能替代 L1-L4 的安全压缩（archive backing / 序列校验 / lifecycle） |
| pico 平行 compaction 机制 | 整套 | 会造成两套 watermark/target/event/fallback/checkpoint 语义，禁止 |

---

## 3. 第三层：harness 数据契约与实验方法

### 3.1 是 FirstCoder 缺失，还是 pico 亮点？

**偏"部分缺失"——FirstCoder 已有 benchmark 基础，pico 的价值是 run-level 证据与专项实验方法。**

**FirstCoder 已有（不用重复建设）：**

- Harbor benchmark 集成：`benchmark/harbor/`（FirstCoderHarborAgent、`firstcoder --benchmark` 非交互模式、verifier/reward 集成）
- **实际结果**：Aider Polyglot 225 tasks / 213 passes from 221 explicit rewards = **96.38% reward-only pass@1**，**94.67% end-to-end Harbor mean**（结果包 `benchmark/runs/harbor/aider-polyglot-feedback-retry-20260726/2026-07-26__12-07-27/`，见 `AGENTS.md:52`）
- 1297 个测试；CompactionEvent / SessionEvent 事件体系

**FirstCoder 缺失的部分（pico 亮点所在）：**

| 能力 | pico 证据 | FirstCoder 现状 |
|---|---|---|
| run-level 证据链（run_id / task_state.json / trace.jsonl / report.json / artifacts/） | `pico/core/run_store.py:19-45, 39-61, 86-96`；`pico/core/task_state.py:12-46` | 有 session JSONL 事件，但无独立 run 概念、无 task_state、无最终 report |
| stop_reason 枚举（9 种：final_answer_returned / step_limit_reached / retry_limit_reached / model_error / tool_timeout / approval_denied / persistence_error / resume_load_error / final_gate_blocked） | `pico/core/task_state.py:17-25` | 无等价枚举（AgentTurnResult 是进程内对象） |
| verification 语义（verifier_suggestions / evidence_summaries / final readiness） | `pico/core/runtime.py:831-858`（report 结构） | `diagnostics` 只执行命令返回 ToolResult（`firstcoder/tools/diagnostics.py`），无"验证覆盖了哪些变更"的归因 |
| context cost 配对实验（treatment/control、claimable_cost_win 只在 actual 且无质量回归时成立） | `pico/evaluation/context_cost.py:23-70, 576-694` | Harbor 只有任务级总分，无"某项设计是否有效"的消融实验 |
| memory 质量指标（recall / precision / stale-use / secret-exposure / abstention / false-resume） | `pico/evaluation/memory_agent_eval.py`（指标见 `:1222, :1307` 附近） | 无（记忆层本身都还没有） |
| dream 质量指标 | `pico/evaluation/dream_quality.py` | 无 |

### 3.2 引入清单

| # | 引入内容 | pico 源路径 | 引入方式 | FirstCoder 落点 |
|---|---|---|---|---|
| H1 | **Run 数据契约**：run 目录布局（task_state.json / trace.jsonl / report.json / artifacts/） | `pico/core/run_store.py:19-45, 39-61, 86-96` | 迁移数据契约；**保留其原子写模式**（`_write_json_atomic`，`run_store.py:98-112`，temp + replace） | 新建 `firstcoder/harness/run_store.py` |
| H2 | **TaskState 字段契约**（run_id/task_id/user_request/status/attempts/tool_steps/last_tool/stop_reason/final_answer/checkpoint_id/changed_paths/artifact_graph/evidence_summaries；序列化 to_dict/from_dict） | `pico/core/task_state.py:12-46, 110-129` | 原样迁移（纯 dataclass，零外部 import，最干净的一层） | `firstcoder/harness/task_state.py` |
| H3 | **trace 事件契约 + redact**（emit_trace 先脱敏再追加 + 持久化） | `pico/core/runtime.py:675-697` | 事件名对照 FirstCoder SessionEvent 映射 | `firstcoder/harness/trace.py` |
| H4 | **report 字段契约**（run_id/task_id/status/stop_reason/final_answer/tool_steps/attempts/prompt_metadata/compactions/verifier_suggestions/evidence_summaries/redacted_env） | `pico/core/runtime.py:831-858` | 迁移字段契约；字段定义为数据契约而非从 agent 拉属性 | `firstcoder/harness/report.py` |
| H5 | **context cost 配对实验方法**（`CostUsage`/`ExperimentRow` 数据结构、`_usage_from_trace` 解析、`_paired_rows` 配对、`_claimable_cost_win` 规则、`write_experiment_artifacts`） | `pico/evaluation/context_cost.py:23-70, 105-170, 488-542, 576-694, 442-456` | 迁移数据结构 + 统计逻辑；**不搬实验运行时**（`_build_*_agent` 构造真实 Pico 的部分，`:705-879` 不搬） | 新建 `firstcoder/harness/experiments/context_cost.py` |
| H6 | **memory 质量指标**（recall / precision / stale-use / secret-exposure / abstention / false-resume 的定义与 fixtures） | `pico/evaluation/memory_agent_eval.py` | 迁移指标定义 + challenge fixtures | `firstcoder/harness/experiments/memory_eval.py` |
| H7 | dream 质量指标 | `pico/evaluation/dream_quality.py` | 迁移指标 | `firstcoder/harness/experiments/dream_quality.py` |

### 3.3 明确不引入

| 内容 | 源路径 | 不搬原因 |
|---|---|---|
| Harbor 之外的 benchmark 执行器 | `pico/evaluation/harnessbench.py` 等 | FirstCoder 已有 Harbor 集成，不重复造 runner |
| `metrics.py` 巨型模块 | `pico/evaluation/metrics.py`（2156 行） | 与 pico 运行时深度耦合，只取指标定义 |
| 实验运行时（构造真实 Pico + ScriptedModelClient + 依赖 pico trace 事件名契约） | `pico/evaluation/context_cost.py:705-879`、`pico/testing.py` | 依赖 pico 运行时，移植前必须先对齐 trace 事件 schema，否则解析全部落到 estimated_proxy |
| `Engine.run_turn` / `completion_governance` | `pico/core/engine.py:76-87` 等 | 与 FirstCoder AgentLoop 生命周期重复，重写接线 |

---

## 4. 铁律：三层都不许碰的 FirstCoder 核心

- **L1-L4 压缩内核**（`firstcoder/context/compaction.py` / `manager.py` / `llm_compact.py` / `checkpoint.py`）——task-boundary compaction 是最大卖点，不许被 pico 的任何机制替换
- **事件溯源事实层**（`store.py` / `writer.py` / `context_builder.py`）——记忆只能以"旁路 + 审计事件"接入，不许污染 session 历史
- **SystemPrompt fingerprint 缓存**（`system_prompt.py`）——记忆注入走 `_request_messages`，不许进 system_prefix
- **依赖规则**（`docs/ARCHITECTURE.md`）：context 不能 import agent；记忆注入必须保持 agent → context → memory 方向
- **权限系统 / 工具注册**（`permission_registry.py` / `session_registry.py`）

---

## 5. 给 Codex 的审查任务

请对照两个项目的实际代码，逐层验证以下问题，输出定稿意见（同意 / 逐条修改 / 新增遗漏）。

### 5.1 通用问题（三层都要）

1. **路径核对**：本清单每一行引用的 pico 源路径（`文件:行号`）是否准确？读取时发现行号漂移或文件不存在，请给出修正后的路径。
2. **判断核对**：每一层"是 FirstCoder 缺失，还是 pico 亮点"的判断是否成立？有没有我们**漏掉的 FirstCoder 已有能力**（导致某条引入其实不必要）？有没有我们**高估或低估的 pico 模块**（导致某条引入过度或不足）？
3. **遗漏依赖**：引入某条内容时，它实际依赖的 pico 内部函数/常量/import 是否在本清单中列出？未列出的请补全（例如共享正则源、辅助函数、配置默认值）。
4. **风险**：每层最可能出问题的 3 个点（架构冲突 / 语义漂移 / 迁移遗漏）。

### 5.2 第一层（记忆）专项问题

1. `pico/features/memory.py` 的纯函数层（744-1510）是否真的零运行时依赖？有没有隐式依赖（如模块级常量、跨函数状态）需要一并迁移？
2. `DurableMemoryStore` 的写盘升级方案（temp+rename + portalocker + CAS）是否完整覆盖了 pico 现有的并发场景（跨进程 dream、多 session 写日志）？
3. 记忆注入走 `AgentLoop._request_messages`（`firstcoder/agent/loop.py:998-1021`）是否会与现有 token 预算计算（`:921-932`）和 task_plan 快照注入冲突？
4. `<memory>` 标签提取挂 `_complete_turn`（`loop.py:728-730`）但必须排除 interrupted/limit/permission 合成回复——具体判定条件应该是什么？
5. 记忆事件（audit-only）与 store 重放的关系：`store.py:92-94` 忽略未知事件，第一版 audit-only 是否真的可行？还是需要一版就补 replay？

### 5.3 第二层（上下文编排）专项问题

1. Section policy（C1）与 FirstCoder 现有 `ContextBudget` 的接合方式：是扩展 `ContextBudget`，还是独立 policy 层在 `_request_messages` 里消费预算结果？给出你认为最不侵入的接法。
2. C2/C3（压力分档 + actual 校准）接入 `ContextWindowManager.compact_if_needed`（`manager.py:109+`）时，会不会与现有高水位触发、熔断、no-effect 去重产生双重触发？如何避免？
3. C4/C5（量化度量 + 渲染报告）的数据该进哪个落点：CompactionEvent？独立 SessionEvent type？run report（H4）？请给出归属建议。
4. C6 三条"取思想"（retention/replacements/handoff）中，哪些其实与 FirstCoder 现有机制（lifecycle / archive / checkpoint）**真重叠**（应直接放弃），哪些才是真增量？

### 5.4 第三层（harness）专项问题

1. run-level 证据（H1-H4）与 FirstCoder 现有 session JSONL 事件（`store.py`）的关系：是**平行双写**，还是 run 作为 session 事件的**投影/聚合视图**？给出与现有架构一致的建议（注意 FirstCoder 的 facts vs view 契约）。
2. `stop_reason` 枚举（H2）与 FirstCoder `AgentTurnResult` / 现有终止路径（正常完成、轮次上限、中断、权限拒绝）的映射表。
3. H5 context cost 配对实验在 FirstCoder 上跑，需要哪些 FirstCoder 侧的最小改动（provider metadata 是否已透出 actual tokens？`_request_messages` 是否可导出 prompt 估算）？
4. H6 memory 指标当前无可跑对象（记忆层尚未实现）——建议实验先在什么尺度上跑（单元级 fixture？集成级？）

### 5.5 最终定稿

1. 本清单的**引入顺序**（M1-M12 → C1-C6 → H1-H7）是否有依赖顺序错误？
2. 缺什么你认为必须引入、而本清单没写的 pico 内容（如果有，说明理由和路径）？
3. 最终定稿意见：**同意 / 有条件同意（列出条件）/ 拒绝（列出原因）**。

---

## 6. 附：核心叙事（文档背景，供审查时对照）

简历/面试主线：

> **FirstCoder 是基底：自研 agent 执行链 + 事件溯源事实层 + task-boundary compaction（核心亮点）+ Harbor benchmark（96.38% pass@1 / 94.67% e2e）。**
> **pico 三层引入：记忆层（补最高频功能缺口）、上下文编排层（让压缩可编排、可度量）、harness 数据契约（给卖点提供可审计证据和消融实验）。**

引入后产品叙事：一个能识别任务变化、主动整理上下文、跨会话保留有价值信息、并且能用 benchmark 和 run evidence 证明效果的 coding agent。

---

## 7. 评审结论与定稿（v2）

> 评审人：Codex（源码级审查，读两项目实际代码，未改文件、未跑测试）+ 人工自查（§5 关键问题逐项代码验证）。双方结论在注入点、双触发规避、C4/C5 落点、C6 取舍、run 投影关系、audit-only 事件等核心判断上完全收敛。

### 7.1 最终意见：有条件同意（Codex 8 条条件 + 自查 3 条修正）

Codex 条件（必须满足）：

1. 不整体移植 pico 运行时、`ContextManager`、`run_dream`、evaluation runner。
2. 记忆迁移定义为"**数据面迁移 + FirstCoder runtime orchestration 重写**"——不是"全量原样移植纯函数"（`memory.py:744-1418` 内含 `DurableMemoryStore`/路径推导/状态规范化，非纯函数；纯函数只是 `default_memory_state`/`_ensure_list`/`_dedupe_preserve_order` 等局部辅助）。工作量评估：**中高等**。
3. RunStore/harness 契约（P1）必须先于 benchmark 和 auto-dream。
4. 安全边界（脱敏/quarantine/路径约束/原子写/跨进程锁）必须先于 memory tools、写盘和 prompt injection。
5. 记忆注入不进入 stable system prefix，且必须参与 FirstCoder 现有预算计算（已验证：`_context_budget_for_view` 调用 `_request_messages`，同一函数注入即自动入预算）。
6. 不引入第二套 compaction；pico 的 section policy / pressure tier 只能扩展 FirstCoder 现有 L1-L4（tier 只生成一个 canonical context decision 交 `compact_if_needed()`，不得与 watermark 各自触发）。
7. auto-dream 不直接复用 FirstCoder 内存态 `BackgroundJobManager`，也不照搬 pico 嵌套 `Pico` 实例。
8. 完成 stop reason、provider usage、verification、report 的统一 schema 后，才能声称融合方案具备可审计 harness 证据。

自查补充修正：

9. **C6 再砍一刀**：handoff 生成机制 ≈ L4 checkpoint 真重叠（`HandoffAdapter` 依赖 pico `complete_model`），只保留 Goal/Next Steps 等 schema 思想；retention 的 protected/bulk 规则 ≈ L1/L2 重叠，仅 failed-tool + changed_paths 语义保护是真增量；replacements 仅跨轮去重账本（event_id+sha256）是真增量。
10. **C1 最不侵入接法**：第一版只做度量（C4/C5 报告），不做主动裁剪——policy 作为外部策略输入、在渲染时消费，不拥有压缩执行权。
11. **C5 补依赖**：`context_report.py:9` 依赖 `ContextUsageAnalyzer`（`context_usage.py`），引入 C5 须一并迁移或剥离该字段。

### 7.2 路径/行号修正（Codex 核对结果）

- `pico/features/memory.py:744-1418` 不能称纯函数层（含 `DurableMemoryStore` :760-1020）；记忆检索实际分段：`_iter_retrieval_notes` :1421-1429 / `_ranked_retrieval_notes` :1431-1450 / `retrieval_view_structured` :1453-1465 / `retrieval_view` :1473-1482（M4 引用行号按此修正）。
- `pico/core/context_budget_summary.py` 实现从 `:6` 起（不是 `:5`）。
- `pico/core/context_orchestrator.py` 完整类延伸至 `:202`（事件写入 :184-202）。
- 记忆命令入口还包括 `pico/cli.py:449-460` 与 `pico/core/runtime.py:640-666`（M12 不能只引 `pico/commands/`）。
- `firstcoder/memory/`、`firstcoder/harness/` 等为拟新增目标路径，实施时须明确标注"目标路径"。
- **Windows 平台坑**：`write_dream_report` 默认时间戳含 `:` 直接拼文件名（`pico/features/memory.py:262-268`）——Windows 上直接失败，迁移时时间戳必须改为合法格式（M10 必改项）。

### 7.3 判断修正与新发现

- "FirstCoder 缺跨 session 记忆 + 独立 harness"判断成立，但表述修正为：**FirstCoder 有 session facts 和 context evidence，缺 durable memory projection 与 run-level audit/report 契约**。现有局部能力：`ContextInspectionReport`（`firstcoder/context/inspector.py:21-45`）、`ToolExecutionEvent`（`firstcoder/agent/tool_execution.py:68-88`）、`AgentTurnResult` 仅 COMPLETED / WAITING_FOR_USER_INPUT（`firstcoder/agent/user_input.py:17-30`）。
- **并发写盘补强**（M1/M3 必改项）：daily log 裸 append（memory.py:102-110）、topic metadata 直接覆盖写（:821-826）、MEMORY.md 覆盖写（:937-945）、topic 文件+metadata 多文件分步发布（:947-968）、pico 锁存在 TOCTOU（:320-337）。须补充：memory root 路径约束、daily log 跨进程追加策略、index/topic/metadata 事务发布或版本校验、stale writer 检测、dream 读取期间快照语义、写失败恢复与审计。FirstCoder 侧先例：`portalocker` 已用于 task plan（`firstcoder/planning/service.py:119-129`）。
- **H2 TaskState 字段增强**：增加 `session_id`、schema version、workspace fingerprint、parent run id。
- **H5 工作量修正**：provider 已透出 actual usage（`ChatResponse.usage` `firstcoder/providers/types.py:167-177`；assistant 事件已保存 usage `firstcoder/agent/session.py:386-405`），但 `TokenUsage` 只有 input/output/total（types.py:52-62），缺 cached tokens、统一 request trace、prompt 估算持久化、protocol/base URL、request↔response 配对 → H5 前必须先建 **provider-call metadata contract**（归入 P1）。
- **stop_reason 映射表**（FirstCoder 现状 → pico 枚举）：
  - 正常最终回答 `loop.py:639-642` → `final_answer_returned`
  - tool round limit `loop_limits.py:9-12` → `step_limit_reached`（保留原始原因）
  - provider call limit `loop.py:1161-1168` / timeout `loop.py:1170-1175` → `step_limit_reached` / `model_error` / `tool_timeout`（工具结果失败或 sandbox timeout）
  - cancel/interrupted `loop.py:669-676, 1232-1240` → 扩展 `cancelled/interrupted`（pico 枚举无此项，须扩展）
  - permission waiting `user_input.py:17-30` → `approval_denied`（permission registry DENY 路径）
  - persistence error（session writer/store 失败）→ `persistence_error`；resume 失败 → `resume_load_error`
  - `final_gate_blocked`：**FirstCoder 当前无等价 final readiness gate，需新增**（P4）
- **H6 三步执行**：① provider-free memory fixture → ② durable store + provenance + quarantine 集成 fixture → ③ AgentLoop 注入、跨 session、真实 provider benchmark。

### 7.4 修订后的实施顺序：P0-P7（替换原 M1-M12 → C1-C6 → H1-H7）

原顺序存在依赖错误（harness 被后置、安全边界排在写入之后），修订为：

- **P0：ports、schema、路径约束、脱敏、quarantine、原子写、跨进程锁**（+ 静态检查与最小契约检查，CI 不等到最后）
- **P1：RunStore、TaskState、trace、report 契约 + AgentLoop stop_reason 映射 + provider-call metadata contract**
- **P2：memory 数据面：daily log、durable topics、metadata、retrieval、provenance**
- **P3：显式 `/remember`、`/memory`、memory tools、受预算约束的 request injection（MemoryProjector）**
- **P4：verification signal、final readiness、context usage、cost report**
- **P5：memory fixture benchmark → AgentLoop 集成 benchmark**
- **P6：受限 auto-dream（独立 scheduler、持久任务状态、崩溃恢复、严格 write scope）**
- **P7：完整 CI 流水线与发布门禁**

### 7.5 H 清单新增（Codex："harness 含金量的核心，不是实验附属品"）

| # | 引入内容 | pico 源路径 | 说明 |
|---|---|---|---|
| H8 | trace 统一字段契约 | `pico/core/runtime_events.py:28-47` | 事件名对照 FirstCoder SessionEvent 映射（H3 的 schema 源） |
| H9 | evidence 汇总 reducer | `pico/core/evidence_summaries.py:18-44` | report 的 evidence_summaries 聚合逻辑 |
| H10 | verification signal + final readiness gate | `pico/core/verification.py:8-38`；`pico/core/final_readiness.py:19-52` | 新增 harness 能力，FirstCoder 无等价物 |
| H11 | 事件消费边界 | `pico/core/runtime_consumers.py` | harness 与运行时解耦的消费者注册模式 |

另：`runtime_secrets.py:9-54` 的环境变量与 **artifact 脱敏**（不只是 env 替换，M8 扩展为读写两侧都脱敏）。

### 7.6 其他遗漏依赖（pico 侧消费点，迁移时逐一对账）

`pico/core/context_manager.py:97-125`（上下文记忆 section）、`pico/core/engine.py:150-163`（检索/读取 trace）、`pico/core/runtime.py:142-179, 640-781`（memory 初始化/工具后更新/durable promotion/report）、`pico/core/session_lifecycle.py:1-50`（resume/clear 重建 memory）、`pico/core/runtime_checkpoints.py:50-73`（文件 freshness checkpoint）、`pico/core/tool_policy.py:58-63`（prior-read policy）、`pico/core/turn_history.py:208-211`（history 渲染读 memory summary）。

### 7.7 定稿

**有条件同意。** 三层结构（记忆 / 上下文编排 / harness 契约）与核心叙事不变；实施按 §7.4 的 P0-P7 执行，满足 §7.1 的 8+3 条条件。代码路径与行号以 §7.2-§7.6 修正后为准。
