# Taiji 生命系统（G4）定型 v1 — 2026-09-23

> 本文件是 **G4 的实现合同**：先定「面板显示什么、什么进上下文、能写什么、读不到时怎么表现」，再动代码。
> 立项依据与判据见 [TAIJI_HARNESS_ADOPTION_BRIEF_20260922.md](TAIJI_HARNESS_ADOPTION_BRIEF_20260922.md) §3（G4：`dsh-life` 面板 + 上下文注入 + 训练/知识并入；判据：面板实时反映 runtime 状态／生命状态出现在请求上下文／训练可在面板启停）。
> 两侧原料与挂点的证据清单（端点表、字段表、槽名、注入 API、装配三面）由本轮两次只读调研产出，关键证据在下文逐条内联。

## 0. 四项裁决（2026-09-23，所有者逐题裁定）

| 议题 | 裁定 | 理由（记录用） |
|---|---|---|
| 面板落点 | **左侧全局面板**：页面注册进 `'main'`(keyed, key=`'life'`)、入口注册进 `'sidebar.panellist'` | 与 G4 措辞「侧栏面板」一致，空间足以容纳生命/训练/知识/健康四区＋控制项；`ui-plugin-manager` 是同构现成范例 |
| 写操作范围 | **训练全量 + 生命调度启停与手动动作** | 训练：启动/暂停/继续/停止/重置；生命：`life/start`、`life/stop`、`feed`/`sleep`/`play`。端点均已存在，且与「生命系统」语义一致 |
| 上下文注入 | **动态 `agent/pre-step` ＋静态 `systemPrompt.section`** | pre-step 走 durable 运行时上下文（Trajectory 可复盘「当时模型看到的生命状态」）；section 只放静态策略/格式 |
| 三处恒假读数 | **先修 runtime 源**，再上面板 | 面板显示恒假值比不显示更坏；这是「失败即读数」原则的直接落地 |

## 1. 数据源与真实性契约（先定来源，再定面板）

### 1.1 三级来源与门控

| 来源 | 端点 | 门控 | 给出什么 |
|---|---|---|---|
| A 常开（主源） | `GET /api/runtime/status`（`api/routes_runtime.py:36`，响应 `models_runtime.py:89-99`） | 无（始终注册，`api/app.py:232,242`） | `health`（state/model_loaded/model_name/is_taiji/is_seed/startup_complete）、`memory`（total_gb/available_gb/used_pct）、`auth`、**`life`**（native homeostasis 优先）、`tools`、**`training`**（is_training/pause_requested/stop_requested/publishing） |
| A′ 常开（训练控制） | `POST /api/train/native`、`/api/train/resume_checkpoint`（SSE）、`pause`/`resume`/`stop`/`reset`、`GET /api/train/checkpoints`（`api/training/*`） | 无（`app.py:239,257`） | 训练启停 + 进度流（`progress{fraction,step,loss,elapsed,eta,epoch,total_epochs,samples_per_sec,total_steps}`、`completed{checkpoint,ticks_added}`）＋检查点清单 |
| B legacy 增强 | `GET /api/life/status`、`/api/taiji/life/status|timeline`、`/api/rag/status`、`/api/taiji/{feed,sleep,play}/status` | `SEED_ENABLE_LEGACY` 且 neuroplex 可导入（`api/legacy_bridge.py:205-219`；`app.py:202-213`） | 五需求（hunger/fatigue/boredom/stress/curiosity，0..100）、dominant_need、心跳/事件计数、时间线、人格/进化阶段、知识库统计（doc_count/chunk_count/has_embeddings/embed_dim，`api/routes_rag.py:195-210`） |

**生命分节的 native/legacy 判别**：`/api/runtime/status.life.status` 为 `"seed"`（native，`seed_platform/runtime_service.py:144-181`）或 `"ok"`（legacy，`:184-219`）。native 只给 `needs{curiosity,fatigue,stress}`（0..1，来自 `runtime.homeostasis_status()`），legacy 给 5 项（0..100）。面板**按来源不同渲染不同量纲**，不换算、不补齐——两套语义不同，换算即编造。

### 1.2 先决修（本仓 Python，G4 第一步，独立可验）

三处恒假读数是面板显示假值的根源，按裁定修源（证据行号来自本轮调研）：
1. `life` 分节（`runtime_service.py:184-219`）：删去永远取不到键的 `total_interactions`/`uptime_seconds`；改读 `scheduler.get_status()` 的真实键（`life_scheduler.py:462-473`：`life_state/is_running/needs/dominant_need/last_heartbeat/last_activity/total_heartbeats/total_events`）；native 分支给 `tick`（`seed_runtime.py:1436-1466` 的 homeostasis.tick），**不**伪造 uptime。
2. `training` 分节（`runtime_service.py:267-281`）：`pause_requested`/`stop_requested` 改读 `app_state.pause_training_requested`/`stop_training_requested`；`publishing` 改读真名 `app_state.publishing`（`seed_platform/app_state.py:36-41`）。
3. 纠正后加定向 pytest 钉住「字段来自哪个属性/函数」，避免再次静默漂移。

### 1.3 新鲜度与降级（面板与注入共用）

- 快照带 `observedAt` 与 `source`；面板显示「来源 + 观测时间 + 轮询状态」，不做「看起来是实时的假象」。
- legacy 关闭时：B 类行整块隐藏并标「legacy 未启用」；**不**用 native 值顶替。
- runtime 不可达：面板显示 `down` + 最后一次成功时间；注入侧整块省略（§3.2）。
- 每类失败进快照的 `unavailable: string[]`（面板可展开看原因），并写一条 warn（每类每次进程内只写一次，避免刷屏）。

## 2. 插件布局与接口面

三个新包（沿用 harness 的 host/client 分面与 `@Remote` 契约，证据见 `docs/api-gateway.md`、`packages/api/workspace-controller/*`）：

### 2.1 `packages/api/life-controller`（host + client 半边）

- **Host**：`LifeController extends TypertRemoteService`；持有对 Python runtime 的 HTTP 客户端（默认 `http://127.0.0.1:8000`，沿用 `llm-taiji` 的 baseURL 约定）与一个轮询器（基线 5s；训练进行中 2s；单次请求超时 2s）。
- **Remote 面**（unary 各带稳定错误码）：
  - `snapshot()` → 一次完整快照（面板首屏/重连用）。
  - `follow(signal)`（`@Remote({ mode: 'stream' })`）→ **首帧 baseline，随后增量**（与 `packages/api/workspace-controller/src/feed.ts:85-96` 同范式）。
  - `trainStart({ dataset?, preset?, parameterBudget?, seed? })` / `trainPause()` / `trainResume()` / `trainStop()` / `trainReset()`。
  - `lifeStart()` / `lifeStop()` / `lifeAction({ action: 'feed'|'sleep'|'play', reason? })`（B 类，legacy 不可用时以 `life/unavailable` 拒绝）。
- **快照形状（v1，字段即为合同）**：
  ```
  {
    source: 'native' | 'legacy' | 'absent',
    observedAt: string, fresh: boolean, pollIntervalMs: number,
    health: { state, modelLoaded, modelName, seedActive, startupComplete } | null,
    memory: { totalGb, availableGb, usedPct } | null,
    life: null | { isRunning, dominantNeed?,
      native?: { tick, needs: { curiosity, fatigue, stress }, drives? },
      legacy?: { needs: { hunger, fatigue, boredom, stress, curiosity }, totalHeartbeats, totalEvents, lastHeartbeat, lifeState } },
    training: { isTraining, pauseRequested, stopRequested, publishing,
      progress?: { fraction, step, loss, elapsed, eta, epoch, totalEpochs, samplesPerSec, totalSteps },
      checkpoints: Array<{ filename, step, bytes, modifiedUtc, savedAtUtc, numEpochs }> },
    knowledge: null | { docCount, chunkCount, hasEmbeddings, embedDim },
    availability: { runtime: 'ok'|'down'|'degraded', legacy: 'ok'|'disabled'|'down', rag: 'ok'|'disabled'|'down', trainingStream: 'idle'|'streaming'|'closed' },
    unavailable: string[]
  }
  ```
- **训练进度**：`trainStart` 在 host 侧消费 SSE 并折叠为 `training.progress` 增量；流断开时把 `availability.trainingStream='closed'` 并保留最后进度（面板显示「流已断开，进度可能落后」+ 可重新查询 checkpoints）。

### 2.2 `packages/context/life-context`（纯 host 插件）

- `inject = ['systemPrompt', 'lifeController']`；订阅快照（同源，不再另开 HTTP）。
- 静态策略段：`ctx.systemPrompt.section({ name: 'life:policy', order: getSectionOrder('LIFE_POLICY'), text: () => LIFE_POLICY })`。
  `LIFE_POLICY`（英文，固定文本，随包发布）：声明「下面是宿主生命系统的当前张力读数；它是内部状态而非事实断言，不得当作证据引用；缺失时忽略」。
- 动态注入：`ctx.on('agent/pre-step', handler)`（`packages/core/agent/src/runtime-types.ts:309-320` 的 waterfall，`next()` 保序），在 `next()` 之后追加一条 durable 运行时上下文消息（机制同 `packages/context/time-context/src/index.ts:188-228`）。
- **注入契约（v1）**：
  - 载荷：单行紧凑块，`<life-state source age needs drives training knowledge>`，**≤ 400 字符**（配置 `maxChars`，超限截断并加 `truncated="1"`）。
  - 节流：默认 `refreshIntervalMs = 30000`；或快照关键量变化超阈值（`dominantNeed` 变化、needs 单项变化 >0.1、训练态翻转）时立即注入。
  - 省略条件：runtime `down`、快照 `fresh === false`（> 60s 未更新）、`life === null` 且训练态未知 → 整块省略（不写空块），每类只 warn 一次。
  - 开关：`enabled`（默认 true）、`refreshIntervalMs`、`maxChars`；关闭时不注册 pre-step handler（section 也不注册），不留半态。

### 2.3 `packages/client/ui-life`（client 半边）

- 注册：页面进 `'main'`（key `'life'`，`MainPanelId`），入口进 `'sidebar.panellist'`（`kind:'list'`，带 icon/order/label），经 `ctx.layout.selectPanel('life')` 切换（范例 `packages/client/ui-plugin-manager/src/client/index.ts:85-106`）。
- 分区（自上而下）：
  1. **来源与新鲜度**：`source` + `observedAt` + 轮询状态 + `unavailable` 展开。
  2. **生命**：native（needs 三项条形 + tick）或 legacy（五需求条形 + dominant + 心跳/事件计数 + life_state）；[启停调度]（legacy）、[喂食/睡眠/玩耍]（legacy）。
  3. **训练**：状态徽标 + 进度条（loss/step/eta/epoch）+ 按钮组（启动/暂停/继续/停止/重置，停止与重置需二次确认）+ checkpoints 表（文件名/step/大小/时间）。
  4. **知识**：doc/chunk/embed_dim/has_embeddings；rag 不可用时整区标 unavailable。
  5. **宿主**：health.state、seed_active、model_name、memory used%。
- 文案：`ctx.locale.register(NS, { zh, en })`（typed、成对，门禁 `verify-client-ui-i18n` 禁止硬编码文案）；错误码 → 可读文案映射。
- 写操作语义：执行中禁用按钮并显示进行态；失败显示 Host 返回的稳定错误码文案（不吐裸 RPC 文本）；成功后以 `follow()` 增量自然刷新（不手工改本地状态）。

## 3. 装配与包文件（缺一即在某阶段失败，证据见 `packages/client/AGENTS.md` 的 checklist）

| 面 | host 侧 | client 侧 |
|---|---|---|
| tsconfig | `tsconfig.host.json` references | `tsconfig.client.json` references |
| bundle | `packages/bundle/base/cordis.patch.yml` 加行 + `base/package.json` 依赖 | `packages/bundle/web-app/cordis.patch.yml` 的 `- insert:` 加 `dsh.client` row + `web-app/package.json` 依赖 |
| 别名 | `pnpm run gen-tsconfig-paths` 生成（不手改 `tsconfig.base.json`） | 同 |
| 包内 | `package.json` 不变量（version/type/main/types/exports/peer+dev 双写/files）、`src/index.ts`、README 对（含 Model Experience 段）、`README.i18n.yaml` | 另需 `exports['./client']`、`dsh.client` manifest、`tsdown.config.ts`（`clientBundle(id, ['lib/types/index.js'])`）、`src/css-modules.d.ts`、`src/client/**`、`locales.ts` |

## 4. 验收映射（G4 三条判据 → 测试与读数）

| 判据 | 实现 | 证据 |
|---|---|---|
| ① 面板实时反映 runtime 状态 | `follow()` baseline+增量 + 轮询 + 训练 SSE 折叠 | host spec：假 runtime（无外部依赖）驱动快照与增量；client spec：渲染各来源/降级态；快照测试：面板首屏；真机：起 `python api/main.py` + harness，观察 needs 与训练进度变化 |
| ② 生命状态出现在请求上下文 | `life:policy` section + `agent/pre-step` 注入 | host spec：假快照 → 断言注入文本（含 needs/training/knowledge）、节流生效、`down`/过期时整块省略且只 warn 一次；Trajectory 中可见该 durable 消息 |
| ③ 训练可在面板启停 | `trainStart/Pause/Resume/Stop/Reset` → Python 端点 | host spec：断言请求路径/方法与错误码映射（含 legacy 不可用时的拒绝）；真机：启动→进度→暂停→继续→停止→checkpoints 出现新条目 |
| ④ 先决修（三处假读数） | §1.2 | 定向 pytest 钉住字段来源；`/api/runtime/status` 真机读数前后对比 |

## 5. 边界（G4 v1 不做，避免范围蔓延）

- 知识库**只读**（不做 upload/rebuild/clear）；生命引擎的 self_mod、进化触发**只读**；训练超参编辑不做（面板只展示 Python 侧推荐的 `selected` 值）。
- 不做「按会话/工作区区分注入」——harness 的 prompt scope 单位是 agent（`packages/core/scope/README.md:27-46`），v1 全局注入，按 agent 取消（`agent/ctx`）留给后续。
- 不引入新的持久化：面板状态全部来自 runtime 快照 + 既有 checkpoints；生命数据落盘仍由 Python 侧负责。

## 6. 实施顺序（每步独立可验、可提交）

1. **先决修**：Python 三处读数 + 定向 pytest（本仓根）。
2. `packages/api/life-controller`（Host 轮询 + Remote 面 + 训练 SSE 折叠）＋ host spec。
3. `packages/context/life-context`（section + pre-step 注入）＋ host spec。
4. `packages/client/ui-life`（面板四区 + 控制）＋ client spec + locale 对。
5. 装配三面 + `gen-tsconfig-paths` + README 对/i18n 记录；`pnpm run build`、`typecheck`、`lint`、`test:gui`、文档门。
6. 真机读数：起 Python runtime + harness web，逐条对判据取读数（面板/上下文/训练启停），并把读数写回 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) §5.7。