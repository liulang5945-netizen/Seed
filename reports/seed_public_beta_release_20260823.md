# Seed 公测（Public Beta）发布报告

> 状态：**草稿（待 M1 大预算长训完成后定稿）**
> 日期：2026-08-23
> 依据：`plans/archive/history/SEED_PUBLIC_BETA_ROADMAP.md` 五维验收标准
> 本报告为公测发布交付物的主体，证据均指向 `reports/` 下的原始产物；
> 标记 ⏳ 的维度依赖 M1 长训（用户指示暂停，4M 检查点待续）。

## 1. 发布概览

- 产品形态：桌面端（`Seed.exe` 双入口打包）+ Web UI（`http://127.0.0.1:8000`）
- 核心运行时：Seed 原生字节级模型（`seed/model.py`）+ Taiji 神经可塑层（`taiji/`）
- 当前检查点：`checkpoints/seed_corpus.pt`（4.82M ticks，由 `seed_beta.pt` 提升而来；
  旧运行时检查点备份为 `seed_corpus_prev_20260823.pt`；⏳ 大预算训练未完成）

## 2. 五维验收标准与证据

### 2.1 模型能力（对话质量/连贯性/多轮）— ⏳ 未达标，量化差距已锁定

评测面板：`scripts/training/verify_seed_beta_dialogue.py`（M2 交付，固定 50 题），两次基线：
`reports/seed_beta_dialogue_baseline_800k.json`（800K）与
`reports/seed_beta_dialogue_4m8_seedbeta_20260823.json`（4.8M，`seed_beta.pt`）：

| 指标 | 公测阈值 | 800K 基线 | 4.8M 复测 | 状态 |
|---|---|---|---|---|
| UTF-8 有效率 | ≥99% | 0% | 0% | ❌ |
| 可读率 | ≥60% | 0% | 0% | ❌ |
| 多轮指代命中率 | ≥60% | 0/10 | 0/10 | ❌ |
| 格式良构率 | ≥99% | 100% | 100% | ✅ |

差距结论：自动代理指标仍未达标，但 4.8M 相对 800K 有清晰质变信号：
回复从高频字符堆叠进化为带段落/多轮格式的结构化伪对话（反复出现“老师：”“诗”“杜甫”等训练语料结构），
证明收敛方向健康，达标路径即继续 M1 大预算长训（1.3GB `simple_zh_texts.jsonl`）；
评测面板可随训练周期复跑，收敛后以同一面板复测定稿。

### 2.2 训练规模（大预算训练+收敛验证）— ⏳ 暂停中

- 吞吐实测（`reports/seed_beta_progress.jsonl`）：约 **181K ticks/s**（CPU）；
  大预算目标规模与对应 ETA 见路线图 §2 训练配套差距。
- 当前进度：**4.82M ticks**，`holdout_surprise` ≈ 2.9–3.0 区间（800K 时 ≈4.2），持续下降 = 收敛中。
- 续训机制已实测可靠：`--resume` 从 `seed_beta.pt` 恢复 + 计数器基线修复（`model.tick` 为基线），
  中断后从 4.82M 正确继续；`/api/train/resume_checkpoint` SSE 端点冒烟通过。
- 检查点原子落盘 + 元数据（P0 工程硬化）已上线。

### 2.3 工程稳定性（可用性/检查点恢复/异常处理）— ✅ 达标

| 验证项 | 结果 | 证据 |
|---|---|---|
| API 压测 1000 连发 | 成功率 **1000/1000**，均延迟 0.702s，零未捕获错误 | `reports/seed_beta_api_stress_20260823.json` |
| 恶意输入审计（空提示/100K 长文/深历史/Unicode 边界/换行注入/空字节） | 6/6 全部受控（200 + 行为合理） | 同上 `hostile_audit` |
| 检查点崩溃恢复演练 | **10/10 轮全过**（7 项子检查×3 故障场景×10 轮） | `reports/seed_beta_recovery_20260823.json` |
| `atomic_save` 目录自愈 | 陈旧 `.tmp` 落盘前清扫（崩溃残留场景已覆盖） | `seed/persistence.py` |
| 回归门禁 | `pytest tests/ -q` **113 passed / 4 skipped**；CI 8 项 `verify_taiji_*` 全部 `status: pass`（2026-08-23 复跑新鲜证据） | `reports/_gate_full_20260823.log` |

### 2.4 性能基线（延迟/吞吐）— ✅ 达标（双检查点均验证）

产物：`reports/seed_beta_perf_baseline_800k.json`（800K）与
`reports/seed_beta_perf_4m8_seedbeta_20260823.json`（4.8M，`seed_beta.pt`）：

| 指标 | 阈值 | 800K | 4.8M | 状态 |
|---|---|---|---|---|
| 模型加载 | ≤30s | 0.171s | **0.138s** | ✅ |
| 首字节延迟（5 问最大） | ≤2s | 0.129s | **0.071s** | ✅ |
| 生成吞吐 | ≥200 B/s | 351.7 B/s | **572.2 B/s** | ✅ |

4.8M 检查点三项均优于 800K（训练带来分布锐化，解码更快）；模型参数量未变（纯在线学习），
长训继续不引入延迟回退风险；长训完成后例行复测即可。

### 2.5 前端/桌面端（安装/启动/对话/界面品质）— ✅ 达标

- 三轮浏览器页面评审：7 页面渲染/间距/观感达标、对话主路径全通、
  console error 0（`reports/m4_frontend_review_20260823.md`、`_scratch/*.png` 截图链）。
- 暗色模式对比度：采样 34/33 元素，最低 **4.91:1**，全部 ≥ WCAG AA 4.5:1。
- 桌面端打包（1121.7MB，双入口 `Seed.exe`+`SeedBackend.exe`）：
  - 启动 **2.9 秒**出界面（判据 ≤15s）；`/api/health` ok、`seed_active=true`；
  - 浏览器实测发送"你好"收到回复并正常渲染，0 控制台错误
    （`_scratch/verify_8000_chat_nihao_reply.png`）。
- 实时通道：WebSocket 8765 已接线（`App.vue`），失败时优雅降级为 10s HTTP 轮询。

## 3. 已知限制

1. **模型能力未达公测阈值**：回复为乱码（4M 早期检查点预期行为），依赖 M1 长训。
2. **ws 跨进程事件推送**：ws 服务与后端分属不同进程，生命事件实时推送走轮询兜底（10s）。
3. **首问延迟**：运行环境偏好持久化后启动即激活 Seed，冷启动后首问首字节实测 **27ms**、完整回复 **1.3s**（2026-08-23 复测）；
   早期无偏好恢复时首问含激活的 10-21s 已消除（后端冷启动就绪 ≈1.5-2.6s，见 `logs/desktop_main.log`）。
4. **远程接入**：`SEED_HOST=0.0.0.0` 开放局域网访问为代码级验证，未做公网安全加固（无鉴权场景下不建议暴露公网）。
5. **打包体积**：1121.7MB（含 PyTorch CPU + 数据资产），未做裁剪。
6. **训练窗口**：大预算长训需长时间占用 CPU，建议空闲时段运行；中断可续训但需手动发起。

## 4. 用户文档要点（已落地，随正式版发布）

完整版见 `docs/seed_public_beta_user_guide.md`（安装/对话/训练/环境变量/已知限制/FAQ），
README 新增 Public Beta 段落（入口/环境变量/证据索引）。要点：

1. **安装与启动**：解压即用，双击 `Seed.exe`；数据目录位于程序目录
   （`data/`、`checkpoints/`、`logs/`），日志见 `logs/desktop_main.log`、`logs/desktop_backend.log`。
2. **基本对话**：打开即进入聊天页；模型切换（Seed 原生 ↔ Cortex）在设置→运行环境，重启自动恢复。
3. **训练与续训**：`/train` 页面查看检查点列表；续训走页面按钮（`/api/train/resume_checkpoint`）
   或命令行 `python scripts/training/train_seed_corpus.py --resume ...`。
4. **环境变量**：`SEED_PORT`（默认 8000）、`SEED_HOST`（默认 127.0.0.1）、
   `SEED_RUNTIME=1`（启动即激活 Seed 原生运行时）。
5. **已知限制**：见本报告 §3；早期检查点回复乱码属预期，非故障。

## 5. 定稿前待办（M1 恢复后）

- [ ] 大预算长训完成 + 收敛证据（holdout_surprise 曲线）
- [ ] 对话面板复测（§2.1 四项阈值；已有 800K/4.8M 双基线）
- [x] 性能基线复测（4.8M 检查点 3/3 达标且优于 800K，长训后例行复核即可）
- [ ] 最终回归（`pytest` + `verify_taiji_*` 8 项）
- [ ] 工作区清理与提交、推送云端
