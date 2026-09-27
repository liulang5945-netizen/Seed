# SPEC-R2-02：解码掩码产品化（产品生成链的 UTF-8 硬约束）——开案预注册

日期：2026-09-27。归属：R2 主线语言能力（owner 裁定"2 之后 3"的第 2 步）。
判据与实现清单**先于代码冻结**。

## 0. 要修的、已量到的那件事（全部有盘上出处）

1. F0：无掩码基底链整句可解码率全 0（`reports/taiji_f0_language_floor_20260927.json`）；
   训练目标方向已试（SPEC-R2-01 分支 3），该预算内修不动。
2. M1 仪器（FROZEN）带 UTF-8 解码掩码：seed_beta＋掩码 ⇒ 成句率 0.3625、
   重训读出臂 A＋掩码 ⇒ 0.8688（`reports/taiji_r2_readout_retrain_verdict_v3_20260923.json`）。
3. 产品表层链（`api/seed_runtime.py:chat`）现状：基底 `generate_input`（无掩码）→
   `errors="replace"` 解码 → `NativeReadableTextLanguageOrgan._readable_surface`
   见到 `\ufffd` 一律拒收 → **用户永远看到占位句**（"当前原生语言表层正在形成稳定表达"）。
   这就是已登记"表层被占位句替换"问题的机制本体。
4. 产品里已有完整且更严的掩码实现（`taiji/language_alignment.py:789`，含 E0/ED/F0/F4
   标量值域界），但只接在语言器官 `_constrained_generate` 的独立路径上，基底链用不到。

## 1. 设计（掩码＝产品替模型写对，如实入账）

* 把 UTF-8 状态机（`advance`＋`allowed`）提为产品核心的一份共享实现，
  `language_alignment._utf8_allowed` 与 `Taiji.generate` 都从它取数——**不许两处各写一份**。
* `Taiji.generate` 新增 `utf8_strict: bool = False`；开启时解码环每步先把非法字节
  的 logit 压 −inf 再取 argmax/采样；收尾截掉悬空的前缀字节（与仪器同法）。
  默认 False ⇒ 基底/仪器/全部既有读数**逐位不变**。
* `generate_input`（Taiji/adapter/Seed）透传该参数；**`SeedRuntime.chat` 的产品调用传 True**
  ——这是本件唯一的默认行为变更，owner 以"2"裁定授权，登记进 PLAN-A-24 §9 已裁清单。
* 判读仪器（`_answer_raw`、F0 探针）默认不传＝无掩码（模型地板线不动）；
  F0 探针加 `--utf8-strict` 旗标，新增"产品通道线"读数——**两条线的数不许互换**（§6 制度 4）。

## 2. 判据（冻结）

| # | 判据 | 读数线 |
|---|---|---|
| **主判据** | 产品链 `chat()` 表层占位句率（32 条固定简单题）：修前 32/32=100% ⇒ 修后 **≤ 16/32**，且被接受表层 **100% 合法可解码** | 真机产品调用，非仪器模拟 |
| 回归门 | ①`utf8_strict=False` 链与修前**逐位相同**（golden 对照）；②既有能力读数不受影响（所有计分链走默认位）；③`tests/taiji_native` 相关套件全绿 | 位级/套件 |
| 次级（报不判） | 产品通道 F0 五任务的 可解码率（按构造应 32/32）／成句率（对照预期 ≈0.36）／T2–T4 命中（首次有意义的表层命中）；learn=True 的 chat 把合法字节写回训练面 ⇒ 登记为"模型地板的合法数据回流"，后续 F0 无掩码重跑时观察 | 新仪器线 |

## 3. 失败与边界（写死）

* 若掩码后表层仍 100% 占位 ⇒ 说明 `_readable_surface` 还卡别的条件（控制符/alnum），
  如实查因入册，不放宽 organ 验收（那是"替它说话"，不是"替它写对"）。
* 掩码只约束合法字节集，不约束内容——成句率不保证过 0.5；**本件不声称能力涨**，
  只声称"产品输出合法中文＋表层从占位句变回模型的话"。能力的账仍记在无掩码 F0 线上。
* 4 字节序列的标量值域界按 `language_alignment` 现有实现（拒绝代理区与超范围），
  仪器旧实现（`probe_taiji_cap0_byte_output`）若与其不一致，**以产品实现为准**，仪器不动（FROZEN）。

## 4. 实现清单

| 文件 | 改动 | 守卫 |
|---|---|---|
| `taiji/utf8_state.py`（新） | `advance_utf8`/`utf8_allowed` 一份共享实现 | 单测：全 256×4 状态枚举对表现有 `language_alignment` 语义逐字节相同 |
| `taiji/model.py` | `generate` 加 `utf8_strict=False`；解码环用共享掩码；收尾截断 | ①off 逐位 golden；②on 任意随机提示×200 fuzz 输出可解码 |
| `taiji/adapter.py`＋`seed/model.py` | `generate_input` 透传 | off 默认位不变 |
| `api/seed_runtime.py` | `chat` 传 `utf8_strict=True` | 产品占位率读数（主判据） |
| `taiji/language_alignment.py` | 两处 staticmethod 改为委托共享实现 | 等价测试（委托前后逐字节相同） |
| `scripts/training/probe_taiji_f0_language_floor.py` | `--utf8-strict` 旗标（产品通道线） | 默认位仍逐位＝已入库判读件 |
| `tests/taiji_native/test_utf8_strict_generation.py`（新） | 上面全部 | — |
