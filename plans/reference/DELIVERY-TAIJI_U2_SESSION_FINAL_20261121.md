# 会话最终交付清单：态极 N 主线（2026-11-20/21；零训练、零 check）

> 文档状态：[交付] — 2026-11-21（D-20261121-242）。**不签模型成型/final/C/D/J；零 check、零 final。**

## 1. 候选与谱系

- **旧核（封存）**：[u2_neural_units.py](../../taiji/u2_neural_units.py)（零改动）；包 22 收束 `package22_closed_partial`，旧 RouteB 23–25 收束；
- **新核（活动）**：[u2_selective_units.py](../../taiji/u2_selective_units.py)（加法式，旧核逐位不变）；`SelectivePersistentGraph`＋`capsule_v2`；
- **唯一谱系**：S0（D-225/227/228）→ S1（D-231/232）→ S1-R 系列（D-233/234/235）→ 结构验证（D-236/237）→ 23-S.0 窄合同（D-238/239/240）→ 有界交付（D-241）。

## 2. 可复核成立项（逐条附证据）

| 项 | 读数 | 证据 |
|---|---|---|
| 旧核单位机制/因果/身份/三技能 | 22.1–22.7 verified；胶囊 diff ≤1e-10；三技能 0.86–1.0 | 包 22/23/24 卡 |
| 新核记忆（合成） | M1 末步 T=5/10/20 1.0；M2 1.0；M3 0.456/0.706/1.0 | D-225 |
| 新核中文 | 32 维零位置嵌入 0.9979/0.9823；Exact A3c 0.8031 | D-227/228 |
| 新核四类纠错 | 专用在线 2,000 步 0.85/1.0/1.0；统一主体 0.875–1.0 | D-238/239 |
| 新核自主选择 | 效用 +0.38～+0.49；路由 0.97/1.0/0.99 | D-235 |
| 新核遮挡物理 | 0.885/0.290 | D-239 |
| 新核恢复 | capsule 0.0、断点重连 1.0 | 多卡 |
| 新核中性生长 | 出生 0、旧技能 0、胶囊 0 | D-236 |
| 中文有界（旧核读出工程） | 语序单留出 1.0、多模板 0.701；词汇接地 0.327 | D-205/220 |

## 3. 明确未成立项

| 项 | 结论 | 证据 |
|---|---|---|
| 单主体 joint 联合 | **未成立**（S1 11/16…23-S.0c 5/13；可训练→破坏、冻结→表征不足） | D-231–240 |
| 生长功能贡献 | **未成立**（lesion 0.03–0.05；与旧线 22.7 一致） | D-236/237 |
| 单位状态跨步记忆（旧核） | **不成立**（z/a/H t≥1≈机会；门控/键值槽失败） | D-213/217/218/219 |
| 动作接地（旧核 A2） | **未过门**（0.784<0.80）；新核 B1 binding 0.91–0.93 有界 | D-215/225 |
| 开放中文/对话/生成 | 未测 | — |
| 真机/多模态行动 | 未测 | — |

## 4. 复现与成本

```powershell
cd E:\Seed
python -m ruff check scripts; python -m black scripts
python -m pytest tests/taiji_native -q
python scripts/training/audit_plan_single_mainline.py --json
# 关键探针（示例）
python scripts/training/verify_taiji_algorithm_holistic_20261121.py
python scripts/training/verify_taiji_cognitive_loop_20261121.py
python scripts/training/verify_taiji_s1_lifeline_20261121.py
python scripts/training/verify_taiji_23s0c_narrow_20261121.py
```

- 回执：`reports/taiji_u2_*.json`（逐卡 CPU/seed/判据/verdict）；卡：`plans/reference/RESULT-*.md`；
- 本会话探针累计约 **10,000–12,000 CPU**（旧核 ≈7,000–8,000 + 新核 ≈2,829 + 结构/窄合同附加）；**零 check / 零 final**；
- 包 22 历史超支已按 B6 记事故；25 主窗释放。

## 5. 待 owner 裁定（阻塞项）

1. **接受有界包**（本清单）为当前交付，停止新核联合探索；
2. **换范式**：每阶段独立主体＋capsule_v2 继承（单一谱系非单实例）——需新预注册与授权；
3. **暂停**新核主线。

在上述裁定作出前，无已授权的新执行卡可开（§1.3 单卡 `training_authority`）。

## 6. 非声明

模型成型、单主体 joint、生长真实收益、开放中文、真机/多模态、自主选择泛化、final/C/D/J 通过；零 check。
