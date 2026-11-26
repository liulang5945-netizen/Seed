# N36 有界交付与用户接管操作手册（`bounded_closed_vocab_and_multimodule_v1`）

> [手册·有界交付] — 2026-11-26（D-20261126-685）。本手册对应 `MAINLINE.md §4.4` 资产⑥，指导操作者在**有界声明范围**内独立构建、启动、交互、备份、故障恢复与退役 `taiji-n36-bounded-delivery-v1` 运行包。

## 1. 声明范围与明确不签署项

- **已包含有界能力**：
  1. `ch.*`：封闭词表（43 token）跨回合自然中文连续指代（`ChineseTurnPolicy(recurrent=True)`；词表外输入一律 fail-closed 拒答 `out_of_bounded_vocab`）；
  2. `phy.*`：两族合成多模态物理因果预测（`ramp_bumper` / `rough_collision`，无 `family_id` 捷径）；
  3. `grow.*`：选择性可塑与生长胶囊状态（`PlasticGrowthNet` + `GrowthPressureController`）；
  4. `ecl.*`：在线 Beta 可信度学习（指数遗忘 `λ=0.995`）；
  5. `unc.*`：学习式不确定性校准与中文四级置信表达；
  6. `SessionStore` ＋ `LongRunHarness` sha256 审计链 ＋ `SealedSplitGuard`。
- **明确不签署（严禁越界宣称）**：
  - `open_chinese = deferred_post_N35`（不支持开放词表自然中文自由对话）；
  - `physical_sensors = synthetic_two_family_only`（不接真机传感器/执行器）；
  - `24h_live_run = not_executed`（未执行 24 小时现场试运行）；
  - `check_reads = 0`、`final_reads = 0`；
  - `not_signed = ["24-S", "25-S", "29", "final", "C", "D", "J"]`。

## 2. 标准接管命令

```powershell
# 1. 构建 100 会话有界运行包（默认交付 seed 6801，零确认 seed）
python scripts/taiji_n36_takeover.py build --bundle-dir <目录> --seed 6801

# 2. 查看状态、审计链、参数量、可信度后验与快照列表
python scripts/taiji_n36_takeover.py status --bundle-dir <目录>

# 3. 封闭词表中文交互（带 --no-persist 时不写会话）
python scripts/taiji_n36_takeover.py chat --bundle-dir <目录> --utterance "请把红色杯子放到桌上。" --utterance "再去检查一下它。"

# 4. 合成物理因果预测（可选 --ablate-barrier 做障碍消融自检）
python scripts/taiji_n36_takeover.py predict-physics --bundle-dir <目录> --family rough_collision

# 5. 在线可信度反馈更新
python scripts/taiji_n36_takeover.py credibility --bundle-dir <目录> --source forum --correct 1

# 6. 生命周期与快照/回滚
python scripts/taiji_n36_takeover.py snapshot --bundle-dir <目录> --name v1
python scripts/taiji_n36_takeover.py pause --bundle-dir <目录>
python scripts/taiji_n36_takeover.py resume --bundle-dir <目录>
python scripts/taiji_n36_takeover.py rollback --bundle-dir <目录> --name v1
python scripts/taiji_n36_takeover.py archive --bundle-dir <目录>
```

## 3. 五类故障恢复操作规程

1. **包缺失或文件篡改（`BundleIntegrityError`）**：`open()` 在 `manifest.json` 缺失、`.session` 摘要不匹配或 `.audit.jsonl` 链断裂时自动 fail-closed 拒载。**处置**：切勿手工编辑 `.session`/`.audit.jsonl`，使用最近已验证快照执行 `rollback --name <snap>` 或从备份目录恢复。
2. **提交前崩溃（`CrashInjected` / 进程异常终止）**：`LongRunHarness` 采用临时文件＋原子替换，提交前崩溃不会污染已提交会话。**处置**：直接重新 `open()` 或 `status`，核对 `chain_valid = true` 后继续。
3. **封存分区越权读取（`SealViolation`）**：任何尝试读取 `check` / `final` 分区都会立即抛出 `SealViolation`。**处置**：保持 `check/final` 封存，仅在授权范围内使用运行推理与在线反馈。
4. **误更新 / 性能退化回滚**：若在线可信度或未结事项变更需撤销，使用 `rollback --name <snap>` 逐位恢复至快照时的 `record_digest` 与 `chain_head`（已执行的外部动作不作伪回滚）。
5. **暂停、撤权与退役封存**：需要停机维护时执行 `pause`（阻断一切写入）；确认退役时执行 `archive`（永久撤除写入与恢复权利，进入不可逆 `Archived` 终态）。
