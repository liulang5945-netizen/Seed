# Seed 项目审计修复交付报告

**日期**：2026-08-23
**范围**：审计报告 P0 全部 + P1 架构项
**验证**：pytest 全量 **115 passed / 4 skipped / 0 failed**；所有改动文件 py_compile 通过；关键数值路径等价性验证通过
**改动规模**：25 个源码/脚本/测试文件，约 +880/-260 行（另含审计前就存在的用户未提交改动）

---

## 一、P0 安全修复（api/）

| 项 | 文件 | 修复内容 |
|---|---|---|
| JWT fail-open | `api/app.py:124-146` | 认证中间件改 fail-closed：非预期异常返回 503；`ImportError` 无认证模块降级保留 |
| 未认证 RCE 面 | `api/routes_agent_workspace.py:223,339` | `/api/workspace/run` 与 `/api/plugins/upload` 补 `_require_admin_auth` |
| SPA 路径穿越 | `api/app.py:330-335` | realpath + `startswith(dist_root + os.sep)` 前缀校验，不通过回退 index.html |
| SSRF | `api/routes_update.py:49-96` | 新增 `_validate_update_url`：http/https 白名单 + 私网/保留地址/localhost 拒绝 + DNS 解析校验防 rebinding + 管理员鉴权 |
| torch.load | `api/seed_runtime.py:60`、`api/training/checkpoints.py:39`、`api/training/resume.py:238` | 默认 `weights_only=True`；老 checkpoint 抛 UnpicklingError 时回退并 warning |
| return→raise | `api/routes_*.py` 十余处 | `return HTTPException` 全清零（grep 验证）；补 4 处 `except HTTPException: raise` 防吞状态码 |
| 终端日志 | `api/routes_terminal.py` | 用户输入日志降 DEBUG 且只记长度；token 不入任何 INFO 日志 |

## 二、P0 安全修复（neuroplex/）

| 项 | 文件 | 修复内容 |
|---|---|---|
| XOR 自制加密 | `neuroplex/core/security.py` | 新路径 Fernet（PBKDF2HMAC-SHA256/600k，盐随机落盘 0600）；读取先 Fernet 失败回退旧 XOR + 告警，下次保存自动升级；公开签名不变 |
| exec 沙箱逃逸 | `neuroplex/life/science_engine.py:626` | 沙箱不可用时拒绝执行（结构化错误），删除黑名单 exec 回退 |
| 裸代码执行 | `neuroplex/body/limbs.py:348` | 优先 sandbox_executor，默认拒绝，`NEUROPLEX_ALLOW_UNSAFE_EXEC=1` 显式放开 |
| shell=True | `neuroplex/tools/desktop.py:86` | 改 shell=False + list 参数 |
| torch.load ×6 | `legacy_checkpoint.py`、`loader.py`、`working_memory.py`、`field_memory.py`（提取 `_torch_load_safe`）、`field_alignment.py` | 先 weights_only=True，回退 + warning |
| pickle 路径 | `neuroplex/tools/rag.py:44,913,940` | `_trusted_pickle_path` realpath 校验必须位于 persist_dir 内 |

## 三、P0 研究可信度修复（scripts/）

| 项 | 文件 | 修复内容 |
|---|---|---|
| holdout 泄漏 | `scripts/training/eval_seed_corpus.py:189` | 同源时改用 `utils.split_train_eval` hash 分桶取评估桶（与训练桶零交集，已验证）；report 新增 `holdout_source/holdout_selection/train_corpus/leakage_risk` 四字段；新增 `--train-corpus`/`--holdout-selection` 参数 |
| verify train==test | `scripts/training/verify_taiji_native_v7.py` | 拆 `train_fit_accuracy`（0.941，仅展示）与 `heldout_accuracy`（卡线，阈值 0.60，实测 0.692，理论期望 2/3）；实跑 status=pass |
| 防腐测试 | `tests/seed/test_seed_corpus_eval.py` | 新增 2 个回归测试：hash 桶互斥 + 泄漏元数据字段 |
| 梯度累积 | `scripts/training/finetune_neuron_dialogue.py:312` | batch 采样移入 micro-step 循环；失败置 `micro_failed` → zero_grad + 跳过 step + 计数告警 |
| 可复现性 | `scripts/training/train_cross_domain_collab.py` | 补 `torch.manual_seed`；resume 跳过 domain 补等量 shuffle 消耗（RNG 流一致）；`clip_grad_norm_` + `--grad-clip`（默认 1.0）；loss 权重提为 `--balance-loss-weight`/`--diversity-loss-weight` |

**重要后续**：holdout 口径已修正，但 **94.12% 需用新口径重测后才能对外宣称**——旧数字产出自泄漏口径。

## 四、P1 架构修复（neuroplex 推理核心，本人亲自施工）

| 项 | 文件 | 修复内容 | 验证 |
|---|---|---|---|
| RMSNorm fp16 溢出 | `layers.py:20-26` | 统计量升 fp32 计算再 cast 回 | fp32 输入与旧实现**逐位一致**；fp16×300 大值不溢出 |
| RoPE 缓存失效 | `layers.py:42-69` | 按 2 的幂桶化缓存 key（最小 128），返回前缀切片 | 同桶切片与直接计算逐位一致，缓存条目稳定为 1 |
| GQA 显存放大 | `layers.py` forward | SDPA 路径用 `enable_gqa=True`（torch≥2.5 探测），手动路径保留 repeat_interleave | 两路径输出 **max diff = 0.0** |
| SDPA 裸 except | `layers.py` 两处 | 收窄为 `(AttributeError, NotImplementedError, RuntimeError)` + 首次回退 warning | — |
| RoPE 绝对位置 | `layers.py` | KV cache 支持 3 元组 `(k, v, abs_len)`：驱逐模式返回绝对长度，新 token 按绝对位置旋转；旧 2 元组兼容 | 驱逐后 len=6/abs=8 断言通过 |
| 双重温度除法 | `cortex.py:2790` | logits 已在 2704-2757 除温，熵计算不再二次除温（注释说明阈值语义恢复） | 全量测试通过 |
| 逐 token re.compile | `cortex.py:38-43` | CJK 正则提升为模块级编译常量 | — |
| .item() 同步风暴 | `ensemble.py:2199-2232` | `activ.tolist()` 单次同步 + `id_to_idx` O(1) 字典 + conf 均值 stack 批量读出 | 数值完全一致（同 float32→float 转换） |

### 刻意未做：cortex 生成主循环接入 KV cache（S3 的设计性结论）

深读 `resonance/neuron.py:528-638` 后确认：**该共振架构的 round 2+ 前向把 field_state 加性注入每个位置的 hidden**（`neuron.py:629+`），任何缓存的 K/V 在下一轮即失效——KV cache 与此架构的多轮共振动力学**根本不兼容**；只有 round 1（无场条件化）理论上可缓存。且 neuroplex 是**冻结基线**，改动推理数学会破坏与 taiji/seed 研究端的可比性。layers.py 的 KV cache 目前是死代码（无任何调用方传入）。结论：S3 不应以"接入 KV cache"修复，正确路径是在未来原生基底设计时规避逐 token 全量场共振；本次已将其记录为设计结论而非遗留 bug。

## 五、P1 其他（desktop/api 性能）

- **desktop/main.py**（+75/-16）：后端启动/健康探测移入 `_RestartWorker(QThread)`，主线程不再可冻结 120s；`_log_handle` 关闭路径；`load_settings` 裸 except 改 warning、`save_settings` 异常保护；端口/路径契约提为模块常量
- **api 限流**：两套限流合并为 middleware/security.py 可配置版；限流字典空列表删键修内存泄漏
- **终端 WS**：queue maxsize=1000 + 满时丢最旧 + 节流 warning；`IDLE_TIMEOUT_SECONDS=300` 经 `asyncio.wait_for` 实际生效
- **SSE 解耦**：`chat_strategies.py:171-207` 新增 `_iterate_sync_gen_in_thread`（线程驱动同步生成器 + asyncio.Queue 桥接，done/error 双哨兵），`_stream_unified` 与 `routes_agent.py` 接入；事件协议/顺序/stop_event 语义不变；正常/异常/停止三路径冒烟通过

## 六、测试基建修复

- `tests/test_cross_domain_eval.py:36`：临时 ckpt 从 tests/ 源码目录改为 `tmp_path`，修复被环境安全删除组件拦截导致的顺序依赖 flaky（修后连跑 3 次全过）

## 已知限制

1. verify 同族脚本 N7/N8/N9/M5/M6/M7 结构各异，未机械修改，建议单独评审（清单在研究端修复汇报中）
2. 熵停止阈值 8.0 语义已恢复为单次除温口径，如生成停止时机变化需重新标定
3. 环境注意：本会话中 E:\Seed 的 .git 目录为沙箱虚拟化提供，真实磁盘上不存在；如需版本控制请 `git init` 重建。本次所有改动均已确认写入真实磁盘
4. P2 项（Cortex 拆分、静默 except 治理、scripts 归档、前端 TS 化等）未动，见审计报告
