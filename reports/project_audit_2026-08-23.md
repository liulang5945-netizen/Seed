# Seed 项目全面代码审计报告

**审计日期**：2026-08-23
**审计范围**：neuroplex/（116 文件 / 43.4k 行）、api/（41 / 7.5k）、taiji/ + seed/（16 / 4.8k）、scripts/（371 / 78k）、tests/（37 / 2.9k）、frontend/src（38 文件）、desktop/（4 / 808 行）
**审计方式**：四路并行深读 + 关键发现人工复核（JWT fail-open、holdout 泄漏、生成主循环重算三条最严重结论已逐行验证属实）

---

## 一、总体结论

项目呈现明显的"双轨质量断层"：**研究端 taiji/seed 核心模块工程纪律良好**（payload 版本校验、原子落盘、frozen dataclass 校验、评估后状态恢复），但**产品端 neuroplex/api 存在多处严重安全漏洞与架构级性能缺陷**，且**头条指标 94.12% 的可信度被评估方法论缺陷直接威胁**。以"冻结基线 + 生产后端"的标准衡量，当前状态不可托付。

**整体质量评分：4.6 / 10**（评分依据见第六节）

---

## 二、严重问题（必须立即修复）

### S1. 评估集泄漏：holdout 就是训练语料的前 32 行 【研究可信度 · 致命】
- **位置**：`scripts/training/eval_seed_corpus.py:154-163`
- **证据**：`_holdout_bytes` 从 `--holdout` 文件**顺序取前 N 行**，而默认 holdout 文件与 `train_seed_corpus.py:43-48` 的 `DEFAULT_CORPUS` 是**同一文件**（`data/simple_zh/dialogue_extended_clean.jsonl`）。模型在线训练过整个语料后，评估读其头部——输出的是训练集复现指标。
- **影响**：对外宣称的 94.12% 准确率若为记忆化拟合度而非泛化能力，整个研究结论站不住。
- **建议**：使用 `utils.split_train_eval` 的 hash 分桶口径取互补集，或单独物理留出文件；在报告中强制声明 train/holdout 来源。

### S2. verify 族脚本 train == test 【研究可信度 · 致命】
- **位置**：`scripts/training/verify_taiji_native_v7.py:82-94, 175`（M5-M7/N7-N9 等族脚本普遍如此）
- **证据**：`learn_bytes(data, epochs=200)` 之后直接 `score_bytes(data)`，以 `after["accuracy"] >= 0.75` 作为通过条件。在训练序列上打分，衡量的是记忆化而非泛化。
- **建议**：所有 verify 报告强制拆分 `train_fit_accuracy` 与 held-out probe 两个字段，后者不过线即失败。

### S3. 生成主循环无 KV cache，逐 token 全量重算 【性能 · 架构级】
- **位置**：`neuroplex/brain/cortex.py:2601-2625`
- **证据**（已复核）：`for ir_step in range(max_tokens)` 内每步对完整 `general_ids` 重建 `torch.tensor(...)`（每步一次 H2D 拷贝）并对全部激活神经元跑完整多轮共振 forward，复杂度 O(L²·rounds·neurons)。而 `layers.py` 明明实现了 KV cache，主路径却未接入。
- **影响**：推理引擎名不副实，长上下文生成成本平方级膨胀。
- **建议**：接入 GQA kv_cache，增量 embedding 只送新 token； RoPE 相位按绝对位置计算。

### S4. JWT 认证中间件 fail-open 【安全 · 可利用】
- **位置**：`api/app.py:124-129`（已复核）
- **证据**：`verify_token` 抛出非预期异常时仅 `logger.warning` 后**直接放行请求**。
- **建议**：认证中间件必须 fail-closed，异常时返回 401/503。

### S5. 未认证 RCE 面：插件上传与代码执行端点 【安全 · 可利用】
- **位置**：`api/routes_agent_workspace.py:337-353`（插件上传无鉴权直接落盘 .py）、`:222-237`（`/api/workspace/run` 执行任意 Python）
- **证据**：`AuthManager` 默认未启用时 JWT 中间件整体放行（`app.py:92-94`），multipart 表单属 CORS 简单请求，构成 网页→localhost:8000 的 RCE 链。
- **建议**：所有写/执行类端点强制管理员认证 + Origin 校验；认证未启用时默认仅 loopback 可写。

### S6. 不安全反序列化遍布全仓 【安全】
- **位置**：生产路径 `api/seed_runtime.py:58`、`api/training/checkpoints.py:37`、`api/training/resume.py:237`；neuroplex 内 `legacy_checkpoint.py:52`、`loader.py:419`、`brain/working_memory.py:135`、`resonance/field_memory.py:73,343`、`resonance/field_alignment.py:72`（均 `weights_only=False`）；`tools/rag.py:913,940` 直接 `pickle.load`。全仓 50+ 处。
- **影响**：加载不可信 .pt/persist 文件即 RCE。讽刺的是正确示范已存在（`neuroplex/loader.py:29` 用 `weights_only=True`）。
- **建议**：统一 loader 收口，weights_only 为默认，逐步迁移 safetensors；遗留路径加白名单与告警。

### S7. 自制 XOR"加密"保护 API Key 【安全】
- **位置**：`neuroplex/core/security.py:168-250`
- **证据**：XOR 密钥流 + 机器指纹派生主密钥 + 硬编码盐 `b"taiji-salt-2024"` + HMAC 截断至 16 字节。能读盘者即可解密全部"加密"密钥。
- **建议**：改用 `cryptography` 库的 Fernet 或 AES-GCM。

### S8. exec 回退沙箱可逃逸 + 裸代码执行 【安全】
- **位置**：`neuroplex/life/science_engine.py:632-642`（黑名单过滤后 `exec`，经典 `__subclasses__` 链可绕过）；`neuroplex/body/limbs.py:359`（`run_python` 无沙箱）；`neuroplex/tools/desktop.py:89`（`shell=True`）
- **建议**：回退路径直接拒绝执行，统一走 sandbox_executor；黑名单式沙箱视为不可防御，废弃。

### S9. 滑窗驱逐破坏 RoPE 位置一致性 【正确性】
- **位置**：`neuroplex/layers.py:209-212`（驱逐后禁用 causal mask）、`144-148`（sink+window 直接 `torch.cat`，window 段 token 的 RoPE 相位按拼接后位置重算而非原始位置）
- **影响**：长上下文推理正确性无保障，且是静默的语义错误。
- **建议**：驱逐时保留原始位置索引，RoPE 按绝对位置应用。

### S10. SPA 兜底路由路径穿越 【安全】
- **位置**：`api/app.py:393-404`
- **证据**：`catchall` 未做 `..` 过滤与 realpath 前缀校验，`/%2e%2e/` 解码后可读 dist 外任意文件。
- **建议**：`realpath` 后强制 `startswith(dist_path)`。

### S11. SSE 主链路同步阻塞事件循环 【性能 · 架构级】
- **位置**：`api/chat_strategies.py:260-294`、`api/routes_agent.py:58-68`
- **证据**：`for event in engine.run_stream(...)` 同步生成器在事件循环线程上逐 next() 跑秒级推理，`asyncio.sleep(0.01)` 只在事件间让出。并发聊天互相阻塞；断连后 stop_event 要等当前推理步结束才被检查。
- **建议**：生成器放 executor + `asyncio.Queue` 桥接；循环内检查 `request.is_disconnected()`。

### S12. 桌面端 GUI 主线程可冻结 120 秒 【可用性】
- **位置**：`desktop/main.py:599-606`（QTimer 主线程调 `backend.start()` → `_wait_for_ready` 内 `urlopen + time.sleep(0.5)` 最长 30s/frozen 120s）、`:183`（`urlopen(timeout=120)`）
- **建议**：看门狗逻辑移入 QThread/worker，或 QTimer 分片探测。

### S13. 未认证 SSRF 【安全】
- **位置**：`api/routes_update.py:76-93`：`repo` 为任意 URL 时无认证、无内网过滤直接抓取，可用于探测内网。
- **建议**：管理员认证 + URL 白名单/禁止私网段。

---

## 三、中等问题（择录最高信号 15 项）

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| M1 | `api/routes_multimodal.py:54,102,123`、`routes_rag.py:45-213` 等 10+ 处 | `return HTTPException(...)` 而非 `raise`，吞异常返回 200 | 全局改 `raise`，加 CI 检查 |
| M2 | `neuroplex/cortex.py:2792-3110`、`resonance/ensemble.py:2221-2225` | 逐 token/逐 step `.item()` GPU 同步，CUDA pipeline 串行化 | 批量 `.tolist()` 或纯张量运算 |
| M3 | `neuroplex/layers.py:229-243` | SDPA 裸 `except Exception` 回退手动 attention，OOM 被吞且语义静默改变 | 只捕获兼容性异常 |
| M4 | `neuroplex/layers.py:38-61` | RoPE LRU 缓存 key 含 seq_len，生成时每步 +1 → 缓存必失效 | 按 max_seq 预计算切片 |
| M5 | `neuroplex/layers.py:196-197` | GQA 用 `repeat_interleave` 复制 KV，显存放大数倍 | 改 `expand` 或 SDPA `enable_gqa=True` |
| M6 | `scripts/training/finetune_neuron_dialogue.py:303-341` | 梯度累积逻辑错误：micro-step 重复同一 batch（等于放大 lr）；失败后仍 step 残余梯度 | batch 采样移入 accum 循环内 |
| M7 | `scripts/training/train_cross_domain_collab.py:971-996, 956` | resume 破坏 shuffle 序列；只设 `random.seed` 无 `torch.manual_seed`；无梯度裁剪 | 补全 RNG、clip_grad_norm、resume 消耗等量 shuffle |
| M8 | `api/app.py:139,174` | RateLimiter 键永不删除内存无界；同时挂两套限流逻辑重复 | 合并一套 + 淘汰空键 |
| M9 | `api/routes_terminal.py:195,208-227` | WS 输出队列无上限无背压；`IDLE_TIMEOUT_SECONDS=300` 声明了却从未实现 | 队列 maxsize + 实现空闲超时 |
| M10 | `api/routes_chat.py:121-156, 262-267` | 会话历史无上限无锁整文件重写；上传先全量读内存再查 20MB 上限 | 限长+原子写；流式计数 |
| M11 | `neuroplex/` 全局 | 30+ 处无锁 `global _global_*` 单例；`except: pass` 数十处静默失败 | 单例收口+锁；至少 logger.debug+指标 |
| M12 | `neuroplex/brain/cortex.py`（3198 行）等 | 上帝对象：Cortex 糅合路由/生成/质量门/记忆；ensemble.py 3953 行、sleep_engine.py 3483 行；`ResonanceEnsemble` 非 nn.Module 导致设备迁移/序列化全靠手写代理 | 按职责拆分；纳入 nn.Module 体系 |
| M13 | `frontend/src/apiClient.js:23` + `chatStore.js:222` | authFetch 对非幂等流式 POST 默认重试 2 次，5xx 静默重发造成重复推理 | stream 调用显式 `retries: 0` |
| M14 | `frontend/src/composables/useMarkdown.js:22-31, 163-166` | code renderer 的 `lang` 未转义拼入 HTML；DOMPurify 放行内联 `style`，可被模型输出注入做 UI 钓鱼 | 转义 lang；移除 style 白名单 |
| M15 | `api/routes_terminal.py:68,242-244` | JWT 走 query 参数（日志泄漏面）；用户终端输入（可能含密码）INFO 级落盘 | 改首帧鉴权；输入审计脱敏 |

## 四、轻微问题（择录）

- `taiji/model.py:759-768`：`parameter_count` 的 `active_only=True/False` 返回同一值，verify 脚本恰好依赖此参数区分口径，语义被静默吞掉。
- `taiji/fabric.py:339-346`：评估时 homeostasis 阈值仍在测试数据上适应（在线评估口径），报告未声明则与静态 PPL 不可比。
- `api/`：鉴权辅助函数近乎逐行重复 4 份（update/agent_workspace/auth/system）；SSE 错误事件协议不一致（裸 JSON vs 结构化 error）。
- `api/chat_strategies.py:90,237`、`routes_rag.py:140-142`：API 层直捅引擎私有成员（`_get_working_memory_context`、`_bm25_index`），重构即碎。
- `frontend`：JWT 存 localStorage（配合 XSS 面可被盗）；blob URL 发送后永不 revoke（内存缓慢增长）；`LogPanel.vue` 深 watch + 索引 key。
- `tests/`：`test_seed_corpus_eval.py:79-80` 是同源恒等断言，永远为真且测不出 S1 泄漏；无 `conftest.py`、pyproject 无 `testpaths`。
- `desktop/main.py:121`：`_log_handle` open 后永不 close；`load_settings` 裸 `except Exception: pass`。
- 命名债：neuroplex 包内残留 Taiji logger / `TaijiNativeTokenizerV2` / `historical_taiji_namespace`；print 与 logger 混用。
- 硬编码：绝对路径 `e:\Seed\checkpoints\...` 散落脚本；魔法数（0.5 EOS bias、8.0 熵阈、512 截断）无配置无日志。

## 五、结构性根本缺陷（设计层面）

1. **脚本生态失控**：371 个脚本 / 7.8 万行 vs 测试 2.9 千行（26:1）。`scripts/archive/diagnostics/` 下 5 个 `_diag_*.py` 互相复制同一段代码；verify 脚本裸名互 import 依赖运行目录。研究迭代留下的"脚本堆"已成为正确性验证的最大噪声源。
2. **冻结基线名不副实**：neuroplex 自称冻结，但核心推理路径未接入自己实现的 KV cache（S3）、RoPE 位置正确性无保障（S9）、上帝对象+全局单例使行为不可复现——一个无法验证正确性的基线无法作为研究对照。
3. **安全姿态"功能齐全但默认关闭"**：认证、限流、沙箱、审计都写了，但默认配置下认证关闭、沙箱有回退、限流双份口径不一、IDLE_TIMEOUT 只声明未实现。攻击面集中在"本地服务被局域网/跨站请求触达"这一现实场景。
4. **指标治理缺失**：没有一份"指标口径白皮书"约束 train/holdout 分离，导致 S1/S2 这类致命泄漏能长期存活，且测试里的恒等断言反而给了虚假信心。

## 六、整体评分与依据

| 模块 | 评分 | 要点 |
|------|------|------|
| neuroplex 核心 | 3.5/10 | layers 骨架方向正确，但主路径无 KV cache、自制加密、上帝对象、海量静默 except |
| api 后端 | 4.5/10 | 部分端点防护认真（ZipSlip/abspath 校验），但 fail-open 认证 + 未认证 RCE/SSRF + 事件循环阻塞 |
| taiji/seed 研究端 | 6/10 | 核心模块纪律好（原子落盘、版本校验、restore），但指标泄漏 + 脚本债 |
| frontend/desktop | 6.5/10 | 架构意识良好（Pinia、DOMPurify、重连退避），TS 缺位 + 主线程冻结 |
| **整体（按产品可托付度加权）** | **4.6/10** | 安全与研究方法论两类致命伤拉低整体；工程亮点集中在 taiji 核心与前端架构 |

## 七、按优先级排序的改进清单

**P0 — 本周内（止血）**
1. 修复评估泄漏（S1/S2）：物理分离 holdout，verify 脚本强制 held-out probe 字段，重测并重新发布可信指标
2. JWT 中间件改 fail-closed（S4）；插件上传/代码执行/系统重启端点强制鉴权 + Origin 校验（S5）
3. 生产路径 `torch.load` 全部改 `weights_only=True`（S6 的 api 部分）
4. 废弃 XOR 自制加密改 Fernet/AES-GCM（S7）；废弃 exec 黑名单回退（S8）
5. SPA 路由加 realpath 前缀校验（S10）；check_update 加认证+私网过滤（S13）

**P1 — 两周内（架构纠偏）**
6. 生成主循环接入 KV cache + 增量 embedding（S3）；修复 RoPE 滑窗位置一致性（S9）
7. SSE 推理链路 executor + Queue 桥接（S11）；消除逐 token `.item()`（M2）
8. `return HTTPException` 全局改 `raise`（M1）；两套限流合并 + 键淘汰（M8）
9. 桌面端看门狗移出主线程（S12）
10. 修复梯度累积与 resume RNG（M6/M7）；补梯度裁剪

**P2 — 一个月内（还债）**
11. Cortex 上帝对象拆分；`ResonanceEnsemble` 纳入 nn.Module；全局单例收口加锁
12. 静默 except 全仓治理（至少 logger + 指标计数）
13. scripts/ 归档清理：建立 `scripts/legacy/` 隔离区，新脚本强制包化可 import
14. 前端：stream 禁重试（M13）、XSS 加固（M14）、store/composable 渐进 TS 化
15. 建立指标口径白皮书 + 测试防腐：删除恒等断言，新增 holdout 分离的回归测试
16. 测试基建：conftest.py、testpaths、核心推理路径覆盖率目标 ≥60%

---

*报告生成：WorkBuddy 高级开发工程师 · 四路并行深读 + 关键发现人工复核*
