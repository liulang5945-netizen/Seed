# DSH 上游差异调研报告（0.1.7-alpha.1 → 0.2.1-alpha.1）

> owner 需求：社区已更新 DKH（DeepSeek Harness），除实验性 Claude Code Mods 兼容层外都要参考。
> **Claude Code Mods 兼容层按 owner 判断排除**（上游自述其目的是验证 CC Mods API 为 DSH 插件的子集，非提供真实兼容）。
> 取证日期：2026-10-09。机制依据：`plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md`。
> 本报告只做调研，未启动换基底工程，未改动任何仓库文件。

## 1 结论先行

1. **值得升级，但这是"整基底换 + 清单重放"的大工程，不是拉取补丁。** playbook 定死：fork 无共同祖先（tarball 落地）⇒ `git merge/cherry-pick upstream` 先天不可用，"只挑有价值的拉"在机制上不成立。
2. **落差不小**：基线 `dsh-v0.1.7-alpha.1`，最新 `dsh-v0.2.1-alpha.1`，中间 6 个 tag（0.1.7-alpha.2 / rc.1 / rc.2 / 0.2.0-rc.1 / rc.2 / 0.2.1-alpha.1）。
3. **必须先由 owner 拍板的一个决策点**：上游在 0.2.x **删除了整套 invariant 机制**（`./invariant` 子路径导出 + 两个 verify 脚本 + tsconfig 8 条路径映射）。而**我们本地的工程约定正在用 invariant**（`packages/AGENTS.md` 有 "Publish ./invariant only for diverging observations" 条目，本地 `tsconfig.base.json` 有 8 条 `@taiji/*/invariant` 映射）。跟删＝放弃本地机制；保留＝重放时手工加回并自行维护与上游的偏离。**这不是技术必然，是取舍。**
4. **一个意外的高价值发现**：上游把 `libreoffice-kit` 从 **0.0.1 升到 0.1.5**（6 个平台包全升）。这正是长期卡着 M7 收束的 **R7 缺陷**（打包布局下 `xlsx→PDF` 报 `loadComponentFromURL returned an empty reference`，docx 通过）所依赖的组件——**0.1.5 很可能已修，但我们无法从公开渠道确认**（见 §6 局限）。

## 2 版本与落差

| 项 | 值 |
|---|---|
| 本地基线 | `dsh-v0.1.7-alpha.1`（G1 tarball 落地 2026-09-22，提交 `82042a2f6`） |
| 上游最新 | `dsh-v0.2.1-alpha.1` |
| 中间 tag | `0.1.7-alpha.2`、`0.1.7-rc.1`、`0.1.7-rc.2`、`0.2.0-rc.1`、`0.2.0-rc.2`（共 6 个） |
| 上游分发方式 | 未发布到公共 npm（`@deepseek-ai/dsh-root` registry 查无此包），只能走 GitHub |

## 3 上游变更清单（按对 Seed 的参考价值分级）

来源：上游 release notes + 锚点文件逐条 diff。

### 3.1 高价值（与 taiji 自主能力同向，建议纳入重放评估）

| 变更 | 观察到的证据 |
|---|---|
| 插件管理页新增「让 Agent 创建插件」入口（创造模式 / 草稿保留） | release notes；web-app patch 未见对应 row（可能在 client 包内） |
| Automation 任务并入 Web，提醒工具按 Standard / Creator / PTC 模式限定 | web-app patch **新增 `schedule` row**（`@deepseek-ai/dsh-schedule`） |
| 开发者工具组合包（会话原始日志、聊天分组↔正文双向定位、内嵌 Host DevTools） | web-app patch **新增 `ui-settings-session-log` row** |
| 新会话预填未发送提示、草稿/工作区切换保留文件与引用 | release notes |

### 3.2 中等

| 变更 | 证据 |
|---|---|
| Web 支持 `--public-url`（对外访问地址、反代入口） | release notes |
| 快捷键体系（新增 `shortcuts` / `ui-shortcuts` rows） | web-app patch |
| 大会话列表性能优化 | release notes |
| 显示实际安装版本 + 说明如何请求精确版本 | release notes |
| 子路径插件不再读独立 `package.json`，元数据取子路径导出 | release notes |

### 3.3 需要决策的破坏性变更

| 变更 | 证据 | 对我们的影响 |
|---|---|---|
| **移除 runtime invariant 插件与全部 `./invariant` 子路径导出** | `tsconfig.base.json` 删掉 8 条 invariant 路径映射（`dsh-invariants` 包 + agent/agent-loop/scope/session/agent-team/host-*/client-* 的 `/invariant`） | **与我们的 invariant 机制正面冲突**，见 §1.3 |
| 删除 `verify-package-invariants` / `verify-built-package-invariants` 脚本 | `package.json` scripts diff | 同上 |
| 删除 `resolve-translation-pairing-conflicts`（`merge-translation-pairing.ts --resolve`） | 同上 | 我们正在用双语配对工作流；`verify-translation-pairing` **保留**，仅删了冲突自动合并模式 |
| `libreoffice-kit` 0.0.1 → 0.1.5（6 平台包） | `pnpm-workspace.yaml` | **可能是 R7 的解**，见 §6 |
| `pi-ai` / `pi-telemetry` 0.85.1 → 0.87.1（且 patch 路径随版本改） | 同上 | LLM 适配层升级，需回归聊天链路 |
| Koffi 消费者钉 3.1.1（注释：3.3.2 源码构建在 GCC 13 下失败） | 同上 | 构建侧约束，重放时须照抄 |

### 3.4 构建流程变化（重放必须跟上，否则本机跑不通）

| 项 | 变化 |
|---|---|
| `build:lib:host` | 末尾追加 `pnpm --filter dsh-desktop run bundle` |
| 所有 `tsdown` 调用 | 追加 `--config-loader native`（**待验证**：本机 tsdown 0.22.2 是否支持该 flag） |
| 新增 `check:ci:unit`（`scripts/run-gates.ts ci-unit`） | 门禁入口变化 |
| 新增 `verify-upgrade-guides` | 上游自带升级指南校验 |
| 新增 `upload:latest:{mac:arm64,mac:x64,win:x64}` | 桌面自动上传（我们暂不需要） |

## 4 H 组对撞面（重放时的真实冲突点）

playbook 锚点与上游 0.2.1 的实际冲突：

| 锚点 | 上游变化 | 后果 |
|---|---|---|
| **H2a `llm-taiji`「紧跟 `llm-deepseek`」** | 上游把 `llm-deepseek` **拆包**为 `llm-deepseek-api-key`，并新增 `llm-deepseek-account`（row id `llm-deepseek-account`） | **锚点失效**，`llm-taiji` 的定位基准变了，必须重新对位 |
| **H2 区 telemetry 行** | 上游新增 `otel` row（`@deepseek-ai/dsh-otel`）；我们已有 `session-telemetry-otel` | 可能重复或需并轨 |
| web-app client 行 | 上游新增 `desktop-product-telemetry`、`product-analytics`、`schedule`、`shortcuts`、`ui-shortcuts`、`ui-settings-session-log` | 与我们的 `ui-settings-*` 系列并存，需核对命名与插入位置 |
| `tsconfig.base.json` / `tsconfig.host.json` | 上游 +25/−44 与 +41/−2 | 我们加的 `@taiji/dsh-web-search-scraper` 等路径映射要重新落位；invariant 段落整体消失（§3.3） |
| `scripts/gen-config-catalog.ts` | 上游 **+171/−75**（大改） | 我们的 `gen-config-catalog-zh.ts` 在上游**不存在**（404 确认）＝纯本仓新增，重放时整份拷回 |
| `vitest.config.ts` / `vitest.e2e.config.ts` | 各 +26/−9、−32 | 测试分组变化；我们改过的 `web-search-round.e2e.ts` 场景接线需重新核对 |

## 5 工作量与风险估计（基于本次取证，不是拍脑袋）

- **机械项（M 组）**：M1 全仓 rename、M2 品牌表、M3 生成物重刷 —— 有脚本/表，成本低。
- **手写项（H 组）**：锚点失效至少 2 处（`llm-taiji` 对位、telemetry 并轨），新增上游 row 6+ 处需判断，invariant 段落需 owner 决策。**这是主要工时**。
- **门禁重验**：删掉的 `verify-package-invariants` 决策后需相应调整 `run-gates.ts` 与 §5 复验集；`--config-loader native` 需在本机 tsdown 验证。
- **网络限制**（见 §6）：本机拉整仓快照极慢（git clone 25 分钟只到 7MB；codeload tarball 9 分钟 4.7MB 且不支持 Range 续传）。**换基底前建议先解决快照获取通道**，否则每轮重放都要付这个代价。

## 6 取证方法与局限（如实记录）

**能做的**：用 `raw.githubusercontent.com` 逐文件取两个 tag 的锚点文件（本轮取到 15 个文件 × 2 tag = 30 份），做行级 diff；用 GitHub API 取 tag 列表。

**做不到的**：
1. **整仓快照拿不到**——codeload tarball 与 `git clone --depth 1` 在本机网络下都极慢/被截断（Range 分块也失败）。因此本报告的差异是**锚点级，不是文件级**。
2. **新增包未穷举**——已知上游新增 `dsh-schedule`、`dsh-otel`、`dsh-client-ui-settings-session-log` 等（从 patch 观察），但包内部实现没核。
3. **libreoffice-kit 0.1.5 是否修 R7 无法确认**——npm registry 对该私有包不可达，拿不到 changelog。

**结论的置信度**：版本落差、破坏性变更清单、H 组对撞点 = 高置信（直接证据）；上游全部新增能力清单 = 中等（可能遗漏包级新增）；R7 修复可能性 = 低（仅版本号推断）。

## 7 建议的下一步

1. owner 就 §1.3（invariant 跟删还是保留）拍板——它决定整个重放的形态。
2. 解决快照获取通道（换出口/代理，或申请预填），否则换基底不可行。
3. 若 owner 只想"拿能力不换基底"，可评估**逐能力移植**（例如先把 `schedule`（Automation）与 `ui-settings-session-log`（开发者工具）按我们自己的设计实现）——这条路不受 playbook 的换基底约束，也没有 invariant 冲突。**但它是新功能，不是"同步上游"，两个账要分开记。**