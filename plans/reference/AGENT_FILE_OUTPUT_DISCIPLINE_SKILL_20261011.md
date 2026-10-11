# Agent 文件输出纪律 Skill 登记（2026-10-11）

## 是什么

用户以 scene#18「Skill 开发」立项：创建统一的多 Agent 文件管理规范，约束所有 Agent 在项目中生成产物时的路径策略。已落地为 WorkBuddy 用户级技能：

- **技能名**：`file-output-discipline`（文件与文件夹管理规范）
- **安装位置**：`C:/Users/23747/.workbuddy/skills/file-output-discipline/SKILL.md`（用户级，跨项目生效）
- **形态**：单 SKILL.md，纯行为规范，无脚本/资产；已通过 skill-creator 官方校验（valid）
- **安全自检**：纯 Markdown 指令、无可执行内容、无网络调用 → P2 安全

## 核心条款（SKILL.md 七节）

1. **红线**：写入路径解析后必须位于当前项目根之下；桌面/下载/文档/主目录/系统目录/其他磁盘/项目根外新目录一律禁止。
2. **唯一输出根** = 当前项目目录；位置先分类（六类：源码/账本/运行时/构建/缓存/草稿）再谈。
3. **产物 → 位置映射表**：脚本→`scripts/`、报告→`reports/`、计划→`plans/`（只追加）、规范→`docs/`、设计→`design/`、临时→`.tmp-<主题>/`、运行时→数据根、构建→构建系统输出。仓库根禁造台账外顶层目录。
4. **命名**：Windows 非法字符（含 `:`）禁用；时间戳 `YYYYMMDD`；临时目录唯一前缀 `.tmp-<主题>/`；规则按命名约定写不按实例名写。
5. **例外仅两条**：用户显式指定外部路径（须复述实际路径）；平台自身约定目录（skills/memory/系统 temp）。
6. **删除纪律**：dry-run 列清单 → 用户确认 → 走既有工具（`scripts/clean_worktree.py`）；注意 `.gitignore` 白名单例外与已跟踪"草稿"。
7. **写前自检五问**。

## 与既有文档的关系（引用，不复制）

本 skill 是 [`docs/FOLDER_STRUCTURE_RULES.md`](../../docs/FOLDER_STRUCTURE_RULES.md)（S1–S8）与 [`docs/REPO_HYGIENE_RULES.md`](../../docs/REPO_HYGIENE_RULES.md)（R1–R10）的 **Agent 行为层泛化**：S1–S8 管"仓库里什么东西放在哪"，skill 管"Agent 写文件前的决策流程"。两套编号不重叠；skill 明确引用台账（S2 表）为仓库内唯一权威，不另立标准，符合"沿用既有规则、不发明新标准"的治理约定。

## 触发背景（为何立项）

真实事故链：构建脚本写到桌面（`C:/Users/23747/Desktop/build-life-update.cmd`）、实验产物散落 `output/`+`outputs/` 约 51 GB、2026-09-21 因临时目录无约定误删 15,754 个文件 / 约 3.5 GB。S1–S8 已治仓库侧，但对"多 Agent 并行工作"缺一层写前约束，本 skill 补上。

## 生效方式

WorkBuddy 会话内 Agent 在写文件/建目录/清理场景自动加载该技能；跨项目通用（用户级安装）。后续若条款修订：skill 侧改 SKILL.md，仓库侧不动（S1–S8 仍为仓库权威）。

---

## 增补 v2（2026-10-11 同日）：文件内容规范 + 大文件拆分

应用户要求在 SKILL.md 追加两节（原 §7 自检顺延为 §9，自检扩为 7 问）：

- **§7 文件内容规范**：七类文件的最低结构——Python（模块 docstring / 导入三段序 / snake_case / ruff+black 门禁）、TS（禁止裸 any / oxlint+tsc 自证编译）、Markdown（frontmatter / 结论先行 / 链接前缀纪律 / 双产出互相引用不复制）、脚本（用法+幂等性+dry-run 默认；一次性脚本不进 scripts/）、数据文件（元数据字段+JSONL 配 .meta.json）、配置（秘密不入库）。
- **§8 大文件拆分**：阈值四档（<500 正常 / 500–1000 警觉 / >1000 须拆分计划或豁免登记 / >3000 高优先债）；真正判据 = 一句话职责。六原则：先锚定后动刀（scoped 回归+黄金等价证明）、按簇拆不按行数切、算法抽离编排留壳（即 Cortex split C-4 原则）、facade 保持导入路径兼容、一刀一回滚点、拆完清场。标准组织形态 = `巨石.py → 巨石/` 包（`__init__.py` facade + core 编排 + 簇子模块 + common），包内依赖单向 core→子模块→common。五步流程：预注册合同 → 簇边界探测 → 逐簇搬运 → 等价验证 → 收尾（更新 AST 契约测试）。存量巨石（`taiji/adapter.py` 12,656 行）按债务排期、不运动式重写，新代码必须守阈值。

条款与 B 分支 Cortex god-object split 的既有方法论同构（黄金等价、按簇一刀一回滚、算法/编排分离），skill 侧将其泛化为通用规则。skill 重新通过 skill-creator 校验。

## 分发包（2026-10-11 同日）：根级 SKILL.md 格式

第三方 Agent 平台上传要求 zip / `.skill` 内**根级直接是 SKILL.md**（skill-creator 官方打包脚本默认带一层目录前缀，故不兼容）。已重新打包并验证（namelist = `['SKILL.md']`，frontmatter name/description 完整，双包逐字节一致）：

- `agent_workspace/file-output-discipline.zip`
- `agent_workspace/file-output-discipline.skill`（同内容换扩展名）

存放于 `agent_workspace/`（C 类运行时区，可随时清理）；WorkBuddy 侧安装本体仍在 `~/.workbuddy/skills/file-output-discipline/`。

## frontmatter YAML 修复（2026-10-11 同日三修）

**根因**：description 为未加引号的 plain scalar，值内含英文冒号+空格（`Trigger on: `）——严格 YAML 解析器报 `ScannerError: mapping values are not allowed here`，第三方平台因此拒收；skill-creator 自带校验器为宽松匹配未拦截（**校验绿 ≠ 平台可解析**）。

**修复**：description 改为 ASCII 双引号包裹 + `Trigger on:` 改全角冒号"触发词："（双保险）。验证链：PyYAML strict 解析 OK（keys=name/description）→ description 280 字符（<1024 平台限）→ skill-creator 校验 valid → 重打根级包 → 字节级三向一致（zip=skill=source）。

**教训入库**：skill frontmatter 的 description 值必须引号包裹或杜绝 ASCII `: `；"官方校验通过"不等于"第三方平台 YAML 解析通过"，分发前须用严格解析器（PyYAML）实测。
