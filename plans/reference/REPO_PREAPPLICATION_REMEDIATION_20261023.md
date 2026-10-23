# 仓库申请前整改结案（2026-10-23）

用户选择落实仓库简介、安全核查、许可证与临时文件四项整改，不提交申请表。

## 项目现状与范围

进场HEAD为`0977f9b91`。项目认知主体是态极（Taiji）原生认知模型，Seed负责承载；N主线当前自然中文多轮三臂训练、恢复及其未提交修改继续由[03](../active/roadmap/03_CURRENT_EXECUTION.md)维护。本轮仅处理仓库展示与卫生，不触碰训练、checkpoint、课程、冻结门或他线修改。

## 已完成

1. GitHub首页About通过已登录浏览器保存并核验为：

   > Open-source runtime and research platform for Taiji (态极), a native cognitive model exploring persistent state, memory, predictive learning, planning and action-feedback loops. Includes Python APIs, experiment runners, tests and reproducible research reports.

2. [安全检查报告](../../reports/repo_preapplication_security_check_20261023.json)：安全守卫6/6通过；根目录三份在位凭据均被ignore，且与处置记录中的三个泄漏指纹不同。扫描范围、排除目录和一处浏览器缓存目录访问失败均记录在报告内。当前本地可达refs没有查到凭据路径blob，因此不能仅用历史比较全绿声称彻底清史；另比对了历史处置记录的指纹。没有轮换凭据或重写历史，没有输出凭据值。当前无dist目录，曾分发到其他机器的安装包、远端缓存和旧clone未核验。

3. 根LICENSE保留Apache-2.0及`Copyright 2026 NeuroPlex Contributors`，与首次开源提交`896f850af`一致。贡献指南解释历史项目名，并明确Harness自身MIT许可证及第三方声明的入口；没有覆盖或转移历史版权署名。

4. 删除两份`C\uF03AUsers23747AppDataLocalTempn1_v6_signcommit{7,8}.txt`。内容仅是git commit输出，删除前确认工作文件blob与HEAD相同；原blob为`70c997fc846a41d91722fe416f1e52a9aa5bb00f`、`19a8f8a25189f5e2d4ff8db683890590851d1638`，可从Git历史恢复。新增按命名族的ignore规则及双向守卫：拒绝异常路径再次入库，同时不忽略正常requirements文本。

## 验证与已知限制

- `python -m pytest tests/test_repo_secret_guard.py -q`：6通过。
- 安全与目录守卫联合检查：13通过、1失败；新增临时文件守卫通过。
- 失败为`test_tracked_machine_reports_have_one_home`：进场HEAD已经在output等历史根跟踪机器报告，数量见安全检查报告的验证补充。本轮未新增此类路径，也不迁移正在被其他工作引用的证据。
- 修改测试Ruff与改动空白检查通过；许可证正文未改。
- GitHub CLI认证返回401；简介改用用户已登录浏览器修改。远端main的已有提交`02e82d9`首页显示检查失败，本轮局部通过不代表全仓CI绿。

## 唯一后继建议

先对既有机器报告建立引用与摘要清单，再以保留原文、更新活动引用的方式修复报告目录守卫；同时保留训练恢复现场。该工作涉及现有研究证据与并行修改，不能通过删除报告或放宽断言消除红项。本轮不据此扩展训练或提交申请。
