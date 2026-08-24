import {
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
} from 'qoder/canvas';

const FINDINGS = [
  ['F01', 'high', '前端 ESLint 为 advisory 模式', '改为 blocking 门禁；修复暴露的 6 处真实代码缺陷（游离 catch 块、重复键等）', '✅'],
  ['F02', 'high', '覆盖率阈值 fail_under=0', '按实测基线设 fail_under=17，当前 19.35% 达标', '✅'],
  ['F03', 'high', 'mypy 未入 CI', 'CI 添加 mypy 步骤（advisory，渐进式策略）', '✅'],
  ['F04', 'medium', 'Vue 组件无测试', '新增 4 个组件测试文件、22 个用例（ConfirmDialog / RuntimeExceptionCenter / MemoryStatusBar / AppSidebar）', '✅'],
  ['F05', 'medium', 'CI 无桌面构建验证', 'Windows CI 编译检查 build.py / release.py / seed.spec', '✅'],
  ['F06', 'medium', 'Playwright 仅截图无断言', '新增 e2e/smoke.cjs：17 项断言（路由巡检、交互、导航、移动端、无未捕获异常）', '✅'],
  ['F07', 'medium', 'Windows CI 无缓存/覆盖率', '添加 cache: pip + coverage-win.xml + artifact 上传', '✅'],
  ['F08', 'medium', 'ruff 缺 I/UP 规则', 'CI advisory 追踪 import 排序与 pyupgrade 债务（存量 ~1886 处）', '✅'],
  ['F09', 'medium', 'OpenAPI 快照自动更新', '严格模式：漂移即失败且不重写基线，需显式 --snapshot-update', '✅'],
  ['F10', 'medium', '无 CVE 扫描', 'CI 添加 pip-audit --desc（advisory）', '✅'],
  ['F11', 'medium', 'verify 输出格式不统一', '_verify_emit.py 统一 {name, status, metrics, checks}；14 个脚本接入；CI 归档 reports/ci_verify/', '✅'],
  ['F12', 'low', '无 Docker 支持', '新增 Dockerfile + docker-compose.yml', '✅'],
  ['F13', 'low', '无性能基准测试', '新增 test_performance_baseline.py（5 用例 + benchmark marker）', '✅'],
  ['F14', 'low', 'import/no-unresolved 被禁用', '升为 error + alias resolver 配置', '✅'],
] as const;

const CHANGED_FILES = [
  ['CI/CD', '.github/workflows/ci.yml', 'ESLint blocking、mypy、pip-audit、I/UP advisory、Windows 缓存/覆盖率、桌面构建验证、verify 报告归档'],
  ['构建配置', 'pyproject.toml', 'fail_under=17；benchmark marker 注册'],
  ['测试基建', 'tests/conftest.py · tests/test_openapi_snapshot.py', '--snapshot-update 选项与快照严格模式'],
  ['测试基建', 'tests/test_performance_baseline.py', '性能基准（加载延迟 / 首字节 / 吞吐）'],
  ['统一输出', 'scripts/training/_verify_emit.py', 'VERIFY_RESULT 归一化（新增）'],
  ['统一输出', 'scripts/training/verify_*.py（14 个）', '全部接入 emit_and_exit'],
  ['前端测试', 'frontend/src/__tests__/*.test.js（4 个新增）', '组件挂载/交互测试 22 用例'],
  ['E2E', 'frontend/e2e/smoke.cjs', '带断言的冒烟测试（新增）'],
  ['Lint', 'frontend/.eslintrc.cjs · package.json', 'import resolver + 依赖；源码缺陷修复 6 处'],
  ['容器化', 'Dockerfile · docker-compose.yml', 'API 服务镜像与编排（新增）'],
] as const;

const VERIFICATION = [
  ['Python 测试', 'python -m pytest tests/ -q --cov', '123 passed, 4 skipped；覆盖率 19.35% ≥ 17% 门禁'],
  ['前端测试', 'npx vitest run', '74/74 通过（原 52 + 组件测试 22）'],
  ['ESLint（blocking）', 'npx eslint src --ext .js,.vue', '0 errors，exit 0（210 warnings 为有意降级存量）'],
  ['前端构建', 'npm run build', '构建成功，exit 0'],
  ['ruff 门禁', 'ruff check .', 'All checks passed'],
  ['black 门禁', 'black --check .', '261 files unchanged'],
  ['版本一致性', 'python scripts/sync_version.py --check', 'OK'],
  ['E2E 冒烟', 'node e2e/smoke.cjs（实机）', '17/17 断言通过'],
] as const;

export default function HarnessFixReport() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>Seed Harness 修复完成报告 — F01–F14 全部闭环</H1>
        <Text tone="secondary">
          依据 .qoder/better-harness/2026-08-24/102809-seed/findings.json · 2026-08-24
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="14 / 14" label="发现项已修复" tone="success" />
        <Stat value="3" label="High 级别清零" tone="success" />
        <Stat value="197" label="测试总数（123 Py + 74 FE）" />
        <Stat value="0" label="门禁失败项" tone="success" />
      </Grid>

      <Divider />

      <H2>逐项修复状态</H2>
      <Table
        headers={['编号', '严重度', '发现', '修复内容', '状态']}
        rows={FINDINGS.map((f) => [...f])}
        rowTone={FINDINGS.map((f) => (f[1] === 'high' ? 'warning' : undefined))}
      />

      <Divider />

      <H2>关键改动文件</H2>
      <Table
        headers={['类别', '文件', '改动说明']}
        rows={CHANGED_FILES.map((f) => [...f])}
      />

      <Divider />

      <H2>验证证据（全部实跑）</H2>
      <Table
        headers={['验证项', '命令', '结果']}
        rows={VERIFICATION.map((v) => [...v])}
      />

      <Divider />

      <Stack gap={8}>
        <H2>关键步骤回顾</H2>
        <Stack gap={4}>
          <Text>1. 探索现状：CI / lint / 测试 / 构建配置全面摸底，量化基线（覆盖率 19%、ESLint 存量、ruff 债务 1886 处）。</Text>
          <Text>2. 第一轮修复（8 项）：blocking 门禁、覆盖率阈值、mypy、pip-audit、快照严格模式、Windows CI、桌面构建验证、import 解析；同步修复 ESLint 暴露的 6 处源码缺陷。</Text>
          <Text>3. 第二轮修复（6 项）：22 个组件测试用例、17 断言 E2E 冒烟、ruff I/UP advisory 追踪、verify 统一输出 schema + CI 归档、Docker、性能基准。</Text>
          <Text>4. 回归处置：修复 verify 脚本被测试以包形式导入时的模块解析问题（import shim）。</Text>
          <Text>5. 完整验证：所有门禁全绿，确认无新问题引入。</Text>
        </Stack>
      </Stack>

      <Callout tone="success" title="最终结果">
        14 项发现全部修复并通过完整验证：Python 123 passed / 覆盖率达标、前端 74/74、ESLint 0 errors、构建成功。
        渐进式策略保留了 advisory 项（mypy / pip-audit / I/UP 债务）以便后续逐步收紧，门禁体系已具备防退化能力。
      </Callout>

      <Text tone="secondary" size="small">
        生成于 better-harness 修复目标完成审计 · Seed 项目
      </Text>
    </Stack>
  );
}
