/**
 * P6-1c browser field canary for the real local Qwen semantic provider.
 *
 * The backend must be started with the explicit semantic-provider environment
 * binding. This script exercises the normal chat UI and records only field
 * evidence; it does not install or mutate a model artifact.
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.SEED_E2E_BASE_URL || 'http://127.0.0.1:5173';
const PROMPT = '读取 s1_upload_test.txt 并确认内容';
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const REPORT = process.env.SEED_P6_1C_REPORT
  || path.join(REPO_ROOT, 'reports', 'taiji_w7_p6_1c_qwen_browser_field_20260831.json');
const FORBIDDEN_CLIENT_FIELDS = new Set([
  'parameter_bindings',
  'patch',
  'before_digest',
  'expected_after_digest',
  'action_intent',
  'intent',
]);

const checks = [];
const requests = [];
const responses = [];
const responseReads = [];
const pageErrors = [];

function check(name, passed, detail = '') {
  checks.push({ name, passed: Boolean(passed), detail });
  console.log(`${passed ? 'PASS' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
}

function forbiddenFields(payload) {
  if (!payload || typeof payload !== 'object') return [];
  return Object.keys(payload).filter((key) => FORBIDDEN_CLIENT_FIELDS.has(key));
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('request', (request) => {
    const url = request.url();
    if (!url.includes('/api/chat/workbench/')) return;
    let payload = null;
    try { payload = request.postDataJSON(); } catch { /* request may have no JSON body */ }
    requests.push({
      path: new URL(url).pathname,
      method: request.method(),
      forbidden_fields: forbiddenFields(payload),
      payload,
    });
  });
  page.on('response', (response) => {
    const url = response.url();
    if (!url.includes('/api/chat/workbench/')) return;
    responseReads.push((async () => {
      let payload = null;
      try { payload = await response.json(); } catch { /* non-JSON response */ }
      responses.push({
        path: new URL(url).pathname,
        status: response.status(),
        payload,
      });
    })());
  });

  let statusCode = 0;
  let interpretation = null;
  let plan = null;
  let execution = null;
  try {
    const response = await page.goto(`${BASE_URL}/#/`, {
      waitUntil: 'domcontentloaded',
      timeout: 15000,
    });
    statusCode = response?.status() || 0;
    check('聊天页可访问', statusCode === 200, `HTTP ${statusCode}`);

    const runtimeStatus = await page.evaluate(async () => {
      const response = await fetch('/api/runtime/status');
      return response.json();
    });
    check(
      'Taiji 运行时已连接',
      runtimeStatus?.health?.state === 'connected' && runtimeStatus?.health?.model_loaded === true,
      `${runtimeStatus?.health?.state || 'unknown'} / model_loaded=${runtimeStatus?.health?.model_loaded}`,
    );

    await page.locator('button[title="工作台"]').click();
    await page.locator('.composer textarea').fill(PROMPT);
    await page.locator('button.send').click();
    await page.locator('.workbench-task-card').waitFor({ state: 'visible', timeout: 120000 });
    check('工作台任务卡片出现', true);

    await page.locator('.task-intake').waitFor({ state: 'visible', timeout: 120000 });
    const intakeText = await page.locator('.task-intake').innerText();
    check('卡片显示 Taiji 目标证据', intakeText.includes('Taiji 目标证据'));

    await page.locator('.semantic-preview').waitFor({ state: 'visible', timeout: 120000 });
    const semanticText = await page.locator('.semantic-preview').innerText();
    check('卡片显示 provider 语义步骤证据', semanticText.includes('语义步骤证据'));
    check('卡片显示真实目标路径', semanticText.includes('s1_upload_test.txt'));
    await page.locator('.task-plan').waitFor({ state: 'visible', timeout: 120000 });
    check('卡片显示 Taiji 执行计划', (await page.locator('.task-plan').innerText()).includes('Taiji 执行计划'));

    const planButton = page.locator('.task-plan button.action-button.primary');
    check('计划无需前端生成执行字段', await planButton.count() > 0);
    if (await planButton.count()) {
      await planButton.click();
      await page.locator('.execution-result').waitFor({ state: 'visible', timeout: 120000 });
      const executionText = await page.locator('.execution-result').innerText();
      check('只读 Workbench 执行完成', executionText.includes('工作台执行完成'));
    }

    const interpretRequest = requests.find((item) => item.path.endsWith('/interpret'));
    const planRequest = requests.find((item) => item.path.endsWith('/natural-language/plan'));
    const executeRequest = requests.find((item) => item.path.endsWith('/natural-language/execute'));
    check('前端发送 interpret 请求', Boolean(interpretRequest));
    check('前端发送 Taiji plan 请求', Boolean(planRequest));
    check('前端发送 Taiji execute 请求', Boolean(executeRequest));
    check('interpret 请求无越权字段', (interpretRequest?.forbidden_fields || []).length === 0);
    check('plan 请求无越权字段', (planRequest?.forbidden_fields || []).length === 0);
    check('execute 请求无越权字段', (executeRequest?.forbidden_fields || []).length === 0);

    interpretation = {
      provider_id: await page.locator('.evidence-meta').innerText().catch(() => ''),
      card_text: intakeText,
      semantic_text: semanticText,
    };
    plan = planRequest?.payload || null;
    execution = executeRequest?.payload || null;
  } catch (error) {
    check('浏览器现场无未处理异常', false, error.message);
  } finally {
    await Promise.all(responseReads);
    check('浏览器无 pageerror', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));
    await browser.close();
  }

  const passed = checks.every((item) => item.passed) && pageErrors.length === 0;
  const report = {
    format: 'taiji-w7-p6-1c-qwen-browser-field-v1',
    base_url: BASE_URL,
    prompt: PROMPT,
    checks,
    requests: requests.map(({ payload, ...item }) => item),
    responses,
    interpretation,
    plan,
    execution,
    page_errors: pageErrors,
    gate: {
      passed,
      criterion: 'real Qwen evidence is visible in the chat Workbench card and Taiji owns plan/execute without client execution-field injection',
    },
  };
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify(report, null, 2), 'utf8');
  console.log(`report=${REPORT}`);
  process.exitCode = passed ? 0 : 1;
}

main().catch((error) => {
  console.error('FATAL', error);
  process.exitCode = 1;
});
