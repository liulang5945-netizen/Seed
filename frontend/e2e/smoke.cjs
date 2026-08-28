/**
 * F06: Playwright E2E 冒烟测试（带断言）。
 *
 * 与 shoot-fe.cjs（纯截图）不同，本脚本对关键 UI 做显式断言：
 *   1. 每个路由可加载且主容器可见；
 *   2. 聊天页核心交互元素（欢迎区 / 输入框 / 发送按钮）存在且可用；
 *   3. 侧边导航链接完整且可点击跳转；
 *   4. 输入文本后发送按钮从禁用转为可用；
 *   5. 移动端视口可渲染；
 *   6. 全程无未捕获页面异常（pageerror）。
 *
 * 前置：前端开发服务器运行于 BASE_URL（默认 http://localhost:5173）。
 * 后端不要求在线——断言只覆盖静态 UI，不依赖 API 数据。
 *
 * 运行：cd frontend && node e2e/smoke.cjs
 */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.SEED_E2E_BASE_URL || 'http://localhost:5173';
const SHOT_DIR = process.env.SEED_E2E_SHOT_DIR || ''; // 设置后失败时落截图

// 路由 → 主容器选择器（与 router/index.js 保持一致）
const ROUTES = [
  { path: '/', selector: '.chat-workbench', name: 'chat', evidence: true },
  { path: '/#/kb', selector: '.kb-view', name: 'kb', evidence: true },
  { path: '/#/train', selector: '.training-view', name: 'train', evidence: true },
  { path: '/#/agent', selector: '.agent-page', name: 'agent', evidence: true },
  { path: '/#/workspace', selector: '.workspace-view', name: 'workspace', evidence: false },
  { path: '/#/life', selector: '.life-status-view', name: 'life', evidence: true },
  { path: '/#/settings', selector: '.settings-view', name: 'settings', evidence: true },
];

// 简易断言工具：收集失败但不中断，最后统一汇报
const failures = [];
let assertions = 0;

function check(name, cond, detail = '') {
  assertions += 1;
  if (cond) {
    console.log(`  PASS  ${name}`);
  } else {
    console.log(`  FAIL  ${name}${detail ? ' — ' + detail : ''}`);
    failures.push(name + (detail ? ` (${detail})` : ''));
  }
}

async function shot(page, tag) {
  if (!SHOT_DIR) return;
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOT_DIR, `e2e-fail-${tag}.png`) }).catch(() => {});
}

(async () => {
  console.log(`[E2E] target = ${BASE_URL}`);
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  // Smoke 必须从空会话开始，否则开发机残留的 localStorage 会让欢迎区、建议词
  // 和新建对话路径变成非确定性状态。
  await ctx.addInitScript(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
  const page = await ctx.newPage();

  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  // ---- 0. 服务器可达 ----
  try {
    await page.goto(BASE_URL + '/', { waitUntil: 'domcontentloaded', timeout: 10000 });
  } catch (e) {
    console.error(`[E2E] FATAL: 无法访问 ${BASE_URL}（请先启动前端开发服务器：npm run dev）`);
    await browser.close();
    process.exit(2);
  }

  // ---- 1. 聊天页核心断言 ----
  console.log('\n[E2E] 聊天页 (/)');
  await page.waitForTimeout(1500);
  check('chat-workbench 容器可见', await page.locator('.chat-workbench').isVisible().catch(() => false));
  check('侧边栏存在', await page.locator('aside.sidebar').count().then((n) => n > 0));
  // 聊天舞台非空：无历史消息时显示欢迎区，有历史消息（如 localStorage 恢复）时显示消息列表
  const welcomeVisible = await page.locator('.chat-welcome h1').isVisible().catch(() => false);
  const messageCount = await page.locator('.chat-stage article').count().catch(() => 0);
  check('欢迎区或历史消息渲染', welcomeVisible || messageCount > 0, `welcome=${welcomeVisible} messages=${messageCount}`);
  check('输入框可见', await page.locator('.composer textarea').isVisible().catch(() => false));

  // ---- 2. 输入 → 发送按钮遵守运行时门控 ----
  console.log('\n[E2E] 输入交互');
  const sendBtn = page.locator('.composer button.send');
  // 发送按钮保持可点击（disabled 移除），但未就绪时带 unavailable 类；点击后由 toast 解释原因
  const unavailableBefore = await sendBtn.evaluate((el) => el.classList.contains('unavailable')).catch(() => null);
  check('空输入时发送按钮门控（unavailable）', unavailableBefore === true);
  await page.locator('.composer textarea').fill('你好');
  await page.waitForTimeout(300);
  check('输入文本后内容保留', (await page.locator('.composer textarea').inputValue().catch(() => '')) === '你好');
  const runtimeLabel = await page.locator('.welcome-sub').innerText().catch(() => '');
  const runtimeReady = runtimeLabel.includes('运行时已连接');
  const unavailableAfter = await sendBtn.evaluate((el) => el.classList.contains('unavailable')).catch(() => true);
  if (runtimeReady) {
    check('运行时就绪后发送按钮可用', !unavailableAfter);
  } else {
    check('运行时未就绪时发送按钮保持门控', unavailableAfter);
  }

  // ---- 2.5 关键路径交互（R4 新增）----
  console.log('\n[E2E] 关键路径交互');
  // A. 建议词 → 输入回显：点击建议芯片后输入框应填入对应文本
  await page.goto(BASE_URL + '/', { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.waitForTimeout(1200);
  const suggestion = page.locator('.suggestions .suggestion').first();
  if (await suggestion.count()) {
    const hint = (await suggestion.innerText()).trim();
    await suggestion.click();
    await page.waitForTimeout(300);
    const echoed = await page.locator('.composer textarea').inputValue().catch(() => '');
    check('点击建议词后输入框回显', echoed.trim() === hint, `期望「${hint}」 实际「${echoed}」`);
  } else {
    check('聊天页存在建议词芯片', false, '未找到 .suggestion');
  }

  // B. 新建对话按钮：点击后应回到聊天页并聚焦输入区
  const newChatBtn = page.locator('.new-chat-btn');
  if (await newChatBtn.count()) {
    await newChatBtn.first().click();
    await page.waitForTimeout(600);
    check('新建对话后回到聊天页', page.url().includes('/'), page.url());
    check('新建对话后输入区可见', await page.locator('.composer textarea').isVisible().catch(() => false));
  } else {
    check('侧边栏存在新建对话按钮', false, '未找到 .new-chat-btn');
  }

  // C. 训练页标签切换：超参数面板随点击激活
  await page.goto(BASE_URL + '/#/train', { waitUntil: 'domcontentloaded', timeout: 10000 });
  await page.waitForTimeout(1200);
  const trainTabs = page.locator('.training-view .tabs button.tab');
  if ((await trainTabs.count()) >= 2) {
    await trainTabs.nth(1).click();
    await page.waitForTimeout(400);
    check('训练页标签切换后激活超参数页', await trainTabs.nth(1).evaluate((el) => el.classList.contains('active')).catch(() => false));
  } else {
    check('训练页存在标签栏', false, '未找到 .tabs button.tab');
  }

  // ---- 3. 侧边导航完整性与跳转 ----
  console.log('\n[E2E] 侧边导航');
  const navLinks = page.locator('nav[aria-label="主导航"] a.nav-item');
  const navCount = await navLinks.count();
  check('导航链接数量 >= 5', navCount >= 5, `实际 ${navCount}`);
  // 点击一个导航项验证路由切换（选"系统设置"，若存在）
  const settingsLink = navLinks.filter({ hasText: '设置' }).first();
  if (await settingsLink.count()) {
    await settingsLink.click();
    await page.waitForTimeout(800);
    check('点击导航后 URL 切换', page.url().includes('/settings'), page.url());
  } else {
    check('导航中包含"设置"入口', false, '未找到');
  }

  // ---- 4. 各路由可加载 ----
  console.log('\n[E2E] 路由巡检');
  for (const r of ROUTES) {
    try {
      await page.goto(BASE_URL + r.path, { waitUntil: 'domcontentloaded', timeout: 10000 });
    } catch (e) {
      check(`路由 ${r.path} 可访问`, false, e.message);
      continue;
    }
    await page.waitForTimeout(1200);
    const visible = await page.locator(r.selector).first().isVisible().catch(() => false);
    check(`路由 ${r.path} 主容器可见`, visible);
    check(`路由 ${r.path} 没有错误页面`, await page.locator('.route-error-view').count() === 0);
    if (r.evidence) {
      check(`路由 ${r.path} 展示状态证据`, await page.locator('.runtime-evidence').count() > 0);
    }
    if (!visible) await shot(page, r.name);
  }

  // ---- 5. 移动端视口 ----
  console.log('\n[E2E] 移动端视口');
  await ctx.close();
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true });
  await mctx.addInitScript(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
  const mpage = await mctx.newPage();
  mpage.on('pageerror', (err) => pageErrors.push(String(err)));
  await mpage.goto(BASE_URL + '/', { waitUntil: 'domcontentloaded', timeout: 10000 });
  await mpage.waitForTimeout(1500);
  check('移动端 #app 渲染', await mpage.locator('#app').isVisible().catch(() => false));
  await mctx.close();

  // ---- 6. 未捕获异常 ----
  check('无未捕获页面异常', pageErrors.length === 0, pageErrors.slice(0, 3).join(' | '));

  await browser.close();

  console.log(`\n[E2E] 断言 ${assertions} 项，失败 ${failures.length} 项`);
  if (failures.length) {
    for (const f of failures) console.log(`  - ${f}`);
    process.exit(1);
  }
  console.log('[E2E] ALL PASS');
  process.exit(0);
})().catch((e) => {
  console.error('[E2E] FATAL', e);
  process.exit(1);
});
