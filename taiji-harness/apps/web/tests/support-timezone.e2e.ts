import { chromium } from 'playwright'
import { expect, it } from 'vitest'
import { newEnglishPage } from './support.ts'

// The premise "Chromium honors the TZ environment variable" holds on POSIX only:
// on Windows the browser always reports the host's real timezone, so this recorded
// browser/ambient isolation cannot be exercised there (G5 §7 (丙), owner-approved
// platform skip 2026-09-27).
it.skipIf(process.platform === 'win32').each(['UTC', 'America/Los_Angeles'])('isolates the recorded browser timezone from %s', async (hostTimeZone) => {
  const browser = await chromium.launch({ env: { ...process.env, TZ: hostTimeZone } })
  try {
    const ambientPage = await browser.newPage()
    expect(await ambientPage.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone)).toBe(hostTimeZone)

    const page = await newEnglishPage(browser)
    expect(await page.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone)).toBe('Asia/Shanghai')
    expect(await page.evaluate(() => navigator.language)).toBe('en-US')
    expect(await ambientPage.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone)).toBe(hostTimeZone)

    await page.close()
    const nextPage = await browser.newPage()
    expect(await nextPage.evaluate(() => Intl.DateTimeFormat().resolvedOptions().timeZone)).toBe(hostTimeZone)
  } finally {
    await browser.close()
  }
})
