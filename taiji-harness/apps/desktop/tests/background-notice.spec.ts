import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { DesktopBackgroundNotice, type DesktopBackgroundNoticeDialogOptions } from '../src/background-notice.ts'

const roots: string[] = []

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'desktop-background-notice-'))
  roots.push(root)
  const markerPath = join(root, 'nested', 'background-close-confirmed')
  const show = vi.fn(async (_options: DesktopBackgroundNoticeDialogOptions) => ({ response: 0 }))
  const focus = vi.fn()
  const notice = new DesktopBackgroundNotice({
    markerPath,
    locale: () => ({
      messages: { productName: 'Seed', backgroundNoticeBody: 'Running tasks will continue.', backgroundNoticeConfirm: 'Confirm' },
    }),
    show,
    focus,
  })
  return { notice, markerPath, show, focus }
}

describe('desktop background notice', () => {
  it('hides only after the acknowledgement and records the marker', async () => {
    const b = fixture()
    const hide = vi.fn()
    b.notice.close(hide)
    expect(hide).not.toHaveBeenCalled()
    await vi.waitFor(() => { expect(hide).toHaveBeenCalledOnce() })
    expect(b.show).toHaveBeenCalledOnce()
    expect(b.show.mock.calls[0]?.[0]).toMatchObject({
      type: 'info',
      title: 'Seed',
      message: 'Running tasks will continue.',
      buttons: ['Confirm'],
      defaultId: 0,
      cancelId: -1,
    })
    expect(existsSync(b.markerPath)).toBe(true)
  })

  it('keeps a cancelled prompt eligible, then hides without asking once the marker exists', async () => {
    const b = fixture()
    b.show.mockResolvedValueOnce({ response: 1 })
    const cancelled = vi.fn()
    b.notice.close(cancelled)
    await vi.waitFor(() => { expect(b.show).toHaveBeenCalledOnce() })
    expect(cancelled).not.toHaveBeenCalled()
    expect(existsSync(b.markerPath)).toBe(false)

    const accepted = vi.fn()
    b.notice.close(accepted)
    await vi.waitFor(() => { expect(accepted).toHaveBeenCalledOnce() })

    const remembered = vi.fn()
    b.notice.close(remembered)
    expect(remembered).toHaveBeenCalledOnce()
    expect(b.show).toHaveBeenCalledTimes(2)
  })

  it('hides without prompting when an earlier run recorded the marker', () => {
    const b = fixture()
    mkdirSync(dirname(b.markerPath), { recursive: true })
    writeFileSync(b.markerPath, '')
    const hide = vi.fn()
    b.notice.close(hide)
    expect(hide).toHaveBeenCalledOnce()
    expect(b.show).not.toHaveBeenCalled()
  })

  it('coalesces repeated closes while the prompt is open', async () => {
    const b = fixture()
    const pending = Promise.withResolvers<{ response: number }>()
    b.show.mockImplementation(() => pending.promise)
    const first = vi.fn()
    const second = vi.fn()
    b.notice.close(first)
    b.notice.close(second)
    expect(b.focus).toHaveBeenCalledOnce()
    expect(b.show).toHaveBeenCalledOnce()
    pending.resolve({ response: 0 })
    await vi.waitFor(() => { expect(first).toHaveBeenCalledOnce() })
    expect(second).not.toHaveBeenCalled()
  })

  it('ignores a late acknowledgement after disposal', async () => {
    const b = fixture()
    const pending = Promise.withResolvers<{ response: number }>()
    b.show.mockImplementation(() => pending.promise)
    const hide = vi.fn()
    b.notice.close(hide)
    b.notice.dispose()
    pending.resolve({ response: 0 })
    await Promise.resolve()
    await Promise.resolve()
    expect(hide).not.toHaveBeenCalled()
    expect(existsSync(b.markerPath)).toBe(false)
  })
})
