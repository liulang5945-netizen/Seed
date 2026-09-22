import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  assertDesktopHostPackageFiles,
  selectDesktopPackageClosure,
  type PackedDesktopPackage,
} from '../scripts/prepare-package-set.ts'

function packed(name: string, manifest: Record<string, unknown> = {}): PackedDesktopPackage {
  return { tarball: `${name}.tgz`, manifest: { name, version: '1.0.0', ...manifest } }
}

describe('desktop package-set selection', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('does not select a packaging target when imported as a library', async () => {
    vi.stubEnv('DSH_DESKTOP_TARGET_PLATFORM', 'linux')
    vi.stubEnv('DSH_DESKTOP_TARGET_ARCH', 'x64')
    vi.resetModules()
    await expect(import('../scripts/prepare-package-set.ts')).resolves.toHaveProperty('prepareDesktopPackageSet')
  })

  it('includes only the available internal production closure', () => {
    const available = new Map<string, PackedDesktopPackage>([
      ['@taiji/dsh', packed('@taiji/dsh', {
        dependencies: { '@taiji/dsh-base': '^1.0.0', external: '^2.0.0' },
        optionalDependencies: { '@taiji/platform-package': '1.0.0', '@taiji/missing-platform': '1.0.0' },
      })],
      ['@taiji/dsh-desktop-host', packed('@taiji/dsh-desktop-host', {
        dependencies: { '@taiji/dsh': '^1.0.0' },
      })],
      ['@taiji/dsh-base', packed('@taiji/dsh-base', {
        peerDependencies: { '@taiji/cordis': '^1.0.0' },
      })],
      ['@taiji/cordis', packed('@taiji/cordis')],
      ['@taiji/platform-package', packed('@taiji/platform-package')],
      ['@taiji/unused', packed('@taiji/unused')],
    ])
    expect(selectDesktopPackageClosure(available).map(entry => entry.manifest.name)).toEqual([
      '@taiji/cordis',
      '@taiji/dsh',
      '@taiji/dsh-base',
      '@taiji/dsh-desktop-host',
      '@taiji/platform-package',
    ])
  })

  it.each([
    '@taiji/dsh-base', '@taiji/cordis', '@taiji/node-addon-system',
  ])('rejects required prepared package %s absent from the packed release inputs', (dependency) => {
    const available = new Map<string, PackedDesktopPackage>([
      ['@taiji/dsh', packed('@taiji/dsh', {
        dependencies: { [dependency]: '^1.0.0' },
      })],
      ['@taiji/dsh-desktop-host', packed('@taiji/dsh-desktop-host', {
        dependencies: { '@taiji/dsh': '^1.0.0' },
      })],
    ])
    expect(() => selectDesktopPackageClosure(available)).toThrow(/unpacked package/u)
    expect(() => selectDesktopPackageClosure(new Map([
      ['@taiji/dsh', packed('@taiji/dsh')],
    ]))).toThrow(/omit @taiji\/dsh-desktop-host/u)
  })

  it('leaves independently published Office packages to npm resolution', () => {
    const available = new Map<string, PackedDesktopPackage>([
      ['@taiji/dsh', packed('@taiji/dsh', {
        dependencies: {
          '@deepseek-ai/libreoffice-kit': '0.0.1',
          '@deepseek-ai/libreoffice-kit-wasm': '0.0.1',
        },
      })],
      ['@taiji/dsh-desktop-host', packed('@taiji/dsh-desktop-host')],
    ])
    expect(selectDesktopPackageClosure(available).map(entry => entry.manifest.name)).toEqual([
      '@taiji/dsh', '@taiji/dsh-desktop-host',
    ])
  })

  it('requires the Desktop Host entry', () => {
    const files = [
      'package/lib/index.js',
    ]
    expect(() => {
      assertDesktopHostPackageFiles(files)
    }).not.toThrow()
    expect(() => {
      assertDesktopHostPackageFiles(files.slice(1))
    }).toThrow(/lib\/index\.js/u)
  })
})
