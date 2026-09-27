// Historical `Fish*` names: this module is the shared brand mark, now drawn as
// the Seed sprout (a seed at the base, one stem, two leaves). The old name is
// kept so the three consumers (sidebar rail, hero, wordmark) and their imports
// stay stable.
import type { IconProps } from './icons/props.ts'

/** Native viewBox of {@link FISH_LOGO_PATH} (width and height in user units). */
export const FISH_LOGO_VIEWBOX = { width: 24, height: 24 }

/**
 * The Seed sprout mark path data (square 24x24 canvas, four closed subpaths:
 * the teardrop seed, the rounded stem, and the left/right leaves). Exported for
 * consumers that compose their own svg (entrance effects, masks) around the
 * same geometry. Render with the default `nonzero` fill rule; the subpaths only
 * touch edge-to-edge, so no lobe cancels.
 */
export const FISH_LOGO_PATH = 'M15 19.6A3 3 0 0 1 9 19.6C9 18.2 10.1 16.8 12 15.6C13.9 16.8 15 18.2 15 19.6ZM11.4 3.6A0.6 0.6 0 0 1 12.6 3.6V15.8H11.4ZM12.6 10.6C15.4 9.4 17.4 6.6 17.6 3.8C14.8 4.4 12.9 6.4 12.6 8.4ZM11.4 10.6C8.6 9.4 6.6 6.6 6.4 3.8C9.2 4.4 11.1 6.4 11.4 8.4Z'

/**
 * Render the brand mark.
 * @param props.size - square edge in px (default 24; the mark is a 24x24 canvas).
 * @param props.className - extra class for layout placement.
 * @returns the logo svg (aria-hidden; pair with the wordmark for accessibility).
 */
export function FishLogo({ size = 24, className }: IconProps) {
  return (
    <svg
      width={size}
      height={(size * FISH_LOGO_VIEWBOX.height) / FISH_LOGO_VIEWBOX.width}
      className={className}
      viewBox={`0 0 ${FISH_LOGO_VIEWBOX.width} ${FISH_LOGO_VIEWBOX.height}`}
      fill="none"
      aria-hidden="true"
    >
      <path d={FISH_LOGO_PATH} fill="currentColor" />
    </svg>
  )
}