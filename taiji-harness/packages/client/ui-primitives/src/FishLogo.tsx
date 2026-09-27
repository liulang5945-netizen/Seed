// Historical `Fish*` names: this module is the shared brand mark, now drawn as
// a stylised tree (a three-lobed canopy above a tapered trunk). The old name is
// kept so the three consumers (sidebar rail, hero, wordmark) and their imports
// stay stable.
import type { IconProps } from './icons/props.ts'

/** Native viewBox of {@link FISH_LOGO_PATH} (width and height in user units). */
export const FISH_LOGO_VIEWBOX = { width: 24, height: 24 }

/**
 * The Seed tree mark path data (square 24x24 canvas, four closed subpaths: the
 * centre canopy lobe, the left and right canopy lobes, and the trunk). Exported
 * for consumers that compose their own svg (entrance effects, masks) around the
 * same geometry. Render with the default `nonzero` fill rule; every subpath is
 * wound the same way, so overlapping lobes stay solid.
 */
export const FISH_LOGO_PATH = 'M17 7.2A5 5 0 1 1 7 7.2A5 5 0 1 1 17 7.2ZM11.3 9.6A3.7 3.7 0 1 1 3.9 9.6A3.7 3.7 0 1 1 11.3 9.6ZM20.1 9.6A3.7 3.7 0 1 1 12.7 9.6A3.7 3.7 0 1 1 20.1 9.6ZM13.3 12C13 15 13.5 17.8 13.8 20.6L10.2 20.6C10.5 17.8 11 15 10.7 12Z'

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