// Historical `Fish*` names: this module is the shared brand mark, now drawn as
// the Seed mark — a tree inside a round seed. The old name is kept so the three
// consumers (sidebar rail, hero, wordmark) and their imports stay stable.
import type { IconProps } from './icons/props.ts'

/** Native viewBox of {@link FISH_LOGO_PATH} (width and height in user units). */
export const FISH_LOGO_VIEWBOX = { width: 24, height: 24 }

/** The round seed disc (a filled r=11 circle on the 24x24 canvas). */
export const FISH_LOGO_DISC_PATH = 'M23 12A11 11 0 1 1 1 12A11 11 0 1 1 23 12Z'

/**
 * The tree that lives inside the seed: three overlapping canopy lobes (centre,
 * left, right) above a tapered trunk. Drawn over the disc with the default
 * `nonzero` fill rule for the coloured artwork.
 */
export const FISH_LOGO_TREE_PATH = 'M17.2 8.4A5.2 5.2 0 1 1 6.8 8.4A5.2 5.2 0 1 1 17.2 8.4ZM11.4 11A3.7 3.7 0 1 1 4 11A3.7 3.7 0 1 1 11.4 11ZM20 11A3.7 3.7 0 1 1 12.6 11A3.7 3.7 0 1 1 20 11ZM13.2 13.6C13.3 16 13.4 18.2 13.5 20.4L10.5 20.4C10.6 18.2 10.7 16 10.8 13.6Z'

/**
 * The monochrome Seed mark: the seed disc with the tree knocked out of it.
 * Render with `fillRule="evenodd"` so the tree reads as a hole in whichever
 * colour the caller supplies via `currentColor`.
 */
export const FISH_LOGO_PATH = `${FISH_LOGO_DISC_PATH}${FISH_LOGO_TREE_PATH}`

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
      <path d={FISH_LOGO_PATH} fill="currentColor" fillRule="evenodd" />
    </svg>
  )
}