// Historical `Fish*` names: this module is the shared brand mark, now drawn as
// a geometric Taiji (yin-yang). The old name is kept so the three consumers
// (sidebar rail, hero, wordmark) and their imports stay stable.
import type { IconProps } from './icons/props.ts'

/** Native viewBox of {@link FISH_LOGO_PATH} (width and height in user units). */
export const FISH_LOGO_VIEWBOX = { width: 24, height: 24 }

/**
 * The Taiji mark path data (square 24x24 canvas: an r=12 disc, the S-divider
 * built from two tangent r=6 semicircles, and two r=1.5 fish-eye dots), exported
 * for consumers that compose their own svg (entrance effects, masks) around the
 * same geometry. Render with `fillRule="evenodd"` so the eye on the dark lobe
 * reads as a hole and the eye on the light lobe reads as a dot.
 */
export const FISH_LOGO_PATH = 'M12 0A12 12 0 0 0 12 24A6 6 0 0 0 12 12A6 6 0 0 1 12 0ZM10.5 6a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0 -3 0ZM10.5 18a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0 -3 0Z'

/**
 * Render the brand mark.
 * @param props.size - square edge in px (default 24; the mark is a 24x24 disc).
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
