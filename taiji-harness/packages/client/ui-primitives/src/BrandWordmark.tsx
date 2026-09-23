import type { IconProps } from './icons/props.ts'
import { FISH_LOGO_PATH } from './FishLogo.tsx'

/** Display options for the official brand wordmark. */
export interface BrandWordmarkProps extends IconProps {
  /** Whether to include the leading brand mark; defaults to true. */
  includeMark?: boolean | undefined
}

/**
 * Render the full brand wordmark: the Taiji mark (shared {@link FISH_LOGO_PATH},
 * scaled to the mark box) followed by the product name set as live text, so the
 * lettering inherits the host font and colour instead of carrying vectorised
 * letterforms.
 * @param props.size - height in px (default 24; width follows the selected artwork).
 * @param props.className - extra class for layout placement.
 * @param props.includeMark - whether to include the leading brand mark.
 * @returns the wordmark svg (aria-hidden decorative brand art).
 */
export function BrandWordmark({ size = 24, className, includeMark = true }: BrandWordmarkProps) {
  const width = includeMark ? 182 : 156
  return (
    <svg
      width={(size * width) / 24}
      height={size}
      className={className}
      viewBox={includeMark ? '0 0 182 24' : '26 0 156 24'}
      fill="none"
      aria-hidden="true"
    >
      {includeMark && (
        <g transform="translate(2 3) scale(0.75)">
          <path d={FISH_LOGO_PATH} fill="currentColor" fillRule="evenodd" />
        </g>
      )}
      <text x="28" y="17.5" fill="currentColor" fontFamily="inherit" fontSize="15" fontWeight="600">
        Taiji Harness
      </text>
    </svg>
  )
}
