import type { IconProps } from './icons/props.ts'
import { FISH_LOGO_PATH } from './FishLogo.tsx'

/** Display options for the official brand wordmark. */
export interface BrandWordmarkProps extends IconProps {
  /** Whether to include the leading brand mark; defaults to true. */
  includeMark?: boolean | undefined
}

/** Product name rendered as live text beside the mark. */
const BRAND_NAME = 'Seed'

/**
 * Render the full brand wordmark: the Seed sprout mark (shared
 * {@link FISH_LOGO_PATH}, scaled to the mark box) followed by the product name
 * set as live text, so the lettering inherits the host font and colour instead
 * of carrying vectorised letterforms.
 * @param props.size - height in px (default 24; width follows the selected artwork).
 * @param props.className - extra class for layout placement.
 * @param props.includeMark - whether to include the leading brand mark.
 * @returns the wordmark svg (aria-hidden decorative brand art).
 */
export function BrandWordmark({ size = 24, className, includeMark = true }: BrandWordmarkProps) {
  const width = includeMark ? 70 : 44
  return (
    <svg
      width={(size * width) / 24}
      height={size}
      className={className}
      viewBox={includeMark ? '0 0 70 24' : '26 0 44 24'}
      fill="none"
      aria-hidden="true"
    >
      {includeMark && (
        <g transform="translate(0 3) scale(0.75)">
          <path d={FISH_LOGO_PATH} fill="currentColor" fillRule="evenodd" />
        </g>
      )}
      <text x="24" y="17.5" fill="currentColor" fontFamily="inherit" fontSize="15" fontWeight="600">
        {BRAND_NAME}
      </text>
    </svg>
  )
}