/**
 * Small helpers that reproduce Python string semantics, so that lengths,
 * budgets and whitespace handling match the reference implementation
 * (`chunkers.py`) exactly.
 */

/** Characters Python's `str.isspace()` / `str.strip()` / `\s` treat as whitespace. */
export const PY_WS =
  "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

const STRIP_RE = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "gu");

/** Python `str.strip()` with no arguments. */
export function pyStrip(s: string): string {
  return s.replace(STRIP_RE, "");
}

/**
 * Python `len(str)`: number of Unicode code points (a JS `.length` counts
 * UTF-16 code units, so astral characters such as emoji would count twice).
 */
export function cpLen(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      const d = s.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) i++;
    }
    n++;
  }
  return n;
}

/**
 * Python `str.isdigit()`: every decimal digit (Unicode category Nd) plus the
 * characters with Numeric_Type=Digit (superscripts, circled digits, ...).
 */
const DIGIT_RE = new RegExp(
  "[\\p{Nd}\\u00B2\\u00B3\\u00B9\\u1369-\\u1371\\u19DA\\u2070\\u2074-\\u2079\\u2080-\\u2089" +
    "\\u2460-\\u2468\\u2474-\\u247C\\u2488-\\u2490\\u24EA\\u24F5-\\u24FD\\u24FF" +
    "\\u2776-\\u277E\\u2780-\\u2788\\u278A-\\u2792\\u{10A40}-\\u{10A43}" +
    "\\u{10E60}-\\u{10E68}\\u{11052}-\\u{1105A}\\u{1F100}-\\u{1F10A}]",
  "gu",
);

/**
 * Digit density ρ(b) = (#digit characters) / len(b). Digit-heavy blocks
 * (fees, deadlines, phone numbers, tariff rows) are "fact dense".
 */
export function digitDensity(text: string): number {
  if (!text) return 0;
  const m = text.match(DIGIT_RE);
  return (m ? m.length : 0) / cpLen(text);
}
