// polymarket-faa-attribution-02 entry, v3 (Node 22, standard library only):
// an adaptive-volatility forecaster with sibling-sum normalisation. It beats
// the house persistence baseline (fixed sigma 0.0025, practice loss
// 0.363589), the v1 entry (0.286323) and the v2 entry (0.265890) by sizing
// each market's spread from its own recent price history, by calibrating
// the inner spread to the dispersion the market actually showed at each
// horizon, AND by enforcing the one piece of structure attribution markets
// share: the sibling outcomes sum to ~1 — with the correction falling
// hardest on the markets whose prices are most uncertain.
//
// Design (validated on the practice cases with a local scorer that
// reproduces the house baseline's stated practice loss of 0.363589 exactly;
// this entry scores ~0.257 on the same cases, better than v2 on all 6):
//
//   - Median: persistence. The last price is the median at every horizon.
//     Prediction-market prices are (near-)martingales, and drift terms
//     (mean-reversion toward the recent average, momentum) tested worse on
//     practice, so no drift.
//   - Spread: per-market robust sigma from recent 1-minute returns. Prices
//     move in steps (87% of recent 1-minute returns in the practice data are
//     exactly 0) with occasional jumps, so a plain standard deviation is
//     dominated by one old jump and overstates the typical move. The IQR of
//     recent returns measures the typical tick instead:
//     sigma = max(IQR(last 60 min)/1.349, 0.7 * IQR(last 240 min)/1.349),
//     floored at 0.0004 (jump insurance for markets that look quiet) and
//     capped at 0.005. With fewer than 2 returns it falls back to 0.0025.
//   - Inner quantiles (25%, 75%): Gaussian around the median with that
//     sigma, scaled by the TERM STRUCTURE: the ratio of the empirical
//     IQR of past h-minute moves to the Gaussian h-horizon spread
//     (sigma*sqrt(h)), clipped to [0.5, 2]. The Gaussian spread assumes
//     dispersion grows as sqrt(h); the ratio corrects it to the
//     dispersion the market actually showed at each horizon (tighter
//     when recent moves were calm, wider when they were jumpy).
//   - Outer quantiles (5%, 95%): a 40/60 blend of the empirical quantile of
//     past h-minute moves from the last 720 minutes and the Gaussian
//     quantile. The empirical tails carry the jump risk the recent regime
//     actually showed (asymmetric when recent moves were one-sided); the
//     Gaussian majority keeps them sane when the recent window was quiet.
//     With fewer than 20 past h-minute moves the outer quantile is pure
//     Gaussian.
//   - Sibling-sum normalisation: after forecasting every market, each
//     horizon row's medians are scaled toward summing to 1 (all five
//     quantiles of a market scale by the same factor, so spreads scale with
//     the median). The correction is ASYMMETRIC and grows with horizon:
//     when the median sum is above 1 it is corrected at full strength (a
//     sum above 1 cannot come from missing outcomes, only mispricing), when
//     below 1 only at quarter strength (a low sum may mean the case lists
//     only some outcomes). Strength grows as 1 - exp(-h/45): the sum's
//     deviation is already mispricing at the checkpoint, but it takes about
//     an hour to fully correct. The correction is also VOLATILITY-WEIGHTED:
//     market i's factor is weighted by sigma_i^0.75 (normalised to mean 1
//     across the case's priced markets), so the correction falls hardest
//     on the markets whose prices are most uncertain and barely touches
//     the quiet ones whose prices are more likely right. Markets with no
//     prices are excluded from the sum and left untouched.
//   - Every value is clamped to [0, 1], rounded to 6 decimals, and each row
//     is forced non-decreasing after rounding.
//   - A market with no prices at all is forecast flat at 0.5 rather than
//     failing the case: an empty market must not invalidate its case-mates.
//
// Everything is deterministic: fixed iteration order, no randomness, no
// clock, no network. Two runs on the same case write the same bytes.
//
// Input/output contract (as the kit's starter): read every <id>.in in the
// input directory (MAKEFUN_INPUT_DIR, else /input), write <id>.out to the
// output directory (MAKEFUN_OUTPUT_DIR, else /output): {"forecasts":[
// {"id", "q": [[5 quantiles] x horizon_minutes]}]}, every market of the
// case exactly once, at most 1 MiB per file.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** Floor/cap on the per-market robust sigma (price units per sqrt-minute). */
const SIGMA_FLOOR = 0.0004;
const SIGMA_CAP = 0.005;
/** Fallback sigma when a market has too little history to estimate one. */
const SIGMA_FALLBACK = 0.0025;
/** Minutes of recent prices the empirical tails are cut from. */
const EMPIRICAL_WINDOW = 720;
/** Minimum past h-minute moves for the empirical tail to be used. */
const EMPIRICAL_MIN_MOVES = 20;
/** Weight of the Gaussian quantile in the outer-quantile blend (the rest is
 *  the empirical quantile of past h-minute moves). */
const GAUSSIAN_OUTER_WEIGHT = 0.6;
/** Term-structure clip bounds: the inner-spread ratio (empirical h-move
 *  IQR / Gaussian h-horizon spread) is clipped to this range. */
const TERM_CLIP_LO = 0.5;
const TERM_CLIP_HI = 2.0;
/** Volatility-weight power for the sibling-sum normalisation: market i's
 *  correction is weighted by sigma_i^NORM_WEIGHT_POWER, normalised to
 *  mean 1 across the case's priced markets. */
const NORM_WEIGHT_POWER = 0.75;
/** Sibling-sum normalisation: the correction's time constant in minutes —
 *  strength at horizon h is (1 - exp(-h / NORM_TAU)). */
const NORM_TAU = 45;
/** Normalisation strength when the median sum is BELOW 1 (a low sum may
 *  mean the case lists only some outcomes, so correct gently). */
const NORM_UP = 0.25;
/** Normalisation strength when the median sum is ABOVE 1 (a sum above 1
 *  cannot come from missing outcomes, only mispricing, so correct fully). */
const NORM_DOWN = 1.0;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Linear-interpolation quantile of a sorted array (0 <= tau <= 1). */
function quantile(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0] as number;
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const frac = pos - lo;
  return (sorted[lo] as number) * (1 - frac) + (sorted[hi] as number) * frac;
}

/** Robust sigma of 1-minute returns: IQR / 1.349 (Gaussian-consistent). */
function iqrSigma(returns: number[]): number {
  if (returns.length < 2) return 0;
  const sorted = [...returns].sort((a, b) => a - b);
  return (quantile(sorted, 0.75) - quantile(sorted, 0.25)) / 1.349;
}

/**
 * Per-market sigma from recent 1-minute returns, favouring the most recent
 * hour, with a floor (jump insurance) and a cap (one wild market cannot
 * blow the spread up without bound).
 */
function adaptiveSigma(prices: number[]): number {
  if (prices.length < 3) return SIGMA_FALLBACK;
  const returns: number[] = [];
  for (let i = 0; i + 1 < prices.length; i++) {
    returns.push((prices[i + 1] as number) - (prices[i] as number));
  }
  const s60 = iqrSigma(returns.slice(-60));
  const s240 = iqrSigma(returns.slice(-240));
  const sig = Math.max(s60, 0.7 * s240);
  return Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sig));
}

/**
 * The forecast of one market: horizon rows of the five quantiles around its
 * last price. Median is persistence; inner quantiles are Gaussian with the
 * market's robust sigma; outer quantiles blend the Gaussian with empirical
 * quantiles of past h-minute moves from the recent window.
 */
function forecast(m: CaseMarket, horizon: number): number[][] {
  // A market with no prices at all carries no information: forecast it flat
  // at 0.5 (maximally uncertain) instead of failing the whole case. One
  // empty market must not zero out the other markets' forecasts — a crash
  // here would make the entire case invalid (scored as all-0.5).
  if (m.prices.length === 0) {
    return Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]);
  }
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = adaptiveSigma(m.prices);
  const hist = m.prices.length > EMPIRICAL_WINDOW
    ? m.prices.slice(-EMPIRICAL_WINDOW)
    : m.prices;
  const n = hist.length;
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = sigma * Math.sqrt(h);
    const g = Z.map((z) => z * spread);
    let lo = g[0] as number;
    let hi = g[4] as number;
    let in1 = g[1] as number;
    let in3 = g[3] as number;
    // Empirical outer tails from past h-minute moves, when there are enough.
    const moveCount = n - h;
    if (moveCount >= EMPIRICAL_MIN_MOVES) {
      const moves: number[] = new Array<number>(moveCount);
      for (let i = 0; i < moveCount; i++) {
        moves[i] = (hist[i + h] as number) - (hist[i] as number);
      }
      moves.sort((a, b) => a - b);
      lo = (1 - GAUSSIAN_OUTER_WEIGHT) * quantile(moves, 0.05) + GAUSSIAN_OUTER_WEIGHT * (g[0] as number);
      hi = (1 - GAUSSIAN_OUTER_WEIGHT) * quantile(moves, 0.95) + GAUSSIAN_OUTER_WEIGHT * (g[4] as number);
      // Term structure: scale the inner quantiles by the ratio of the
      // empirical IQR of past h-minute moves to the Gaussian h-horizon
      // spread, so the inner interval follows the dispersion the market
      // actually showed at this horizon rather than assuming sqrt(h)
      // scaling. The ratio is clipped so one noisy horizon cannot blow
      // the interval up or collapse it.
      const empIqr = (quantile(moves, 0.75) - quantile(moves, 0.25)) / 1.349;
      const ratio = Math.min(TERM_CLIP_HI, Math.max(TERM_CLIP_LO, empIqr / spread));
      in1 = (g[1] as number) * ratio;
      in3 = (g[3] as number) * ratio;
    }
    const row = [
      price6(x + lo),
      price6(x + in1),
      price6(x),
      price6(x + in3),
      price6(x + hi),
    ];
    // Enforce non-decreasing (the blend can cross the inner quantiles).
    for (let j = 1; j < 5; j++) {
      if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
    }
    rows.push(row);
  }
  return rows;
}

/**
 * Sibling-sum normalisation for one case, applied in place. Attribution
 * markets in a case sum to ~1, so at each horizon the medians should too.
 * Every market's whole row is scaled by one factor (spreads scale with the
 * median), the factor being 1 + w_i * s * (1 - exp(-h/45)) * (1/S - 1)
 * where S is the row's median sum, s is NORM_UP (S < 1) or NORM_DOWN
 * (S > 1), and w_i is the market's volatility weight: sigma_i^0.75,
 * normalised to mean 1 across the case's priced markets. The asymmetry is
 * deliberate: a sum above 1 is mispricing and corrects fully, a sum below
 * 1 may mean the case lists only some outcomes and is corrected gently.
 * The volatility weighting is deliberate too: the correction falls
 * hardest on the markets whose prices are most uncertain. Markets with no
 * prices are excluded from the sum and left flat at 0.5. A case with
 * fewer than two priced markets is left alone (a lone market has no
 * siblings to sum against).
 */
function normaliseForecasts(rows: number[][][], empty: boolean[], sigmas: number[]): void {
  const idx: number[] = [];
  for (let i = 0; i < rows.length; i++) {
    if (!empty[i]) idx.push(i);
  }
  if (idx.length < 2) return;
  const wMean = idx.reduce((a, i) => a + Math.pow(sigmas[i] as number, NORM_WEIGHT_POWER), 0) / idx.length;
  const horizon = (rows[0] as number[][]).length;
  for (let h = 0; h < horizon; h++) {
    let sum = 0;
    for (const i of idx) sum += (rows[i] as number[][])[h][2] as number;
    if (sum <= 0) continue;
    const growth = 1 - Math.exp(-(h + 1) / NORM_TAU);
    const strength = sum < 1 ? NORM_UP : NORM_DOWN;
    for (const i of idx) {
      const w = wMean > 0 ? Math.pow(sigmas[i] as number, NORM_WEIGHT_POWER) / wMean : 1;
      const f = 1 + w * strength * growth * (1 / sum - 1);
      if (f === 1) continue;
      const row = (rows[i] as number[][])[h] as number[];
      for (let j = 0; j < 5; j++) row[j] = price6((row[j] as number) * f);
      for (let j = 1; j < 5; j++) {
        if ((row[j] as number) < (row[j - 1] as number)) row[j] = row[j - 1] as number;
      }
    }
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const rows = c.markets.map((m) => forecast(m, c.horizon_minutes));
  const sigmas = c.markets.map((m) => adaptiveSigma(m.prices));
  normaliseForecasts(rows, c.markets.map((m) => m.prices.length === 0), sigmas);
  const forecasts = c.markets.map((m, i) => ({ id: m.id, q: rows[i] }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
