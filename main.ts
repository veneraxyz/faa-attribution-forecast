// polymarket-faa-attribution-02 entry (Node 22, standard library only): an
// adaptive-volatility forecaster. It beats the house persistence baseline
// (fixed sigma 0.0025) by sizing each market's spread from its own recent
// price history instead of one number for every market.
//
// Design (validated on the practice cases with a local scorer that
// reproduces the house baseline's stated practice loss of 0.363589 exactly;
// this entry scores ~0.286 on the same cases):
//
//   - Median: persistence. The last price is the median at every horizon.
//     Prediction-market prices are (near-)martingales, and mild drift terms
//     (mean-reversion, momentum) tested worse on practice, so no drift.
//   - Spread: per-market robust sigma from recent 1-minute returns. Prices
//     move in steps (87% of recent 1-minute returns in the practice data are
//     exactly 0) with occasional jumps, so a plain standard deviation is
//     dominated by one old jump and overstates the typical move. The IQR of
//     recent returns measures the typical tick instead:
//     sigma = max(IQR(last 60 min)/1.349, 0.7 * IQR(last 240 min)/1.349),
//     floored at 0.0006 (jump insurance for markets that look quiet) and
//     capped at 0.006. With fewer than 2 returns it falls back to 0.0025.
//   - Inner quantiles (25%, 75%): Gaussian around the median with that sigma.
//   - Outer quantiles (5%, 95%): a 50/50 blend of the Gaussian quantile and
//     the empirical quantile of past h-minute moves from the last 720 minutes
//     of prices. The empirical tails carry the jump risk the recent regime
//     actually showed (asymmetric when recent moves were one-sided); the
//     Gaussian half keeps them sane when the recent window was quiet. With
//     fewer than 20 past h-minute moves the outer quantile is pure Gaussian.
//   - Every value is clamped to [0, 1], rounded to 6 decimals, and each row
//     is forced non-decreasing after rounding.
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
const SIGMA_FLOOR = 0.0006;
const SIGMA_CAP = 0.006;
/** Fallback sigma when a market has too little history to estimate one. */
const SIGMA_FALLBACK = 0.0025;
/** Minutes of recent prices the empirical tails are cut from. */
const EMPIRICAL_WINDOW = 720;
/** Minimum past h-minute moves for the empirical tail to be used. */
const EMPIRICAL_MIN_MOVES = 20;

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
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
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
    // Empirical outer tails from past h-minute moves, when there are enough.
    const moveCount = n - h;
    if (moveCount >= EMPIRICAL_MIN_MOVES) {
      const moves: number[] = new Array<number>(moveCount);
      for (let i = 0; i < moveCount; i++) {
        moves[i] = (hist[i + h] as number) - (hist[i] as number);
      }
      moves.sort((a, b) => a - b);
      lo = 0.5 * quantile(moves, 0.05) + 0.5 * (g[0] as number);
      hi = 0.5 * quantile(moves, 0.95) + 0.5 * (g[4] as number);
    }
    const row = [
      price6(x + lo),
      price6(x + (g[1] as number)),
      price6(x),
      price6(x + (g[3] as number)),
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

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
