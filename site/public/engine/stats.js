/*  Confidence intervals and significance tests.
 *
 *  Port of metrics/stats.py. Everything here is exact arithmetic with no
 *  randomness, so it matches Python to the last digit.
 *
 *  The bootstrap is deliberately absent. metrics/stats.py seeds it with
 *  CPython's Mersenne Twister, and matching that in JavaScript would mean
 *  reimplementing MT19937 plus _randbelow's rejection sampling. It is not
 *  worth it: at the 2000 resamples run_decisive.py uses, the published
 *  interval is not even stable under Python's own RNG -- across 300 seeds it
 *  comes out as [-0.375, +0.125] only 242 times. Porting the PRNG would pin
 *  a number that is a coin-flip from being different. Where this page needs
 *  an interval it uses Wilson, which is exact.
 */
"use strict";

const Z_TABLE = { 0.10: 1.6448536269514722, 0.05: 1.959963984540054,
                  0.01: 2.5758293035489004 };

function zFor(alpha){
  if (alpha in Z_TABLE) return Z_TABLE[alpha];
  return invNorm(1 - alpha / 2);          // invNorm lives in engines.js
}

/**
 * Wilson score interval. Behaves where the normal approximation does not:
 * at 0 or n successes it still returns a bounded interval instead of a point.
 * `point` is the raw proportion, not the shifted centre.
 */
function wilson(successes, n, alpha = 0.05){
  if (n <= 0) return { point: 0, low: 0, high: 1, n: 0, method: "wilson" };
  const z = zFor(alpha);
  const p = successes / n;
  const denom = 1 + z * z / n;
  const centre = (p + z * z / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return { point: p, low: Math.max(0, centre - half), high: Math.min(1, centre + half),
           n, method: "wilson" };
}

/** Binomial coefficient, iteratively. n stays <= 24 on the exact path. */
function comb(n, k){
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let c = 1;
  for (let i = 1; i <= k; i++) c = c * (n - k + i) / i;
  return c;
}

function binomTwoSided(b, c){
  const n = b + c;
  if (n === 0) return 1.0;
  const k = Math.min(b, c);
  let tail = 0;
  for (let i = 0; i <= k; i++) tail += comb(n, i);
  return Math.min(1.0, 2 * (tail / Math.pow(2, n)));
}

/** Abramowitz-Stegun 7.1.26 complementary error function. */
function erfc(x){
  const z = Math.abs(x);
  const t = 1 / (1 + z / 2);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196
    + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/**
 * McNemar's test on paired binary outcomes.
 *
 * Only DISCORDANT pairs carry information -- cases where both systems agree
 * contribute nothing, which is why 24 branches yielded 9 usable ones in the
 * decisive run. Below 25 discordant the exact binomial is used, and at that
 * size 2**n stays well inside a double.
 */
function mcnemar(correctA, correctB){
  if (correctA.length !== correctB.length) throw new Error("unpaired inputs");
  let b = 0, c = 0, bothRight = 0, bothWrong = 0;
  for (let i = 0; i < correctA.length; i++){
    const x = !!correctA[i], y = !!correctB[i];
    if (x && !y) b++;
    else if (!x && y) c++;
    else if (x && y) bothRight++;
    else bothWrong++;
  }
  let p, method;
  if (b + c < 25){
    p = binomTwoSided(b, c);
    method = "exact binomial";
  } else {
    const chi2 = Math.pow(Math.abs(b - c) - 1, 2) / (b + c);
    p = Math.min(1.0, erfc(Math.sqrt(chi2 / 2)));
    method = "chi-square, continuity-corrected";
  }
  return { b, c, bothRight, bothWrong, p, method,
           nDiscordant: b + c, n: b + c + bothRight + bothWrong };
}

/**
 * Holm-Bonferroni step-down. Controls the family-wise error rate without
 * Bonferroni's loss of power.
 *
 * Two details that are easy to get wrong: adjusted p values are forced
 * non-decreasing down the ordering, and the rejection flag is cleared BEFORE
 * the current entry is recorded, so the first entry at or above alpha is
 * itself not rejected and neither is anything after it.
 */
function holmAdjust(pValues, alpha = 0.05){
  const entries = Object.entries(pValues);
  if (!entries.length) return {};
  entries.sort((x, y) => x[1] - y[1]);          // stable in modern JS
  const m = entries.length;
  const out = {};
  let running = 0, stillRejecting = true;
  entries.forEach(([name, p], i) => {
    const adjusted = Math.min(1.0, Math.max(running, (m - i) * p));
    running = adjusted;
    if (adjusted >= alpha) stillRejecting = false;
    out[name] = { adjusted, rejected: stillRejecting && adjusted < alpha };
  });
  return out;
}
