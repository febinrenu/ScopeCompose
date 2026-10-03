/*  Branch matching and preservation scoring.
 *
 *  Port of metrics/branch_match.py and the deterministic paths of
 *  metrics/preservation.py. Checked against Python over the whole annotated
 *  corpus by site/tools/verify_engines.js.
 *
 *  Several of the constants below look arbitrary and are not. Each one is a
 *  fixed bug, and the comments say which, because the obvious "tidier"
 *  rewrite reintroduces it:
 *
 *    - "no" and "not" are deliberately NOT stopwords. When they were,
 *      "no fee applies" and "a fee applies" scored a Jaccard similarity of
 *      1.00 -- a branch and its exact opposite judged identical.
 *    - polarity parity is taken over a SET, so two different negators cancel.
 *    - _NUM_RE is greedy over \s*%?, so "3 %" and "3%" are distinct strings.
 *      Three different _NUM_RE patterns exist across the Python codebase with
 *      different lookarounds; they must not be unified.
 */
"use strict";

const OUTCOME_MATCH = 0.45;
const OUTCOME_CONTAINMENT = 0.6;
const APPLICABILITY_MATCH = 0.35;

const _STOP = new Set(["a","all","an","and","any","are","be","for","in","is","it",
                       "of","on","or","that","the","this","to","with"]);
const _NEGATORS = new Set(["cannot","neither","never","no","non","none","nor",
                           "not","nothing","without"]);
const _CARDINALS = {zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,
                    eight:8,nine:9,ten:10,eleven:11,twelve:12,thirteen:13,
                    fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,
                    nineteen:19};
const _TENS = {twenty:20,thirty:30,forty:40,fifty:50,sixty:60,seventy:70,
               eighty:80,ninety:90};

const _WORD_RE = /[a-z0-9']+/g;
const _NUM_RE = /\d+(?:[.,]\d+)?\s*%?/g;
const _QUANTIFIER_RE =
  /\b(?:no|not)\s+(?:more|less|fewer|later|earlier|sooner|longer)\s+than\b/gi;

const BJ = { PRESERVED: "preserved", SUPPRESSED: "suppressed", DISTORTED: "distorted" };

function _words(text){ return String(text || "").toLowerCase().match(_WORD_RE) || []; }

function _tokens(text){
  const out = new Set();
  for (const w of _words(text)) if (!_STOP.has(w)) out.add(w);
  return out;
}

function _inter(a, b){ let n = 0; for (const v of a) if (b.has(v)) n++; return n; }

/** Symmetric Jaccard over content tokens. */
function similarity(a, b){
  const ta = _tokens(a), tb = _tokens(b);
  if (!ta.size || !tb.size) return 0.0;
  const i = _inter(ta, tb);
  return i / (ta.size + tb.size - i);
}

/** Asymmetric: the fraction of GOLD's words that appear in pred. */
function containment(gold, pred){
  const tg = _tokens(gold), tp = _tokens(pred);
  if (!tg.size) return 0.0;
  return _inter(tg, tp) / tg.size;
}

/** Figures mentioned, digits and spelled cardinals alike. */
function _numbers(text){
  const found = new Set();
  for (const m of String(text || "").match(_NUM_RE) || []) found.add(m.trim());
  const words = _words(text);
  let i = 0;
  while (i < words.length){
    const w = words[i];
    if (w in _TENS){
      const nxt = i + 1 < words.length ? words[i + 1] : "";
      // "twenty eight" -> 28, and the bare 20 is deliberately NOT added.
      if (nxt in _CARDINALS && _CARDINALS[nxt] >= 1 && _CARDINALS[nxt] <= 9){
        found.add(String(_TENS[w] + _CARDINALS[nxt]));
        i += 2;
        continue;
      }
      found.add(String(_TENS[w]));
    } else if (w in _CARDINALS){
      found.add(String(_CARDINALS[w]));
    }
    i++;
  }
  return found;
}

/** True when both sides cite figures and share none. */
function numbers_conflict(a, b){
  const na = _numbers(a), nb = _numbers(b);
  return na.size > 0 && nb.size > 0 && _inter(na, nb) === 0;
}

/** "no more than 20 hours" is a quantifier, not a negation. */
function _strip_quantifiers(text){
  return String(text || "").replace(_QUANTIFIER_RE, " ");
}

/**
 * One side negates, the other does not, and they are otherwise talking about
 * the same thing. A hard veto on outcome matching: without it, "no fee
 * applies" and "a fee applies" matched.
 */
function polarity_conflict(a, b){
  const ta = _tokens(_strip_quantifiers(a)), tb = _tokens(_strip_quantifiers(b));
  // Parity over the SET: two DIFFERENT negators cancel, which is why
  // "does not lapse" and "never lapses" are not a conflict.
  if ((_inter(ta, _NEGATORS) % 2) === (_inter(tb, _NEGATORS) % 2)) return false;
  const coreA = new Set([...ta].filter(t => !_NEGATORS.has(t)));
  const coreB = new Set([...tb].filter(t => !_NEGATORS.has(t)));
  if (!coreA.size || !coreB.size) return false;
  // 0.6 here is a literal in the Python, NOT OUTCOME_CONTAINMENT.
  return _inter(coreA, coreB) / Math.min(coreA.size, coreB.size) >= 0.6;
}

/**
 * Do two outcomes say the same thing? Order is load-bearing: the figure and
 * polarity vetoes override every similarity signal beneath them.
 * The NLI arm of the Python is omitted -- the browser cannot run that model.
 */
function outcomes_match(gold, pred){
  if (numbers_conflict(gold, pred) || polarity_conflict(gold, pred)) return false;
  if (similarity(gold, pred) >= OUTCOME_MATCH) return true;
  if (containment(gold, pred) >= OUTCOME_CONTAINMENT) return true;
  return false;
}

/**
 * preservation.py's TEXTUAL judging path: prose in, a judgement per gold
 * branch out. The thresholds 0.6 and 0.25 are inline literals in the Python.
 *
 * This path is a LOWER BOUND on preservation, as the source says -- it can
 * only see what the answer's words overlap with.
 */
function judgeTextual(answerText, goldBranches){
  return (goldBranches || []).map(b => {
    if (containment(b.out, answerText) >= 0.6) return BJ.PRESERVED;
    if (similarity(b.out, answerText) >= 0.25) return BJ.DISTORTED;
    return BJ.SUPPRESSED;
  });
}

/** PR / SR / distortion over a flat list of judgements. */
function rates(judgements){
  const n = judgements.length;
  if (!n) return { PR: 0, SR: 0, DR: 0, n: 0, preserved: 0, suppressed: 0, distorted: 0 };
  const c = { preserved: 0, suppressed: 0, distorted: 0 };
  for (const j of judgements) c[j]++;
  return { PR: c.preserved / n, SR: c.suppressed / n, DR: c.distorted / n, n, ...c };
}
