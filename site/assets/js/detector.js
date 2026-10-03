/*  Detection: lexical features, the heuristic NLI stand-in, and stage 1.
 *
 *  Port of detection/features.py, detection/nli.py::HeuristicNLI and
 *  detection/stage1.py::_rule_score.
 *
 *  TWO WARNINGS THE SOURCE INSISTS ON, carried here because the UI must
 *  repeat them:
 *
 *    HeuristicNLI is "a fixture, not a model". It is lexical overlap wearing
 *    the shape of an entailment scorer, and it never produces a reported
 *    number.
 *
 *    _rule_score is "a scaffold rather than a model: measured on mock data it
 *    scores at chance, so its 'resolved locally' pairs are resolved WRONGLY."
 *    Over the 42 annotated instances it tops out at 0.2206 against an
 *    escalation floor of 0.35 and flags zero conflicts. That is not a bug in
 *    this port; it is the measured behaviour, and showing it is the point.
 *
 *  Note the regexes differ from the ones in match.js on purpose. There are
 *  three distinct number patterns across the Python codebase with different
 *  lookarounds, and unifying them reintroduces bugs each was written to fix.
 */
"use strict";

const STRONG_EXCEPTION_CUES = ["except","unless","excluding","other than","save for",
  "apart from","does not apply","do not apply","shall not apply","is not applicable",
  "notwithstanding","provided that","provided, that","only if","only when","only for",
  "solely for","waived for","exempt from","exemption","carve-out","carve out",
  "unless otherwise","with the exception of","save that"];

const WEAK_CONDITION_CUES = ["subject to","in the case of","where the","if the",
  "limited to","provided","for the purposes of","in respect of"];

const RESTRICTION_CUES = ["tier","premium","platinum","signature","gold","silver",
  "bronze","student","senior","resident","non-resident","eligible","qualifying",
  "category","class","type","holders","members","customers who","accounts with",
  "cards with"];

const TEMPORAL_CUES = ["effective","as of","from 20","until","prior to","before 20",
  "after 20","with effect from","no longer","previously","currently",
  "has been updated","revised"];

const OPINION_CUES = ["we believe","in our view","arguably","many consider",
  "most advisers","some argue","regard","widely seen","generally considered",
  "practitioners","critics","reportedly","may be seen as"];

const EXCEPTION_CUES = STRONG_EXCEPTION_CUES.concat(WEAK_CONDITION_CUES);

const _F_NUM_RE = /\d+(?:[.,]\d+)?\s*%?/g;

// NOT a typo, and not worth "fixing" here. Python's pattern is
// `\b(19|20)\d{2}\b` with a CAPTURING group, and re.findall returns the
// group rather than the whole match -- so every 20xx year collapses to the
// string "20" and two different years in the same century never clash.
// features.py's year_clash is therefore almost always 0. This port
// reproduces that, because the site's job is to show what the pipeline
// actually does, not a tidied-up version of it.
const _F_YEAR_RE = /\b(19|20)\d{2}\b/g;
const _F_MODAL_RE = /\b(may|must|shall|can|cannot|will not|won't|should)\b/gi;
const _F_WORD_RE = /[a-z']+/g;

/* ---------------------------------------------------------------- NLI --- */

const _NEGATIONS = new Set(["cannot","denied","does","doesn't","excluded","exempt",
  "may","never","no","not","prohibited","waived","won't"]);
const _NLI_NUM_RE = /\d+(?:\.\d+)?%?/g;      // no comma branch, unlike match.js

function _set(re, text){ return new Set(String(text || "").match(re) || []); }

/** features.py strips each number; branch_match.py's copy does too, but the
 *  NLI one does not. Keeping them separate is deliberate. */
function _numSet(text){
  const out = new Set();
  for (const m of String(text || "").match(_F_NUM_RE) || []) out.add(m.trim());
  return out;
}

/** Python's findall returns the capturing group -- see _F_YEAR_RE above. */
function _yearSet(text){
  const out = new Set();
  for (const m of String(text || "").matchAll(_F_YEAR_RE)) out.add(m[1]);
  return out;
}

function _setsEqual(a, b){
  if (a.size !== b.size) return false;
  for (const v of a) if (!b.has(v)) return false;
  return true;
}
function _lowerSet(text){ return new Set(String(text || "").toLowerCase().match(_F_WORD_RE) || []); }
function _overlapCount(a, b){ let n = 0; for (const v of a) if (b.has(v)) n++; return n; }

/** Five discrete buckets, checked in this exact order. */
function heuristicNLI(premise, hypothesis){
  const p = _lowerSet(premise), h = _lowerSet(hypothesis);
  if (!p.size || !h.size) return { entailment: 0.0, neutral: 1.0, contradiction: 0.0 };

  const overlap = _overlapCount(p, h) / Math.max(1, h.size);
  const pn = _set(_NLI_NUM_RE, premise), hn = _set(_NLI_NUM_RE, hypothesis);
  const numericClash = pn.size > 0 && hn.size > 0 && _overlapCount(pn, hn) === 0;
  const polarityClash = (_overlapCount(p, _NEGATIONS) > 0) !== (_overlapCount(h, _NEGATIONS) > 0);

  if (numericClash  && overlap > 0.3) return { entailment: 0.05, neutral: 0.15, contradiction: 0.80 };
  if (polarityClash && overlap > 0.3) return { entailment: 0.10, neutral: 0.25, contradiction: 0.65 };
  if (overlap > 0.7)                  return { entailment: 0.75, neutral: 0.20, contradiction: 0.05 };
  if (overlap > 0.35)                 return { entailment: 0.25, neutral: 0.65, contradiction: 0.10 };
  return { entailment: 0.05, neutral: 0.90, contradiction: 0.05 };
}

/**
 * The 0.25 weight on neutral is the premise of the whole project: a genuine
 * conditional conflict reads as NEUTRAL to an entailment model, not as
 * CONTRADICTION, so a signal built only on contradiction cannot see it.
 */
function conflictSignal(s){ return s.contradiction + 0.25 * s.neutral; }

/* ----------------------------------------------------------- features --- */

/** Distinct cues PRESENT, not occurrences. */
function countCues(text, cues){
  const low = String(text || "").toLowerCase();
  let n = 0;
  for (const c of cues) if (low.includes(c)) n++;
  return n;
}

function qualifierCount(text){
  const low = String(text || "").toLowerCase();
  const modals = (low.match(_F_MODAL_RE) || []).length;
  const commas = (low.match(/,/g) || []).length;
  return countCues(low, STRONG_EXCEPTION_CUES) + countCues(low, WEAK_CONDITION_CUES)
       + countCues(low, RESTRICTION_CUES) + modals + commas;
}

/**
 * 17 features, every one symmetric or absolute-valued, so that reordering the
 * two passages cannot change them. Order invariance has to survive feature
 * extraction or it is not a property of the system.
 */
function extractPairFeatures(textI, textJ){
  const ei = countCues(textI, STRONG_EXCEPTION_CUES);
  const ej = countCues(textJ, STRONG_EXCEPTION_CUES);
  const ri = countCues(textI, RESTRICTION_CUES);
  const rj = countCues(textJ, RESTRICTION_CUES);
  const qi = qualifierCount(textI), qj = qualifierCount(textJ);

  const numsI = _numSet(textI), numsJ = _numSet(textJ);
  const shared = _overlapCount(numsI, numsJ);
  const union = numsI.size + numsJ.size - shared;
  const yearsI = _yearSet(textI), yearsJ = _yearSet(textJ);
  const sameYears = _setsEqual(yearsI, yearsJ);

  const tokI = _lowerSet(textI), tokJ = _lowerSet(textJ);
  const tShared = _overlapCount(tokI, tokJ);
  const tUnion = tokI.size + tokJ.size - tShared;

  const lenI = Math.max(1, (String(textI || "").split(/\s+/).filter(Boolean)).length);
  const lenJ = Math.max(1, (String(textJ || "").split(/\s+/).filter(Boolean)).length);

  return {
    exception_cues_i: ei, exception_cues_j: ej,
    exception_cues_max: Math.max(ei, ej),
    exception_cues_asymmetry: Math.abs(ei - ej),
    restriction_cues_i: ri, restriction_cues_j: rj,
    restriction_asymmetry: Math.abs(ri - rj),
    qualifiers_i: qi, qualifiers_j: qj,
    specificity_asymmetry: Math.abs(qi - qj),
    length_ratio: Math.min(lenI, lenJ) / Math.max(lenI, lenJ),
    numeric_overlap: union ? shared / union : 0.0,
    numeric_clash: (numsI.size && numsJ.size && shared === 0) ? 1.0 : 0.0,
    year_clash: (yearsI.size && yearsJ.size && !sameYears) ? 1.0 : 0.0,
    temporal_cues_max: Math.max(countCues(textI, TEMPORAL_CUES), countCues(textJ, TEMPORAL_CUES)),
    opinion_cues_max: Math.max(countCues(textI, OPINION_CUES), countCues(textJ, OPINION_CUES)),
    token_jaccard: tUnion ? tShared / tUnion : 0.0,
  };
}

/* ------------------------------------------------------------ stage 1 --- */

const ESCALATE_LOW = 0.35;      // config/hardware.yaml, thresholds_tuned: false
const ESCALATE_HIGH = 0.75;

/**
 * The whole offline detector. Weights sum to 1.15 before the topicality
 * multiplier damps anything phrased in different vocabulary toward zero --
 * which is exactly why it finds nothing on this corpus.
 */
function ruleScore(nli, f){
  let score = 0.6 * conflictSignal(nli);
  score += 0.20 * Math.min(1.0, f.exception_cues_max / 2.0);
  score += 0.10 * Math.min(1.0, f.restriction_asymmetry / 2.0);
  score += 0.15 * f.numeric_clash;
  score += 0.10 * f.year_clash;
  score *= 0.35 + 0.65 * Math.min(1.0, f.token_jaccard * 3.0);
  return Math.min(1.0, Math.max(0.0, score));
}

/** Score one passage pair exactly as Stage1Filter does offline. */
function screenPair(textI, textJ){
  const nli = heuristicNLI(textI, textJ);
  const features = extractPairFeatures(textI, textJ);
  const p = ruleScore(nli, features);
  return {
    score: p, nli, features,
    isConflict: p >= 0.5,
    confidence: p >= 0.5 ? p : 1.0 - p,
    needsEscalation: p >= ESCALATE_LOW && p <= ESCALATE_HIGH,
  };
}
