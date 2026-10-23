/*  Inputs that are not benchmark records.
 *
 *  Every example here is ILLUSTRATIVE and the page labels it so wherever it
 *  appears. What is real is the computation: these inputs are run through the
 *  same verified engines as the corpus. An illustrative input to a real engine
 *  is still an illustration -- it is never quoted as a measurement.
 */
import type { Branch, Passage, Relation } from "../engine/types";

export interface Example {
  key: string;
  label: string;
  /** Always shown next to the example. */
  provenance: "Illustrative input" | string;
  q: string;
  p: Passage[];
  b: Branch[];
  ty: "conditional";
  /** The relation this example was written to show; the page shows what the engine returns. */
  intended?: Relation;
}

/** The story's example. Source types are assigned (illustratively) so the
 *  real selection fallback, mostCredible(), has something to rank. */
export const FEE: Example = {
  key: "fee",
  label: "International fee with a premium carve-out",
  provenance: "Illustrative input",
  q: "Do I pay an international transaction fee?",
  ty: "conditional",
  intended: "refinement",
  p: [
    { id: "p0", t: "International transactions incur a 3% fee.", st: "official_policy", dt: "2024-03", d: "card_terms" },
    { id: "p1", t: "Fees are waived for premium-tier cardholders.", st: "product_terms", dt: "2025-01", d: "premium_benefits" },
  ],
  b: [
    { id: "b0", out: "a 3% fee applies to international transactions", desc: "all cases", def: true, sup: "p0" },
    { id: "b1", out: "no international transaction fee is charged", desc: "premium-tier cardholders",
      def: false, sup: "p1", cond: "the cardholder is on the premium tier",
      at: [{ n: "card_tier", k: "ordered_tier", v: ["premium"] }] },
  ],
};

/** One small example per relation, for the geometry explorer. */
export const RELATION_EXAMPLES: Record<Relation, Example> = {
  refinement: FEE,
  disjoint: {
    key: "disjoint", label: "Two rules that never meet", provenance: "Illustrative input",
    q: "What notice period applies?", ty: "conditional", intended: "disjoint",
    p: [
      { id: "p0", t: "Fixed-term contracts require one month of notice.", st: "official_policy" },
      { id: "p1", t: "Permanent contracts require three months of notice.", st: "official_policy" },
    ],
    b: [
      { id: "b0", out: "one month of notice is required", desc: "fixed-term contracts", def: false, sup: "p0",
        cond: "the contract is fixed-term", at: [{ n: "contract_type", k: "categorical", v: ["fixed_term"] }] },
      { id: "b1", out: "three months of notice is required", desc: "permanent contracts", def: false, sup: "p1",
        cond: "the contract is permanent", at: [{ n: "contract_type", k: "categorical", v: ["permanent"] }] },
    ],
  },
  redundant: {
    key: "redundant", label: "A restatement, not an exception", provenance: "Illustrative input",
    q: "Do I pay an international transaction fee?", ty: "conditional", intended: "redundant",
    p: [
      { id: "p0", t: "International transactions incur a 3% fee.", st: "official_policy" },
      { id: "p1", t: "Standard cardholders pay a 3% fee on international transactions.", st: "product_terms" },
    ],
    b: [
      { id: "b0", out: "a 3% fee applies to international transactions", desc: "all cases", def: true, sup: "p0" },
      { id: "b1", out: "a 3% fee applies to international transactions", desc: "standard cardholders",
        def: false, sup: "p1", cond: "the cardholder is on the standard tier",
        at: [{ n: "card_tier", k: "ordered_tier", v: ["standard"] }] },
    ],
  },
  opposed: {
    key: "opposed", label: "Overlap, and they disagree there", provenance: "Illustrative input",
    q: "Do I pay an international transaction fee?", ty: "conditional", intended: "opposed",
    p: [
      { id: "p0", t: "Premium-tier cardholders pay no international transaction fee.", st: "product_terms" },
      { id: "p1", t: "All cards issued in the EU incur a 3% international fee.", st: "official_policy" },
    ],
    b: [
      { id: "b0", out: "no international transaction fee is charged", desc: "premium-tier cardholders",
        def: false, sup: "p0", cond: "the cardholder is on the premium tier",
        at: [{ n: "card_tier", k: "ordered_tier", v: ["premium"] }] },
      { id: "b1", out: "a 3% fee applies to international transactions", desc: "cards issued in the EU",
        def: false, sup: "p1", cond: "the card was issued in the EU",
        at: [{ n: "issuing_region", k: "categorical", v: ["eu"] }] },
    ],
  },
};

/** Bring-your-own presets, carried over from the original byo.js. */
export const BYO_PRESETS = {
  fee: {
    label: "Fee with a product carve-out",
    q: "Is there a monthly fee on my current account?",
    d: { out: "a monthly account fee of 12.00 is payable", desc: "all cases", attr: "", val: "" },
    e: { out: "no monthly charge applies", desc: "Graduate Scheme accounts", attr: "account_product", val: "graduate" },
  },
  visa: {
    label: "Work ban with a sponsorship exception",
    q: "Can I work while studying?",
    d: { out: "employment is not permitted during the validity of the leave", desc: "all cases", attr: "", val: "" },
    e: { out: "work is allowed up to 20 hours per week during term time",
         desc: "students sponsored by a provider with track record status", attr: "sponsor_status", val: "track_record" },
  },
  disjoint: {
    label: "Two rules that never meet",
    q: "What notice period applies?",
    d: { out: "one month of notice is required", desc: "fixed-term contracts", attr: "contract_type", val: "fixed_term" },
    e: { out: "three months of notice is required", desc: "permanent contracts", attr: "contract_type", val: "permanent" },
  },
} as const;
