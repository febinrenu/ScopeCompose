"""A2 and A3: pairwise conflict detection and five-class classification.

A2 is a two-stage detector -- a local fast filter over all C(5,2) pairs, with
selective LLM escalation for the uncertain band only. A3 classifies the
conflicting pairs into the five-class taxonomy, and does so three different
ways so the architecture choice is settled empirically rather than assumed.
"""

from detection.features import PairFeatures, extract_pair_features, more_specific
from detection.head import HeadConfig, PairHead, interaction_features
from detection.nli import CrossEncoderNLI, HeuristicNLI, NLIScores, get_nli
from detection.pipeline import DetectionStats, TwoStageDetector
from detection.stage1 import PairCandidate, Stage1Filter, enumerate_pairs, feature_names
from detection.stage2 import Stage2Judge, Stage2Judgement

__all__ = [
    "CrossEncoderNLI",
    "DetectionStats",
    "HeadConfig",
    "HeuristicNLI",
    "NLIScores",
    "PairCandidate",
    "PairFeatures",
    "PairHead",
    "Stage1Filter",
    "Stage2Judge",
    "Stage2Judgement",
    "TwoStageDetector",
    "enumerate_pairs",
    "extract_pair_features",
    "feature_names",
    "get_nli",
    "interaction_features",
    "more_specific",
]
