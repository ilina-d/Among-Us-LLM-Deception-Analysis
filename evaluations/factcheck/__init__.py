"""Fact-check what players say in Among Us games against what really happened.

Pipeline:  logs.py (load) -> truth.py (ground truth) -> claims.py (extract claims
from speeches) -> verify.py (check claims) -> analysis.py (rates, judge agreement).
"""
from .logs import Entry, load_experiment, load_log_file
from .truth import GameTruth, build_truth
from .claims import Claim, extract_claims
from .verify import verify_claim

__all__ = ["Entry", "load_experiment", "load_log_file", "GameTruth", "build_truth",
           "Claim", "extract_claims", "verify_claim"]
