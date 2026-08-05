"""External FairChem / FXPU monkeypatches (do not edit sqs_sampling)."""

from patches.phase1_force_correctness import (
    apply,
    apply_all_phase1_patches,
    apply_edge_degree_embedding_ckpt_seq,
    apply_edgewise_ckpt_clone,
)
from patches import phase2_xccl
from patches import phase3_launch
from patches import phase4_collectives

__all__ = [
    "apply",
    "apply_all_phase1_patches",
    "apply_edge_degree_embedding_ckpt_seq",
    "apply_edgewise_ckpt_clone",
    "phase2_xccl",
    "phase3_launch",
    "phase4_collectives",
]
