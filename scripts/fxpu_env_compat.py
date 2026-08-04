#!/usr/bin/env python3
"""Map legacy HEN_* FairChem/UMA env vars to FXPU_* (one-shot, process-local).

HEN = High Entropy Nitride (materials project). FairChem/UMA infra uses FXPU_*.
Call early in scripts if old job scripts still export HEN_*.
"""
from __future__ import annotations

import os

_PREFIX = "HEN_"
_NEW = "FXPU_"
# Path to the nitride workspace — not FairChem infra branding.
_SPECIAL = {
    "HEN_ROOT": "PROJECT_ROOT",
    "HEN_XPU": "FXPU_PREFIX",
    "HEN_XPU_ENV": "FXPU_ENV",
}


def apply_hen_to_fxpu_env_compat(*, warn: bool = False) -> list[str]:
    mapped: list[str] = []
    for old, new in _SPECIAL.items():
        if old in os.environ and new not in os.environ:
            os.environ[new] = os.environ[old]
            mapped.append(f"{old}->{new}")
    for key, val in list(os.environ.items()):
        if not key.startswith(_PREFIX):
            continue
        if key in _SPECIAL:
            continue
        new = _NEW + key[len(_PREFIX) :]
        if new not in os.environ:
            os.environ[new] = val
            mapped.append(f"{key}->{new}")
    if warn and mapped:
        print(
            "[fxpu] mapped legacy HEN_* env vars:",
            ", ".join(mapped[:12]),
            ("..." if len(mapped) > 12 else ""),
            flush=True,
        )
    return mapped


if __name__ == "__main__":
    apply_hen_to_fxpu_env_compat(warn=True)
