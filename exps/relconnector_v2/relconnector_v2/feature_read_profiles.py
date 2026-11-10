"""Frozen phase-one SQL read settings for the ten RelBench datasets."""

from __future__ import annotations

from typing import NamedTuple


class FeatureReadProfile(NamedTuple):
    window_batches: int
    density_threshold: float


# Selected by the CPU grid (seeds 42 and 43), then checked on seed 45.
TUNED_FEATURE_READ_PROFILES: dict[str, FeatureReadProfile] = {
    "rel-amazon": FeatureReadProfile(32, 0.6),
    "rel-arxiv": FeatureReadProfile(32, 0.6),
    "rel-avito": FeatureReadProfile(32, 0.4),
    "rel-event": FeatureReadProfile(16, 1.0),
    "rel-f1": FeatureReadProfile(32, 0.6),
    "rel-hm": FeatureReadProfile(32, 0.4),
    "rel-ratebeer": FeatureReadProfile(32, 1.0),
    "rel-salt": FeatureReadProfile(32, 0.8),
    "rel-stack": FeatureReadProfile(32, 0.8),
    "rel-trial": FeatureReadProfile(32, 0.6),
}


def tuned_feature_read_profile(dataset: str) -> FeatureReadProfile:
    try:
        return TUNED_FEATURE_READ_PROFILES[dataset]
    except KeyError as error:
        raise ValueError(f"No tuned feature-read profile for {dataset!r}") from error
