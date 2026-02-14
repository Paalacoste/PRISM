"""Configurable perturbation schedules for each experiment.

References:
    master.md section 6 (experiment protocols)
"""

from dataclasses import dataclass


@dataclass
class PerturbationEvent:
    """A single scheduled perturbation."""
    episode: int
    ptype: str
    kwargs: dict


class PerturbationSchedule:
    """Ordered sequence of perturbation events."""

    def __init__(self, events: list[PerturbationEvent]):
        raise NotImplementedError("Phase 1")

    @classmethod
    def exp_a(cls) -> "PerturbationSchedule":
        """Schedule for Experiment A (calibration)."""
        raise NotImplementedError("Phase 2")

    @classmethod
    def exp_c(cls) -> "PerturbationSchedule":
        """Schedule for Experiment C (adaptation)."""
        raise NotImplementedError("Phase 3")
