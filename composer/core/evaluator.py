# Copyright 2021 MosaicML. All Rights Reserved.
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from torchmetrics import Metric, MetricCollection

if TYPE_CHECKING:
    from composer.core.types import DataLoader, Metrics


class Evaluator:
    """Wrapper for a dataloader to include metrics that apply to a specific dataset.

    Attributes:
        label (str): Name of the Evaluator
        dataloader (DataLoader): Dataloader for evaluation data
        metrics (Metrics): Metrics to log. The metrics will be deep-copied to ensure that
            each evaluator updates only its metrics.
    """

    def __init__(self, *, label: str, dataloader: DataLoader, metrics: Metrics):
        self.label = label
        self.dataloader = dataloader

        # Forcing metrics to be a MetricCollection simplifies logging results
        metrics = copy.deepcopy(metrics)
        if isinstance(metrics, Metric):
            self.metrics = MetricCollection([metrics])
        else:
            self.metrics = metrics
