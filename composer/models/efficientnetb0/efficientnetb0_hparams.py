# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp

from composer.models import EfficientNetB0
from composer.models.model_hparams import ModelHparams


@dataclass
class EfficientNetB0Hparams(ModelHparams):
    drop_connect_rate: float = hp.optional(
        doc="Probability of dropping a sample within a block before identity connection.",
        default=0.2,
    )

    def initialize_object(self):
        return EfficientNetB0(num_classes=self.num_classes, drop_connect_rate=self.drop_connect_rate)
