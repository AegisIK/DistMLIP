"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    UMATask,
)

from fairchem.core import FAIRChemCalculator
from DistMLIP.implementations.uma.escn_md import from_existing

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.units.mlip_unit import MLIPPredictUnit


class FAIRChemCalculator_Dist(FAIRChemCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_existing(cls, existing):
        # Call from_existing on underlying predictor
        dist_calc = cls.__new__(cls)
        dist_calc.__dict__ = existing.__dict__.copy()
        
        dist_calc.dist_enabled = False
        dist_calc.predictor.model.module.backbone = from_existing(existing.predictor.model.module.backbone)
        return dist_calc

    def enable_distributed_mode(self, gpus):
        self.predictor.model.module.backbone.enable_distributed_mode(gpus)
        self.dist_enabled = True
