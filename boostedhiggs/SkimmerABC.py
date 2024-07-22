"""
Skimmer Base Class - common functions for all skimmers.
Author(s): Raghav Kansal
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from coffea import processor

#from HHbbVV.hh_vars import LUMI

from . import corrections

logging.basicConfig(level=logging.INFO)



def pileup_cutoff(self, events, year, cutoff: float = 4):
    pweights = corrections.get_pileup_weight_raghav(year, events.Pileup.nPU.to_numpy())
    pw_pass = (
            (pweights["nominal"] <= cutoff)
            * (pweights["up"] <= cutoff)
            * (pweights["down"] <= cutoff)
        )
    logging.info(f"Passing pileup weight cut: {np.sum(pw_pass)} out of {len(events)} events")
    events = events[pw_pass]
    return events

