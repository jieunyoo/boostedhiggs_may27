import warnings

import awkward as ak
import numpy as np
from coffea import processor

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class LumiProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        output_location="./outfiles/",
    ):
        self._year = year

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""
        dataset = events.metadata["dataset"]
        # isMC = hasattr(events, "genWeight")
        # nevents = len(events)

        # lumilist = coffea.lumi_tools.LumiList(events.run.to_numpy(), events.luminosityBlock.to_numpy())
        lumilist = set(zip(events.run, events.luminosityBlock))

        # TODO: if possible, get lumi value per file and accumulate
        # return dictionary with cutflows
        return {dataset: {self._year: {"lumilist": lumilist}}}

    def postprocess(self, accumulator):
        return accumulator
