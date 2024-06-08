import warnings

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, column_accumulator

from boostedhiggs.corrections import (
    add_lepton_weight,
    add_pileup_weight,
    add_VJets_kFactors,
)
from boostedhiggs.utils import match_H

# from boostedhiggs.utils import getParticles

# we suppress ROOT warnings where our input ROOT tree has duplicate branches - these are handled correctly.
warnings.filterwarnings("ignore", message="Found duplicate branch ")


def getParticles(genparticles, lowid=22, highid=25, flags=["fromHardProcess", "isLastCopy"]):
    """
    returns the particle objects that satisfy a low id,
    high id condition and have certain flags
    """
    absid = abs(genparticles.pdgId)
    return genparticles[((absid >= lowid) & (absid <= highid)) & genparticles.hasFlags(flags)]


# def simple_match_HWW(genparticles, candidatefj):
#     """
#     return the number of matched objects (hWW*),daughters,
#     and gen flavor (enuqq, munuqq, taunuqq)
#     """
#     higgs = getParticles(genparticles, 25)  # genparticles is the full set... this function selects Higgs particles
#     # W~24 so we get H->WW (limitation: only picking one W and assumes the other will be there)
#     is_hWW = ak.all(abs(higgs.children.pdgId) == 24, axis=2)

#     higgs = higgs[is_hWW]

#     matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)  # choose higgs closest to fj

#     return matchedH


def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


class TriggerEfficienciesProcessor(ProcessorABC):
    """Accumulates yields from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self, year="2017"):
        super(TriggerEfficienciesProcessor, self).__init__()
        self._year = year
        self._trigger_dict = {
            "2017": {
                "ele35": ["Ele35_WPTight_Gsf"],
                "ele115": ["Ele115_CaloIdVT_GsfTrkIdT"],
                "Photon200": ["Photon200"],
                "Mu50": ["Mu50"],
                "IsoMu27": ["IsoMu27"],
                "OldMu100": ["OldMu100"],
                "TkMu100": ["TkMu100"],
            }
        }[self._year]
        self._triggers = {
            "ele": ["ele35", "ele115", "Photon200"],
            "mu": ["Mu50", "IsoMu27", "OldMu100", "TkMu100"],
        }

        self._channels = ["ele", "mu"]

    def pad_val(
        self,
        arr: ak.Array,
        target: int,
        value: float,
        axis: int = 0,
        to_numpy: bool = True,
    ):
        """pads awkward array up to `target` index along axis `axis` with value `value`,
        optionally converts to numpy array"""
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=True), value)
        return ret.to_numpy() if to_numpy else ret

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger histograms from input NanoAOD events"""
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.weights = Weights(nevents, storeIndividual=True)
        self.weights_per_ch = {}

        def pad_val_nevents(arr: ak.Array):
            """pad values with the length equal to the number of events"""
            return self.pad_val(arr, nevents, -1)

        # skimmed events for different channels
        out = {}
        for channel in self._channels:
            out[channel] = {}
            out[channel]["triggers"] = {}

        """ Save OR of triggers as booleans """
        for channel in self._channels:
            HLT_triggers = {}
            for t in self._triggers[channel]:
                HLT_triggers["HLT_" + t] = np.any(
                    np.array([events.HLT[trigger] for trigger in self._trigger_dict[t] if trigger in events.HLT.fields]),
                    axis=0,
                )
            out[channel]["triggers"] = {**out[channel]["triggers"], **HLT_triggers}

        """ basic definitions """
        # DEFINE MUONS
        loose_muons = (
            (((events.Muon.pt > 30) & (events.Muon.pfRelIso04_all < 0.25)) | (events.Muon.pt > 55))
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (events.Muon.pt > 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (np.abs(events.Muon.dz) < 0.1)
            & (np.abs(events.Muon.dxy) < 0.05)
            & (events.Muon.sip3d <= 4.0)
            & events.Muon.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # DEFINE ELECTRONS
        loose_electrons = (
            (((events.Electron.pt > 38) & (events.Electron.pfRelIso03_all < 0.25)) | (events.Electron.pt > 120))
            & (np.abs(events.Electron.eta) < 2.4)
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            (events.Electron.pt > 38)
            & (np.abs(events.Electron.eta) < 2.4)
            & ((np.abs(events.Electron.eta) < 1.44) | (np.abs(events.Electron.eta) > 1.57))
            & (np.abs(events.Electron.dz) < 0.1)
            & (np.abs(events.Electron.dxy) < 0.05)
            & (events.Electron.sip3d <= 4.0)
            & (events.Electron.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # get candidate lepton
        goodleptons = ak.concatenate(
            [events.Muon[good_muons], events.Electron[good_electrons]], axis=1
        )  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt
        candidatelep = ak.firsts(goodleptons)  # pick highest pt

        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        # DEFINE JETS
        goodjets = events.Jet[
            (events.Jet.pt > 30) & (abs(events.Jet.eta) < 5.0) & events.Jet.isTight & (events.Jet.puId > 0)
        ]
        # reject EE noisy jets for 2017
        if self._year == "2017":
            goodjets = goodjets[(goodjets.pt > 50) | (abs(goodjets.eta) < 2.65) | (abs(goodjets.eta) > 3.139)]

        # fatjets
        fatjets = events.FatJet

        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        # n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        # for lep channel: first clean jets and leptons by removing overlap, then pick candidate_fj closest to the lepton
        lep_in_fj_overlap_bool = good_fatjets.delta_r(candidatelep_p4) > 0.1
        good_fatjets = good_fatjets[lep_in_fj_overlap_bool]
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        """ Baseline weight """
        self.weights.add("genweight", events.genWeight)
        self.weights.add(
            "L1Prefiring",
            events.L1PreFiringWeight.Nom,
            events.L1PreFiringWeight.Up,
            events.L1PreFiringWeight.Dn,
        )
        add_pileup_weight(self.weights, self._year, "", nPU=ak.to_numpy(events.Pileup.nPU))
        add_VJets_kFactors(self.weights, events.GenPart, dataset)

        # """ Baseline selection """
        # # define selections for different channels
        for channel in self._channels:
            selection = PackedSelection()
            if channel == "mu":
                add_lepton_weight(self.weights, candidatelep, self._year, "muon")
                selection.add(
                    "onemuon",
                    (
                        (n_good_muons == 1)
                        & (n_good_electrons == 0)
                        & (n_loose_electrons == 0)
                        & ~ak.any(loose_muons & ~good_muons, 1)
                    ),
                )
                selection.add("muonkin", (candidatelep.pt > 30))
            elif channel == "ele":
                add_lepton_weight(self.weights, candidatelep, self._year, "electron")
                selection.add(
                    "oneelectron",
                    (
                        (n_good_muons == 0)
                        & (n_loose_muons == 0)
                        & (n_good_electrons == 1)
                        & ~ak.any(loose_electrons & ~good_electrons, 1)
                    ),
                )
                selection.add("electronkin", (candidatelep.pt > 40))

            """Define other variables to save"""
            out[channel]["vars"] = {}
            out[channel]["vars"]["fj_pt"] = pad_val_nevents(candidatefj.pt)
            out[channel]["vars"]["fj_msoftdrop"] = pad_val_nevents(candidatefj.msoftdrop)
            out[channel]["vars"]["lep_pt"] = pad_val_nevents(candidatelep.pt)

            if "HToWW" in dataset:
                genVars, _ = match_H(events.GenPart, candidatefj)
                matchedH_pt = genVars["fj_genH_pt"]
            else:
                matchedH_pt = ak.zeros_like(candidatefj.pt)
            out[channel]["vars"]["fj_genH_pt"] = pad_val_nevents(matchedH_pt)

            out[channel]["weights"] = {}
            for key in self.weights._weights.keys():
                # store the individual weights (ONLY for now until we debug)
                out[channel]["weights"][f"weight_{key}"] = self.weights.partial_weight([key])
                if channel in self.weights_per_ch.keys():
                    self.weights_per_ch[channel].append(key)

            # use column accumulators
            for key_ in out[channel].keys():
                out[channel][key_] = {
                    key: column_accumulator(value[selection.all(*selection.names)])
                    for (key, value) in out[channel][key_].items()
                }

        return {self._year: {dataset: {"nevents": nevents, "skimmed_events": out}}}

    def postprocess(self, accumulator):
        for year, datasets in accumulator.items():
            for dataset, output in datasets.items():
                for channel in output["skimmed_events"].keys():
                    for key_ in output["skimmed_events"][channel].keys():
                        output["skimmed_events"][channel][key_] = {
                            key: value.value for (key, value) in output["skimmed_events"][channel][key_].items()
                        }

        return accumulator
