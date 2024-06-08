import importlib.resources
import json
import logging
import os
import pathlib
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate

logger = logging.getLogger(__name__)

from boostedhiggs.corrections import (
    add_HiggsEW_kFactors,
    add_lepton_weight,
    add_pileup_weight,
    add_pileupid_weights,
    add_ps_weight,
    add_VJets_kFactors,
    btagWPs,
    corrected_msoftdrop,
    get_btag_weights,
    get_jec_jets,
    get_jmsr,
    getJECVariables,
    getJMSRVariables,
    met_factory,
)
from boostedhiggs.utils import VScore, match_H, match_Top, match_V, sigs

from .run_tagger_inference import runInferenceTriton

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")


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


class HwwProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        systematics=False,
        getLPweights=False,
        uselooselep=False,
        fakevalidation=False,
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._systematics = systematics
        self._getLPweights = getLPweights
        self._uselooselep = uselooselep
        self._fakevalidation = fakevalidation

        self._output_location = output_location

        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        if self._year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }

        self.jecs = {
            "JES": "JES_jes",
            "JER": "JER",
            # individual sources
            "JES_FlavorQCD": "JES_FlavorQCD",
            "JES_RelativeBal": "JES_RelativeBal",
            "JES_HF": "JES_HF",
            "JES_BBEC1": "JES_BBEC1",
            "JES_EC2": "JES_EC2",
            "JES_Absolute": "JES_Absolute",
            f"JES_BBEC1_{self._year}": f"JES_BBEC1_{self._year}",
            f"JES_RelativeSample_{self._year}": f"JES_RelativeSample_{self._year}",
            f"JES_EC2_{self._year}": f"JES_EC2_{self._year}",
            f"JES_HF_{self._year}": f"JES_HF_{self._year}",
            f"JES_Absolute_{self._year}": f"JES_Absolute_{self._year}",
            "JES_Total": "JES_Total",
        }

        # for tagger inference
        self._inference = inference
        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = self._channels if channel == "all" else [channel]

        for ch in channels:
            if ch not in self._channels:
                logger.warning(f"Attempted to add selection to unexpected channel: {ch} not in %s" % (self._channels))
                continue

            # add selection
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            if self.isMC:
                weight = self.weights[ch].partial_weight(["genweight"])
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]
        self.isMC = hasattr(events, "genWeight")

        nevents = len(events)
        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents

        # sum LHE weight
        sumlheweight = {}
        if "LHEScaleWeight" in events.fields and self.isMC:
            if len(events.LHEScaleWeight[0]) == 9:
                for i in range(len(events.LHEScaleWeight[0])):
                    sumlheweight[i] = ak.sum(events.LHEScaleWeight[:, i] * events.genWeight)

        # sum PDF weight
        sumpdfweight = {}
        if "LHEPdfWeight" in events.fields and self.isMC and "HToWW" in dataset:
            for i in range(len(events.LHEPdfWeight[0])):
                sumpdfweight[i] = ak.sum(events.LHEPdfWeight[:, i] * events.genWeight)

        # add genweight before filling cutflow
        if self.isMC:
            for ch in self._channels:
                self.weights[ch].add("genweight", events.genWeight)

        ######################
        # Trigger
        ######################

        trigger = {}
        for ch in ["ele", "mu_lowpt", "mu_highpt"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]

        trigger["ele"] = trigger["ele"] & (~trigger["mu_lowpt"]) & (~trigger["mu_highpt"])
        trigger["mu_highpt"] = trigger["mu_highpt"] & (~trigger["ele"])
        trigger["mu_lowpt"] = trigger["mu_lowpt"] & (~trigger["ele"])

        ######################
        # METFLITERS
        ######################

        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        ######################
        # OBJECT DEFINITION
        ######################

        # OBJECT: taus
        loose_taus_mu = (events.Tau.pt > 20) & (abs(events.Tau.eta) < 2.3) & (events.Tau.idAntiMu >= 1)  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiEleDeadECal >= 2)  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        # OBJECT: muons
        muons = ak.with_field(events.Muon, 0, "flavor")

        loose_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
            & (((muons.pfRelIso04_all < 0.25) & (muons.pt < 55)) | (muons.pt >= 55))
        )

        tight_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )

        n_loose_muons = ak.sum(loose_muons, axis=1)
        n_tight_muons = ak.sum(tight_muons, axis=1)

        if self._uselooselep:
            good_muons = loose_muons
        else:
            good_muons = tight_muons

        n_good_muons = ak.sum(good_muons, axis=1)

        # OBJECT: electrons
        electrons = ak.with_field(events.Electron, 1, "flavor")

        loose_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WPL)
            & (((electrons.pfRelIso03_all < 0.25) & (electrons.pt < 120)) | (electrons.pt >= 120))
        )

        tight_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WP90)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
            # additional cuts
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
        )

        n_loose_electrons = ak.sum(loose_electrons, axis=1)
        n_tight_electrons = ak.sum(tight_electrons, axis=1)

        if self._uselooselep:
            good_electrons = loose_electrons
        else:
            good_electrons = tight_electrons

        n_good_electrons = ak.sum(good_electrons, axis=1)

        # OBJECT: candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        good_fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True
        )

        # OBJECT: candidate fatjet
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        jmsr_shifted_fatjetvars = get_jmsr(good_fatjets[fj_idx_lep], num_jets=1, year=self._year, isData=not self.isMC)

        # VH jet
        minDeltaR = ak.argmin(candidatelep_p4.delta_r(good_fatjets), axis=1)  # similar to fj_idx_lep but without keepdims
        fatJetIndices = ak.local_index(good_fatjets, axis=1)
        mask_candidatefj = fatJetIndices != minDeltaR

        allScores = VScore(good_fatjets)
        masked = allScores[mask_candidatefj]
        VH_fj = ak.firsts(good_fatjets[allScores == ak.max(masked, axis=1)])

        # OBJECT: AK4 jets
        jets, jec_shifted_jetvars = get_jec_jets(events, events.Jet, self._year, not self.isMC, self.jecs, fatjets=False)
        met = met_factory.build(events.MET, jets, {}) if self.isMC else events.MET

        ht = ak.sum(jets.pt, axis=1)

        jet_selector = (
            (jets.pt > 30)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
        )
        goodjets = jets[jet_selector]
        ak4_outside_ak8_selector = jets.delta_r(candidatefj) > 0.8
        ak4_outside_ak8 = jets[ak4_outside_ak8_selector]

        # OBJECT: VBF variables
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass

        # OBJECT: b-jets (only for jets with abs(eta)<2.5)
        bjet_selector = (jet_selector) & (jets.delta_r(candidatefj) > 0.8) & (abs(jets.eta) < 2.5)
        ak4_bjet_candidate = jets[bjet_selector]

        NumFatjets = ak.num(good_fatjets)
        FirstFatjet = ak.firsts(good_fatjets[:, 0:1])
        SecondFatjet = ak.firsts(good_fatjets[:, 1:2])
        NumOtherJets = ak.num(ak4_outside_ak8)

        n_bjets_L = ak.sum(
            ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["L"],
            axis=1,
        )
        n_bjets_M = ak.sum(
            ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"],
            axis=1,
        )
        n_bjets_T = ak.sum(
            ak4_bjet_candidate.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"],
            axis=1,
        )

        # delta R between AK8 jet and lepton
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        mt_lep_met = np.sqrt(
            2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # delta phi MET and higgs candidate
        met_fj_dphi = candidatefj.delta_phi(met)

        ######################
        # Store variables
        ######################

        variables = {
            # candidatefj
            "fj_lsf3": candidatefj.lsf3,
            "fj_VScore": VScore(candidatefj),
            # lepton
            "lep_pt": candidatelep.pt,
            "lep_eta": candidatelep.eta,
            # others
            "lep_isolation": lep_reliso,
            "lep_misolation": lep_miso,
            "lep_fj_dr": lep_fj_dr,
            "lep_met_mt": mt_lep_met,
            "met_fj_dphi": met_fj_dphi,
            "met_pt": met.pt,
            "deta": deta,
            "mjj": mjj,
            "ht": ht,
            # bjets
            "n_bjets_L": n_bjets_L,
            "n_bjets_M": n_bjets_M,
            "n_bjets_T": n_bjets_T,
            "NumFatjets": NumFatjets,
            "NumOtherJets": NumOtherJets,
            # leading fatjet
            "FirstFatjet_pt": FirstFatjet.pt,
            "FirstFatjet_eta": FirstFatjet.eta,
            "FirstFatjet_phi": FirstFatjet.phi,
            "FirstFatjet_msd": FirstFatjet.msdcorr,
            "FirstFatjet_Vscore": VScore(SecondFatjet),
            # second leading fatjet
            "SecondFatjet_pt": SecondFatjet.pt,
            "SecondFatjet_eta": SecondFatjet.eta,
            "SecondFatjet_phi": SecondFatjet.phi,
            "SecondFatjet_msd": SecondFatjet.msdcorr,
            "SecondFatjet_Vscore": VScore(FirstFatjet),
            # number
            "n_loose_electrons": n_loose_electrons,
            "n_tight_electrons": n_tight_electrons,
            "n_loose_muons": n_loose_muons,
            "n_tight_muons": n_tight_muons,
            # second fatjet after candidate jet
            "VH_fj_pt": VH_fj.pt,
            "VH_fj_eta": VH_fj.eta,
            "VH_fj_VScore": VScore(VH_fj),
        }

        fatjetvars = {
            "fj_pt": candidatefj.pt,
            "fj_eta": candidatefj.eta,
            "fj_phi": candidatefj.phi,
            "fj_mass": candidatefj.msdcorr,
        }

        variables = {**variables, **fatjetvars}

        if self._systematics and self.isMC:
            fatjetvars_sys = {}
            # JEC vars
            for shift, vals in jec_shifted_fatjetvars["pt"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_pt{shift}"] = ak.firsts(vals[fj_idx_lep])

            # JMSR vars
            for shift, vals in jmsr_shifted_fatjetvars["msoftdrop"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_mass{shift}"] = ak.firsts(vals)

            variables = {**variables, **fatjetvars_sys}
            fatjetvars = {**fatjetvars, **fatjetvars_sys}

            # add variables affected by JECs/MET
            mjj_shift = {}
            for shift, vals in jec_shifted_jetvars["pt"].items():
                if shift != "":
                    pt_1 = jet1.pt
                    pt_2 = jet2.pt
                    try:
                        pt_1 = vals[ak4_outside_ak8_selector][:, 0]
                    except Exception:
                        pt_1 = jet1.pt
                    try:
                        pt_2 = vals[ak4_outside_ak8_selector][:, 1]
                    except Exception:
                        pt_2 = jet2.pt

                    jet1_shift = ak.zip(
                        {
                            "pt": pt_1,
                            "eta": jet1.eta,
                            "phi": jet1.phi,
                            "mass": jet1.mass,
                            "charge": 0,
                        },
                        with_name="PtEtaPhiMCandidate",
                        behavior=candidate.behavior,
                    )
                    jet2_shift = ak.zip(
                        {
                            "pt": pt_2,
                            "eta": jet2.eta,
                            "phi": jet2.phi,
                            "mass": jet2.mass,
                            "charge": 0,
                        },
                        with_name="PtEtaPhiMCandidate",
                        behavior=candidate.behavior,
                    )
                    mjj_shift[f"mjj{shift}"] = (ak.firsts(jet1_shift) + ak.firsts(jet2_shift)).mass
            variables = {**variables, **mjj_shift}

            for met_shift in ["UES_up", "UES_down"]:
                jecvariables = getJECVariables(fatjetvars, candidatelep_p4, met, pt_shift=None, met_shift=met_shift)
                variables = {**variables, **jecvariables}

        for shift in jec_shifted_fatjetvars["pt"]:
            if shift != "" and not self._systematics:
                continue
            jecvariables = getJECVariables(fatjetvars, candidatelep_p4, met, pt_shift=shift, met_shift=None)
            variables = {**variables, **jecvariables}

        for shift in jmsr_shifted_fatjetvars["msoftdrop"]:
            if shift != "" and not self._systematics:
                continue
            jmsrvariables = getJMSRVariables(fatjetvars, candidatelep_p4, met, mass_shift=shift)
            variables = {**variables, **jmsrvariables}

        ######################
        # Selection
        ######################

        for ch in self._channels:
            # trigger
            if ch == "mu":
                self.add_selection(
                    name="Trigger",
                    sel=((candidatelep.pt < 55) & trigger["mu_lowpt"]) | ((candidatelep.pt >= 55) & trigger["mu_highpt"]),
                    channel=ch,
                )
            else:
                self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)

        self.add_selection(name="METFilters", sel=metfilters)
        self.add_selection(name="OneLep", sel=(n_good_muons == 1) & (n_loose_electrons == 0), channel="mu")
        self.add_selection(name="OneLep", sel=(n_loose_muons == 0) & (n_good_electrons == 1), channel="ele")
        self.add_selection(name="NoTaus", sel=(n_loose_taus_mu == 0), channel="mu")
        self.add_selection(name="NoTaus", sel=(n_loose_taus_ele == 0), channel="ele")
        self.add_selection(name="AtLeastOneFatJet", sel=(NumFatjets >= 1))

        fj_pt_sel = candidatefj.pt > 250
        if self.isMC:  # make an OR of all the JECs
            for k, v in self.jecs.items():
                for var in ["up", "down"]:
                    fj_pt_sel = fj_pt_sel | (candidatefj[v][var].pt > 250)
        self.add_selection(name="CandidateJetpT", sel=(fj_pt_sel == 1))

        self.add_selection(name="LepInJet", sel=(lep_fj_dr < 0.8))
        self.add_selection(name="JetLepOverlap", sel=(lep_fj_dr > 0.03))
        self.add_selection(name="dPhiJetMET", sel=(np.abs(met_fj_dphi) < 1.57))

        if self._fakevalidation:
            self.add_selection(name="MET", sel=(met.pt < 20))
        else:
            self.add_selection(name="MET", sel=(met.pt > 20))

        # gen-level matching
        signal_mask = None
        if self.isMC:
            if ("HToWW" in dataset) or ("HWW" in dataset) or ("ttHToNonbb" in dataset):
                genVars, signal_mask = match_H(events.GenPart, candidatefj, fatjet_pt=FirstFatjet)
                # add signal mask and modify sum of genweights to only consider those events that pass the mask
                self.add_selection(name="Signal", sel=signal_mask)
            elif "HToTauTau" in dataset:
                genVars, signal_mask = match_H(events.GenPart, candidatefj, dau_pdgid=15)
                self.add_selection(name="Signal", sel=signal_mask)
            elif ("WJets" in dataset) or ("ZJets" in dataset) or ("DYJets" in dataset):
                genVars, _ = match_V(events.GenPart, candidatefj)
                genVars["LHE_HT"] = events.LHE.HT
                genVars["LHE_Vpt"] = events.LHE.Vpt
            elif "TT" in dataset:
                genVars, _ = match_Top(events.GenPart, candidatefj)
            else:
                genVars = {}
            # save gen jet mass (not msd)
            genVars["fj_genjetmass"] = candidatefj.matched_gen.mass
            genVars["fj_genjetpt"] = candidatefj.matched_gen.pt
            variables = {**variables, **genVars}

        # hem-cleaning selection
        if self._year == "2018":
            hem_veto = ak.any(
                ((goodjets.eta > -3.2) & (goodjets.eta < -1.3) & (goodjets.phi > -1.57) & (goodjets.phi < -0.87)),
                -1,
            ) | ak.any(
                (
                    (electrons.pt > 30)
                    & (electrons.eta > -3.2)
                    & (electrons.eta < -1.3)
                    & (electrons.phi > -1.57)
                    & (electrons.phi < -0.87)
                ),
                -1,
            )

            hem_cleaning = (
                ((events.run >= 319077) & (not self.isMC))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (hem_veto)

            self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

        if self.isMC:
            for ch in self._channels:
                if self._year in ("2016", "2017"):
                    self.weights[ch].add(
                        "L1Prefiring",
                        events.L1PreFiringWeight.Nom,
                        events.L1PreFiringWeight.Up,
                        events.L1PreFiringWeight.Dn,
                    )
                add_pileup_weight(
                    self.weights[ch],
                    self._year,
                    self._yearmod,
                    nPU=ak.to_numpy(events.Pileup.nPU),
                )

                add_pileupid_weights(self.weights[ch], self._year, self._yearmod, goodjets, events.GenJet, wp="L")

                if ch == "mu":
                    add_lepton_weight(self.weights[ch], candidatelep, self._year + self._yearmod, "muon")
                elif ch == "ele":
                    add_lepton_weight(self.weights[ch], candidatelep, self._year + self._yearmod, "electron")

                ewk_corr, qcd_corr, alt_qcd_corr = add_VJets_kFactors(self.weights[ch], events.GenPart, dataset, events)
                # add corrections for plotting
                variables["weight_ewkcorr"] = ewk_corr
                variables["weight_qcdcorr"] = qcd_corr
                variables["weight_altqcdcorr"] = alt_qcd_corr

                if "HToWW" in dataset:
                    add_HiggsEW_kFactors(self.weights[ch], events.GenPart, dataset)

                if "HToWW" in dataset or "TT" in dataset or "WJets" in dataset:
                    """
                    For the QCD acceptance uncertainty:
                    - we save the individual weights [0, 1, 3, 5, 7, 8]
                    - postprocessing: we obtain sum_sumlheweight
                    - postprocessing: we obtain LHEScaleSumw: sum_sumlheweight[i] / sum_sumgenweight
                    - postprocessing:
                      obtain histograms for 0, 1, 3, 5, 7, 8 and 4: h0, h1, ... respectively
                       weighted by scale_0, scale_1, etc
                      and normalize them by  (xsec * luminosity) / LHEScaleSumw[i]
                    - then, take max/min of h0, h1, h3, h5, h7, h8 w.r.t h4: h_up and h_dn
                    - the uncertainty is the nominal histogram * h_up / h4
                    """
                    scale_weights = {}
                    if "LHEScaleWeight" in events.fields:
                        # save individual weights
                        if len(events.LHEScaleWeight[0]) == 9:
                            for i in [0, 1, 3, 5, 7, 8, 4]:
                                scale_weights[f"weight_scale{i}"] = events.LHEScaleWeight[:, i]
                    variables = {**variables, **scale_weights}

                if "HToWW" in dataset:
                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values
                    - The last two values: alpha_s variations.
                    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
                    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
                    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
                    """
                    pdf_weights = {}
                    if "LHEPdfWeight" in events.fields:
                        # save individual weights
                        for i in range(len(events.LHEPdfWeight[0])):
                            pdf_weights[f"weight_pdf{i}"] = events.LHEPdfWeight[:, i]
                    variables = {**variables, **pdf_weights}

                if "HToWW" in dataset:
                    add_ps_weight(
                        self.weights[ch],
                        events.PSWeight if "PSWeight" in events.fields else [],
                    )

                # store the final weight per ch
                variables[f"weight_{ch}"] = self.weights[ch].weight()
                if self._systematics:
                    for systematic in self.weights[ch].variations:
                        variables[f"weight_{ch}_{systematic}"] = self.weights[ch].weight(modifier=systematic)

                # store b-tag weight
                for wp_ in ["T"]:
                    variables = {
                        **variables,
                        **get_btag_weights(
                            self._year,
                            events.Jet,
                            bjet_selector,
                            wp=wp_,
                            algo="deepJet",
                            systematics=self._systematics,
                        ),
                    }

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {}
                for var, item in variables.items():
                    # pad all the variables that are not a cut with -1
                    # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    # fill out dictionary
                    out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}

                if self._getLPweights:
                    from boostedhiggs.corrections import getLPweights

                    (pf_cands, gen_parts_eta_phi, ak8_jets) = getLPweights(
                        dataset,
                        events[selection_ch],
                        candidatefj[selection_ch],
                        fj_idx_lep[selection_ch],
                        candidatelep_p4[selection_ch],
                    )

                    lpvars = {}
                    for pfcandidx in range(pf_cands.shape[1]):
                        lpvars[f"LP_pfcand{pfcandidx}_px"] = pf_cands[:, pfcandidx, 0]
                        lpvars[f"LP_pfcand{pfcandidx}_py"] = pf_cands[:, pfcandidx, 1]
                        lpvars[f"LP_pfcand{pfcandidx}_pz"] = pf_cands[:, pfcandidx, 2]
                        lpvars[f"LP_pfcand{pfcandidx}_energy"] = pf_cands[:, pfcandidx, 3]

                    for quarkidx in range(gen_parts_eta_phi.shape[1]):
                        lpvars[f"LP_quark{quarkidx}_eta"] = gen_parts_eta_phi[:, quarkidx, 0]
                        lpvars[f"LP_quark{quarkidx}_phi"] = gen_parts_eta_phi[:, quarkidx, 1]

                    lpvars["LP_fj_pt"] = ak8_jets[:, 0]
                    lpvars["LP_fj_eta"] = ak8_jets[:, 1]
                    lpvars["LP_fj_phi"] = ak8_jets[:, 2]
                    lpvars["LP_fj_energy"] = ak8_jets[:, 3]

                    output[ch] = {**output[ch], **lpvars}

                # fill inference
                if self._inference:
                    for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
                        pnet_vars = runInferenceTriton(
                            self.tagger_resources_path,
                            events[selection_ch],
                            fj_idx_lep[selection_ch],
                            model_name=model_name,
                        )
                        pnet_df = self.ak_to_pandas(pnet_vars)
                        scores = {"fj_ParT_score": pnet_df[sigs].sum(axis=1).values}

                        hidNeurons = {}
                        for key in pnet_vars:
                            if "hidNeuron" in key:
                                hidNeurons[key] = pnet_vars[key]

                        reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}
                        output[ch] = {**output[ch], **scores, **reg_mass, **hidNeurons}

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

            for var_ in [
                "rec_higgs_m",
                "rec_higgs_pt",
                "rec_W_qq_m",
                "rec_W_qq_pt",
                "rec_W_lnu_m",
                "rec_W_lnu_pt",
            ]:
                if var_ in output[ch].keys():
                    output[ch][var_] = np.nan_to_num(output[ch][var_], nan=-1)

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year
                + self._yearmod: {
                    "sumgenweight": sumgenweight,
                    "sumlheweight": sumlheweight,
                    "sumpdfweight": sumpdfweight,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
