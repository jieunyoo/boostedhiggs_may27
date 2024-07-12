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
    add_TopPtReweighting,
)
from boostedhiggs.utils import VScore, match_H, match_Top, match_V, sigs, get_pid_mask

from .run_tagger_inference import runInferenceTriton

from .SkimmerABC import *

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

def VScore(goodFatJetsSelected):
    num = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
    )
    den = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
        + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score



#class HwwProcessor(processor.ProcessorABC):
class vhProcessor(processor.ProcessorABC):
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
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._systematics = systematics
        self._getLPweights = getLPweights
        self._uselooselep = uselooselep
        #self._fakevalidation = fakevalidation

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

        print('events',events)

        #if self.isMC: 
         #   cutoff = 4
          #  pweights = corrections.get_pileup_weight_raghav(self._year, events.Pileup.nPU.to_numpy())
           # pw_pass = ((pweights["nominal"] <= cutoff)*(pweights["up"] <= cutoff)*(pweights["down"] <= cutoff))
           # print('pw_pass', pw_pass)
           # events = events[pw_pass]


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

        # OBJECT: muons
        muons = ak.with_field(events.Muon, 0, "flavor")

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
            # additional cuts
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.02)
        )

        n_good_muons = ak.sum(good_muons, axis=1)

        # OBJECT: electrons
        electrons = ak.with_field(events.Electron, 1, "flavor")

        good_electrons = (
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

        ngood_leptons = ak.num(goodleptons, axis=1)

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        #fatjet_selector = (fatjets.pt > 250) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt
        NumFatjets = ak.num(good_fatjets)

        #JETS**************************************************
        #this applies JEC to all the fat jets
        good_fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True
        )
        #******************************************************

        # OBJECT: candidate fatjet
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        #this applies jmsr to higgs jet - need to fix to apply to V
        #jmsr_shifted_fatjetvars = get_jmsr(good_fatjets[fj_idx_lep], num_jets=1, year=self._year, isData=not self.isMC)

        #*************************************************************************
        # VH jet   /differs from HWW processor, but Farouks added this into hww processor now
        deltaR_lepton_all_jets = candidatelep_p4.delta_r(good_fatjets)
        minDeltaR = ak.argmin(deltaR_lepton_all_jets, axis=1)
        fatJetIndices = ak.local_index(good_fatjets, axis=1)
        mask1 = fatJetIndices != minDeltaR
        allScores = VScore(good_fatjets)
        masked = allScores[mask1]
        secondFJ = good_fatjets[allScores == ak.max(masked, axis=1)]
        second_fj = ak.firsts(secondFJ)
        VCandidateVScore = VScore(second_fj)
        VCandidate_Mass = second_fj.msdcorr

        dr_two_jets = candidatefj.delta_r(second_fj)

        #only for V boson, since need up and down
        #if self.isMC:
        Vboson_Jet, jec_shifted_fatjetvars_V = get_jec_jets(events, secondFJ, self._year, not self.isMC, self.jecs, fatjets=True) 
        VbosonIndex = ak.local_index(Vboson_Jet,axis=1)

        Vboson_Jet_mass, jmsr_shifted_fatjetvars = get_jmsr(secondFJ, num_jets=1, year=self._year, isData=not self.isMC)
        #correctedVbosonNominalMass = ak.firsts(Vboson_Jet_mass)

        correctedVbosonNominalMass = candidatefj.mass
        #print('correctedMass', ak.to_list(correctedVbosonNominalMass)[0:200])
        #else:
         #   Vboson_Jet = second_fj

        #*************************************************************************

        # OBJECT: AK4 jets
        jets, jec_shifted_jetvars = get_jec_jets(events, events.Jet, self._year, not self.isMC, self.jecs, fatjets=False)
        met = met_factory.build(events.MET, jets, {}) if self.isMC else events.MET

        jet_selector = (
            (jets.pt > 30)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
        )

        goodjets = jets[jet_selector]

        # OBJECT: b-jets (only for jets with abs(eta)<2.5)
        bjet_selector = (jet_selector) & (jets.delta_r(candidatefj) > 0.8) & (abs(jets.eta) < 2.5)
        ak4_bjet_candidate = jets[bjet_selector]

        # bjet counts for SR and TTBar Control Region
        #V H version
        dr_ak8Jets_HiggsCandidateJet = goodjets.delta_r(candidatefj)
        dr_ak8Jets_VCandidateJet = goodjets.delta_r(second_fj)
        ak4_outsideBothJets = goodjets[ (dr_ak8Jets_HiggsCandidateJet > 0.8) & (dr_ak8Jets_VCandidateJet  > 0.8) ]

        NumOtherJetsOutsideBothJets = ak.num(ak4_outsideBothJets)
        n_bjets_M_OutsideBothJets = ak.sum(
            ak4_outsideBothJets.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"],
            axis=1,
        )
        n_bjets_T_OutsideBothJets = ak.sum(
            ak4_outsideBothJets.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"],
            axis=1,
        )

        ak4_outsideHiggs = goodjets[(dr_ak8Jets_HiggsCandidateJet > 0.8)]
        ak4_outsideV = goodjets[(dr_ak8Jets_VCandidateJet  > 0.8)]
        n_bjets_M_OutsideHiggs = ak.sum( ak4_outsideHiggs.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"], axis=1,)
        n_bjets_T_OutsideHiggs = ak.sum( ak4_outsideHiggs.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"], axis=1,)
        n_bjets_M_OutsideV = ak.sum( ak4_outsideV.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"], axis=1,)
        n_bjets_T_OutsideV = ak.sum( ak4_outsideV.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"], axis=1,)




        mt_lep_met = np.sqrt(
            2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # delta phi MET and higgs candidate
        met_fj_dphi = candidatefj.delta_phi(met)



        if self.isMC: 
            cutoff = 4.
            cutOnPU = np.ones(nevents,dtype='bool')
            pweights = corrections.get_pileup_weight_raghav(self._year, events.Pileup.nPU.to_numpy())
            pw_pass = ((pweights["nominal"] <= cutoff)*(pweights["up"] <= cutoff)*(pweights["down"] <= cutoff))
            #print('pw_pass', pw_pass)

           # events = events[pw_pass]

        ######################
        # Store variables
        ######################

        variables = {
            "n_good_electrons": n_good_electrons, # n_good_electrons = ak.sum(good_electrons, axis=1)
            "n_good_muons": n_good_muons, #     n_good_muons = ak.sum(good_muons, axis=1)
            "lep_pt": candidatelep.pt,
            "lep_eta": candidatelep.eta,
            "lep_isolation": lep_reliso,
            "lep_misolation": lep_miso,
  
            "lep_fj_dr": lep_fj_dr, #  lep_fj_dr = candidatefj.delta_r(candidatelep_p4)
            "lep_met_mt": mt_lep_met, 
            "met_fj_dphi": met_fj_dphi,
            "met_pt": met.pt,

            "NumFatjets": NumFatjets, # NumFatjets = ak.num(good_fatjets)
            "ReconHiggsCandidateFatJet_pt": candidatefj.pt, #THIS IS THE UNCORRECTED PT!!
            "ReconVCandidateFatJetVScore": VCandidateVScore, # VCandidateVScore = VScore(second_fj)
            "ReconVCandidateMass": VCandidate_Mass,  #VCandidate_Mass = second_fj.msdcorr
        
            "numberAK4JetsOutsideFatJets": NumOtherJetsOutsideBothJets,
            "numberBJets_Medium_OutsideFatJets": n_bjets_M_OutsideBothJets,
	    "numberBJets_Tight_OutsideFatJets": n_bjets_T_OutsideBothJets,

            "dr_TwoFatJets": dr_two_jets, #dr_two_jets = candidatefj.delta_r(second_fj)
       
       #check JEC
            "VbosonJECcorrectedPT": Vboson_Jet.pt,
            "VbosonJetregularPT": second_fj.pt,

            "numberBJets_Medium_OutsideHiggs": n_bjets_M_OutsideHiggs,
            "numberBJets_Tight_OutsideHiggs": n_bjets_T_OutsideHiggs,
            "numberBJets_Medium_OutsideV": n_bjets_M_OutsideV,
            "numberBJets_Tight_OutsideV": n_bjets_T_OutsideV,

            "pileupWeightCheck": pw_pass


        }

        if self.isMC:
            fatjetvars = {
            "fj_eta": second_fj.eta,
            "fj_phi": second_fj.phi,
            "fj_pt":second_fj.pt, 
            #"fj_mass": ak.firsts(correctedVbosonNominalMass), #corrected for msdcorr, and then JMR/JMS
            #"fj_mass": correctedVbosonNominalMass, #corrected for msdcorr, and then JMR/JMS
            
            "fj_mass": candidatefj.msdcorr, #corrected for msdcorr, and then JMR/JMS
            }
        else:
            fatjetvars = {
            "fj_eta": second_fj.eta,
            "fj_phi": second_fj.phi,
            "fj_pt": second_fj.pt,  #don't correct data!
            "fj_mass": second_fj.msdcorr
            }


        variables = {**variables, **fatjetvars}

        if self._systematics and self.isMC:
            fatjetvars_sys = {}
#farouk applies pt shift to only the higgs, i need to apply it to the V,
            for shift, vals in jec_shifted_fatjetvars_V["pt"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_pt{shift}"] = ak.firsts(vals[VbosonIndex])  #to do: change this to the V
                    #print('fj pt shift', ak.to_list(fatjetvars_sys[f"fj_pt{shift}"])[0:100]) 

            for shift, vals in jmsr_shifted_fatjetvars["msoftdrop"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_mass{shift}"] = ak.firsts(vals)

            variables = {**variables, **fatjetvars_sys}
            fatjetvars = {**fatjetvars, **fatjetvars_sys}

        #deleted farouk's code: re JEC for the other two jets outside Higgs for his VBF case
#            for met_shift in ["UES_up", "UES_down"]:
#                jecvariables = getJECVariables(fatjetvars, candidatelep_p4, met, pt_shift=None, met_shift=met_shift)
#                variables = {**variables, **jecvariables}
 
#7/6 4:52 pm, putting back in JEC variables function =try this to get the pt shifts on the mass
        
#7/11 11:25 am commenting out the shifts since we don't need these per cristina
        #for shift in jec_shifted_fatjetvars_V["pt"]:
        #    if shift != "" and not self._systematics:
        #        continue
        #    jecvariables = getJECVariables(fatjetvars, met, pt_shift=shift, met_shift=None)
        #    variables = {**variables, **jecvariables}

        for shift in jmsr_shifted_fatjetvars["msoftdrop"]:
            if shift != "" and not self._systematics:
                continue
            jmsrvariables = getJMSRVariables(fatjetvars, met, mass_shift=shift)
            variables = {**variables, **jmsrvariables}

        # Selection ***********************************************************************************************************************************************

        #only for MC! need to fix this so it applies only for MC
        self.add_selection(name = "PileupWeight", sel=pw_pass)

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
        self.add_selection(name="OneLep", sel=(n_good_muons == 1) & (n_good_electrons == 0), channel="mu")
        self.add_selection(name="OneLep", sel=(n_good_electrons == 1) & (n_good_muons == 0), channel="ele")
        self.add_selection(name="GreaterTwoFatJets", sel=(NumFatjets >= 2))

        #*************************
        #fj_pt_sel = candidatefj.pt > 250   Farouk
        fj_pt_sel = second_fj.pt > 250   # put back, this now for V only.... should be for higgs as well?
        if self.isMC:  # make an OR of all the JECs
            for k, v in self.jecs.items():
                for var in ["up", "down"]:
                    #fj_pt_sel = fj_pt_sel | (candidatefj[v][var].pt > 200) #Farouk uses candidatefj
                    fj_pt_sel = fj_pt_sel | (second_fj[v][var].pt > 250) #changed to V
        self.add_selection(name="CandidateJetpT", sel=(fj_pt_sel == 1))
        #*************************

        self.add_selection(name="LepInJet", sel=(lep_fj_dr < 0.8))
        self.add_selection(name="JetLepOverlap", sel=(lep_fj_dr > 0.03))
        self.add_selection(name="VmassCut", sel=( VCandidate_Mass > 30 )) #keeping at 30, can increase to 40 in post-processing to check
        self.add_selection(name="MET", sel=(met.pt > 30))


        # gen-level matching
        signal_mask = None
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

# IF MC**********************************************************************************************************************************************************
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

                #add top pt reweighting from farouk's repo, june 29th: 3:50 pm version
                #https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
                if "TT" in dataset:
                    tops = events.GenPart[get_pid_mask(events.GenPart, 6, byall=False) * events.GenPart.hasFlags(["isLastCopy"])]
                    variables["top_reweighting"] = add_TopPtReweighting(tops.pt)

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

                # store b-tag weight  #i am using MEDIUM, changing to "M"
                for wp_ in ["M"]:
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


                # fill inference
                if self._inference:
                    print('running inference')
                    for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
                        pnet_vars = runInferenceTriton(
                            self.tagger_resources_path,
                            events[selection_ch],
                            fj_idx_lep[selection_ch],
                            model_name=model_name,
                        )
                        pnet_df = self.ak_to_pandas(pnet_vars)
                        scores = {"fj_ParT_score": pnet_df[sigs].sum(axis=1).values}
                        #print('scores', scores)

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
                #"rec_higgs_m",
                #"rec_higgs_pt",
                "rec_V_m",
                "rec_V_pt",
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
