import importlib.resources
import json
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
					    get_jmsr, met_factory, add_pdf_weight,

)
from boostedhiggs.utils import match_H, match_Top, match_V, sigs

from .run_tagger_inference import runInferenceTriton

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")

import logging

logger = logging.getLogger(__name__)


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


def VScoreCC(goodFatJetsSelected):
    num = goodFatJetsSelected.particleNetMD_Xcc
    den = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
        + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score


def VScoreBB(goodFatJetsSelected):
    num = goodFatJetsSelected.particleNetMD_Xbb
    den = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
        + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score


def VScoreQQ(goodFatJetsSelected):
    num = goodFatJetsSelected.particleNetMD_Xqq
    den = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
        + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score


def VScoreWH(goodFatJetsSelected): #to do: change this variable's name
    num = goodFatJetsSelected.particleNetMD_Xqq + goodFatJetsSelected.particleNetMD_Xcc
    den = (
        goodFatJetsSelected.particleNetMD_Xbb
        + goodFatJetsSelected.particleNetMD_Xcc
        + goodFatJetsSelected.particleNetMD_Xqq
        + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score


class vhProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        systematics=False,
        region="signal",
    ):
        """
        region can take ["signal", "zll", "qcd", "wjets"].
        """
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._region = region
        self._systematics = systematics
        print(f"Will apply selections applicable to {region} region")

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
        }

        # do inference
        self.inference = inference
        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(
                    table, self._output_location + ch + "/parquet/" + fname + ".parquet"
                )

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
                logger.warning(
                    f"Attempted to add selection to unexpected channel: {ch} not in %s"
                    % (self._channels)
                )
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
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.weights = {
            ch: Weights(nevents, storeIndividual=True) for ch in self._channels
        }
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        sumgenweight = ak.sum(events.genWeight) if self.isMC else 0

        # trigger
        trigger = {}
        trigger_noiso = {}
        trigger_iso = {}
        for ch in self._channels:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            trigger_noiso[ch] = np.zeros(nevents, dtype="bool")
            trigger_iso[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    if "Iso" in t or "WPTight_Gsf" in t:
                        trigger_iso[ch] = trigger_iso[ch] | events.HLT[t]
                    else:
                        trigger_noiso[ch] = trigger_noiso[ch] | events.HLT[t]
                    trigger[ch] = trigger[ch] | events.HLT[t]

        # metfilters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        # taus
        loose_taus_mu = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idAntiMu >= 1)
        )  # loose antiMu ID
        loose_taus_ele = (
            (events.Tau.pt > 20)
            & (abs(events.Tau.eta) < 2.3)
            & (
                events.Tau.idAntiEleDeadECal >= 2
            )  # loose Anti-electron MVA discriminator V6 (2018) ?
        )
        n_loose_taus_mu = ak.sum(loose_taus_mu, axis=1)
        n_loose_taus_ele = ak.sum(loose_taus_ele, axis=1)

        muons = ak.with_field(events.Muon, 0, "flavor")
        electrons = ak.with_field(events.Electron, 1, "flavor")

        # muons
        loose_muons = (
            (((muons.pt > 30) & (muons.pfRelIso04_all < 0.25)) | (muons.pt > 55))
            & (np.abs(muons.eta) < 2.4)
            & (muons.looseId)
        )
        n_loose_muons = ak.sum(loose_muons, axis=1)

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
            & (((muons.pfRelIso04_all < 0.15) & (muons.pt < 55)) | (muons.pt >= 55))
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        # electrons
        loose_electrons = (
            (
                ((electrons.pt > 38) & (electrons.pfRelIso03_all < 0.25))
                | (electrons.pt > 120)
            )
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.cutBased >= electrons.LOOSE)
        )
        n_loose_electrons = ak.sum(loose_electrons, axis=1)

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (electrons.mvaFall17V2noIso_WP90)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        # get candidate lepton
        goodleptons = ak.concatenate(
            [muons[good_muons], electrons[good_electrons]], axis=1
        )  # concat muons and electrons
        # goodleptons = ak.concatenate(muons, electrons, axis=1)
        goodleptons = goodleptons[
            ak.argsort(goodleptons.pt, ascending=False)
        ]  # sort by pt

        ngood_leptons = ak.num(goodleptons, axis=1)
        # print('ngoodleptons', ngood_leptons)
        # print('nloosetaumu', ak.to_list(n_loose_taus_mu)[0:200])

        candidatelep = ak.firsts(goodleptons)  # pick highest pt

        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton
        lep_reliso = (
            candidatelep.pfRelIso04_all
            if hasattr(candidatelep, "pfRelIso04_all")
            else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton

        # ak4 jets
        ak4_jet_selector_no_btag = (
            (events.Jet.pt > 30)
            & (abs(events.Jet.eta) < 5.0)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        )
        # reject EE noisy jets for 2017
        if self._year == "2017":
            ak4_jet_selector_no_btag = ak4_jet_selector_no_btag & (
                (events.Jet.pt > 50)
                | (abs(events.Jet.eta) < 2.65)
                | (abs(events.Jet.eta) > 3.139)
            )

        goodjets = events.Jet[ak4_jet_selector_no_btag]
        goodjets, jec_shifted_jetvars = get_jec_jets(
            events, goodjets, self._year, not self.isMC, self.jecs, fatjets=False
        )

        ht = ak.sum(goodjets.pt, axis=1)

        # fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)

        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight

        n_fatjets = ak.sum(good_fatjets, axis=1)

        good_fatjets = fatjets[good_fatjets]  # select good fatjets

        good_fatjets = good_fatjets[
            ak.argsort(good_fatjets.pt, ascending=False)
        ]  # sort them by pt
        # print('nfatjets', ak.to_list(n_fatjets)[0:200])

        good_fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True
        )

        # ************************************************************************************
        # jieun added below for VH
        deltaR_lepton_all_jets = candidatelep_p4.delta_r(good_fatjets)
        minDeltaR = ak.argmin(deltaR_lepton_all_jets, axis=1)
        fatJetIndices = ak.local_index(good_fatjets, axis=1)

        mask1 = fatJetIndices != minDeltaR

        fj_idx_lep = ak.argmin(
            good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True
        )
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        # CHOOSE PN Xqq discriminator
        allScores = VScore(good_fatjets)
        masked = allScores[mask1]
        secondFJ = good_fatjets[allScores == ak.max(masked, axis=1)]
        second_fj = ak.firsts(secondFJ)
        HiggsCandidateVScore = VScore(candidatefj)
        VCandidateVScore = VScore(second_fj)
        VCandidate_Mass = second_fj.msdcorr


        remainingPTValues = good_fatjets.pt
        masked_ForN2B1 = remainingPTValues[mask1]
        secondFJ_ForN2B1 = good_fatjets[remainingPTValues == ak.max(masked_ForN2B1, axis=1)]
        second_highestPT_fatJet = ak.firsts(secondFJ_ForN2B1)

        reconV_n2b1score = second_highestPT_fatJet.n2b1
        reconV_n2b1_softdropmass = second_highestPT_fatJet.msdcorr


        # PN Xcc discriminator
        allScoresCC = VScoreCC(good_fatjets)
        maskedCC = allScoresCC[mask1]
        secondFJCC = good_fatjets[allScoresCC == ak.max(maskedCC, axis=1)]
        second_fjCC = ak.firsts(secondFJCC)
        VCandidateVScoreCC = VScoreCC(second_fjCC)
        VCandidate_MassCC = second_fjCC.msdcorr

        # PN Xbb discriminator
        allScoresBB = VScoreBB(good_fatjets)
        maskedBB = allScoresBB[mask1]
        secondFJBB = good_fatjets[allScoresBB == ak.max(maskedBB, axis=1)]
        second_fjBB = ak.firsts(secondFJBB)
        VCandidateVScoreBB = VScoreBB(second_fjBB)
        VCandidate_MassBB = second_fjBB.msdcorr

        # PN XQQ discriminator
        allScoresQQ = VScoreQQ(good_fatjets)
        maskedQQ = allScoresQQ[mask1]
        secondFJQQ = good_fatjets[allScoresQQ == ak.max(maskedQQ, axis=1)]
        second_fjQQ = ak.firsts(secondFJQQ)
        VCandidateVScoreQQ = VScoreQQ(second_fjQQ)
        VCandidate_MassQQ = second_fjQQ.msdcorr

        # PN WH discriminator
        allScoresWH = VScoreWH(good_fatjets)
        maskedWH = allScoresWH[mask1]
        secondFJWH = good_fatjets[allScoresWH == ak.max(maskedWH, axis=1)]
        second_fjWH = ak.firsts(secondFJWH)
        VCandidateVScoreWH = VScoreWH(second_fjWH)
        VCandidate_MassWH = second_fjWH.msdcorr
        # ************************************************************************************

        # choose candidate fatjet
        # fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        # candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # MET
        met = met_factory.build(events.MET, goodjets, {}) if self.isMC else events.MET

        mt_lep_met = np.sqrt(
            2.0
            * candidatelep_p4.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )
        # delta phi MET and higgs candidate
        met_fjlep_dphi = candidatefj.delta_phi(met)
        # delta R between AK8 jet and lepton
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)
        # mohammad requested
        dr_two_jets = candidatefj.delta_r(second_fj)

        # bjet studies
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

        #if self.isMC:
         #   lheHT = events.LHE.HT
          #  print("lheHT", lheHT)


        #check cuts for inversion
        leptonDZ = candidatelep.dz
        #print('leptonDZ', leptonDZ)

        variables = {
            "lepton_dz": candidatelep.dz,
            "lepton_dxy": candidatelep.dxy,
            "lepton_sip3d": candidatelep.sip3d,
            "fj_msoftdrop": candidatefj.msdcorr,
            "lep_pt": candidatelep.pt,
            "lep_isolation": lep_reliso,
            "lep_misolation": lep_miso,
            "lep_fj_dr": lep_fj_dr,
            "lep_met_mt": mt_lep_met,
            "met_fj_dphi": met_fjlep_dphi,
            "met_pt": met.pt,
            "ht": ht,
	    "lep_eta": candidatelep.eta,
            # "deta": deta,
            # "mjj": mjj,
            # jieun below
           # "lhe_pt": lheHT,
            "numberLeptons": ngood_leptons,
            "numbertau_mu": n_loose_taus_mu,
            "numbertau_ele": n_loose_taus_ele,
            #  "numberBHadrons": numBHadrons,
            "numberFatJet": n_fatjets,
            "ReconLepton_pt": candidatelep.pt,
            "ReconLepton_flavor": candidatelep.flavor,
            "ReconHiggsCandidateFatJet_pt": candidatefj.pt,
            "ReconVCandidateFatJet_pt": second_fj.pt,
            "DR_ReconHiggsCandidateJetReconLepton": lep_fj_dr,
            # "ReconVCandidateFatJet_N2B1score": VCandidateN2B1,
            "ReconVCandidateFatJetVScore": VCandidateVScore,
            "ReconVCandidateMass": VCandidate_Mass,
            "ReconV_n2b1Score": reconV_n2b1score,
	    "numberAK4JetsOutsideFatJets": NumOtherJetsOutsideBothJets,
	    "numberBJets_Medium_OutsideFatJets": n_bjets_M_OutsideBothJets,
	    "numberBJets_Tight_OutsideFatJets": n_bjets_T_OutsideBothJets,
            "dr_TwoFatJets": dr_two_jets,
            "ReconHiggsCandidateFatJet_eta": candidatefj.eta,
            "ReconHiggsCandidateFatJet_phi": candidatefj.phi,
            "ReconVCandidateFatJet_eta": second_fj.eta,
            "ReconVCandidateFatJet_phi": second_fj.phi,

            "ReconV_n2b1_selected_pt": second_highestPT_fatJet.pt,
            "ReconV_n2b1_selected_mass": reconV_n2b1_softdropmass,



            # other CASES
            "ReconVCandidateFatJet_ptCC": second_fjCC.pt,
            "ReconVCandidateFatJet_ptBB": second_fjBB.pt,
            "ReconVCandidateFatJet_ptQQ": second_fjQQ.pt,

            "ReconVCandidateFatJetVScoreCC": VCandidateVScoreCC,
            "ReconVCandidateMassCC": VCandidate_MassCC,
            "ReconVCandidateFatJetVScoreBB": VCandidateVScoreBB,
            "ReconVCandidateMassBB": VCandidate_MassBB,
            "ReconVCandidateFatJetVScoreQQ": VCandidateVScoreQQ,
            "ReconVCandidateMassQQ": VCandidate_MassQQ,
            "ReconVCandidateFatJetVScoreWH": VCandidateVScoreWH,
            "ReconVCandidateMassWH": VCandidate_MassWH,
            "ReconV_SoftDropMass": second_fj.msdcorr,
        }

        fatjetvars = {
            "fatjetPt": candidatefj.pt,
            "fatjetEta": candidatefj.eta,
            "fatjetPhi": candidatefj.phi,
            "fatjetMass": candidatefj.msdcorr,
        }
        for shift, vals in jec_shifted_fatjetvars["pt"].items():
            if shift != "":
                fatjetvars[f"fatjetPt{shift}"] = ak.firsts(vals[fj_idx_lep])

        def getJECVariables(
            fatjetvars, candidatelep_p4, met, pt_shift=None, met_shift=None
        ):
            """
            get variables affected by JES_up, JES_down, JER_up, JER_down, UES_up, UES_down
            """
            variables = {}

            ptlabel = pt_shift if pt_shift is not None else ""
            if met_shift is not None:
                if met_shift == "UES_up":
                    metvar = met.MET_UnclusteredEnergy.up
                elif met_shift == "UES_down":
                    metvar = met.MET_UnclusteredEnergy.down
                metlabel = met_shift
            else:
                if ptlabel != "":
                    metlabel = ""
                    if ptlabel == "JES_up":
                        metvar = met.JES_jes.up
                    elif ptlabel == "JES_down":
                        metvar = met.JES_jes.down
                    elif ptlabel == "JER_up":
                        metvar = met.JER.up
                    elif ptlabel == "JER_down":
                        metvar = met.JER.down
                else:
                    metvar = met
                    metlabel = ""
            shift = ptlabel + metlabel

            candidatefj = ak.zip(
                {
                    "pt": fatjetvars[f"fatjetPt{ptlabel}"],
                    "eta": fatjetvars["fatjetEta"],
                    "phi": fatjetvars["fatjetPhi"],
                    "mass": fatjetvars["fatjetMass"],
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )
            candidateNeutrino = ak.zip(
                {
                    "pt": metvar.pt,
                    "eta": candidatelep_p4.eta,
                    "phi": met.phi,
                    "mass": 0,
                    "charge": 0,
                },
                with_name="PtEtaPhiMCandidate",
                behavior=candidate.behavior,
            )
            rec_W_lnu = candidatelep_p4 + candidateNeutrino
            rec_W_qq = candidatefj - candidatelep_p4
            rec_higgs = rec_W_qq + rec_W_lnu

            variables[f"fj_minus_lep_m{shift}"] = (candidatefj - candidatelep_p4).mass
            variables[f"fj_pt{shift}"] = candidatefj.pt
            variables[f"rec_higgs_m{shift}"] = rec_higgs.mass
            return variables

        # add variables affected by JECs/MET
        for shift in jec_shifted_fatjetvars["pt"]:
            if shift != "" and not self._systematics:
                continue
            jecvariables = getJECVariables(
                fatjetvars, candidatelep_p4, met, pt_shift=shift, met_shift=None
            )
            variables = {**variables, **jecvariables}
        if self._systematics and self.isMC:
            for met_shift in ["UES_up", "UES_down"]:
                jecvariables = getJECVariables(
                    fatjetvars, candidatelep_p4, met, pt_shift=None, met_shift=met_shift
                )
                variables = {**variables, **jecvariables}

        self.add_selection("all", np.ones(nevents, dtype="bool"))

        # apply trigger
        for ch in self._channels:
            self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)

        # apply selections
        self.add_selection(name="METFilters", sel=metfilters)
        #self.add_selection(name="LepKin", sel=(candidatelep.pt > 30), channel="mu")
        #self.add_selection(name="LepKin", sel=(candidatelep.pt > 40), channel="ele")
        self.add_selection(name="GreaterTwoFatJets", sel=n_fatjets >= 2)

        self.add_selection(
            name="LeptonCut",
            sel=(ngood_leptons == 1)
	    & (n_good_muons == 1), channel="mu",
        )

        self.add_selection(
            name="LeptonCut",
            sel=(ngood_leptons == 1)
	    & (n_good_electrons == 1), channel="ele",
        )

        # lepton misolation
        self.add_selection(
            name="LepMisolation",
            sel=((candidatelep.pt < 55) | ((lep_miso < 0.2) & (candidatelep.pt >= 55))),
            channel="mu",
        )


        """
        HEM issue: Hadronic calorimeter Endcaps Minus (HEM) issue.
        The endcaps of the hadron calorimeter failed to cover the phase space at -3 < eta < -1.3 and -1.57 < phi < -0.87
        during the 2018 data C and D.
        The transverse momentum of the jets in this region is typically under-measured, this results in over-measured MET.
        It could also result on new electrons.
        We must veto the jets and met in this region.
        Should we veto on AK8 jets or electrons too?
        Let's add this as a cut to check first.
        """
        if self._year == "2018":
            hem_cleaning = (
                (
                    (events.run >= 319077) & (not self.isMC)
                )  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (
                ak.any(
                    (
                        (events.Jet.pt > 30.0)
                        & (events.Jet.eta > -3.2)
                        & (events.Jet.eta < -1.3)
                        & (events.Jet.phi > -1.57)
                        & (events.Jet.phi < -0.87)
                    ),
                    -1,
                )
                | (
                    (events.MET.phi > -1.62)
                    & (events.MET.pt < 470.0)
                    & (events.MET.phi < -0.62)
                )
            )
            self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

        if self.isMC:
            for ch in self._channels:
                self.weights[ch].add("genweight", events.genWeight)
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
                #if ch == "mu":
                #    add_lepton_weight(
                #        self.weights[ch],
                #        candidatelep,
                #        self._year + self._yearmod,
                #        "muon",
                #    )
               # elif ch == "ele":
               #     add_lepton_weight(
               #         self.weights[ch],
               #         candidatelep,
               #         self._year + self._yearmod,
               #         "electron",
               #     )

                # add_btag_weights(self.weights[ch], self._year, events.Jet, ak4_jet_selector_no_btag)

                add_VJets_kFactors(self.weights[ch], events.GenPart, dataset, events)

                #if "HToWW" in dataset and self._region == "signal":
                 #   add_HiggsEW_kFactors(self.weights[ch], events.GenPart, dataset)
                 #   add_scalevar_7pt(
                 #       self.weights[ch],
                 #       events.LHEScaleWeight
                 #       if "LHEScaleWeight" in events.fields
                 #       else [],
                 #   )
                 #   add_scalevar_3pt(
                 #       self.weights[ch],
                 #       events.LHEScaleWeight
                 #       if "LHEScaleWeight" in events.fields
                 #       else [],
                 #   )
                 #   add_ps_weight(
                 #       self.weights[ch],
                 #       events.PSWeight if "PSWeight" in events.fields else [],
                  #  )
                  #  add_pdf_weight(
                  #      self.weights[ch],
                  #      events.LHEPdfWeight if "LHEPdfWeight" in events.fields else [],
                  #  )

              #  if "EWK" in dataset and self._region == "signal":
              #      add_pdf_weight(
              #          self.weights[ch],
              #          events.LHEPdfWeight if "LHEPdfWeight" in events.fields else [],
              #      )

                # store the final weight per ch
                variables[f"weight_{ch}"] = self.weights[ch].weight()
                if self._systematics:
                    for systematic in self.weights[ch].variations:
                        variables[f"weight_{ch}_{systematic}"] = self.weights[
                            ch
                        ].weight(modifier=systematic)

                # store the individual weights (for DEBUG)
                # for key in self.weights[ch]._weights.keys():
                #    if f"weight_{key}" not in variables.keys():
                #        variables[f"weight_{key}"] = self.weights[ch].partial_weight([key])

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
                if self.inference:
                    for model_name in [
                         "ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes",
                    ]:
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

            if "rec_higgs_m" in output[ch].keys():
                output[ch]["rec_higgs_m"] = np.nan_to_num(
                    output[ch]["rec_higgs_m"], nan=-1
                )

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
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
