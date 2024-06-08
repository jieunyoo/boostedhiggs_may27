"""
Skimmer for ParticleNet tagger inputs.

Author(s): Cristina Mantilla Suarez, Raghav Kansal, Farouk Mokhtar.
"""

import os
import pathlib
import warnings
from typing import Dict

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import candidate
from coffea.processor import ProcessorABC, dict_accumulator

from .corrections import btagWPs
from .run_tagger_inference import runInferenceTriton
from .tagger_gen_matching import match_H, match_QCD, match_Top, match_V
from .utils import FILL_NONE_VALUE, add_selection_no_cutflow, sigs

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
np.seterr(invalid="ignore")


class InputProcessor(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, year, output_location="./outfiles/"):
        self._year = year
        self._output_location = output_location

        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

        # self._accumulator = dict_accumulator({})

        self.GenPartvars = [
            "fj_genjetmass",
            # higgs classes
            "fj_isggF",
            "fj_isVBF",
            "fj_isJHUVariableWMass",
            "fj_isBulkGraviton",
            # higgs properties
            "fj_genRes_pt",
            "fj_genRes_eta",
            "fj_genRes_phi",
            "fj_genRes_mass",
            "fj_genH_pt",
            "fj_genH_jet",
            "fj_genV_dR",
            "fj_genVstar",
            "genV_genVstar_dR",
            "fj_isHVV",
            "fj_isHVV_Matched",
            "fj_isHVV_4q",
            "fj_isHVV_elenuqq",
            "fj_isHVV_munuqq",
            "fj_isHVV_taunuqq",
            "fj_isHVV_Vlepton",
            "fj_isHVV_Vstarlepton",
            "fj_nquarks",
            "fj_lepinprongs",
            # wjets
            "fj_isV",
            "fj_isV_Matched",
            "fj_isV_2q",
            "fj_isV_elenu",
            "fj_isV_munu",
            "fj_isV_taunu",
            "fj_nprongs",
            "fj_lepinprongs",
            "fj_ncquarks",
            "fj_isV_lep",
            # ttbar
            "fj_isTop",
            "fj_isTop_Matched",
            "fj_Top_numMatched",
            "fj_isTop_W_lep_b",
            "fj_isTop_W_lep",
            "fj_isTop_W_ele_b",
            "fj_isTop_W_ele",
            "fj_isTop_W_mu_b",
            "fj_isTop_W_mu",
            "fj_isTop_W_tau_b",
            "fj_isTop_W_tau",
            "fj_Top_nquarksnob",
            "fj_Top_nbquarks",
            "fj_Top_ncquarks",
            "fj_Top_nleptons",
            "fj_Top_nele",
            "fj_Top_nmu",
            "fj_Top_ntau",
            "fj_Top_taudecay",
            # qcd
            "fj_isQCD",
            "fj_isQCD_Matched",
            "fj_isQCDb",
            "fj_isQCDbb",
            "fj_isQCDc",
            "fj_isQCDcc",
            "fj_isQCDothers",
        ]

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, df, fname):
        if self._output_location is not None:
            PATH = f"{self._output_location}/parquet/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            table = pa.Table.from_pandas(df)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, f"{PATH}/{fname}.parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def dump_root(self, skimmed_vars: Dict[str, np.array], fname: str) -> None:
        """
        Saves ``jet_vars`` dict as a rootfile to './outroot'
        """
        local_dir = os.path.abspath(os.path.join(self._output_location, "outroot"))
        os.system(f"mkdir -p {local_dir}")

        with uproot.recreate(f"{local_dir}/{fname}.root", compression=uproot.LZ4(4)) as rfile:
            rfile["Events"] = ak.Array(skimmed_vars)
            rfile["Events"].show()

    def process(self, events: ak.Array):
        import time

        start = time.time()

        genparts = events.GenPart
        dataset = events.metadata["dataset"]

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

        met = events.MET

        # lepton
        electrons = events["Electron"][events["Electron"].pt > 40]
        muons = events["Muon"][events["Muon"].pt > 30]
        leptons = ak.concatenate([electrons, muons], axis=1)
        leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
        candidatelep = ak.firsts(leptons)
        candidatelep_p4 = build_p4(candidatelep)

        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton

        lep_met_mt = np.sqrt(
            2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met)))
        )

        # fatjet
        fatjets = events["FatJet"]
        good_fatjets = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight

        good_fatjets = fatjets[good_fatjets]  # select good fatjets
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt

        NumFatjets = ak.num(good_fatjets)
        FirstFatjet = ak.firsts(good_fatjets[:, 0:1])
        SecondFatjet = ak.firsts(good_fatjets[:, 1:2])

        # candidatefj
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])

        # ak4 jets
        ak4_jet_selector_no_btag = (
            (events.Jet.pt > 30) & (abs(events.Jet.eta) < 5.0) & events.Jet.isTight & (events.Jet.puId > 0)
        )
        # reject EE noisy jets for 2017
        if self._year == "2017":
            ak4_jet_selector_no_btag = ak4_jet_selector_no_btag & (
                (events.Jet.pt > 50) | (abs(events.Jet.eta) < 2.65) | (abs(events.Jet.eta) > 3.139)
            )

        goodjets = events.Jet[ak4_jet_selector_no_btag]
        ht = ak.sum(goodjets.pt, axis=1)

        dr_jet_lepfj = goodjets.delta_r(candidatefj)
        ak4_outside_ak8 = goodjets[dr_jet_lepfj > 0.8]
        NumOtherJets = ak.num(ak4_outside_ak8)

        # VBF variables
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass
        jj_pt = (ak.firsts(jet1) + ak.firsts(jet2)).pt

        # rec_higgs
        candidateNeutrino = ak.zip(
            {
                "pt": met.pt,
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

        # INPUT VARIABLES
        # CANDIDATE JET
        fj_vars = [
            "eta",
            "phi",
            "mass",
            "pt",
            "msoftdrop",
            "lsf3",
        ]

        FatJetVars = {f"fj_{var}": ak.fill_none(candidatefj[var], FILL_NONE_VALUE) for var in fj_vars}

        # CANDIDATE LEPTON
        LepVars = {}
        LepVars["lep_fj_dr"] = candidatelep_p4.delta_r(candidatefj).to_numpy().filled(fill_value=0)
        LepVars["lep_pt"] = (candidatelep_p4.pt).to_numpy().filled(fill_value=0)
        LepVars["lep_pt_ratio"] = (candidatelep_p4.pt / candidatefj.pt).to_numpy().filled(fill_value=0)
        LepVars["lep_reliso"] = lep_reliso.to_numpy().filled(fill_value=0)
        LepVars["lep_miso"] = lep_miso.to_numpy().filled(fill_value=0)

        # MET
        METVars = {}
        METVars["met_pt"] = met.pt
        METVars["met_relpt"] = met.pt / candidatefj.pt
        METVars["met_fj_dphi"] = met.delta_phi(candidatefj)
        METVars["abs_met_fj_dphi"] = np.abs(met.delta_phi(candidatefj))
        METVars["lep_met_mt"] = lep_met_mt.to_numpy().filled(fill_value=0)

        # OTHERS

        # bjet
        Others = {}
        Others["n_bjets_L"] = (
            ak.sum(
                ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["L"],
                axis=1,
            )
            .to_numpy()
            .filled(fill_value=0)
        )
        Others["n_bjets_M"] = (
            ak.sum(
                ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"],
                axis=1,
            )
            .to_numpy()
            .filled(fill_value=0)
        )
        Others["n_bjets_T"] = (
            ak.sum(
                ak4_outside_ak8.btagDeepFlavB > btagWPs["deepJet"][self._year]["T"],
                axis=1,
            )
            .to_numpy()
            .filled(fill_value=0)
        )

        # RECONSTRUCTED MASS
        Others["rec_W_lnu_pt"] = rec_W_lnu.pt.to_numpy().filled(fill_value=0)
        Others["rec_W_lnu_m"] = rec_W_lnu.mass.to_numpy().filled(fill_value=0)
        Others["rec_W_qq_pt"] = rec_W_qq.pt.to_numpy().filled(fill_value=0)
        Others["rec_W_qq_m"] = rec_W_qq.mass.to_numpy().filled(fill_value=0)
        Others["rec_higgs_pt"] = rec_higgs.pt.to_numpy().filled(fill_value=0)
        Others["rec_higgs_m"] = rec_higgs.mass.to_numpy().filled(fill_value=0)

        # ggF & VBF
        Others["mjj"] = mjj.to_numpy().filled(fill_value=0)
        Others["jj_pt"] = jj_pt.to_numpy().filled(fill_value=0)
        Others["deta"] = deta.to_numpy().filled(fill_value=0)
        Others["j1_pt"] = ak.firsts(jet1).pt.to_numpy().filled(fill_value=0)
        Others["j2_pt"] = ak.firsts(jet2).pt.to_numpy().filled(fill_value=0)
        Others["j1_m"] = ak.firsts(jet1).mass.to_numpy().filled(fill_value=0)
        Others["j2_m"] = ak.firsts(jet2).mass.to_numpy().filled(fill_value=0)

        # ggF & VBF
        Others["ht"] = ht.to_numpy()
        Others["NumFatjets"] = NumFatjets.to_numpy()
        Others["NumOtherJets"] = NumOtherJets.to_numpy()
        Others["FirstFatjet_pt"] = FirstFatjet.pt.to_numpy().filled(fill_value=0)
        Others["FirstFatjet_m"] = FirstFatjet.mass.to_numpy().filled(fill_value=0)
        Others["SecondFatjet_pt"] = SecondFatjet.pt.to_numpy().filled(fill_value=0)
        Others["SecondFatjet_m"] = SecondFatjet.mass.to_numpy().filled(fill_value=0)

        # last but not least, gen info
        if ("HToWW" in dataset) | ("BulkGraviton" in dataset) | ("JHUVariableWMass" in dataset):
            print("match_H")
            GenVars, _ = match_H(genparts, candidatefj)
            if "VBF" in dataset:
                print("VBF")
                GenVars["fj_isVBF"] = np.ones(len(genparts))
            elif "GluGluHToWW" in dataset:
                print("ggF")
                GenVars["fj_isggF"] = np.ones(len(genparts))
            elif "JHUVariableWMass" in dataset:
                print("JHUVariableWMass")
                GenVars["fj_isJHUVariableWMass"] = np.ones(len(genparts))
            elif "BulkGraviton" in dataset:
                print("BulkGraviton")
                GenVars["fj_isBulkGraviton"] = np.ones(len(genparts))
        elif "TTToSemiLeptonic" in dataset:
            print("match_Top")
            GenVars, _ = match_Top(genparts, candidatefj)
        elif "WJets" in dataset:
            print("match_V")
            GenVars, _ = match_V(genparts, candidatefj)
        elif "QCD" in dataset:
            print("match_QCD")
            GenVars, _ = match_QCD(genparts, candidatefj)

        # genjet_vars, matched_gen_jet_mask = get_genjet_vars(events, fatjets)
        # AllGenVars = {**GenVars, **genjet_vars}

        AllGenVars = {
            **GenVars,
            **{"fj_genjetmass": candidatefj.matched_gen.mass},
        }  # add gen jet mass

        # loop to keep only the specified variables in `self.GenPartvars`
        # if `GenVars` doesn't contain a variable, that variable is not applicable to this sample so fill with 0s
        GenVars = {key: AllGenVars[key] if key in AllGenVars.keys() else np.zeros(len(genparts)) for key in self.GenPartvars}
        for key, item in GenVars.items():
            try:
                GenVars[key] = GenVars[key].to_numpy()
            except Exception:
                continue

        # combine all the input variables
        skimmed_vars = {**FatJetVars, **GenVars, **METVars, **LepVars, **Others}

        # apply selections
        selection = PackedSelection()
        add_selection_no_cutflow("fjselection", (candidatefj.pt > 200), selection)

        if np.sum(selection.all(*selection.names)) == 0:
            return {}

        skimmed_vars = {
            key: np.squeeze(np.array(value[selection.all(*selection.names)])) for (key, value) in skimmed_vars.items()
        }
        for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
            pnet_vars = runInferenceTriton(
                self.tagger_resources_path,
                events[selection.all(*selection.names)],
                fj_idx_lep[selection.all(*selection.names)],
                model_name=model_name,
            )

            # pnet_df = self.ak_to_pandas(pnet_vars)
            pnet_df = pd.DataFrame(pnet_vars)

            scores = {"fj_ParT_score": (pnet_df[sigs].sum(axis=1)).values}
            reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}

            hidNeurons = {}
            for key in pnet_vars:
                if "hidNeuron" in key:
                    hidNeurons[key] = pnet_vars[key]

            skimmed_vars = {**skimmed_vars, **scores, **reg_mass, **hidNeurons}

        for key in skimmed_vars:
            skimmed_vars[key] = skimmed_vars[key].squeeze().tolist()

        # convert output to pandas
        df = pd.DataFrame(skimmed_vars)

        df = df.dropna()  # very few events would have genjetmass NaN for some reason

        print(f"convert: {time.time() - start:.1f}s")

        print(df)

        # save the output
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        self.save_dfs_parquet(df, fname)

        print(f"dump parquet: {time.time() - start:.1f}s")

        return {}

    def postprocess(self, accumulator):
        pass
