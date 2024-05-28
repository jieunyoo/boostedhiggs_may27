#!/usr/bin/python

import glob
import json
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
from hist import Hist
from typing import List
import onnxruntime as ort
import scipy
import onnx
import hist

warnings.filterwarnings("ignore", message="Found duplicate branch ")

def blindBins(h: hist2, blind_region: List, blind_samples: List[str] = []):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_samples`` specified, only blind those samples, else blinds all.

    CAREFUL: assumes axis=0 is samples, axis=3 is mass_axis

    """

    h = h.copy()

    massbins = h.axes["mass_observable"].edges
    print('massbins', massbins)
    

    lv = int(np.searchsorted(massbins, blind_region[0], "right"))
    rv = int(np.searchsorted(massbins, blind_region[1], "left") + 1)
    print('lv', lv)
    print('rv', rv)
    if len(blind_samples) >= 1:
        for blind_sample in blind_samples:
            sample_index = np.argmax(np.array(list(h.axes[0])) == blind_sample)
            h.view(flow=True)[sample_index, :, :, lv:rv] = 0

    else:
        h.view(flow=True)[:, :, :, lv:rv] = 0

    return h



def get_sum_sumgenweight(pkl_files, year, sample):
    sum_sumgenweight = 0
    for ifile in pkl_files:
        # load and sum the sumgenweight of each
        with open(ifile, "rb") as f:
            metadata = pkl.load(f)
            print('metadata', metadata)
        sum_sumgenweight = sum_sumgenweight + metadata[sample][year]["sumgenweight"]
    return sum_sumgenweight


def get_xsecweight(pkl_files, year, sample, is_data, luminosity):
    if not is_data:
        # find xsection
        f = open("../fileset/xsec_pfnano.json")
        xsec = json.load(f)
        f.close()
        try:
            print('sample', sample)
            xsec = eval(str((xsec[sample])))
        except ValueError:
            print(f"sample {sample} doesn't have xsecs defined in xsec_pfnano.json so will skip it")
            return None

        # get overall weighting of events.. each event has a genweight...
        # sumgenweight sums over events in a chunk... sum_sumgenweight sums over chunks
       # print('getsumsumgenweight', get_sum_sumgenweight)
       # print('pkl_files', pkl_files)
       # print('year', year)
       # print('sample', sample)
        print('xsec*lum', xsec * luminosity)
        xsec_weight = (xsec * luminosity) / get_sum_sumgenweight(pkl_files, year, sample)
    else:
        xsec_weight = 1
    return xsec_weight


axis_dict = {    #you need to make these titles for each histogram you make, many of them are obsolete or for various tests, just am leaving them here now

    "ReconV_SoftDropMass": hist2.axis.Regular(20, 0, 200, name="var", label=r"V candidate soft drop mass [GeV]", overflow=True,),
    #  "lep_pt": hist2.axis.Regular(20, 30, 200, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),

          "lep_pt": hist2.axis.Regular(20, 30, 400, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
      #    "lep_pt_loose": hist2.axis.Regular(20, 30, 200, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
              "lep_pt_loose": hist2.axis.Regular(20, 30, 400, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
    "lep_eta": hist2.axis.Regular(20, 0, 2.5, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),

        
 "lep_isolation": hist2.axis.Regular(25, 0, 0.5, name="var", label=r"lep_isolation", overflow=True),    
 "lep_misolation": hist2.axis.Regular(20, 0, 0.1, name="var", label=r"lep_misolation", overflow=True),    
    
      "lepton_dz": hist2.axis.Regular(25, 0, 0.5, name="var", label=r"lepton dz", overflow=True),    
    "lepton_dxy": hist2.axis.Regular(20, 0, 0.1, name="var", label=r"lepton dxy", overflow=True),
    "lepton_sip3d": hist2.axis.Regular(20, 0, 10, name="var", label=r"lepton sip3d", overflow=True),
    

    "higgsPT": hist2.axis.Regular(30, 0, 1000, name="var", label=r"Higgs true $p_T$ [GeV]", overflow=True),

          "prob_H" : hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"prob_H", overflow=True),
      "fj_ParT_score" : hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"HWW tagger score", overflow=True),
    "lhe_pt": hist2.axis.Regular(100, 0, 2500, name="var", label=r"lhe $p_T$ [GeV]", overflow=True),
     "fraction": hist2.axis.Regular(10, 0, 1, name="var", label=r"fraction", overflow=True),    
    "numberAK4JetsOutsideFatJets": hist2.axis.Regular(8, 0, 8, name="var", label=r"number of ak4 jets outside fat jets ", overflow=True),

     "numberBJets_Tight_OutsideFatJets": hist2.axis.Regular(4, 0, 4, name="var", label=r"number of tight b-tagged jets outside fat jets", overflow=True),

     "numberBJets_Medium_OutsideFatJets": hist2.axis.Regular(4, 0, 4, name="var", label=r"number of medium b-tagged ak4 jets outside fat jets", overflow=True),
    
      
     "dr_TwoFatJets": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R(recon Higgs, recon V)$", overflow=True),

    "ReconHiggsCandidateFatJet_phi": hist2.axis.Regular(35, 0, 3.14, name="var", label=r"recon. Higgs jet phi", overflow=True ),

    "ReconHiggsCandidateFatJet_eta": hist2.axis.Regular(40, 0, 5, name="var", label=r"recon. Higgs jet phi", overflow=True ),


    
      "ReconV_n2b1Score": hist2.axis.Regular(20, 0, 0.5, name="var", label=r"n2b1 score of V jet", overflow=True),

      "numberLeptons": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),

       "numbertau_mu": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),

       "numbertau_ele": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),
   
      "fj_ParT_inclusive_score" : hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"HWW tagger score", overflow=True),
      "fj_ParT_all_score" : hist2.axis.Regular(20, 0.5, 1.0, name="var", label=r"HWW tagger score", overflow=True),
      "fatjet0": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"HWW tagger score", overflow=True),
      "ReconVCandidateFatJetVScore": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"V tagger score", overflow=True),
     "ReconVCandidateFatJetVScoreCC": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"PN Xcc/total score of candidate V jet", overflow=True),
     "ReconVCandidateFatJetVScoreBB": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"PN Xbb/total score of candidate V jet", overflow=True),
     "ReconVCandidateFatJetVScoreQQ": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"PN Xqq/total score of candidate V jet", overflow=True),
     "ReconVCandidateFatJetVScoreWH": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"PN Xcc+Xqq/total score of candidate V jet", overflow=True),
    
    "ReconVCandidateMass": hist2.axis.Regular(20, 0, 200, name="var", label=r"V candidate mass [GeV]", overflow=False),
        
  #      "ReconVCandidateMass": hist2.axis.Regular(25, 0, 250, name="var", label=r"V candidate mass [GeV]", overflow=False),
    
     "ReconVCandidateMassQQ": hist2.axis.Regular(30, 0, 300, name="var", label=r"V candidate mass [GeV]", overflow=True),
     "ReconVCandidateMassBB": hist2.axis.Regular(30, 0, 300, name="var", label=r"V candidate mass [GeV]", overflow=True),
      "ReconVCandidateMassCC": hist2.axis.Regular(30, 0, 300, name="var", label=r"V candidate mass [GeV]", overflow=True),    
     "ReconVCandidateMassWH": hist2.axis.Regular(30, 0, 300, name="var", label=r"V candidate mass [GeV]", overflow=True),

    
    "NScore": hist2.axis.Regular(40, 0.1, 0.5, name="var", label=r"n2b1 score candidate V jet", overflow=True),

     "ReconHiggsCandidateJetReconLepton": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R (Recon.Higgs Jet,Recon.HWW lepton)$", overflow=True),
  
    "numberFatJet": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number of fat jets", overflow=True),
   "VJetCandidatePT": hist2.axis.Regular(40, 0, 1000, name="var", label=r"V candidate $p_T$ [GeV]", overflow=True),
  
    
    "ReconLepton_pt": hist2.axis.Regular(20, 0, 800, name="var", label=r"recon HWW lepton $p_T$ [GeV]", overflow=True),
    
    "ReconHiggsCandidateFatJet_pt": hist2.axis.Regular(25, 0, 1000, name="var", label=r"recon Higgs candidate fat jet $p_T$ [GeV]", overflow=True),
    
    
    
    "ReconVCandidateFatJet_pt": hist2.axis.Regular(40, 0, 1000, name="var", label=r"recon V boson fat jet $p_T$ [GeV]", overflow=True),
    "DR_Higgs_CandidateHiggsJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R(true Higgs, candidate Higgs Jet)$", overflow=True),
    "DR_Higgs_CandidateVJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R (true Higgs, candidate V Jet))$", overflow=True),
    "DR_TrueLep_CandidateHiggsJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R (true lep, candidate Higgs jet)$", overflow=True),
    "DR_TrueLep_CandidateVJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R (true lep, candidate V Jet)$", overflow=True),
    "ReconHiggsCandidateFatJet_Zscore": hist2.axis.Regular(20, -1.0, 1.0, name="var", label=r"recon Higgs jet n2b1", overflow=True),
    "ReconVCandidateFatJet_Zscore": hist2.axis.Regular(20, -1.0, 1.0, name="var", label=r"recon V candidate jet n2b1", overflow=True),
  
    "ReconHiggsCandidateFatJetZscore": hist2.axis.Regular(20, -1.0, 1.0, name="var", label=r"recon Higgs jet Z score", overflow=True),
    "ReconVCandidateFatJetZscore": hist2.axis.Regular(20, -1.0, 1.0, name="var", label=r"recon V candidate jet Z score", overflow=True),
  
"DR_ReconHiggsCandidateJetReconLepton": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R(true Higgs, candidate Higgs Jet)$", overflow=True),
    
    
    
    
    "PT_highestScoringJet":hist2.axis.Regular(20, 200, 400, name="var", label=r"$p_T$ [GeV]", overflow=True),
    "highestScore": hist2.axis.Regular(20, 0.0, 1.0, name="var", label=r"score", overflow=True),
     "Lep1_NearestAllJets": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "Lep2_NearestAllJets": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "Lep3_NearestAllJets": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
    "numberOSSF_Pairs": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),
      "ReconLepton_flavor": hist2.axis.Regular(3, 0.0, 3.0, name="var", label=r"flavor (0==muon,1==electron)", overflow=True),
   "dR_TrueHiggs_HighestPT_ReconJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
      "dR_TrueHiggs_TrueZ": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
    "3LepMass": hist2.axis.Regular(30, 30, 1000, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
    
   "dRHiggs_ClosestJet_AfterZClean": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
  "dRHiggs_HighestPT_Jet_AfterZClean": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
      "drLeptonClosestJet_AfterZClean": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "dRLeptonHighestPTJet_AfterZClean": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "dRtrueLepNon_ZLep_Recon": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
  # "": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
   "dR_ZLeptons": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
    "DeltaR_AK8recon_anyLep": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "dR_Higgs_HighestPTGenJet": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "dR_ZtrueLeps": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
    "dRHiggsTrueLep": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
   "WdeltaR": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R)$", overflow=True),
     "numberGoodLeptons": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),
      "numberSL_ZLepDecayAK8GenJets": hist2.axis.Regular(5, 0.0, 5.0, name="var", label=r"number", overflow=True),
          
     "JetPT_AfterMatchClosestJetToLepton_AFterZClean": hist2.axis.Regular(40, 30, 450, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
     "Jet_PT_HighestPT_AfterZClean": hist2.axis.Regular(40, 30, 450, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
 
    "JetPT_AfterdRMatchedJetPT_Higgs": hist2.axis.Regular(40, 30, 450, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
   "Z_truePT": hist2.axis.Regular(40, 30, 450, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
    "Zmass": hist2.axis.Regular(15, 50, 150, name="var", label=r"Zmass [GeV]", overflow=True),
    "fj_msoftdrop_nocorr": hist2.axis.Regular(35, 20, 250, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
     "Z_leadingLepPT": hist2.axis.Regular(40, 30, 450, name="var", label=r"leading lepton $p_T$ [GeV]", overflow=True),
      "Z_subleadingLepPT": hist2.axis.Regular(40, 30, 450, name="var", label=r"sub-leading lepton $p_T$ [GeV]", overflow=True),
    "ht": hist2.axis.Regular(30, 0, 1200, name="var", label=r"ht [GeV]", overflow=True),
    "lepton_pT": hist2.axis.Regular(40, 30, 450, name="var", label=r"Lepton $p_T$ [GeV]", overflow=True),
 
     "met_pt": hist2.axis.Regular(30, 0, 300, name="var", label=r"MET_pt", overflow=True),
    "fj_minus_lep_m": hist2.axis.Regular(35, 0, 280, name="var", label=r"Jet - Lepton mass [GeV]", overflow=True),
    "fj_minus_lep_pt": hist2.axis.Regular(40, 0, 450, name="var", label=r"Jet - Lepton $p_T$ [GeV]", overflow=True),
    "lep_met_mt": hist2.axis.Regular(40, 0, 200, name="var", label=r"$m_T(lep, p_T^{miss})$ [GeV]", overflow=True),
    "fj_bjets_ophem": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB (opphem)", overflow=True),
    "fj_bjets": hist2.axis.Regular(35, 0, 1, name="var", label=r"max btagFlavB", overflow=True),
    "lep_fj_dr": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R(Jet, Lepton)$", overflow=True),
    "lep_fj_dr2": hist2.axis.Regular(20, 0.0, 4.0, name="var", label=r"$\Delta R(Jet, Lepton)$", overflow=True),
  
    
    
    "mu_mvaId": hist2.axis.Variable([0, 1, 2, 3, 4, 5], name="var", label="Muon MVAID", overflow=True),
    "ele_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Electron high pT ID", overflow=True),
    "mu_highPtId": hist2.axis.Regular(5, 0, 5, name="var", label="Muon high pT ID", overflow=True),
    "fj_pt": hist2.axis.Regular(30, 200, 600, name="var", label=r"Jet $p_T$ [GeV]", overflow=True),
    "fj_msoftdrop": hist2.axis.Regular(35, 20, 250, name="var", label=r"Jet $m_{sd}$ [GeV]", overflow=True),
    "rec_higgs_m": hist2.axis.Regular(35, 0, 480, name="var", label=r"Higgs reconstructed mass [GeV]", overflow=True),
    "rec_higgs_pt": hist2.axis.Regular(30, 0, 1000, name="var", label=r"Higgs reconstructed $p_T$ [GeV]", overflow=True),
    "fj_pt_over_lep_pt": hist2.axis.Regular(35, 1, 10, name="var", label=r"$p_T$(Jet) / $p_T$(Lepton)", overflow=True),
    "rec_higgs_pt_over_lep_pt": hist2.axis.Regular(
        35, 1, 10, name="var", label=r"$p_T$(Recontructed Higgs) / $p_T$(Lepton)", overflow=True
    ),
    "golden_var": hist2.axis.Regular(35, 0, 10, name="var", label=r"$p_{T}(W_{l\nu})$ / $p_{T}(W_{qq})$", overflow=True),
    "rec_dphi_WW": hist2.axis.Regular(
        35, 0, 3.14, name="var", label=r"$\left| \Delta \phi(W_{l\nu}, W_{qq}) \right|$", overflow=True
    ),
    "fj_ParT_mass": hist2.axis.Regular(20, 0, 250, name="var", label=r"ParT regressed mass [GeV]", overflow=True),
    "fj_ParticleNet_mass": hist2.axis.Regular(
        35, 0, 250, name="var", label=r"fj_ParticleNet regressed mass [GeV]", overflow=True
    ),
    "met_fj_dphi": hist2.axis.Regular(5, -1, 4, name="var", label=r"$\left| \Delta \phi(MET, Jet) \right|$", overflow=True ),
}


# new stuff
combine_samples = {
     "HZJ_HToWW_M-125": "VH",
    "HWplusJ_HToWW_M-125" : "VH",
    "HWminusJ_HToWW_M-125": "VH",
    "GluGluZH": "VH",
    #"GluGluZH_HToWW_ZTo2L_M-125" : "VH",
    #"WZ": "WZ",
    #"QCD_Pt": "QCD",
    "QCD_": "QCD",
    "DYJets": "DYJets",
    "WJetsToLNu_": "WJetsLNu",
    "JetsToQQ": "WZQQ",
    
    "TT": "TTbar",  
    "ST_": "SingleTop",
    "WW": "Diboson",
    "WZ": "Diboson",
    "ZZ": "Diboson",
    "GluGluHToTauTau": "HTauTau",
    "ttHToNonbb_M125": "ttH",
    
    "SingleElectron_": "Data",
    "SingleMuon_": "Data",
    "EGamma_": "Data",

    
}
#signals = ["HWW", "ttH", "VH", "VBF"]
signals = ["VH"]

data_by_ch = {
    "ele": "SingleElectron",
    "mu": "SingleMuon",
}


weights = {
    "mu": {
        "weight_genweight": 1,
        "weight_L1Prefiring": 1,
        "weight_pileup": 1,
        "weight_trigger_iso_muon": 1,
        "weight_trigger_noniso_muon": 1,
        "weight_isolation_muon": 1,
        "weight_id_muon": 1,
        "weight_vjets_nominal": 1,
    },
    "ele": {
        "weight_genweight": 1,
        "weight_L1Prefiring": 1,
        "weight_pileup": 1,
        "weight_trigger_electron": 1,
        "weight_reco_electron": 1,
        "weight_id_electron": 1,
        "weight_vjets_nominal": 1,
    },
}

def event_skimmer(
    year,
    channels,
    samples_dir,
    samples,
    columns="all",
    add_inclusive_score=False,
    add_qcd_score=False,
    add_top_score=False,
):
    events_dict = {}
    for ch in channels:
        events_dict[ch] = {}
   
       # print('samples', samples)
        # get lumi
        with open("../fileset/luminosity.json") as f:
            luminosity = json.load(f)[ch][year]
            print('luminosity', luminosity)

        condor_dir = os.listdir(samples_dir)
      #  print('condor_dir', condor_dir)
        for sample in condor_dir:
            #print('sample', sample)
            # get a combined label to combine samples of the same process
            #print('combine_samples', combine_samples)
            for key in combine_samples:
                #print('combine_samples', combine_samples)
               # print('key', key)
                if key in sample:
                   # print('key in sample', key)
                    sample_to_use = combine_samples[key]
                    break
                else:
             #       print('in else statememt')
                    sample_to_use = sample
              #      print('sample_to_use', sample_to_use, sample)

            if sample_to_use not in samples:
                print(f"ATTENTION!: {sample} will be skipped")
                continue

            is_data = False
            # if sample_to_use == data_by_ch[ch]:
            if sample_to_use == "Data":
                is_data = True

           # print(f"Finding {sample} samples and should combine them under {sample_to_use}")
#       parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            #out_files = f"{samples_dir}"
            #parquet_files = glob.glob(f"{out_files}/*.parquet")
            #pkl_files = glob.glob(f"{out_files}/*.pkl")
            
            out_files = f"{samples_dir}/{sample}/outfiles/"
            parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            pkl_files = glob.glob(f"{out_files}/*.pkl")
    

            if not parquet_files:
                print(f"No parquet file for {sample}")
                continue

            data = pd.read_parquet(parquet_files)
            if len(data) == 0:
                continue

            # replace the weight_pileup of the strange events with the mean weight_pileup of all the other events
         #   if not is_data:
          #      strange_events = data["weight_pileup"] > 6
           #     if len(strange_events) > 0:
            #        data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

            # get event_weight
            if not is_data:
                print("---> Accumulating event weights.")
                event_weight = get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                event_weight *= data[f"weight_{ch}"]

                print("---> Done with accumulating event weights.")
            else:
                #event_weight = np.ones_like(data["fj_pt"])
                event_weight = np.ones_like(data["lep_pt"])

            data["event_weight"] = event_weight



#FAROUK - finetunning - this section uses farouk's fine-tuned model; it runs on the post-processed files (which have the saved hiddden neuron values)

            PATH = "/home/jieun201/may18/boostedhiggs/python/model.onnx"
            input_dict = {
                "highlevel": data.loc[:, "fj_ParT_hidNeuron000":"fj_ParT_hidNeuron127"].values.astype("float32"),
            }
            print('input_dict', input_dict)
            onnx_model = onnx.load(PATH)
            onnx.checker.check_model(onnx_model)
            
            ort_sess = ort.InferenceSession(
                PATH,
                providers=["AzureExecutionProvider"],
            )
            outputs = ort_sess.run(None, input_dict)
            prob_H = scipy.special.softmax(outputs[0], axis=1)[:, 0]   # recall: [class_H, class_W, class_Top, class_QCD
           # print('prob_H', prob_H)

            data["prob_H"] = prob_H
            

            # add tagger scores  
            if add_qcd_score:
                data["QCD_score"] = disc_score(data, new_sig, qcd_bkg)
            if add_top_score:
                data["Top_score"] = disc_score(data, new_sig, top_bkg)
            if add_inclusive_score:
                data["inclusive_score"] = disc_score(data, new_sig, inclusive_bkg)

            print(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
            print(f"tot event weight {data['event_weight'].sum()} \n")

            if columns == "all":
                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data])
            else:
                # specify columns to keep
                cols = columns + ["event_weight"]
                if add_qcd_score:
                    cols += ["QCD_score"]
                if add_top_score:
                    cols += ["Top_score"]
                if add_inclusive_score:
                    cols += ["inclusive_score"]

                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data[cols]
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data[cols]])

    return events_dict

def event_skimmer_noInference(
    year,
    channels,
    samples_dir,
    samples,
    columns="all",
    add_inclusive_score=False,
    add_qcd_score=False,
    add_top_score=False,
):
    events_dict = {}
    for ch in channels:
        events_dict[ch] = {}
   
       # print('samples', samples)
        # get lumi
        with open("../fileset/luminosity.json") as f:
            luminosity = json.load(f)[ch][year]
            print('luminosity', luminosity)

        condor_dir = os.listdir(samples_dir)
      #  print('condor_dir', condor_dir)
        for sample in condor_dir:
            #print('sample', sample)
            # get a combined label to combine samples of the same process
            #print('combine_samples', combine_samples)
            for key in combine_samples:
                #print('combine_samples', combine_samples)
               # print('key', key)
                if key in sample:
                   # print('key in sample', key)
                    sample_to_use = combine_samples[key]
                    break
                else:
             #       print('in else statememt')
                    sample_to_use = sample
              #      print('sample_to_use', sample_to_use, sample)

            if sample_to_use not in samples:
                print(f"ATTENTION!: {sample} will be skipped")
                continue

            is_data = False
            # if sample_to_use == data_by_ch[ch]:
            if sample_to_use == "Data":
                is_data = True

           # print(f"Finding {sample} samples and should combine them under {sample_to_use}")
#       parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            #out_files = f"{samples_dir}"
            #parquet_files = glob.glob(f"{out_files}/*.parquet")
            #pkl_files = glob.glob(f"{out_files}/*.pkl")
            
            out_files = f"{samples_dir}/{sample}/outfiles/"
            parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
            pkl_files = glob.glob(f"{out_files}/*.pkl")
    

            if not parquet_files:
                print(f"No parquet file for {sample}")
                continue

            data = pd.read_parquet(parquet_files)
            if len(data) == 0:
                continue

            # replace the weight_pileup of the strange events with the mean weight_pileup of all the other events
         #   if not is_data:
          #      strange_events = data["weight_pileup"] > 6
           #     if len(strange_events) > 0:
            #        data["weight_pileup"][strange_events] = data[~strange_events]["weight_pileup"].mean(axis=0)

            # get event_weight
            if not is_data:
                print("---> Accumulating event weights.")
                event_weight = get_xsecweight(pkl_files, year, sample, is_data, luminosity)
                event_weight *= data[f"weight_{ch}"]

                print("---> Done with accumulating event weights.")
            else:
                #event_weight = np.ones_like(data["fj_pt"])
                event_weight = np.ones_like(data["lep_pt"])

            data["event_weight"] = event_weight



            # add tagger scores  
            if add_qcd_score:
                data["QCD_score"] = disc_score(data, new_sig, qcd_bkg)
            if add_top_score:
                data["Top_score"] = disc_score(data, new_sig, top_bkg)
            if add_inclusive_score:
                data["inclusive_score"] = disc_score(data, new_sig, inclusive_bkg)

           # print(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
            print(f"tot event weight {data['event_weight'].sum()} \n")

            if columns == "all":
                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data])
            else:
                # specify columns to keep
                cols = columns + ["event_weight"]
                if add_qcd_score:
                    cols += ["QCD_score"]
                if add_top_score:
                    cols += ["Top_score"]
                if add_inclusive_score:
                    cols += ["inclusive_score"]

                # fill the big dataframe
                if sample_to_use not in events_dict[ch]:
                    events_dict[ch][sample_to_use] = data[cols]
                else:
                    events_dict[ch][sample_to_use] = pd.concat([events_dict[ch][sample_to_use], data[cols]])

    return events_dict
