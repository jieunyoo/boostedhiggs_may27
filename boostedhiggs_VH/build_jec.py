import contextlib
import importlib.resources

from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor

"""Twikis:
    - Recommendations: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC
    - Explanation: https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources

I copied Dylan's file https://github.com/drankincms/boostedhiggs/blob/ULv2/boostedhiggs/build_jec.py and I updated
the corrections "UncertaintySources" by the corresponding "Regrouped" unc.

"""

jec_name_map = {
    "JetPt": "pt",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "METpt": "pt",
    "METphi": "phi",
    "JetPhi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}


def jet_factory_factory(files):
    ext = extractor()
    with contextlib.ExitStack() as stack:
        # this would work even in zipballs but since extractor keys on file extension and
        # importlib make a random tempfile, it won't work. coffea needs to enable specifying the type manually
        # for now we run this whole module as $ python -m boostedhiggs.build_jec boostedhiggs/data/jec_compiled.pkl.gz
        # so the compiled value can be loaded using the importlib tool in corrections.py
        real_files = [stack.enter_context(importlib.resources.path("boostedhiggs.data", f)) for f in files]
        ext.add_weight_sets([f"* * {file}" for file in real_files])
        ext.finalize()

    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


jet_factory = {
    "2016preVFPmc": jet_factory_factory(
        files=[
            "Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs.jec.txt",
            "Summer19UL16APV_V7_MC_L2Relative_AK4PFchs.jec.txt",
            # "Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt",
            "RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL16APV_V7_MC_Uncertainty_AK4PFchs.junc.txt",
            "Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
            "Summer20UL16APV_JRV3_MC_SF_AK4PFchs.jersf.txt",
        ]
    ),
    "2016postVFPmc": jet_factory_factory(
        files=[
            # "Summer16_07Aug2017_V11_MC_L1FastJet_AK4PFchs.jec.txt.gz",
            # "Summer16_07Aug2017_V11_MC_L2Relative_AK4PFchs.jec.txt.gz",
            # "RegroupedV2_Summer16_07Aug2017_V11_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            # "Summer16_07Aug2017_V11_MC_Uncertainty_AK4PFchs.junc.txt.gz",
            # "Summer16_25nsV1b_MC_PtResolution_AK4PFchs.jr.txt.gz",
            # "Summer16_25nsV1b_MC_SF_AK4PFchs.jersf.txt.gz",
            "Summer19UL16_V7_MC_L1FastJet_AK4PFchs.jec.txt",
            "Summer19UL16_V7_MC_L2Relative_AK4PFchs.jec.txt",
            # "Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt",
            "RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL16_V7_MC_Uncertainty_AK4PFchs.junc.txt",
            "Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
            "Summer20UL16_JRV3_MC_SF_AK4PFchs.jersf.txt",
        ]
    ),
    "2017mc": jet_factory_factory(
        files=[
            # "Fall17_17Nov2017_V32_MC_L1FastJet_AK4PFchs.jec.txt.gz",
            # "Fall17_17Nov2017_V32_MC_L2Relative_AK4PFchs.jec.txt.gz",
            # "RegroupedV2_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            # "Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFchs.junc.txt.gz",
            # "Fall17_V3b_MC_PtResolution_AK4PFchs.jr.txt.gz",
            # "Fall17_V3b_MC_SF_AK4PFchs.jersf.txt.gz",
            "Summer19UL17_V5_MC_L1FastJet_AK4PFchs.jec.txt",
            "Summer19UL17_V5_MC_L2Relative_AK4PFchs.jec.txt",
            # "Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
            "RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL17_V5_MC_Uncertainty_AK4PFchs.junc.txt",
            "Summer19UL17_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
            "Summer19UL17_JRV3_MC_SF_AK4PFchs.jersf.txt",
        ]
    ),
    "2018mc": jet_factory_factory(
        files=[
            # "Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt.gz",
            # "Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt.gz",
            # "RegroupedV2_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            # "Autumn18_V19_MC_Uncertainty_AK4PFchs.junc.txt.gz",
            # "Autumn18_V7b_MC_PtResolution_AK4PFchs.jr.txt.gz",
            # "Autumn18_V7b_MC_SF_AK4PFchs.jersf.txt.gz",
            "Summer19UL18_V5_MC_L1FastJet_AK4PFchs.jec.txt",
            "Summer19UL18_V5_MC_L2Relative_AK4PFchs.jec.txt",
            # "Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
            "RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL18_V5_MC_Uncertainty_AK4PFchs.junc.txt",
            "Summer19UL18_JRV2_MC_PtResolution_AK4PFchs.jr.txt",
            "Summer19UL18_JRV2_MC_SF_AK4PFchs.jersf.txt",
        ]
    ),
}

fatjet_factory = {
    "2016preVFPmc": jet_factory_factory(
        files=[
            # "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            # "Summer16_07Aug2017_V11_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Summer16_07Aug2017_V11_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            # "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            # "Summer16_25nsV1b_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            # "Summer16_25nsV1b_MC_SF_AK8PFPuppi.jersf.txt.gz",
            "Summer19UL16APV_V7_MC_L1FastJet_AK8PFPuppi.jec.txt",
            "Summer19UL16APV_V7_MC_L2Relative_AK8PFPuppi.jec.txt",
            # "Summer19UL16APV_V7_MC_UncertaintySources_AK8PFPuppi.junc.txt",
            "RegroupedV2_Summer19UL16APV_V7_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL16APV_V7_MC_Uncertainty_AK8PFPuppi.junc.txt",
            "Summer20UL16APV_JRV3_MC_PtResolution_AK8PFPuppi.jr.txt",
            "Summer20UL16APV_JRV3_MC_SF_AK8PFPuppi.jersf.txt",
        ]
    ),
    "2016postVFPmc": jet_factory_factory(
        files=[
            # "Summer16_07Aug2017_V11_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            # "Summer16_07Aug2017_V11_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Summer16_07Aug2017_V11_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            # "Summer16_07Aug2017_V11_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            # "Summer16_25nsV1b_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            # "Summer16_25nsV1b_MC_SF_AK8PFPuppi.jersf.txt.gz",
            "Summer19UL16_V7_MC_L1FastJet_AK8PFPuppi.jec.txt",
            "Summer19UL16_V7_MC_L2Relative_AK8PFPuppi.jec.txt",
            # "Summer19UL16_V7_MC_UncertaintySources_AK8PFPuppi.junc.txt",
            "RegroupedV2_Summer19UL16_V7_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL16_V7_MC_Uncertainty_AK8PFPuppi.junc.txt",
            "Summer20UL16_JRV3_MC_PtResolution_AK8PFPuppi.jr.txt",
            "Summer20UL16_JRV3_MC_SF_AK8PFPuppi.jersf.txt",
        ]
    ),
    "2017mc": jet_factory_factory(
        files=[
            # "Fall17_17Nov2017_V32_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            # "Fall17_17Nov2017_V32_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Fall17_17Nov2017_V32_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            # "Fall17_17Nov2017_V32_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            # "Fall17_V3b_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            # "Fall17_V3b_MC_SF_AK8PFPuppi.jersf.txt.gz",
            "Summer19UL17_V5_MC_L1FastJet_AK8PFPuppi.jec.txt",
            "Summer19UL17_V5_MC_L2Relative_AK8PFPuppi.jec.txt",
            # "Summer19UL17_V5_MC_UncertaintySources_AK8PFPuppi.junc.txt",
            "RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL17_V5_MC_Uncertainty_AK8PFPuppi.junc.txt",
            "Summer19UL17_JRV3_MC_PtResolution_AK8PFPuppi.jr.txt",
            "Summer19UL17_JRV3_MC_SF_AK8PFPuppi.jersf.txt",
        ]
    ),
    "2018mc": jet_factory_factory(
        files=[
            # "Autumn18_V19_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            # "Autumn18_V19_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Autumn18_V19_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            # "Autumn18_V19_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            # "Autumn18_V7b_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            # "Autumn18_V7b_MC_SF_AK8PFPuppi.jersf.txt.gz",
            "Summer19UL18_V5_MC_L1FastJet_AK8PFPuppi.jec.txt",
            "Summer19UL18_V5_MC_L2Relative_AK8PFPuppi.jec.txt",
            # "Summer19UL18_V5_MC_UncertaintySources_AK8PFPuppi.junc.txt",
            "RegroupedV2_Summer19UL18_V5_MC_UncertaintySources_AK4PFchs.junc.txt",  # added by Farouk
            "Summer19UL18_V5_MC_Uncertainty_AK8PFPuppi.junc.txt",
            "Summer19UL18_JRV2_MC_PtResolution_AK8PFPuppi.jr.txt",
            "Summer19UL18_JRV2_MC_SF_AK8PFPuppi.jersf.txt",
        ]
    ),
}

met_factory = CorrectedMETFactory(jec_name_map)

# TODO: build a "jet_met_factory" under a single function so that it's using the same set of random numbers.


if __name__ == "__main__":
    import argparse
    import gzip

    # jme stuff not pickleable in coffea
    import cloudpickle

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", default="jec_compiled.pkl.gz", type=str)
    args = parser.parse_args()

    with gzip.open(args.output, "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "fatjet_factory": fatjet_factory,
                "met_factory": met_factory,
            },
            fout,
        )
