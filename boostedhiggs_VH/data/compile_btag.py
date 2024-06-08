import awkward as ak
import hist
import numpy as np
from coffea import util as cutil
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

btagWPs = {
    "deepJet": {
        "2016preVFP_UL": {
            "L": 0.0508,
            "M": 0.2598,
            "T": 0.6502,
        },
        "2016postVFP_UL": {
            "L": 0.0480,
            "M": 0.2489,
            "T": 0.6377,
        },
        "2017_UL": {
            "L": 0.0532,
            "M": 0.3040,
            "T": 0.7476,
        },
        "2018_UL": {
            "L": 0.0490,
            "M": 0.2783,
            "T": 0.7100,
        },
    },
    "deepCSV": {
        "2016preVFP_UL": {
            "L": 0.2027,
            "M": 0.6001,
            "T": 0.8819,
        },
        "2016postVFP_UL": {
            "L": 0.1918,
            "M": 0.5847,
            "T": 0.8767,
        },
        "2017_UL": {
            "L": 0.1355,
            "M": 0.4506,
            "T": 0.7738,
        },
        "2018_UL": {
            "L": 0.1208,
            "M": 0.4168,
            "T": 0.7665,
        },
    },
}

# single TT files to derive efficiency
tt_files = {
    "2016preVFP_UL": "/store/user/lpcpfnano/cmantill/v2_3/2016APV/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic/220808_173625/0000/nano_mc2016pre_3-146.root",
    "2016postVFP_UL": "/store/user/lpcpfnano/cmantill/v2_3/2016/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic/220808_181840/0000/nano_mc2016post_3-30.root",
    "2017_UL": "/store/user/lpcpfnano/rkansal/v2_3/2017/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic/220705_160227/0000/nano_mc2017_227.root",
    "2018_UL": "/store/user/lpcpfnano/cmantill/v2_3/2018/TTbar/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic/220808_151244/0000/nano_mc2018_2-15.root",
}

taggerBranch = {"deepJet": "btagDeepFlavB", "deepCSV": "btagDeepB"}

for year, fname in tt_files.items():
    events = NanoEventsFactory.from_root(
        f"root://cmsxrootd-site.fnal.gov/{fname}",
        entry_stop=100_000,
        schemaclass=NanoAODSchema,
    ).events()

    # b-tagging only applied for jets with |eta| < 2.5
    phasespace_cuts = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.5)
    jets = ak.flatten(events.Jet[phasespace_cuts])

    for tag in ["deepJet", "deepCSV"]:
        for wp in ["L", "M", "T"]:
            efficiencyinfo = (
                hist.Hist.new.Reg(20, 40, 300, name="pt")
                .Reg(4, 0, 2.5, name="abseta")
                .IntCat([0, 4, 5], name="flavor")
                .Bool(name="passWP")
                .Double()
                .fill(
                    pt=jets.pt,
                    abseta=abs(jets.eta),
                    flavor=jets.hadronFlavour,
                    passWP=jets[taggerBranch[tag]] > btagWPs[tag][year][wp],
                )
            )
            eff = efficiencyinfo[{"passWP": True}] / efficiencyinfo[{"passWP": sum}]
            efflookup = dense_lookup(eff.values(), [ax.edges for ax in eff.axes])
            print(tag, wp, efflookup(np.array([42, 60]), 0.2, 2))

            cutil.save(efflookup, f"btageff_{tag}_{wp}_{year}.coffea")
