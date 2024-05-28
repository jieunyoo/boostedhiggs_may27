"""
Builds hist.Hist templates after adding systematics for all samples

Author: Farouk Mokhtar

JY: altered to use V Mass, not use ralpha, alter to read in QCD estimate from new files postprocessed with nonprompt_event_weight
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings
import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import utils
import yaml

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

weights = {
    "weight_pileup": {"mu": "weight_mu_pileup", "ele": "weight_ele_pileup"},
    "weight_isolation": {"mu": "weight_mu_isolation_muon", "ele": "weight_ele_isolation_electron"},
    "weight_id": {"mu": "weight_mu_id_muon", "ele": "weight_ele_id_electron"},
    "weight_reco_ele": {"mu": "", "ele": "weight_ele_reco_electron"},
    "weight_L1Prefiring": {"mu": "weight_mu_L1Prefiring", "ele": "weight_ele_L1Prefiring"},
    "weight_trigger_ele": {"mu": "", "ele": "weight_ele_trigger_electron"},
    "weight_trigger_iso_mu": {"mu": "weight_mu_trigger_iso_muon", "ele": ""},
    "weight_trigger_noniso_mu": {"mu": "weight_mu_trigger_noniso_muon", "ele": ""},
    # ggF & VBF
    "weight_aS_weight": {"mu": "weight_mu_aS_weight", "ele": "weight_ele_aS_weight"},
    "weight_UEPS_FSR": {"mu": "weight_mu_UEPS_FSR", "ele": "weight_ele_UEPS_FSR"},
    "weight_UEPS_ISR": {"mu": "weight_mu_UEPS_ISR", "ele": "weight_ele_UEPS_ISR"},
    "weight_PDF_weight": {"mu": "weight_mu_PDF_weight", "ele": "weight_ele_PDF_weight"},
    "weight_PDFaS_weight": {"mu": "weight_mu_PDFaS_weight", "ele": "weight_ele_PDFaS_weight"},
    "weight_scalevar_3pt": {"mu": "weight_mu_scalevar_3pt", "ele": "weight_ele_scalevar_3pt"},
    "weight_scalevar_7pt": {"mu": "weight_mu_scalevar_7pt", "ele": "weight_ele_scalevar_7pt"},
    # WJetsLNu & DY
    "weight_d1kappa_EW": {"mu": "weight_mu_d1kappa_EW", "ele": "weight_ele_d1kappa_EW"},
    # WJetsLNu
    "weight_d1K_NLO": {"mu": "weight_mu_d1K_NLO", "ele": "weight_ele_d1K_NLO"},
    "weight_d2K_NLO": {"mu": "weight_mu_d2K_NLO", "ele": "weight_ele_d2K_NLO"},
    "weight_d3K_NLO": {"mu": "weight_mu_d3K_NLO", "ele": "weight_ele_d3K_NLO"},
    "weight_W_d2kappa_EW": {"mu": "weight_mu_W_d2kappa_EW", "ele": "weight_ele_W_d2kappa_EW"},
    "weight_W_d3kappa_EW": {"mu": "weight_mu_W_d3kappa_EW", "ele": "weight_ele_W_d3kappa_EW"},
    # DY
    "weight_Z_d2kappa_EW": {"mu": "weight_mu_Z_d2kappa_EW", "ele": "weight_ele_Z_d2kappa_EW"},
    "weight_Z_d3kappa_EW": {"mu": "weight_mu_Z_d3kappa_EW", "ele": "weight_ele_Z_d3kappa_EW"},
}


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path):
#def get_templates(years, channels, samples, samples_dir, regions_sel):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "QCD", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (fj_ParT_score>0.97)}`)
        model_path [str]: path to the ParT finetuned model.onnx

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (samples, systematic, ptbin, mass_observable)

    """

    # add filter to load parquets faster  (only if needed)... we only apply lepiso anyway
    lepiso_filter = {
        "lepiso": {
            "mu": [
                [("lep_pt", "<", 55), (("lep_isolation", "<", 0.15))],
                [("lep_misolation", "<", 0.2), ("lep_pt", ">=", 55)],
            ],
            "ele": [
                [("lep_pt", "<", 120), (("lep_isolation", "<", 0.15))],
                [("lep_pt", ">=", 120)],
            ],
        },
        "lepisoinv": {
            "mu": [
                [("lep_pt", "<", 55), (("lep_isolation", ">", 0.15))],
                [("lep_misolation", ">", 0.2), ("lep_pt", ">=", 55)],
            ],
            "ele": [
                [("lep_pt", "<", 120), (("lep_isolation", ">", 0.15))],
                [("lep_pt", ">=", 120)],
            ],
        },
    }

    # add extra selections to preselection
    presel = {
        "mu": {
            "lep_fj_dr003": "( ( lep_fj_dr>0.03) )",
            "lep_fj_dr08": "( ( lep_fj_dr<0.8) )",
        },
        "ele": {
            "lep_fj_dr003": "( ( lep_fj_dr>0.03) )",
            "lep_fj_dr08": "( ( lep_fj_dr<0.8) )",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable( list(range(20,220 , mass_binning)),
            name="mass_observable", label=r"V reconstructed soft drop mass [GeV]", overflow=True, ), storage=hist2.storage.Weight(),
    )

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]
            for sample in os.listdir(samples_dir[year]):
                # if sample == "QCD_Pt_170to300":
                #     print(f"Skipping sample {sample}")
                #     continue
                for key in utils.combine_samples:
                    if key in sample:
                        sample_to_use = utils.combine_samples[key]
                        break
                    else:
                        sample_to_use = sample
                if sample_to_use not in samples:
                    continue
                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")
                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                if "postprocess" in samples_dir[year]:
                    parquet_files = glob.glob(f"{out_files}/{ch}.parquet")
                else:
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")
                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue
                try:
                    #data = pd.read_parquet(parquet_files, filters=lepiso_filter["lepiso"][ch])
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["fj_ParT_score_finetuned"] = utils.get_finetuned_score(data, model_path)
                #print('fjpart t', data["fj_ParT_score_finetuned"])

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                data["abs_met_fj_dphi"] = np.abs(data["met_fj_dphi"])  # relevant variable

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                # get event_weight
                #if sample_to_use != "Data":
                if (sample_to_use != "Data") and (sample_to_use!= 'QCD'): #don't do this for QCD since QCD is data-driven
                    event_weight = utils.get_xsecweight(pkl_files, year, sample, False, luminosity)   #this is only for MC, this is where xsec weight is applied
                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    df = data.copy()
                    print('region', region)
                    logging.info(f"Applying {region} selection on {len(data)} events")
                    df = data.copy()
                    df = df.query(region_sel)
                    print('region_sel', region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")
                #REAL data***************************************************************************************************************************
                    if sample_to_use == "Data":  # for data (fill as 1)
                        hists.fill( Sample=sample_to_use, Systematic="nominal", Region=region, mass_observable=df["ReconV_SoftDropMass"],
                            weight=np.ones_like(df["fj_pt"]), #filling it with ones
                        )
                    if sample_to_use == "QCD":  # for data (fill as 1)
                        nominal = df[f"nonprompt_event_weight_{ch}"]  #assumes pre-processed QCD samples that have this new variable called nonprompt_event_weight
                        hists.fill(Sample=sample_to_use, Systematic="nominal", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=nominal,
                        ) 
                    else: #for MC are multiplying each weight of channel with event weight
                        nominal = df[f"weight_{ch}"] * event_weight
                        hists.fill(Sample=sample_to_use, Systematic="nominal", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=nominal,
                        )
                    #now take care of weights for data and MC
                    for weight in weights:     # up and down weights
                        if sample_to_use == "Data":  # for data (fill as 1)
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Up", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=np.ones_like(df["fj_pt"]),)
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Down", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=np.ones_like(df["fj_pt"]),)
                        if sample_to_use == "QCD":  # (fill as 1)
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Up", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=np.ones_like(df["fj_pt"]),)
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Down", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=np.ones_like(df["fj_pt"]),)
                        else: #MC***************************************************************************************************************************
                            try:
                                syst = df[f"{weights[weight][ch]}Up"] * event_weight
                            except KeyError:
                                syst = nominal
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Up", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=syst,)
                            try:
                                syst = df[f"{weights[weight][ch]}Down"] * event_weight
                            except KeyError:
                                syst = nominal
                            hists.fill( Sample=sample_to_use, Systematic=f"{weight}Down", Region=region, mass_observable=df["ReconV_SoftDropMass"], weight=syst,)
    logging.info(hists)
    return hists


def fix_neg_yields(h):
    """ Will set the bin yields of a process to 0 if the nominal yield is negative, and will set the yield to 0 for the full Systematic axis.   """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 0
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years, channels, config["samples"], config["samples_dir"], config["regions_sel"], config["model_path"]
        #years, channels, config["samples"], config["samples_dir"], config["regions_sel"]
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
