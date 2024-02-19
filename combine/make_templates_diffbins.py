"""
Builds hist.Hist templates after adding systematics for all samples

Author: Farouk Mokhtar
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


# ("key", "value"): the "key" is the common naming (to commonalize over both channels)
weights = {
    # common for all samples
    "weight_btagSFlightCorrelated": {"mu": "weight_btagSFlightCorrelated", "ele": "weight_btagSFlightCorrelated"},
    "weight_btagSFbcCorrelated": {"mu": "weight_btagSFbcCorrelated", "ele": "weight_btagSFbcCorrelated"},
    "weight_btagSFlight2016": {"mu": "weight_btagSFlight2016", "ele": "weight_btagSFlight2016"},
    "weight_btagSFbc2016": {"mu": "weight_btagSFbc2016", "ele": "weight_btagSFbc2016"},
    "weight_btagSFlight2016APV": {"mu": "weight_btagSFlight2016APV", "ele": "weight_btagSFlight2016APV"},
    "weight_btagSFbc2016APV": {"mu": "weight_btagSFbc2016APV", "ele": "weight_btagSFbc2016APV"},
    "weight_btagSFlight2017": {"mu": "weight_btagSFlight2017", "ele": "weight_btagSFlight2017"},
    "weight_btagSFbc2017": {"mu": "weight_btagSFbc2017", "ele": "weight_btagSFbc2017"},
    "weight_btagSFlight2018": {"mu": "weight_btagSFlight2018", "ele": "weight_btagSFlight2018"},
    "weight_btagSFbc2018": {"mu": "weight_btagSFbc2018", "ele": "weight_btagSFbc2018"},
    "weight_pileup": {"mu": "weight_mu_pileup", "ele": "weight_ele_pileup"},
    "weight_pileupIDSF": {"mu": "weight_mu_pileupIDSFDown", "ele": "weight_ele_pileupIDSFDown"},
    "weight_isolation": {"mu": "weight_mu_isolation_muon", "ele": "weight_ele_isolation_electron"},
    "weight_id": {"mu": "weight_mu_id_muon", "ele": "weight_ele_id_electron"},
    "weight_reco_ele": {"mu": "", "ele": "weight_ele_reco_electron"},
    "weight_L1Prefiring": {"mu": "weight_mu_L1Prefiring", "ele": "weight_ele_L1Prefiring"},
    "weight_trigger_ele": {"mu": "", "ele": "weight_ele_trigger_electron"},
    "weight_trigger_iso_mu": {"mu": "weight_mu_trigger_iso_muon", "ele": ""},
    "weight_trigger_noniso_mu": {"mu": "weight_mu_trigger_noniso_muon", "ele": ""},
    # ggF & VBF
    "weight_PSFSR": {"mu": "weight_mu_PSFSR", "ele": "weight_ele_PSFSR_weight"},
    "weight_PSISR": {"mu": "weight_mu_PSISR", "ele": "weight_ele_PSISR_weight"},
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

AK8_systs = [
    "rec_higgs_mUES_up",
    "rec_higgs_mUES_down",
    "rec_higgs_mJES_up",
    "rec_higgs_mJES_down",
    "rec_higgs_mJER_up",
    "rec_higgs_mJER_down",
    # these
    "rec_higgs_mJMS_up",
    "rec_higgs_mJMS_down",
    "rec_higgs_mJMR_up",
    "rec_higgs_mJMR_down",
]

# shape_weights = {
#     # "fj_pt": [
#     #     "fj_ptJES_up",
#     #     "fj_ptJES_down",
#     #     "fj_ptJER_up",
#     #     "fj_ptJER_down",
#     # ],
#     # "fj_mass": [
#     #     "fj_massJMS_up",
#     #     "fj_massJMS_down",
#     #     "fj_massJMR_up",
#     #     "fj_massJMR_down",
#     # ],
#     # "mjj": [
#     #     "mjjJES_up",
#     #     "mjjJES_down",
#     #     "mjjJER_up",
#     #     "mjjJER_down",
#     # ],
#     "rec_higgs_m": [
#         "rec_higgs_mUES_up",
#         "rec_higgs_mUES_down",
#         "rec_higgs_mJES_up",
#         "rec_higgs_mJES_down",
#         "rec_higgs_mJER_up",
#         "rec_higgs_mJER_down",


#         "rec_higgs_mJMS_up",
#         "rec_higgs_mJMS_down",
#         "rec_higgs_mJMR_up",
#         "rec_higgs_mJMR_down",
#     ],
#     # "rec_higgs_pt": [
#     #     "rec_higgs_ptUES_up",
#     #     "rec_higgs_ptUES_down",
#     #     "rec_higgs_ptJES_up",
#     #     "rec_higgs_ptJES_down",
#     #     "rec_higgs_ptJER_up",
#     #     "rec_higgs_ptJER_down",
#     #     "rec_higgs_ptJMS_up",
#     #     "rec_higgs_ptJMS_down",
#     #     "rec_higgs_ptJMR_up",
#     #     "rec_higgs_ptJMR_down",
#     # ],
# }


def get_templates(years, channels, samples, samples_dir, regions_sel, regions_massbins, model_path):
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

    # add extra selections to preselection
    presel = {
        "mu": {
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
        },
        "ele": {
            "tagger>0.5": "fj_ParT_score_finetuned>0.5",
        },
    }

    hists = {}
    for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
        hists[region] = hist2.Hist(
            hist2.axis.StrCategory([], name="Sample", growth=True),
            hist2.axis.StrCategory([], name="Systematic", growth=True),
            hist2.axis.Variable(
                list(range(50, 240, regions_massbins[region])),
                name="mass_observable",
                label=r"Higgs reconstructed mass [GeV]",
                overflow=True,
            ),
            storage=hist2.storage.Weight(),
        )

        for year in years:  # e.g. 2018, 2017, 2016APV, 2016
            for ch in channels:  # e.g. mu, ele
                logging.info(f"Processing year {year} and {ch} channel")

                with open("../fileset/luminosity.json") as f:
                    luminosity = json.load(f)[ch][year]

                for sample in os.listdir(samples_dir[year]):

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
                    parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                    pkl_files = glob.glob(f"{out_files}/*.pkl")

                    if not parquet_files:
                        logging.info(f"No parquet file for {sample}")
                        continue

                    try:
                        data = pd.read_parquet(parquet_files)
                    except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                        continue

                    if len(data) == 0:
                        continue

                    # use hidNeurons to get the finetuned scores
                    data["fj_ParT_score_finetuned"] = utils.get_finetuned_score(data, model_path)

                    # drop hidNeurons which are not needed anymore
                    data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                    # apply selection
                    for selection in presel[ch]:
                        logging.info(f"Applying {selection} selection on {len(data)} events")
                        data = data.query(presel[ch][selection])

                    # get event_weight
                    if sample_to_use == "Data":
                        is_data = True
                    else:
                        is_data = False

                    event_weight = utils.get_xsecweight(pkl_files, year, sample, is_data, luminosity)

                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(data)} events")

                    df = df.query(region_sel)

                    if not is_data:
                        W = df[f"weight_{ch}"]

                        if "bjets" in region_sel:  # add btag SF
                            W *= df["weight_btag"]

                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                    # add nominal weight
                    if is_data:  # for data (nominal is 1)
                        nominal = np.ones_like(df["fj_pt"])
                    else:
                        nominal = W * event_weight
                    hists[region].fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

                    # add up/down variations
                    for weight in weights:

                        if is_data:  # for data (fill as 1 for up and down variations)
                            w = nominal

                        # retrieve UP variations for MC
                        if not is_data:
                            try:
                                w = df[f"{weights[weight][ch]}Up"] * event_weight
                                if "btag" in weight:
                                    w *= W
                            except KeyError:
                                w = nominal

                        hists[region].fill(
                            Sample=sample_to_use,
                            Systematic=f"{weight}_up",
                            mass_observable=df["rec_higgs_m"],
                            weight=w,
                        )

                        # retrieve DOWN variations for MC
                        if not is_data:
                            try:
                                w = df[f"{weights[weight][ch]}Down"] * event_weight
                                if "btag" in weight:
                                    w *= W
                            except KeyError:
                                w = nominal

                        hists[region].fill(
                            Sample=sample_to_use,
                            Systematic=f"{weight}_down",
                            mass_observable=df["rec_higgs_m"],
                            weight=w,
                        )

                    for rec_higgs_m_variation in AK8_systs:

                        if is_data:
                            x = "rec_higgs_m"
                        else:
                            x = rec_higgs_m_variation

                        hists[region].fill(
                            Sample=sample_to_use,
                            Systematic=rec_higgs_m_variation,
                            mass_observable=df[x],
                            weight=nominal,
                        )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h:
        for sample in h[region].axes["Sample"]:
            neg_bins = np.where(h[region][{"Sample": sample, "Systematic": "nominal"}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h[region].axes["Sample"]) == sample)

                for neg_bin in neg_bins:
                    h[region].view(flow=True)[sample_index, :, neg_bin + 1].value = 0
                    h[region].view(flow=True)[sample_index, :, neg_bin + 1].variance = 0


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates_diffbins.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["regions_massbins"],
        config["model_path"],
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates_diffbins.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v5

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")

    args = parser.parse_args()

    main(args)
