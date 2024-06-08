"""
Builds hist.Hist templates after adding systematics for all samples.

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


def get_common_sample_name(sample):
    # first: check if the sample is in one of combine_samples_by_name
    sample_to_use = None
    for key in utils.combine_samples_by_name:
        if key in sample:
            sample_to_use = utils.combine_samples_by_name[key]
            break

    # second: if not, combine under common label
    if sample_to_use is None:
        for key in utils.combine_samples:
            if key in sample:
                sample_to_use = utils.combine_samples[key]
                break
            else:
                sample_to_use = sample
    return sample_to_use


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path, add_fake=False):
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
            "tagger>0.50": "THWW>0.50",
        },
        "ele": {
            "tagger>0.50": "THWW>0.50",
        },
    }

    mass_binning = 20

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            list(range(55, 255, mass_binning)),
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

                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                if sample_to_use == "Data":
                    is_data = True
                else:
                    is_data = False

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
                data["THWW"] = utils.get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = utils.get_xsecweight(
                    pkl_files, year, sample, sample_to_use, is_data, luminosity
                )

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    df = data.copy()

                    logging.info(f"Applying {region} selection on {len(data)} events")
                    df = df.query(region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                    # ------------------- Nominal -------------------
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * xsecweight

                        if "bjets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                            nominal *= df["weight_btag"]

                    ###################################
                    if sample_to_use == "EWKvjets":
                        threshold = 20
                        avg_good_weight = nominal[nominal < threshold].mean()
                        nominal[nominal > threshold] = avg_good_weight
                    ###################################

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=nominal,
                    )

                    # ------------------- Systematics correlated across both channels  -------------------

                    # ("key", "value"): the "key" is the systematic, the "value" is the samples to apply the systematic for
                    SYSTEMATICS_lep = {
                        "weight_pileup": utils.sigs + utils.bkgs,
                        "weight_pileupIDSF": utils.sigs + utils.bkgs,
                        "weight_L1Prefiring": utils.sigs + utils.bkgs,
                        # ggF & VBF
                        "weight_PSFSR": ["ggF", "VBF"],
                        "weight_PSISR": ["ggF", "VBF"],
                        # WJetsLNu & DY
                        "weight_d1kappa_EW": ["WJetsLNu", "DYJets"],
                        # WJetsLNu
                        "weight_d1K_NLO": ["WJetsLNu"],
                        "weight_d2K_NLO": ["WJetsLNu"],
                        "weight_d3K_NLO": ["WJetsLNu"],
                        "weight_W_d2kappa_EW": ["WJetsLNu"],
                        "weight_W_d3kappa_EW": ["WJetsLNu"],
                        # DY
                        "weight_Z_d2kappa_EW": ["DYJets"],
                        "weight_Z_d3kappa_EW": ["DYJets"],
                    }

                    for syst in SYSTEMATICS_lep:
                        if (year == "2018") and ("weight_L1Prefiring" in syst):
                            continue

                        if (is_data) | (sample_to_use not in SYSTEMATICS_lep[syst]):
                            shape_up = nominal
                            shape_down = nominal
                        else:
                            shape_up = df[f"{syst.replace('weight', f'weight_{ch}')}Up"] * xsecweight
                            shape_down = df[f"{syst.replace('weight', f'weight_{ch}')}Down"] * xsecweight

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_down,
                        )

                    # ------------------- Systematics uncorrelated across both channels  -------------------

                    # systematics for electron channel
                    SYSTEMATICS_ele = [
                        "weight_ele_isolation_electron",
                        "weight_ele_id_electron",
                        "weight_ele_reco_electron",
                        "weight_ele_trigger_electron",
                    ]
                    for syst in SYSTEMATICS_ele:

                        if (is_data) | (ch == "mu"):
                            shape_up = nominal
                            shape_down = nominal
                        else:
                            shape_up = df[f"{syst}Up"] * xsecweight
                            shape_down = df[f"{syst}Down"] * xsecweight

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_down,
                        )

                    # systematics for muon channel
                    SYSTEMATICS_mu = [
                        "weight_mu_isolation_muon",
                        "weight_mu_id_muon",
                        "weight_mu_trigger_iso_muon",
                        "weight_mu_trigger_noniso_muon",
                    ]

                    for syst in SYSTEMATICS_mu:

                        if (is_data) | (ch == "ele"):
                            shape_up = nominal
                            shape_down = nominal
                        else:
                            shape_up = df[f"{syst}Up"] * xsecweight
                            shape_down = df[f"{syst}Down"] * xsecweight

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_down,
                        )

                    # ------------------- PDF acceptance -------------------

                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values
                    - The last two values: alpha_s variations.
                    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
                    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
                    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
                    """
                    if sample_to_use in ["ggF", "VBF", "VH", "ZH"]:
                        pdfweights = []
                        for weight_i in sumpdfweights:

                            # noqa: get the normalization factor per variation i (ratio of sumpdfweights_i/sumgenweights)
                            R_i = sumpdfweights[weight_i] / sumgenweights

                            pdfweight = df[f"weight_pdf{weight_i}"].values * nominal / R_i
                            pdfweights.append(pdfweight)

                        pdfweights = np.swapaxes(np.array(pdfweights), 0, 1)  # so that the shape is (# events, variation)

                        abs_unc = np.linalg.norm((pdfweights - nominal.values.reshape(-1, 1)), axis=1)
                        # cap at 100% uncertainty
                        rel_unc = np.clip(abs_unc / nominal, 0, 1)
                        shape_up = nominal * (1 + rel_unc)
                        shape_down = nominal * (1 - rel_unc)

                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_pdf_acceptance_up",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_pdf_acceptance_down",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_down,
                    )

                    # ------------------- QCD scale -------------------

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
                    if sample_to_use in ["ggF", "VBF", "VH", "ZH", "WJetsLNu", "TTbar"]:

                        scaleweights = []
                        for weight_i in sumscaleweights:
                            if weight_i == 4:
                                continue

                            # noqa: get the normalization factor per variation i (ratio of sumscaleweights_i/sumgenweights)
                            R_i = sumscaleweights[weight_i] / sumgenweights
                            scaleweight_i = df[f"weight_scale{weight_i}"].values * nominal / R_i
                            scaleweight_i = scaleweight_i

                            scaleweights.append(scaleweight_i)

                        scaleweights = np.array(scaleweights)

                        scaleweights = np.swapaxes(
                            np.array(scaleweights), 0, 1
                        )  # so that the shape is (# events, variation)

                        shape_up = np.max(scaleweights, axis=1)
                        shape_down = np.min(scaleweights, axis=1)

                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_qcd_scale_up",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_qcd_scale_down",
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=shape_down,
                    )

                    # ------------------- btag syst. -------------------

                    # systematics correlated across all years
                    BTAG_systs_correlated = [
                        "weight_btagSFlightCorrelated",
                        "weight_btagSFbcCorrelated",
                    ]

                    for syst in BTAG_systs_correlated:
                        if is_data:
                            shape_up = nominal
                            shape_down = nominal
                        else:
                            shape_up = df[f"{syst}Up"] * nominal
                            shape_down = df[f"{syst}Down"] * nominal

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_down,
                        )

                    # systematics uncorrelated across all years and stored in the parquets per year
                    if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
                        yearlabel = "2016"
                    else:
                        yearlabel = year

                    BTAG_systs_uncorrelated = [
                        f"weight_btagSFlight{yearlabel}",
                        f"weight_btagSFbc{yearlabel}",
                    ]
                    for syst in BTAG_systs_uncorrelated:
                        if is_data:
                            shape_up = nominal
                            shape_down = nominal
                        else:
                            shape_up = df[f"{syst}Up"] * nominal
                            shape_down = df[f"{syst}Down"] * nominal

                        if year == "2016APV":
                            syst = syst.replace("2016", "2016APV")

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["rec_higgs_m"],
                            weight=shape_down,
                        )

                    # ------------------- JECs -------------------

                    # systematics correlated across all years
                    JEC_systs_correlated = [
                        "UES_up",
                        "UES_down",
                        # individual sources
                        "JES_FlavorQCD_up",
                        "JES_FlavorQCD_down",
                        "JES_RelativeBal_up",
                        "JES_RelativeBal_down",
                        "JES_HF_up",
                        "JES_HF_down",
                        "JES_BBEC1_up",
                        "JES_BBEC1_down",
                        "JES_EC2_up",
                        "JES_EC2_down",
                        "JES_Absolute_up",
                        "JES_Absolute_down",
                    ]
                    for syst in JEC_systs_correlated:
                        if is_data:
                            x = df["rec_higgs_m"]
                        else:
                            x = df[f"rec_higgs_m{syst}"]

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=syst,
                            Region=region,
                            mass_observable=x,
                            weight=nominal,
                        )

                    # systematics uncorrelated across all years and NOT stored in the parquets per year
                    JEC_systs_uncorrelated = [
                        "JER_up",
                        "JER_down",
                        "JMR_up",
                        "JMR_down",
                        "JMS_up",
                        "JMS_down",
                    ]
                    for syst in JEC_systs_uncorrelated:
                        if is_data:
                            x = df["rec_higgs_m"]
                        else:
                            x = df[f"rec_higgs_m{syst}"]

                        syst = syst.replace("_", f"_{year}_")

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=syst,
                            Region=region,
                            mass_observable=x,
                            weight=nominal,
                        )

                    # systematics uncorrelated across all years and stored in the parquets per year
                    if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
                        yearlabel = "2016"
                    else:
                        yearlabel = year

                    JEC_systs_uncorrelated = [
                        f"JES_BBEC1_{yearlabel}_up",
                        f"JES_BBEC1_{yearlabel}_down",
                        f"JES_RelativeSample_{yearlabel}_up",
                        f"JES_RelativeSample_{yearlabel}_down",
                        f"JES_EC2_{yearlabel}_up",
                        f"JES_EC2_{yearlabel}_down",
                        f"JES_HF_{yearlabel}_up",
                        f"JES_HF_{yearlabel}_down",
                        f"JES_Absolute_{yearlabel}_up",
                        f"JES_Absolute_{yearlabel}_down",
                    ]
                    for syst in JEC_systs_uncorrelated:
                        if is_data:
                            x = df["rec_higgs_m"]
                        else:
                            x = df[f"rec_higgs_m{syst}"]

                        if year == "2016APV":
                            syst = syst.replace("2016", "2016APV")

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=syst,
                            Region=region,
                            mass_observable=x,
                            weight=nominal,
                        )

    if add_fake:
        for year in years:
            data = pd.read_parquet(f"{samples_dir[year]}/fake_{year}_ele.parquet")

            # apply selection
            for selection in presel["ele"]:
                logging.info(f"Applying {selection} selection on {len(data)} events")
                data = data.query(presel["ele"][selection])

            for region in hists.axes["Region"]:
                df = data.copy()

                logging.info(f"Applying {region} selection on {len(data)} events")
                df = df.query(regions_sel[region])
                logging.info(f"Will fill the histograms with the remaining {len(data)} events")

                for syst in hists.axes["Systematic"]:
                    hists.fill(
                        Sample="Fake",
                        Systematic=syst,
                        Region=region,
                        mass_observable=df["rec_higgs_m"],
                        weight=df["event_weight"],
                    )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 1e-3
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 1e-3


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
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["model_path"],
        args.add_fake,
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1 --add-fake

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")
    parser.add_argument("--add-fake", dest="add_fake", action="store_true")

    args = parser.parse_args()

    main(args)
