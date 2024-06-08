"""
Loads the config from `config_make_stacked_hists.yaml`, and postprocesses
the condor output to make stacked histograms.

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


def make_events_dict(years, channels, samples_dir, samples, presel, add_THWW=True):
    """
    Postprocess the parquets by applying preselections, saving an `event_weight` column, and
    a tagger score column in a big concatenated dataframe.

    Args
        years [list]: years to postprocess and save in the output (e.g. ["2016APV", "2016"])
        channels [list]: channels to postprocess and save in the output (e.g. ["ele", "mu"])
        samples_dir [str]: points to the path of the parquets
        samples [list]: samples to postprocess and save in the output (e.g. ["HWW", "QCD", "Data"])
        presel [dict]: selections to apply per ch (e.g. `presel = {"ele": {"pt cut": fj_pt>250}}`)

    Returns
        a dict() object events_dict[year][channel][samples] that contains big dataframes of procesed events

    """

    events_dict = {}
    for year in years:
        events_dict[year] = {}

        for ch in channels:
            events_dict[year][ch] = {}

            # get lumi
            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir):

                # get a combined label to combine samples of the same process
                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                parquet_files = glob.glob(f"{samples_dir}/{sample}/outfiles/*_{ch}.parquet")
                pkl_files = glob.glob(f"{samples_dir}/{sample}/outfiles/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                if "met_fj_dphi" in data.keys():
                    data["abs_met_fj_dphi"] = np.abs(data["met_fj_dphi"])

                # get event_weight
                if sample_to_use != "Data":
                    xsecweight = utils.get_xsecweight(pkl_files, year, sample, False, luminosity)
                    data["event_weight"] = xsecweight * data[f"weight_{ch}"]
                else:
                    data["event_weight"] = np.ones_like(data["fj_pt"])

                if add_THWW:
                    # use hidNeurons to get the finetuned scores
                    data["THWW"] = utils.get_finetuned_score(data, modelv="v35_30")

                    # drop hidNeuron columns for memory purposes
                    data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                logging.info(f"Will fill the {sample_to_use} dataframe with the remaining {len(data)} events")
                logging.info(f"tot event weight {data['event_weight'].sum()} \n")

                # fill the big dataframe
                if sample_to_use not in events_dict[year][ch]:
                    events_dict[year][ch][sample_to_use] = data
                else:
                    events_dict[year][ch][sample_to_use] = pd.concat([events_dict[year][ch][sample_to_use], data])

    return events_dict


def make_hists_from_events_dict(events_dict, samples_to_plot, vars_to_plot, selections):
    """
    Takes an `events_dict` object that was processed by `make_events_dict` and starts filling histograms.

    Args
        events_dict [dict]: see output of `make_events_dict()`
        samples_to_plot [list]: which samples to use when plotting
        vars_to_plot [list]: which variables to plot

    Returns
        hist.Hist object

    """

    hists = {}
    for var in vars_to_plot:
        hists[var] = hist2.Hist(
            hist2.axis.StrCategory([], name="samples", growth=True),
            utils.axis_dict[var],
        )

        for sample in samples_to_plot:
            for year in events_dict:
                for ch in events_dict[year]:
                    df = events_dict[year][ch][sample]

                    for sel, value in selections[ch].items():
                        df = df.query(value)

                    hists[var].fill(samples=sample, var=df[var], weight=df["event_weight"])

    return hists


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    os.system(f"cp config_make_stacked_hists.yaml {args.outpath}/")

    # load config from yaml
    with open("config_make_stacked_hists.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.make_events_dict:
        events_dict = make_events_dict(
            years,
            channels,
            args.samples_dir,
            config["samples"],
            config["presel"],
        )
        with open(f"{args.outpath}/events_dict.pkl", "wb") as fp:
            pkl.dump(events_dict, fp)
    else:
        try:
            with open(f"{args.outpath}/events_dict.pkl", "rb") as fp:
                events_dict = pkl.load(fp)
        except FileNotFoundError:
            logging.info("Event dictionary not found. Run command with --make_events_dict option")
            exit()

    if args.plot_hists:
        PATH = args.outpath + f"stacked_hists_{args.tag}"
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        os.system(f"cp config_make_stacked_hists.yaml {PATH}/")

        logging.info("##### SELECTIONS")
        for ch in config["sel"]:
            logging.info(f"{ch} CHANNEL")
            for sel, value in config["sel"][ch].items():
                logging.info(f"{sel}: {value}")
            logging.info("-----------------------------")

        hists = make_hists_from_events_dict(events_dict, config["samples_to_plot"], config["vars_to_plot"], config["sel"])

        utils.plot_hists(
            years,
            channels,
            hists,
            config["vars_to_plot"],
            config["add_data"],
            config["logy"],
            config["add_soverb"],
            config["only_sig"],
            config["mult"],
            outpath=PATH,
        )


if __name__ == "__main__":
    # e.g.
    # python finetuned_make_stacked_hists.py --years 2017 --channels ele,mu --plot_hists --make_events_dict --tag v1
    # python finetuned_make_stacked_hists.py --years 2017 --channels ele,mu --plot_hists --tag v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas")
    parser.add_argument("--samples_dir", dest="samples_dir", default="../eos/Jul21_2017", help="path to parquets", type=str)
    parser.add_argument("--outpath", dest="outpath", default="hists/", help="path of the output", type=str)
    parser.add_argument("--tag", dest="tag", default="test/", help="path of the output", type=str)
    parser.add_argument("--make_events_dict", dest="make_events_dict", help="Make events dictionary", action="store_true")
    parser.add_argument("--plot_hists", dest="plot_hists", help="Plot histograms", action="store_true")

    args = parser.parse_args()

    main(args)
