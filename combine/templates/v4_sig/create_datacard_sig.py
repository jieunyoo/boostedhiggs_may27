"""
Creates "combine datacards" using hist.Hist templates.

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py

Same as `create_datacard.py` except it fills the systematics with None.
"""

from __future__ import division, print_function

import argparse
import logging
import os
import warnings

import pandas as pd
import rhalphalib as rl
from utils import get_template, labels, load_templates, samples, sigs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


def create_datacard(hists_templates, add_ttbar_constraint=True, add_wjets_constraint=True):

    # define the model
    model = rl.Model("testModel")

    # define the signal and control regions
    SIG_regions = list(hists_templates.axes["Region"])

    SIG_regions.remove("TopCR")
    SIG_regions.remove("WJetsCR")

    CONTROL_regions = ["TopCR", "WJetsCR"]

    if add_ttbar_constraint:
        ttbarnormSF = rl.IndependentParameter("ttbarnormSF", 1.0, 0, 10)

    if add_wjets_constraint:
        wjetsnormSF = rl.IndependentParameter("wjetsnormSF", 1.0, 0, 10)

    # fill datacard with systematics and rates
    for ChName in SIG_regions + CONTROL_regions:
        Samples = samples.copy()

        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in Samples:

            templ = get_template(hists_templates, sName, ChName)
            if templ == 0:
                continue
            stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

            if "CR" in ChName:
                sample.autoMCStats(lnN=True)

            ch.addSample(sample)

        # add data
        data_obs = get_template(hists_templates, "Data", ChName)
        ch.setObservation(data_obs)

        if "CR" not in ChName:
            ch.autoMCStats()

    if add_ttbar_constraint:
        failCh = model["TopCR"]

        ttbarfail = failCh["ttbar"]
        ttbarfail.setParamEffect(ttbarnormSF, 1 * ttbarnormSF)

        for sig_region in SIG_regions:

            passCh = model[sig_region]

            ttbarpass = passCh["ttbar"]
            ttbarpass.setParamEffect(ttbarnormSF, 1 * ttbarnormSF)

    if add_wjets_constraint:
        failCh = model["WJetsCR"]

        wjetsfail = failCh["wjets"]
        wjetsfail.setParamEffect(wjetsnormSF, 1 * wjetsnormSF)

        for sig_region in SIG_regions:
            passCh = model[sig_region]

            wjetspass = passCh["wjets"]
            wjetspass.setParamEffect(wjetsnormSF, 1 * wjetsnormSF)

    return model


def main(args):
    years = args.years.split(",")
    lep_channels = args.channels.split(",")

    hists_templates = load_templates(years, lep_channels, args.outdir)

    model = create_datacard(hists_templates)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard_sig.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", default="templates/test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
