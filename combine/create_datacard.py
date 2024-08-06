"""
Creates "combine datacards" using hist.Hist templates.

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
"""

from __future__ import division, print_function

import argparse
import logging
import math
import os
import warnings

import pandas as pd
import rhalphalib as rl
from systematics import systs_from_parquets, systs_not_from_parquets
from utils import get_template, labels, load_templates, samples, shape_to_num, sigs, bkgs

rl.ParametericSample.PreferRooParametricHist = True
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

CMS_PARAMS_LABEL = "CMS_HWW_boosted"


#def create_datacard(hists_templates, years, lep_channels, add_ttbar_constraint=True, add_wjets_constraint=False):
def create_datacard(hists_templates, years, lep_channels, add_ttbar_constraint=True, add_wjets_constraint=True):
    # define the systematics
    systs_dict, systs_dict_values = systs_not_from_parquets(years, lep_channels)
    sys_from_parquets = systs_from_parquets(years)

    # define the model
    model = rl.Model("testModel")

    # define the signal and control regions ********this is different from Faroku
    SIG_regions = ["SR1"]
    CONTROL_regions = ["TopCR", "WJetsCR"]

    if add_ttbar_constraint:
        ttbarnormSF = rl.IndependentParameter("ttbarnormSF", 1.0, 0, 10)

    if add_wjets_constraint:
        wjetsnormSF = rl.IndependentParameter("wjetsnormSF", 1.0, 0, 10)

    # fill datacard with systematics and rates
    for ChName in SIG_regions + CONTROL_regions:
        
        ch = rl.Channel(ChName)
        model.addChannel(ch)

        for sName in sigs + bkgs:
            print('sName', sName)

            if (sName in sigs) and (ChName in CONTROL_regions):
                continue

            templ = get_template(hists_templates, sName, ChName)
            print('templ', templ)
            #if templ == 0:
             #   continue
            stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + labels[sName], stype, templ)

            #if "CR" in ChName:
             #   sample.autoMCStats(lnN=True)

            #sample.autoMCStats(lnN=True)
            #sample.autoMCStats()

            # SYSTEMATICS NOT FROM PARQUETS
            for syst_on_sample in ["all_samples", sName]:  # apply common systs and per sample systs
                for sys_name, sys_value in systs_dict[syst_on_sample].items():
                    if systs_dict_values[syst_on_sample][sys_name][1] is None:  # if up and down are the same
                        sample.setParamEffect(sys_value, systs_dict_values[syst_on_sample][sys_name][0])
                    else:
                        sample.setParamEffect(
                            sys_value,
                            systs_dict_values[syst_on_sample][sys_name][0],
                            systs_dict_values[syst_on_sample][sys_name][1],
                        )

            # SYSTEMATICS FROM PARQUETS
            for sys_value, (sys_name, list_of_samples) in sys_from_parquets.items():
                #print('sys_value', sys_value)
                if sName in list_of_samples:
                    syst_up = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": sys_name + "_up"}].values()
                    syst_do = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": sys_name + "_down"}].values()
                    nominal = hists_templates[{"Sample": sName, "Region": ChName, "Systematic": "nominal"}].values()

                    if sys_value.combinePrior == "lnN":
                        eff_up = shape_to_num(syst_up, nominal)
                        eff_do = shape_to_num(syst_do, nominal)

                        if math.isclose(eff_up, eff_do, rel_tol=1e-2):  # if up and down are the same
                            sample.setParamEffect(sys_value, max(eff_up, eff_do))
                        else:
                            sample.setParamEffect(sys_value, max(eff_up, eff_do), min(eff_up, eff_do))

                    else:
                        nominal[nominal == 0] = 1  # to avoid invalid value encountered in true_divide in "syst_up/nominal"
                        sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))


            ch.addSample(sample)

#FAKES templates ***********
   #     sName = "Fake"
   #     templ = get_template(hists_templates, sName, ChName)
   #     sample = rl.TemplateSample(ch.name + "_" + labels[sName], rl.Sample.BACKGROUND, templ)

   #     sample.setParamEffect(rl.NuisanceParameter("Fake_rate_unc", "lnN"), 1.6)

   #     for sys_name in ["fakes_SF"]:
   #         sys_value = rl.NuisanceParameter(sys_name, "shape")
   #         syst_up = hists_templates[{"Sample": "Fake", "Region": ChName, "Systematic": sys_name + "_Up"}].values()
#            syst_do = hists_templates[{"Sample": "Fake", "Region": ChName, "Systematic": sys_name + "_Down"}].values()
#            nominal = hists_templates[{"Sample": "Fake", "Region": ChName, "Systematic": "nominal"}].values()

 #           nominal[nominal == 0] = 1  # to avoid invalid value encountered in true_divide in "syst_up/nominal"
 #           sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))



        #ch.addSample(sample)

#end fakes*********************
        # add data
        data_obs = get_template(hists_templates, "Data", ChName)
        ch.setObservation(data_obs)

        ch.autoMCStats(channel_name=f"{CMS_PARAMS_LABEL}_{ChName}",)
        #ch.autoMCStats(channel_name=f"{CMS_PARAMS_LABEL}_{ChName}",lnN=True)

        #if "CR" not in ChName:
         #   ch.autoMCStats()

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

    model = create_datacard(hists_templates, years, lep_channels)

    model.renderCombine(os.path.join(str("{}".format(args.outdir)), "datacards"))


if __name__ == "__main__":
    # e.g.
    # python create_datacard.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", default="templates/test", type=str, help="name of template directory")

    args = parser.parse_args()

    main(args)
