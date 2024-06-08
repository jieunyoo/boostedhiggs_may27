"""
Methods for deriving input variables for the tagger.
Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan, Farouk Mokhtar
"""

# import json
from typing import Dict

import awkward as ak
import numpy as np
import numpy.ma as ma
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods.base import NanoEventsArray


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


def get_pfcands_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    fj_idx_lep,
    fatjet_label: str = "FatJet",
    pfcands_label: str = "FatJetPFCands",
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extracts the pf_candidate features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    jet = ak.firsts(preselected_events[fatjet_label][fj_idx_lep])

    msk = preselected_events[pfcands_label].jetIdx == ak.firsts(fj_idx_lep)
    jet_ak_pfcands = preselected_events[pfcands_label][msk]
    jet_pfcands = preselected_events.PFCands[jet_ak_pfcands.pFCandsIdx]

    # sort them by pt
    pfcand_sort = ak.argsort(jet_pfcands.pt, ascending=False)
    jet_pfcands = jet_pfcands[pfcand_sort]

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.ones_like(jet_pfcands.eta)
    eta_sign = eta_sign * (ak.values_astype(jet.eta > 0, int) * 2 - 1)
    feature_dict["pfcand_etarel"] = eta_sign * (jet_pfcands.eta - jet.eta)
    feature_dict["pfcand_phirel"] = jet.delta_phi(jet_pfcands)
    feature_dict["pfcand_abseta"] = np.abs(jet_pfcands.eta)

    feature_dict["pfcand_pt_log_nopuppi"] = np.log(jet_pfcands.pt)
    feature_dict["pfcand_e_log_nopuppi"] = np.log(jet_pfcands.energy)

    pdgIds = jet_pfcands.pdgId
    feature_dict["pfcand_isEl"] = np.abs(pdgIds) == 11
    feature_dict["pfcand_isMu"] = np.abs(pdgIds) == 13
    feature_dict["pfcand_isChargedHad"] = np.abs(pdgIds) == 211
    feature_dict["pfcand_isGamma"] = np.abs(pdgIds) == 22
    feature_dict["pfcand_isNeutralHad"] = np.abs(pdgIds) == 130

    feature_dict["pfcand_charge"] = jet_pfcands.charge
    feature_dict["pfcand_VTX_ass"] = jet_pfcands.pvAssocQuality
    feature_dict["pfcand_lostInnerHits"] = jet_pfcands.lostInnerHits
    feature_dict["pfcand_quality"] = jet_pfcands.trkQuality

    feature_dict["pfcand_normchi2"] = np.floor(jet_pfcands.trkChi2)

    if "Cdz" in jet_ak_pfcands.fields:
        feature_dict["pfcand_dz"] = jet_ak_pfcands["Cdz"][pfcand_sort]
        feature_dict["pfcand_dxy"] = jet_ak_pfcands["Cdxy"][pfcand_sort]
        feature_dict["pfcand_dzsig"] = jet_ak_pfcands["Cdzsig"][pfcand_sort]
        feature_dict["pfcand_dxysig"] = jet_ak_pfcands["Cdxysig"][pfcand_sort]
    else:
        # this is for old PFNano (<= v2.3)
        feature_dict["pfcand_dz"] = jet_pfcands.dz
        feature_dict["pfcand_dxy"] = jet_pfcands.d0
        feature_dict["pfcand_dzsig"] = jet_pfcands.dz / jet_pfcands.dzErr
        feature_dict["pfcand_dxysig"] = jet_pfcands.d0 / jet_pfcands.d0Err

    feature_dict["pfcand_px"] = jet_pfcands.px
    feature_dict["pfcand_py"] = jet_pfcands.py
    feature_dict["pfcand_pz"] = jet_pfcands.pz
    feature_dict["pfcand_energy"] = jet_pfcands.energy

    # work-around to match PKU
    # https://github.com/colizz/DNNTuples/blob/dev-UL-hww/Ntupler/plugins/CustomDeepBoostedJetTagInfoProducer.cc#L486-L496
    # jet_pfcands_p4 = ak.zip(
    #     {
    #         "pt": jet_pfcands.pt,
    #         "eta": jet_pfcands.eta,
    #         "phi": jet_pfcands.delta_phi(jet),
    #         "energy": jet_pfcands.energy,
    #     },
    #     with_name="PtEtaPhiELorentzVector",
    # )
    # feature_dict["pfcand_px"] = jet_pfcands_p4.px
    # feature_dict["pfcand_py"] = jet_pfcands_p4.py
    # feature_dict["pfcand_pz"] = jet_pfcands_p4.pz
    # feature_dict["pfcand_energy"] = jet_pfcands_p4.energy

    # btag vars
    for var in tagger_vars["pf_features"]["var_names"]:
        if "btag" in var:
            feature_dict[var] = jet_ak_pfcands[var[len("pfcand_") :]][pfcand_sort]

    # pfcand mask
    feature_dict["pfcand_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["pfcand_abseta"],
                    tagger_vars["pf_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)

    # if no padding is needed, mask will = 1.0
    if isinstance(feature_dict["pfcand_mask"], np.float32):
        feature_dict["pfcand_mask"] = np.ones(
            (
                len(feature_dict["pfcand_abseta"]),
                tagger_vars["pf_features"]["var_length"],
            )
        ).astype(np.float32)

    repl_values_dict = {
        "pfcand_normchi2": [-1, 999],
        "pfcand_dz": [-1, 0],
        "pfcand_dzsig": [1, 0],
        "pfcand_dxy": [-1, 0],
        "pfcand_dxysig": [1, 0],
    }

    # convert to numpy arrays and normalize features
    if "pf_vectors" in tagger_vars.keys():
        variables = set(tagger_vars["pf_features"]["var_names"] + tagger_vars["pf_vectors"]["var_names"])
    else:
        variables = tagger_vars["pf_features"]["var_names"]

    for var in variables:
        a = (
            ak.pad_none(
                feature_dict[var],
                tagger_vars["pf_features"]["var_length"],
                axis=1,
                clip=True,
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)
        a = np.nan_to_num(a)

        # replace values to match PKU's
        if var in repl_values_dict:
            vals = repl_values_dict[var]
            a[a == vals[0]] = vals[1]

        # print(var)
        # print(a[11])

        if normalize:
            if var in tagger_vars["pf_features"]["var_names"]:
                info = tagger_vars["pf_features"]["var_infos"][var]
            else:
                info = tagger_vars["pf_vectors"]["var_infos"][var]

            # print(info)
            # print("\n")

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_svs_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    fj_idx_lep,
    fatjet_label: str = "FatJet",
    svs_label: str = "FatJetSVs",
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extracts the sv features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """

    feature_dict = {}

    jet = ak.firsts(preselected_events[fatjet_label][fj_idx_lep])
    msk = preselected_events[svs_label].jetIdx == ak.firsts(fj_idx_lep)

    jet_svs = preselected_events.SV[preselected_events[svs_label].sVIdx[(preselected_events[svs_label].sVIdx != -1) * (msk)]]

    # sort by dxy significance
    jet_svs = jet_svs[ak.argsort(jet_svs.dxySig, ascending=False)]

    # negative eta jets have -1 sign, positive eta jets have +1
    eta_sign = ak.values_astype(jet_svs.eta > 0, int) * 2 - 1
    feature_dict["sv_etarel"] = eta_sign * (jet_svs.eta - jet.eta)
    feature_dict["sv_phirel"] = jet_svs.delta_phi(jet)
    feature_dict["sv_abseta"] = np.abs(jet_svs.eta)
    feature_dict["sv_mass"] = jet_svs.mass
    feature_dict["sv_pt_log"] = np.log(jet_svs.pt)

    feature_dict["sv_ntracks"] = jet_svs.ntracks
    feature_dict["sv_normchi2"] = jet_svs.chi2
    feature_dict["sv_dxy"] = jet_svs.dxy
    feature_dict["sv_dxysig"] = jet_svs.dxySig
    feature_dict["sv_d3d"] = jet_svs.dlen
    feature_dict["sv_d3dsig"] = jet_svs.dlenSig
    svpAngle = jet_svs.pAngle
    feature_dict["sv_costhetasvpv"] = -np.cos(svpAngle)

    feature_dict["sv_px"] = jet_svs.px
    feature_dict["sv_py"] = jet_svs.py
    feature_dict["sv_pz"] = jet_svs.pz
    # feature_dict["sv_energy"] = jet_svs.E
    feature_dict["sv_energy"] = jet_svs.energy

    feature_dict["sv_mask"] = (
        ~(
            ma.masked_invalid(
                ak.pad_none(
                    feature_dict["sv_etarel"],
                    tagger_vars["sv_features"]["var_length"],
                    axis=1,
                    clip=True,
                ).to_numpy()
            ).mask
        )
    ).astype(np.float32)
    if isinstance(feature_dict["sv_mask"], np.float32):
        feature_dict["sv_mask"] = np.ones((len(feature_dict["sv_abseta"]), tagger_vars["sv_features"]["var_length"])).astype(
            np.float32
        )

    # convert to numpy arrays and normalize features
    if "sv_vectors" in tagger_vars.keys():
        variables = set(tagger_vars["sv_features"]["var_names"] + tagger_vars["sv_vectors"]["var_names"])
    else:
        variables = tagger_vars["sv_features"]["var_names"]

    for var in variables:
        a = (
            ak.pad_none(
                feature_dict[var],
                tagger_vars["sv_features"]["var_length"],
                axis=1,
                clip=True,
            )
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)
        a = np.nan_to_num(a)

        # print(var)
        # print(a[11])

        if normalize:
            if var in tagger_vars["sv_features"]["var_names"]:
                info = tagger_vars["sv_features"]["var_infos"][var]
            else:
                info = tagger_vars["sv_vectors"]["var_infos"][var]

            # print(info)
            # print("\n")

            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict


def get_lep_features(
    tagger_vars,
    preselected_events,
    jet,
    lepton,
) -> Dict[str, np.ndarray]:
    """
    Extracts the lepton features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    feature_dict["lep_dR_fj"] = lepton.delta_r(jet).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt"] = (lepton.pt).to_numpy().filled(fill_value=0)
    feature_dict["lep_pt_ratio"] = (lepton.pt / jet.pt).to_numpy().filled(fill_value=0)

    return feature_dict


def get_met_features(
    tagger_vars: dict,
    preselected_events: NanoEventsArray,
    jet,
    met_label,
    normalize=True,
) -> Dict[str, np.ndarray]:
    """
    Extracts the MET features specified in the ``tagger_vars`` dict from the
    ``preselected_events`` and returns them as a dict of numpy arrays
    """
    feature_dict = {}

    met = preselected_events[met_label]

    # get features
    feature_dict["met_relpt"] = met.pt / jet.pt
    feature_dict["met_relphi"] = met.delta_phi(jet)

    for var in tagger_vars["met_features"]["var_names"]:
        a = (
            # ak.pad_none(
            #     feature_dict[var], tagger_vars["met_features"]["var_length"], axis=1, clip=True
            # )
            feature_dict[var]  # just 1d, no pad_none
            .to_numpy()
            .filled(fill_value=0)
        ).astype(np.float32)

        if normalize:
            info = tagger_vars["met_features"]["var_infos"][var]
            a = (a - info["median"]) * info["norm_factor"]
            a = np.clip(a, info.get("lower_bound", -5), info.get("upper_bound", 5))

        feature_dict[var] = a

    return feature_dict
