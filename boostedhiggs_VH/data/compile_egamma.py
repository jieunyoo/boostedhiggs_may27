import correctionlib.schemav2 as schema
import uproot
from correctionlib.schemav2 import Correction, CorrectionSet

corrections = {
    "trigger": {
        "electron": {  # derived by D. Rankin
            "2016preVFP_UL": "egammaEffi_txt_trigger_EGM2D_UL2016preVFP.root:EGamma_SF2D",
            "2016postVFP_UL": "egammaEffi_txt_trigger_EGM2D_UL2016postVFP.root:EGamma_SF2D",
            "2017_UL": "egammaEffi_txt_trigger_EGM2D_UL2017.root:EGamma_SF2D",
            "2018_UL": "egammaEffi_txt_trigger_EGM2D_UL2018.root:EGamma_SF2D",
        },
    },
}


# Function to extract SFs from EGamma standard root files
def getSFs(fn="filename", IsSF="sf"):
    fo = uproot.open(fn)
    values = fo.values().flatten().tolist()
    edges_x = fo.axis(0).edges().tolist()
    edges_y = fo.axis(1).edges().tolist()
    errors = fo.errors().flatten().tolist()
    if IsSF == "sf":
        valSFs = schema.MultiBinning.parse_obj(
            {
                "inputs": ["eta", "pt"],
                "nodetype": "multibinning",
                "edges": [
                    edges_x,
                    edges_y,
                ],
                "content": values,
                "flow": "error",
            }
        )

        return valSFs
    if IsSF == "sfup":
        valerrorsup = schema.MultiBinning.parse_obj(
            {
                "inputs": ["eta", "pt"],
                "nodetype": "multibinning",
                "edges": [
                    edges_x,
                    edges_y,
                ],
                "content": [m + n for m, n in zip(values, errors)],
                "flow": "error",
            }
        )
        return valerrorsup
    if IsSF == "sfdown":
        valerrorsdown = schema.MultiBinning.parse_obj(
            {
                "inputs": ["eta", "pt"],
                "nodetype": "multibinning",
                "edges": [
                    edges_x,
                    edges_y,
                ],
                "content": [m - n for m, n in zip(values, errors)],
                "flow": "error",
            }
        )
        return valerrorsdown


def SFyearwise(files=[], names=[], valtypes=["sf", "sfup", "sfdown"]):
    # content = [convert.from_uproot_THx(files[name]) for name in names]

    output = schema.Category.parse_obj(
        {
            "nodetype": "category",
            "input": "ValType",
            "content": [
                schema.CategoryItem.parse_obj(
                    {
                        "key": val,
                        "value": schema.Category.parse_obj(
                            {
                                "nodetype": "category",
                                "input": "WorkingPoint",
                                "content": [
                                    schema.CategoryItem.parse_obj({"key": name, "value": getSFs(fn=files[name], IsSF=val)})
                                    for name in names
                                ],
                            }
                        ),
                    }
                )
                for val in valtypes
            ],
        }
    )
    return output


"""
2016preVFP/2016postVFP: Ele27_WPTight_Gsf | Ele115_CaloIdVT_GsfTrkIdT | Photon175,
2017': Ele35_WPTight_Gsf | Ele115_CaloIdVT_GsfTrkIdT | Photon200,
2018: Ele32_WPTight_Gsf | Ele115_CaloIdVT_GsfTrkIdT | Photon200.
"""
for corrName, corrDict in corrections.items():
    for lepton_type, leptonDict in corrDict.items():
        for year, ystring in leptonDict.items():
            corrs = []
            corr = Correction.parse_obj(
                {
                    "version": 2,
                    "name": "UL-Electron-Trigger-SF",
                    "description": f"These are the electron Trigger Scale Factors (nominal, up or down) for {year} Ultra Legacy dataset. They are dependent on the transverse momenta and pseudorapidity of the electron.",  # noqa
                    "inputs": [
                        {"name": "year", "type": "string", "description": "year/scenario: example 2016preVFP, 2017 etc"},
                        {
                            "name": "ValType",
                            "type": "string",
                            "description": "sf/sfup/sfdown (sfup = sf + syst, sfdown = sf - syst) ",
                        },
                        {
                            "name": "WorkingPoint",
                            "type": "string",
                            "description": "Working Point of choice : Loose, Medium etc.",
                        },
                        {"type": "real", "name": "eta", "description": "supercluster eta"},
                        {"name": "pt", "type": "real", "description": "electron pT"},
                    ],
                    "output": {
                        "name": "weight",
                        "type": "real",
                        "description": "value of scale factor (nominal, up or down)",
                    },
                    "data": schema.Category.parse_obj(
                        {
                            "nodetype": "category",
                            "input": "year",
                            "content": [
                                schema.CategoryItem.parse_obj(
                                    {"key": year, "value": SFyearwise(files={corrName: ystring}, names=[corrName])}
                                )
                            ],
                        }
                    ),
                }
            )
            corrs.append(corr)

            # Save JSON
            cset = CorrectionSet(
                schema_version=2,
                corrections=corrs,
                description=f"These are the electron ID Scale Factors (nominal, up or down) for {year} Ultra Legacy dataset. They are available for the cut-based and MVA IDs. They are dependent on the transverse momenta and pseudorapidity of the electron.",  # noqa
            )

            with open(f"{lepton_type}_{corrName}_{year}.json", "w") as fout:
                fout.write(cset.json(exclude_unset=True, indent=4))
