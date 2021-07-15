import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep as hep
hep.style.use(hep.style.CMS)

import hist as hist2
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights

import warnings
warnings.filterwarnings("ignore", message="Found duplicate branch")


def getParticles(genparticles,lowid=22,highid=25,flags=['fromHardProcess', 'isLastCopy']):
    """
    returns the particle objects that satisfy a low id, 
    high id condition and have certain flags
    """
    absid = abs(genparticles.pdgId)
    return genparticles[
        ((absid >= lowid) & (absid <= highid))
        & genparticles.hasFlags(flags)
    ]

def match_HWWlepqq(genparticles,candidatefj):
    """
    return the number of matched objects (hWW*),daughters, 
    and gen flavor (enuqq, munuqq, taunuqq) 
    """
    # all the higgs bosons in the event (pdgID=25)
    higgs = getParticles(genparticles,25)
    
    # select our higgs to be all WW decays
    is_hWW = ak.all(abs(higgs.children.pdgId)==24,axis=2)
    higgs = higgs[is_hWW]
    
    # select higgs's children
    higgs_wstar = higgs.children[ak.argmin(higgs.children.mass,axis=2,keepdims=True)]
    higgs_w = higgs.children[ak.argmax(higgs.children.mass,axis=2,keepdims=True)]
    
    # select electrons, muons and taus
    prompt_electron = getParticles(genparticles,11,11,['isPrompt','isLastCopy'])
    prompt_muon = getParticles(genparticles,13,13,['isPrompt', 'isLastCopy'])
    prompt_tau = getParticles(genparticles,15,15,['isPrompt', 'isLastCopy'])
    
    # choosing the quarks that not only are coming from a hard process (e.g. the WW decay)
    # but also the ones whose parent is a W (pdgId=24), this avoids select quarks whose parent is a gluon 
    # who also happened to be produced in association with the Higgs
    prompt_q = getParticles(genparticles,0,5,['fromHardProcess', 'isLastCopy'])
    prompt_q = prompt_q[abs(prompt_q.distinctParent.pdgId) == 24]
    
    # counting the number of gen particles 
    n_electrons = ak.sum(prompt_electron.pt>0,axis=1)
    n_muons = ak.sum(prompt_muon.pt>0,axis=1)
    n_taus = ak.sum(prompt_tau.pt>0,axis=1)
    n_quarks = ak.sum(prompt_q.pt>0,axis=1)
    
    # we define the `flavor` of the Higgs decay
    # 4(elenuqq),6(munuqq),8(taunuqq)
    hWWlepqq_flavor = (n_quarks==2)*1 + (n_electrons==1)*3 + (n_muons==1)*5 + (n_taus==1)*7

    # since our jet has a cone size of 0.8, we use 0.8 as a dR threshold
    matchedH = candidatefj.nearest(higgs, axis=1, threshold=0.8)
    matchedW = candidatefj.nearest(higgs_w, axis=1, threshold=0.8)
    matchedWstar = candidatefj.nearest(higgs_wstar, axis=1, threshold=0.8) 

    # matched objects
    # 1 (H only), 4(W), 6(W star), 9(H, W and Wstar)
    hWWlepqq_matched = (ak.sum(matchedH.pt>0,axis=1)==1)*1 + (ak.sum(ak.flatten(matchedW.pt>0,axis=2),axis=1)==1)*3 + (ak.sum(ak.flatten(matchedWstar.pt>0,axis=2),axis=1)==1)*5    
    
    # let's concatenate all the daughters
    dr_fj_quarks = candidatefj.delta_r(prompt_q)
    dr_fj_electrons = candidatefj.delta_r(prompt_electron)
    dr_fj_muons = candidatefj.delta_r(prompt_muon)
    dr_fj_taus = candidatefj.delta_r(prompt_tau)
    dr_daughters = ak.concatenate([dr_fj_quarks,dr_fj_electrons,dr_fj_muons,dr_fj_taus],axis=1)
    
    #  number of visible daughters
    hWWlepqq_nprongs = ak.sum(dr_daughters<0.8,axis=1)
    
    return hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs

class HwwSignalProcessor(processor.ProcessorABC):
    def __init__(self,jet_arbitration='pt'):
        self._jet_arbitration = jet_arbitration
        
        # output
        self.make_output = lambda: {
            'sumw': 0.,
            'signal_kin': hist2.Hist(
                hist2.axis.IntCategory([0, 2, 4, 6, 8], name='genflavor', label='gen flavor'),
                hist2.axis.IntCategory([0, 1, 4, 6, 9], name='genHflavor', label='higgs matching'),
                hist2.axis.Regular(100, 200, 1200, name='pt', label=r'Jet $p_T$'),
                hist2.axis.IntCategory([0, 1, 2, 3, 4], name='nprongs', label='Jet n prongs'),
                hist2.storage.Weight(),
            ),
            "signal_iso": hist2.Hist(
                hist2.axis.Regular(25, 0, 1, name="eleminiIso", label="$e$ miniIso"),
                hist2.axis.Regular(25, 0, 1, name="elerelIso", label="$e$ Rel Iso"),
                hist2.axis.Regular(25, 0, 1, name="muminiIso", label="$\mu$ miniIso"),
                hist2.axis.Regular(25, 0, 1, name="murelIso", label="$\mu$ Rel Iso"),
                hist2.storage.Weight(),
            )
        }
        
    def process(self, events):
        dataset = events.metadata['dataset']
        weights = Weights(len(events), storeIndividual=True)
        weights.add('genweight', events.genWeight)
        
        output = self.make_output()
        output['sumw'] = ak.sum(events.genWeight)
            
        # leptons
        goodmuon = (
            (events.Muon.pt > 25)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.mediumId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        lowptmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.looseId
        )
        nlowptmuons = ak.sum(lowptmuon, axis=1)
            
        goodelectron = (
            (events.Electron.pt > 25)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.mvaFall17V2noIso_WP80)
        )
        nelectrons = ak.sum(goodelectron, axis=1)
        lowptelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nlowptelectrons = ak.sum(lowptelectron, axis=1)

        # concatenate leptons and select leading one
        goodleptons = ak.concatenate([events.Muon[goodmuon], events.Electron[goodelectron]], axis=1)
        candidatelep = ak.firsts(goodleptons[ak.argsort(goodleptons.pt)])
        candidatelep_p4 = ak.zip(
            {
                "pt": candidatelep.pt,
                "eta": candidatelep.eta,
                "phi": candidatelep.phi,
                "mass": candidatelep.mass,
                "charge": candidatelep.charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        
        # met
        met = events.MET
        
        # jets
        fatjets = events.FatJet
        candidatefj = fatjets[
            (fatjets.pt > 200)
        ]

        dphi_met_fj = abs(candidatefj.delta_phi(met))
        dr_lep_fj = candidatefj.delta_r(candidatelep_p4)
    
        if self._jet_arbitration == 'pt':
            candidatefj = ak.firsts(candidatefj)
        elif self._jet_arbitration == 'met':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dphi_met_fj,axis=1,keepdims=True)])
        elif self._jet_arbitration == 'lep':
            candidatefj = ak.firsts(candidatefj[ak.argmin(dr_lep_fj,axis=1,keepdims=True)])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")
    
        # match HWWlepqq 
        hWWlepqq_flavor,hWWlepqq_matched,hWWlepqq_nprongs = match_HWWlepqq(events.GenPart,candidatefj)
        
        # function to normalize arrays after a cut or selection
        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
                
        # select only leptons inside the jet
        dr_lep_jet_cut = candidatefj.delta_r(candidatelep_p4) < 0.8
        dr_lep_jet_cut = ak.fill_none(dr_lep_jet_cut, False)
        
        # here we fill our histogram
        output['signal_kin'].fill(
        	genflavor=normalize(hWWlepqq_flavor, dr_lep_jet_cut),
             	genHflavor=normalize(hWWlepqq_matched, dr_lep_jet_cut),
             	pt = normalize(candidatefj.pt, dr_lep_jet_cut),
             	nprongs = normalize(hWWlepqq_nprongs, dr_lep_jet_cut),
              	weight=weights.weight()[dr_lep_jet_cut],
        )
        output['signal_iso'].fill(
        	eleminiIso = normalize(ele_miniIso, dr_lep_jet_cut),
            	elerelIso = normalize(ele_relIso, dr_lep_jet_cut),
            	muminiIso = normalize(mu_miniIso,dr_lep_jet_cut),
            	murelIso = normalize(mu_relIso,dr_lep_jet_cut),
            	weight=weights.weight()[dr_lep_jet_cut],
        )

        return {dataset: output}
            
    def postprocess(self, accumulator):
        return accumulator


# Dask client
from dask.distributed import Client

client = Client("tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.coffea.casa:8786")
client

# executing the processor for all arbitrations
fileset = {
    "HWW": ["root://xcache/" + file for file in np.loadtxt("data.txt", dtype=str)][:5] 
}
 
for arbitration in ["pt", "met", "lep"]:
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=HwwSignalProcessor(jet_arbitration=arbitration),
        executor=processor.dask_executor,#iterative_executor,
        executor_args={
            "schema": processor.NanoAODSchema,
            "client": client,
        },
        maxchunks=30,
    )
    
    # Plots
    title = f"arbitration: {'$p_T$' if arbitration=='pt' else arbitration}"
    match = ["None",r"$H$",r"$W$",r"$W*$",r"$HWW*$"]
    
    # hWWlepqq_matched
    gen_Hflavor = out["HWW"]["signal_kin"][{"genflavor": sum, "pt": sum, "nprongs":sum}]

    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )
    gen_Hflavor.plot1d(
        ax=ax,
        histtype="fill",
        density=True
    )
    ax.set(
        title=title,
        ylabel="Events",
        xlabel="Matched jets",
        xticklabels=match
    )
    fig.savefig(f"matched_{arbitration}.png")
    
    
    # hWWlepqq_flavor
    gen_flavor = out["HWW"]["signal_kin"][{"genHflavor": sum, "pt": sum, "nprongs":sum}]

    fig, ax = plt.subplots(
        figsize=(8,7),
        constrained_layout=True
    )
    gen_flavor.plot1d(
        ax=ax,
        density=True,
        histtype="fill",
    )
    ax.set(
        title=title,
        ylabel="Events",
        xlabel="gen flavor jets",
        xlim=(1.5,5.5),
        xticklabels=["","",r"$e\nu_e qq$", r"$\mu\nu_{\mu}qq$", r"$\tau\nu_{\tau}qq$"]
    )
    fig.savefig(f"genflavor_{arbitration}.png")
    
    # number of daus
    nprongs = out["HWW"]["signal_kin"][{"genHflavor":sum, "pt": sum, "genflavor": sum}]

    fig, ax = plt.subplots(
        figsize=(8,7),
        constrained_layout=True
    )
    nprongs.plot1d(
        ax=ax,
        histtype="fill",
        density=True
    )
    ax.set(
        title=title,
        ylabel="Events",
        xlabel="nprongs"
    )
    fig.savefig(f"nprongs_{arbitration}.png")
    
    
    # hWWlepqq_matched and jet pt
    h = out["HWW"]["signal_kin"][{"nprongs":sum, "genflavor":sum}]

    fig, ax = plt.subplots(
        figsize=(10,7),
        constrained_layout=True
    )
    for i in range(5): 
        h[i,:].plot1d(ax=ax)
        
    ax.set(
        title=title,
        ylabel="Events",
        xlim=(180,600),
        xlabel="jet $p_T$ [GeV]"
    )
    ax.legend(match, title="matched")
    fig.savefig(f"matchvspt_{arbitration}.png")
    
    
    # hWWlepqq_nprongs and jet pt
    h = out["HWW"]["signal_kin"][{"genHflavor": sum, "genflavor":sum}]

    fig, ax = plt.subplots(
        figsize=(10,7),
        constrained_layout=True
    )
    h.plot1d(ax=ax)
    
    ax.set(
        title=title,
        ylabel="Events",
        xlim=(180,600),
        xlabel="jet $p_T$ [GeV]"
    )
    ax.legend(title="nprongs")
    fig.savefig(f"nprongsvspt_{arbitration}.png")
    
    
    # hWWlepqq_nprongs and jet pt with HWW match
    h = out["HWW"]["signal_kin"][{"genflavor":sum}]

    fig, ax = plt.subplots(
        figsize=(10,7),
        constrained_layout=True
    )

    h[-1,:,:].plot1d(ax=ax)

    ax.set(
        title=title,
        xlim=(180,600),
        xlabel="jet $p_T$ [GeV]",
        ylabel="Events"
    )
    ax.legend(title="nprongs[HWW*]") 
    fig.savefig(f"nprongs[HWW]vspt_{arbitration}.png")
    
    # isolation
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"murelIso":sum, "elerelIso":sum, "eleminiIso":sum}].plot1d(ax=ax)

    ax.set(
        title=title,
        ylabel="Events"
    )
    plt.savefig(f"hww_muminiIso_{arbitration}.png")
    
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"muminiIso":sum,"elerelIso":sum, "eleminiIso":sum}].plot1d(ax=ax)

    ax.set(
        title=title,
        ylabel="Events"
    )
    plt.savefig(f"hww_murelIso_{arbitration}.png")
                
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"murelIso":sum,"elerelIso":sum, "eleminiIso":sum}].plot1d(ax=ax, label="mini")
    out["HWW"]["signal_iso"][{"muminiIso":sum,"elerelIso":sum, "eleminiIso":sum}].plot1d(ax=ax, label="rel")

    ax.set(
        title=title,
        ylabel="Events",
        xlabel="$\mu$ Isolation",
    )
    ax.legend(title="Isolation")
    plt.savefig(f"hww_muIso_{arbitration}.png")
    
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"elerelIso":sum,"murelIso":sum, "muminiIso":sum}].plot1d(ax=ax)

    ax.set(
        title=title,
        ylabel="Events"
    )
    plt.savefig(f"hww_eleminiIso_{arbitration}.png")
                
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"eleminiIso":sum,"murelIso":sum, "muminiIso":sum}].plot1d(ax=ax)

    ax.set(
        title=title,
        ylabel="Events"
    )
    plt.savefig(f"hww_elerelIso_{arbitration}.png")
    
    fig, ax = plt.subplots(
        figsize=(8,7), 
        constrained_layout=True
    )

    out["HWW"]["signal_iso"][{"elerelIso":sum,"murelIso":sum, "muminiIso":sum}].plot1d(ax=ax, label="mini")
    out["HWW"]["signal_iso"][{"eleminiIso":sum,"murelIso":sum, "muminiIso":sum}].plot1d(ax=ax, label="rel")

    ax.set(
        title=title,
        ylabel="Events",
        xlabel="$e$ Isolation",
    )
    ax.legend(title="Isolation")
    plt.savefig(f"hww_eleIso_{arbitration}.png")
