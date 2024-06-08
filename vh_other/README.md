# directions for running VH 

the vhprocessor.py that is being used is in the boostedhiggs directory (called vhprocessor.py)

in the directory above this one, you can run 

- locally without inference: `python run.py --year 2018 --sample HZJ_HToWW_M-125 --processor vh --pfnano v2_2 --starti 0 --channels ele,mu --local --config test.yaml --n 1 --no-inference`
- locally with inference: `python run.py --year 2018 --sample HZJ_HToWW_M-125 --processor vh --pfnano v2_2 --starti 0 --channels ele,mu --local --config test.yaml --n 1 --inference`
- on condor, with systematics and inference:
  `python3 condor/submit.py --year 2018 --tag june8_208pm --submit --processor vh --channels ele,mu --config sample_MC_june8.yaml --key mc --systematics --inference`
  - without inference, just change to --no-inference, and without systematics change to --no-systematics

- note: in order to run on EAF need the versions of run_tagger_inference.py and the full directory tagger_resources from vh_other
  - this tells condor to run inference with the EAF server at FNAL
  - note to self: if i pull from Farouk's directory, make sure I re-copy over these two files
  
