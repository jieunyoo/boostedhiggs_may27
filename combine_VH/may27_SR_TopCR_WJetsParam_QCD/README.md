# about

python3 make_templatesWithQCD.py --years 2018 --channels mu,ele --outdir templates/test

python3 create_datacard.py --years 2018 --channels mu,ele --outdir templates/test


manually add rateParams to cards before combining

combineCards.py signal_region=SR1.txt ttbar_cr=TopCR.txt &> run1_combined.txt

text2workspace.py run1_combined.txt -o workspace.root

combine -M FitDiagnostics workspace.root -t -1 --expectSignal=1


