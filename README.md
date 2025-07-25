# PTB_drugscreen
# [A predictive simulation framework for immunomodulatory drug interventions in pregnancy]

**Authors**: Jakob Einhaus\*, Peter Neidlinger\*, Olivier Fondeur\*, Masaki Sato\*, Brice Gaudilliere

We present a prediction ensemble that combines outcome modeling and treatment evaluation on simulated single-cell data to rapidly evaluate multiple drug candidates for their potential to shift the timing of labor and prevent PTB.

We trained CellOT, a neural optimal transport (OT) model, to learn single-cell perturbation behavior for 28 immune cell types. We then trained a Stabl model to identify a sparse number of features able to accurately predict the time to labor for each sample. Finally, we used the trained model on simulated treatment effect to quantify the effect of the drugs on time to labor. Simulated treatment effects shifted predicted time to labor, enabling individual-level evaluation and personalized treatment selection. 

This repository contains the CellOT scripts to predict single-cell perturbation behavior and the Stabl scripts to predict the time to labor.

## Installation

To setup the corresponding `conda` environment run:
```
conda create --name cellot python=3.9.5
conda activate cellot

conda update -n base -c defaults conda
pip install --upgrade pip
```
Install requirements and dependencies via:
```
pip install -r requirements.txt
```
To install `CellOT` run:
```
python setup.py develop
```
Package requirements and dependencies are listed in `requirements.txt`. Installation takes < 5 minutes and has been tested on Linux (CentOS Linux release 7.9.2009), macOS (Version 12.4, with Apple M1 Pro and Version 11.3, with 2.6 GHz Intel Core i7). 

## Datasets
You can download the preprocessed data [here]().

## Experiments

The CellOT models can be trained for each stim-cell type combination using the scripts in CellOT/sbatch_files such as CellOT/sbatch_files/train_cellot_cellwise_cv.sh.

For example, to run CellOT on the ptb data for PI stimulation on cMCS using the healthy patients DMSO sample the script would be:

```
python ../cellot/scripts/train.py \
      --config ../configs/models/cellot.yaml \
      --config ../configs/tasks/ptb_final_cv_original/HV/ptb_PI_cMCs_HV_train.yaml \
      --outdir ../results/cross_validation_original/PI/cMCs/model-PI_cMCs \
      --config.data.target PI \
      --config.data.drug_used DMSO
```

The Stabl models can be trained using the scripts in Stabl_Launch/Bash_Files such as batch_stabl model.sh. The other .sh files launch a cross-validation on features already selected to perform a grid-search on XGBoost hyperparameters (e.g. ```batch_gridsearch.sh```) or to compare the time to labour prediction for each drug treated data (e.g. ```run_drug_comparison.sh```).

To run a Stabl model on predicted perturbation and unstim data, for a knockoff artifical type and an xgboost final model, the script would be:
```
python ../run_regression_cv.py \
    --features_path ../Data/ina_13OG_final_long_allstims_filtered.csv \
    --results_dir ../Results/run_stabl_XGB_KO_ptb_data \
    --artificial_type knockoff \
    --model_chosen xgboost
```
which corresponds to running :
```
bash batch_stablmodel.sh
```
The final models available are linear regression, XGBoost and Random Forest, with XGBoost giving the best results.

## Citation

In case you found our work useful, please consider citing us:
```
```

## Contact
In case you have questions, please contact us via the Github Issues.