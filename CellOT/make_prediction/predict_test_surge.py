import os
import anndata as ad
from cellot.utils.helpers import load_config
import pandas as pd
from cellot.utils.evaluate import load_all_inputs, transport
from cellot.utils.loaders import load_model
import os


result_path = "/Users/MacBook/stanford/cellot/results/test_new/model-cellot"
unstim_data_path = (
    "/Users/MacBook/stanford/cellot/datasets/atest_data/unstim_Bcell_to_predict.h5ad"
)
output_path = "/Users/MacBook/stanford/cellot/datasets/atest_data/prediction.csv"
hkpt = os.path.join(result_path, "cache/model.pt")
config_path = os.path.join(result_path, "config.yaml")
config = load_config(config_path)

control, treated, to_pushfwd, obs, model_kwargs = load_all_inputs(
    config, setting="iid", embedding=None, where="data_space"
)
model, *_ = load_model(config, restore=hkpt, **model_kwargs)

imputed = transport(config, model, to_pushfwd).to_df()
