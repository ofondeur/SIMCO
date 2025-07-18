from itertools import product
import json
import os

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [2, 4, 10],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.5, 0.7],
    "colsample_bytree": [0.5, 0.8],
    "gamma": [0, 1],
    "reg_alpha": [0],
    "reg_lambda": [1],
}

param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

output_dir = "param_grid_files"
os.makedirs(output_dir, exist_ok=True)

for i, values in enumerate(param_combinations):
    param_dict = dict(zip(param_keys, values))
    with open(f"{output_dir}/params_{i}.json", "w") as f:
        json.dump(param_dict, f, indent=2)

print(f"Saved {len(param_combinations)} parameter sets to {output_dir}/")
