#!/usr/bin/python3

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cellot.models import load_autoencoder_model
from cellot.utils import load_config
from cellot.data.utils import cast_dataset_to_loader
from cellot.utils.helpers import nest_dict


class AnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        value = self.adata.X[idx]

        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        if self.include_index:
            return self.adata.obs_names[idx], value

        return value


def read_list(arg):

    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg

    return list(lst)


def read_single_anndata(config, path=None):
    if path is None:
        path = config.data.path

    data = anndata.read_h5ad(path)
    data.obs_names_make_unique()
    if "features" in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()
    # select subgroup of individuals
    if "drug_used" in config.data:
        drug_available= data.obs['drug'].unique()
        print(f"drug available: {drug_available}")
        if config.data.drug_used!='DMSO':
            data = data[(data.obs['drug']==config.data.drug_used) | (data.obs['drug'].str.startswith("DMSO"))].copy()
            
            if "batch_correction" in config.data:
                df1 = pd.read_csv(config.data.batch_correction)
                dmso_target=df1[df1['Treatment']==config.data.drug_used]["use_DMSO"].unique()
                if len(dmso_target)!=1:
                    print('bad dmso target length, found {len(dmso_target)}')
                dmso_target = dmso_target[0]
                data = data[(data.obs['drug']==config.data.drug_used) | (data.obs['drug']==dmso_target)].copy()
            drug_used= data.obs['drug'].unique()
            print(f"drug used: {drug_used}")
        else:
            data = data[data.obs['drug'].str.startswith("DMSO")].copy()
            drug_used= data.obs['drug'].unique()
            print(f"drug used: {drug_used}")
    
    if "patients_excluded" in config.data:
        excluded_raw = config.data.patients_excluded
        if isinstance(excluded_raw, str):
            excluded_list = [x.strip() for x in excluded_raw.strip('[]').split(',') if x.strip()]
        else:
            excluded_list = excluded_raw
        print(config.data.patients_excluded)
        data = data[~data.obs["patient"].isin(excluded_list)].copy()
        print(data.obs["patient"].unique())
    if "batch_correction" in config.data:
        df = pd.read_csv(config.data.batch_correction)
    
        present_patients = data.obs["patient"].unique()
    
        
        df = df[df["Patient"].isin(present_patients)]
    
        
        markers = list(set(df["Marker"]).intersection(data.var_names))
        print(f"Correcting markers: {markers}")
        
        for patient in present_patients:
            for population in data.obs["cell_type"].unique():
                mask_patient = (data.obs["patient"] == patient)
                mask_pop = (data.obs["cell_type"] == population)
    
                for marker in markers:
                    if "drug_used" not in config.data:
                        raise ValueError("No drug specified, need one to batch correct")
                    
                    if config.data.drug_used == "DMSO":
                        for treat in ['DMSO1','DMSO2','DMSO3']:
                            mask_treat = (data.obs["drug"] == treat)
                            for stim in ["Unstim", config.data.target]:
                                mask_stim = (data.obs[config.data.condition] == stim)
                                mask = mask_patient & mask_pop & mask_treat & mask_stim
    
                                if not mask.any():
                                    print('mask seems pretty empty')
                                    continue
    
                                row = df[
                                    (df["Patient"] == patient)
                                    & (df["Treatment"] == treat)
                                    & (df["Stim"] == stim)
                                    & (df["Marker"] == marker)
                                    & (df["Population"] == population)
                                ]
                                if row.empty:
                                    print(f"[EMPTY] Patient={patient}, Treatment={treat}, Stim={stim}, Marker={marker}, Pop={population}")
                                if not row.empty:
                                    corrected = row.iloc[0]["correctedMedian"]
                                    real = np.median(data[mask, marker].X)
                                    shift = corrected - real
                                    data[mask, marker].X += shift
                                    new_median = np.median(data[mask, marker].X)
                                    if not np.isclose(new_median, corrected, atol=1e-6):
                                        print(f"[WARNING] Correction mismatch: expected {corrected}, got {new_median}")
                                    else:
                                        print(f"[OK] Corrected median matches: {corrected}")
    
                    else:
                        
                        mask_stim = (data.obs["stim"] == config.data.stim)
                        drug_used= data.obs['drug'].unique()
                        row1 = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == config.data.drug_used)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]
                        if not row1.empty:
                            mask_stim = (data.obs["stim"] == config.data.stim)
                            mask_treat = (data.obs["drug"] == config.data.drug_used)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row1.iloc[0]["correctedMedian"]
                            dmso_target_feat = row1.iloc[0]["use_DMSO"]
                            if dmso_target_feat != dmso_target:
                                print('[WARNING] Found a different dmso target for a feature')
                                continue
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data.X[mask.to_numpy(), data.var_names.get_loc(marker)] += shift
                            new_median = np.median(data[mask, marker].X)
                            if not np.isclose(new_median, corrected, atol=1e-6):
                                print(f"[WARNING] Correction mismatch: expected {corrected}, got {new_median}, for {config.data.stim}, {patient}, {marker}, {population}")
                            else:
                                print(f"[OK] Corrected median matches: {corrected}")
                            
                        row = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == dmso_target)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]
                        if not row.empty:
                            mask_treat = (data.obs["drug"] == dmso_target)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row.iloc[0]["correctedMedian"]
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data.X[mask.to_numpy(), data.var_names.get_loc(marker)] += shift
                            new_median = np.median(data[mask, marker].X)
                            if not np.isclose(new_median, corrected, atol=1e-6):
                                print(f"[WARNING] Correction mismatch: expected {corrected}, got {new_median}, for {config.data.stim}, {patient}, {marker}, {population}")
                            else:
                                print(f"[OK] Corrected median matches: {corrected}")
    if config.data.condition=='drug':
        data=data[data.obs['stim']==config.data.stim].copy()
    stims_in_data=data.obs['stim'].unique().tolist()
    drugs_in_data=data.obs['drug'].unique().tolist()
    cell_type_in_data=data.obs['cell_type'].unique().tolist()
    
    print(f"The categories of the anndata before the source, target split is: stim {stims_in_data}, drugs {drugs_in_data}, cells {cell_type_in_data}, with a shape of {data.X.shape}")
    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    transport_mapper = dict()
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)

    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    mask = data.obs["transport"].notna()
    assert "subset" not in config.data
    if "subset" in config.datasplit:
        for key, value in config.datasplit.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # write train/test/valid into split column
    data = data[mask].copy()
    if "datasplit" in config:
        data.obs["split"] = split_cell_data(data, **config.datasplit)

    return data


def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",
    include_model_kwargs=False,
    pair_batch_on=None,
    **kwargs
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition

    if data is None:
        if config.data.type == "cell":
            data = read_single_anndata(config, **kwargs)
        else:
            raise ValueError

    if config.data.get("select") is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    if "dimension_reduction" in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == "pca":
            dims = config.data.dimension_reduction.get(
                "dims", data.obsm["X_pca"].shape[1]
            )

            data = anndata.AnnData(
                data.obsm["X_pca"][:, :dims], obs=data.obs.copy(), uns=data.uns.copy()
            )
            data.uns["genes"] = genes

    if "ae_emb" in config.data:
        # load path to autoencoder
        assert config.get("model.name", "cellot") == "cellot"
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {"input_dim": data.n_vars}
        config_ae = load_config(path_ae / "config.yaml")
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
        )

        inputs = torch.Tensor(
            data.X if not sparse.issparse(data.X) else data.X.todense()
        )

        genes = data.var_names.to_list()
        data = anndata.AnnData(
            ae_model.eval().encode(inputs).detach().numpy(),
            obs=data.obs.copy(),
            uns=data.uns.copy(),
        )
        data.uns["genes"] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs["input_dim"] = data.n_vars

    if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

    if "training" in config:
        pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

    if split_on is None:
        if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
            split_on = ["split", "transport"]
            if pair_batch_on is not None:
                split_on.append(pair_batch_on)

        elif (config.model.name == "scgen" or config.model.name == "cae"
              or config.model.name == "popalign"):
            split_on = ["split"]

        else:
            raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else ".".join(key)): data[index]
            for key, index in data.obs[split_on].groupby(split_on, observed=False).groups.items()
        }

        dataset = nest_dict(
            {
                key: AnnDataDataset(val.copy(), **dataset_args)
                for key, val in splits.items()
            },
            as_dot_dict=True,
        )

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if "loader" in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault("drop_last", True)
        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "anndata":
            returns.append(data)

        elif key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def split_cell_data_train_test(
    data, groupby=None, random_state=0, holdout=None, subset=None, **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby, observed=False).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(
    data,
    test_size=0.15,
    eval_size=0.15,
    groupby=None,
    random_state=0,
    holdout=None,
    **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size
        )

        trainobs, testobs = train_test_split(
            training, random_state=random_state, test_size=test_size
        )

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):

    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """

    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype("category")
