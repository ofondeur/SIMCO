import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader


def predict_from_unstim_data(result_path, unstim_data_path, output_path,stim):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")

    # load the config and then the model (f,g)
    config = load_config(config_path)
    (_, g), _, _ = load(config, restore=chkpt)
    g.eval()

    # load the data to predict and filter with the interzsting markers
    unstim_anndata_to_predict = ad.read(unstim_data_path)
    features = ['149Sm_pCREB', '167Er_pERK12', '164Dy_IkB', '159Tb_pMAPKAPK2', '166Er_pNFkB', '151Eu_pp38','155Gd_pS6', '153Eu_pSTAT1', '154Sm_pSTAT3', '150Nd_pSTAT5', '168Yb_pSTAT6', '174Yb_HLADR', '169Tm_CD25']

    unstim_anndata_to_predict = unstim_anndata_to_predict[:, features].copy()

    # predict the data (first put it in the dataset format)
    dataset_args = {}
    dataset = AnnDataDataset(unstim_anndata_to_predict.copy(), **dataset_args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )
    unstim_data=unstim_anndata_to_predict.copy()
    unstim_data.obs["state"] = "true_corrected"
    
    predicted=predicted[predicted.obs['drug']==stim]
    predicted.obs["state"] = "predicted"
    concatenated = ad.concat([predicted, unstim_data], axis=0)
    
    # save the prediction in the desired format
    if output_path.endswith(".csv"):
        concatenated = concatenated.to_df()
        concatenated.to_csv(output_path)

    elif output_path.endswith(".h5ad"):
        print(output_path)
        concatenated.write(output_path)
    return

perio_stim_list_=['TNFa','P._gingivalis']
perio_cell_list_=['Granulocytes_(CD45-CD66+)','B-Cells_(CD19+CD3-)','Classical_Monocytes_(CD14+CD16-)','MDSCs_(lin-CD11b-CD14+HLADRlo)','mDCs_(CD11c+HLADR+)','pDCs(CD123+HLADR+)','Intermediate_Monocytes_(CD14+CD16+)','Non-classical_Monocytes_(CD14-CD16+)','CD56+CD16-_NK_Cells','CD56loCD16+NK_Cells','NK_Cells_(CD7+)','CD4_T-Cells','Tregs_(CD25+FoxP3+)','CD8_T-Cells','CD8-CD4-_T-Cells']
for cell_type in perio_cell_list_:
    for stim in perio_stim_list_:
        print(f"Predicting {cell_type} for {stim}")
        if stim=='P._gingivalis':
            doms_stim='LPS'
            unstim_data_path = f'./datasets/surge_corrected/surge_concatenated_{doms_stim}_{cell_type}.h5ad'
        else:
            doms_stim=stim
            unstim_data_path = f'./datasets/surge_corrected/surge_concatenated_{stim}_{cell_type}.h5ad'
        result_path = f"./results/perio_surge_training_corrected/{stim}_{cell_type}/model-cellot"
        output_path = f'./results/perio_surge_training_corrected/{stim}_{cell_type}/model-cellot/pred_surge_corrected.h5ad'
        if [stim, cell_type] not in [['P._gingivalis', 'Non-classical_Monocytes_(CD14-CD16+)'],['P._gingivalis', 'NK_Cells_(CD7+)'],['TNFa', 'Granulocytes_(CD45-CD66+)']]:
            ada = predict_from_unstim_data(result_path, unstim_data_path, output_path,doms_stim)
        