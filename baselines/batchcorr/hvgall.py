import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

import torch
import numpy as np
from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config
from tqdm import tqdm
import argparse
from sklearn.decomposition import IncrementalPCA
from scipy.sparse import csr_matrix
import pandas as pd
import h5torch
import anndata as ad
import bbknn
import scib
import scanpy as sc

def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Training script for modality prediction.",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("config_path", type=str, metavar="config_path", help="config_path")
    parser.add_argument(
        "output_h5ad",
        type=str,
        metavar="output_h5ad",
        help="results will be written in a h5ad file",
    )
    parser.add_argument(
        "batch_col",
        type=str,
        metavar="batch_col",
        help="index of column in 0/obs in h5t file where batch effects are",
    )
    parser.add_argument(
        "ct_col",
        type=str,
        metavar="ct_col",
        help="index of column in 0/obs in h5t file where celltypes are",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Data file. Overrides value in config file if specified")

    args = parser.parse_args()

    config = Config(args.config_path)

    if args.data_path is not None:
        config["data_path"] = args.data_path

    dm = BentoDataModule(
        config
    )
    dm.setup(None)

    ipca = IncrementalPCA(n_components=50)

    for batch in tqdm(dm.predict_dataloader()):
        ipca.partial_fit(batch["gene_counts"])

    embeds = []
    obs = []
    for batch in tqdm(dm.predict_dataloader()):
        embeds.append(ipca.transform(batch["gene_counts"]))
        obs.append(batch["0/obs"])
        

    embeds = torch.cat(embeds).numpy()
    obs = torch.cat(obs).numpy()

    f = h5torch.File(args.data_path)
    f = f.to_dict()

    matrix = csr_matrix((f["central/data"][:],f["central/indices"][:],f["central/indptr"][:]), shape = (f["0/obs"].shape[0], f["1/var"].shape[0]))

    adata = ad.AnnData(matrix)
    adata.obs = pd.DataFrame(f["0/obs"], columns=np.arange(f["0/obs"].shape[1]).astype(str))
    adata.var = pd.DataFrame(f["1/var"], columns=np.arange(f["1/var"].shape[1]).astype(str))

    adata.obs[args.batch_col] = adata.obs[args.batch_col].astype("category")
    adata.obs[args.ct_col] = adata.obs[args.ct_col].astype("category")

    adata.obsm["X_emb"] = embeds

    embeds_pca = adata.obsm["X_emb"]

    adata.obsm["X_pca"] = embeds_pca

    bbknn.bbknn(adata, batch_key=args.batch_col)

    sc.tl.umap(adata)

    clisi, ilisi = scib.me.lisi_graph(adata, batch_key=args.batch_col, label_key=args.ct_col, type_="knn", n_cores=16)
    graph_conn = scib.me.graph_connectivity(adata, label_key=args.ct_col)

    scib.cl.cluster_optimal_resolution(adata, cluster_key="iso_label", label_key=args.ct_col)
    iso_f1 = scib.me.isolated_labels_f1(adata, batch_key=args.batch_col, label_key=args.ct_col, embed=None)
    ari = scib.me.ari(adata, cluster_key="iso_label", label_key=args.ct_col)
    nmi = scib.me.nmi(adata, cluster_key="iso_label", label_key=args.ct_col)

    adata.uns["scores"] = {
        "iLISI" : ilisi,
        "Graph Connectivity" : graph_conn,
        "cLISI" : clisi,
        "ARI" : ari,
        "NMI" : nmi,
        "Isolated F1": iso_f1,
    }

    adata.write(args.output_h5ad)


if __name__ == "__main__":
    main()

