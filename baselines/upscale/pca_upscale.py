import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

from bento_sc.data import BentoDataModule
from bento_sc.utils.config import Config
from sklearn.decomposition import IncrementalPCA
from bento_sc.utils.metrics import pearson_batch_masked
import torch
import numpy as np
from tqdm import tqdm
import sys


train_config = str(sys.argv[1])
test_config = str(sys.argv[2])
main_scTab_file = str(sys.argv[3])
scTab_sub_val = str(sys.argv[4])
scTab_sub_test = str(sys.argv[5])

config = Config(train_config)


config["data_path"] = main_scTab_file

dm = BentoDataModule(
    config
)
dm.setup(None)

ipca = IncrementalPCA(n_components=50)

iter_ = iter(dm.train_dataloader())


for i in tqdm(range(10_000), total=10_000):
    batch = next(iter_)
    ipca.partial_fit(batch["gene_counts"])

config = Config(test_config)

config["data_path"] = scTab_sub_val

dm = BentoDataModule(
    config
)
dm.setup(None)

pearsons = []
for batch in dm.predict_dataloader():

    mat = torch.zeros((len(batch["gene_counts"]), 19331))

    for b in range(len(batch["gene_index"])):
        gene_ix = batch["gene_index"][b]
        gene_c = batch["gene_counts"][b]

        mask = gene_c != -1
        mat[b][gene_ix[mask]] = gene_c[mask]

    preds = ipca.inverse_transform(ipca.transform(mat.numpy()))

    preds = torch.tensor(preds[np.arange(len(preds))[:, None], batch["gene_index"].numpy()])

    pearsons.append(
        pearson_batch_masked(torch.expm1(preds), batch["gene_counts_true"]).numpy()
    )
print(np.concatenate(pearsons).mean())


config = Config(test_config)

config["data_path"] = scTab_sub_test

dm = BentoDataModule(
    config
)
dm.setup(None)

pearsons = []
for batch in dm.predict_dataloader():

    mat = torch.zeros((len(batch["gene_counts"]), 19331))

    for b in range(len(batch["gene_index"])):
        gene_ix = batch["gene_index"][b]
        gene_c = batch["gene_counts"][b]

        mask = gene_c != -1
        mat[b][gene_ix[mask]] = gene_c[mask]

    preds = ipca.inverse_transform(ipca.transform(mat.numpy()))

    preds = torch.tensor(preds[np.arange(len(preds))[:, None], batch["gene_index"].numpy()])

    pearsons.append(
        pearson_batch_masked(torch.expm1(preds), batch["gene_counts_true"]).numpy()
    )
print(np.concatenate(pearsons).mean())