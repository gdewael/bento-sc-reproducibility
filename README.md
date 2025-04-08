<div align="center">

<img src="https://raw.githubusercontent.com/gdewael/bento-sc/refs/heads/main/assets/bento.svg" align="center" width="450" alt="bento-sc" href="https://github.com/gdewael/bento-sc">

<h1></h1>

# Reproducibility

</div>

This repository contains supplementary code and config files to reproduce all experimental results presented in the bento-sc paper. Please refer to:
- The main [`bento-sc` repository](https://github.com/gdewael/bento-sc) for the majority of single-cell language modeling code and scripts.
- The main [documentation page](https://bento-sc.readthedocs.io/en/latest/index.html) for instructions on getting started with `bento-sc` in general.


## Installing and preparing data

For installation and data preparation instructions, please refer to the [Getting started section](https://bento-sc.readthedocs.io/en/latest/getting_started.html) of our documentation.

## scLM pre-training and evaluation

All config files to pre-train and evaluate all models are in `./scLM_configs/`.
Config files are categorized per experiment in our original study. For example, config files for the "base" and "final" scLMs in our study are located under: `./scLM_configs/exp_A/binned/` and `./scLM_configs/exp_D/mlm_ctclf_contr/`, respectively.

Here, by example, we provide all steps needed to fully reproduce pre-training and evaluation of these two models.

**Note:** you may need to change values around in the config.yaml files to suit your set-up.
In addition, many of the scripts come with additional flags that are relevant for only some of the scLM models. Before running, make sure you are using the appropriate optional arguments for your model set-up.

### Base design

Pre-training:
```bash
bentosc_pretrain ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ --lr 0.0003 --data_path /path/to/scTab.h5t
```

Evaluation:
```bash
# upscaling on val and test
bentosc_task_upscale ./scLM_configs/exp_A/binned/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_path /path/to/scTab_upsc_val.h5t --clf_output True
bentosc_task_upscale ./scLM_configs/exp_A/binned/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_path /path/to/scTab_upsc_test.h5t --clf_output True

# grn infer on val and test
bentosc_task_grninfer embed ./scLM_configs/exp_A/binned/grninfer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_val/ --data_path /path/to/scTab_grn_val.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_A/binned/grninfer.yaml /path/to/embeds_grn_binned_val/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_val.h5t --test_mode val

bentosc_task_grninfer embed ./scLM_configs/exp_A/binned/grninfer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_test/ --data_path /path/to/scTab_grn_test.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_A/binned/grninfer.yaml /path/to/embeds_grn_binned_test/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_test.h5t --test_mode test

# post perturbation expression pred
bentosc_task_perturb ./scLM_configs/exp_A/binned/perturb.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_perturb/ --data_path /path/to/perturb.h5t --init_factor 1 --batch_size 32 --lr 0.00007

# celltype id
bentosc_task_celltypeid ./scLM_configs/exp_A/binned/celltypeid.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_celltypeid/ --lr 0.0003 --data_path /path/to/scTab.h5t

# prot conc pred
bentosc_task_protconc ./scLM_configs/exp_A/binned/protconc.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_protconc/ --lr 0.0001 --data_path /path/to/citeseq.h5t

# batch correction on 3 datasets.
bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_el.npz --data_path /path/to/batchcorr_el.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_el.h5t /path/to/embeds_el.npz /path/to/batchcorr_el_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ci.npz --data_path /path/to/batchcorr_ci.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ci.h5t /path/to/embeds_ci.npz /path/to/batchcorr_ci_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ga.npz --data_path /path/to/batchcorr_ga.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ga.h5t /path/to/embeds_ga.npz /path/to/batchcorr_ga_results.h5ad 0 3
```

### Final design

Pre-training:
```bash
bentosc_pretrain ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ --lr 0.0003 --data_path /path/to/scTab.h5t
```

Evaluation:
```bash
# upscaling on val and test
bentosc_task_upscale ./scLM_configs/exp_D/mlm_ctclf_contr/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_path /path/to/scTab_upsc_val.h5t --clf_output False
bentosc_task_upscale ./scLM_configs/exp_D/mlm_ctclf_contr/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_path /path/to/scTab_upsc_test.h5t --clf_output False

# grn infer on val and test
bentosc_task_grninfer embed ./scLM_configs/exp_D/mlm_ctclf_contr/grninfer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_val/ --data_path /path/to/scTab_grn_val.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_D/mlm_ctclf_contr/grninfer.yaml /path/to/embeds_grn_binned_val/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_val.h5t --test_mode val

bentosc_task_grninfer embed ./scLM_configs/exp_D/mlm_ctclf_contr/grninfer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_test/ --data_path /path/to/scTab_grn_test.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_D/mlm_ctclf_contr/grninfer.yaml /path/to/embeds_grn_binned_test/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_test.h5t --test_mode test

# post perturbation expression pred
bentosc_task_perturb ./scLM_configs/exp_D/mlm_ctclf_contr/perturb.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_perturb/ --data_path /path/to/perturb.h5t --init_factor 1 --batch_size 32 --lr 0.00007

# celltype id
bentosc_task_celltypeid ./scLM_configs/exp_D/mlm_ctclf_contr/celltypeid.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_celltypeid/ --lr 0.0003 --data_path /path/to/scTab.h5t

# prot conc pred
bentosc_task_protconc ./scLM_configs/exp_D/mlm_ctclf_contr/protconc.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_protconc/ --lr 0.0001 --data_path /path/to/citeseq.h5t

# batch correction on 3 datasets.
bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_el.npz --data_path /path/to/batchcorr_el.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_el.h5t /path/to/embeds_el.npz /path/to/batchcorr_el_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ci.npz --data_path /path/to/batchcorr_ci.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ci.h5t /path/to/embeds_ci.npz /path/to/batchcorr_ci_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/batchcorr.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ga.npz --data_path /path/to/batchcorr_ga.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ga.h5t /path/to/embeds_ga.npz /path/to/batchcorr_ga_results.h5ad 0 3
```

## Baselines

All baseline scripts are listed under `./baselines/`

**Note**: you may need to change values around in the used config.yaml files to suit your set-up.

### Upscaling

For the PCA upscaling baseline:
```bash
python ./baselines/upscale/pca_upscale.py ./baselines/upscale/pca_train.yaml ./baselines/upscale/pca_test.yaml /path/to/scTab.h5t /path/to/scTab_upsc_val.h5t /path/to/scTab_upsc_test.h5t
```

### GRN Inference

For the raw count co-expression baseline:
```bash
python ./baselines/grninfer/coexpr_raw.py ./baselines/grninfer/raw.yaml /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather /path/to/scTab_grn_val.h5t val
python ./baselines/grninfer/coexpr_raw.py ./baselines/grninfer/raw.yaml /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather /path/to/scTab_grn_test.h5t test
```

For the pseudobulked co-expression baseline
```bash
python ./baselines/grninfer/coexpr_pseudobulk.py ./baselines/grninfer/pseudobulk.yaml /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather /path/to/scTab_grn_val.h5t val
python ./baselines/grninfer/coexpr_pseudobulk.py ./baselines/grninfer/pseudobulk.yaml /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather /path/to/scTab_grn_test.h5t test
```

### Post-perturbation expression prediction

For the MLP-Mixer:
```bash
python ./baselines/perturb/mlp_mixer.py ./baselines/perturb/mlp_mixer.yaml /path/to/logs_perturb_mlpmixer/ --lr 0.01 --init_factor 1 --batch_size 32 --data_path /path/to/perturb.h5t
```

For the unpre-trained transformer:
```bash
python ./baselines/perturb/unpretrained.py ./baselines/perturb/unpretrained.yaml /path/to/logs_perturb_unpre/ --lr 0.00007 --init_factor 1 --batch_size 32 --data_path /path/to/perturb.h5t 
```

### Cell-type identification

For the LR model:
```bash
python ./baselines/celltypeid/logistic.py ./baselines/celltypeid/logistic.yaml /path/to/logs_celltypeid_logistic/ --lr 0.0007 --data_path /path/to/scTab.h5t --batch_size 1024 --n_workers 8 --prefetch_factor 2
```

For the MLP:
```bash
python ./baselines/celltypeid/mlp.py ./baselines/celltypeid/mlp.yaml /path/to/logs_celltypeid_mlp/ --lr 0.007 --data_path /path/to/scTab.h5t --batch_size 1024 --n_workers 8 --prefetch_factor 2
```

For the unpre-trained transformer:
```bash
python ./baselines/celltypeid/unpretrained.py ./baselines/celltypeid/unpretrained.yaml /path/to/logs_celltypeid_unpre/ --data_path /path/to/scTab.h5t --lr 0.0003
```

### Protein concentration prediction

For the LR model:
```bash
python ./baselines/protconc/logistic.py ./baselines/protconc/logistic.yaml /path/to/logs_protconc_logistic/ --lr 0.0003 --batch_size 256 --n_workers 8 --data_path /path/to/citeseq.h5t
```

For the MLP:
```bash
python ./baselines/protconc/mlp.py ./baselines/protconc/mlp.yaml /path/to/logs_protconc_mlp/ --lr 0.007 --batch_size 128 --n_workers 8 --data_path /path/to/citeseq.h5t
```

For the unpre-trained transformer:
```bash
python ./baselines/protconc/unpretrained.py ./baselines/protconc/unpretrained.yaml /path/to/logs_protconc_unpre/ --lr 0.0001 --data_path /path/to/citeseq.h5t
```

### Batch correction

HVG2000:
```bash
python ./baselines/batchcorr/hvg2000.py ./baselines/batchcorr/hvg2000_el.yaml /path/to/batchcorr_hvg2000_el_res.h5ad 0 2 --data_path /path/to/batchcorr_el.h5t
python ./baselines/batchcorr/hvg2000.py ./baselines/batchcorr/hvg2000_ci.yaml /path/to/batchcorr_hvg2000_ci_res.h5ad 0 2 --data_path /path/to/batchcorr_ci.h5t
python ./baselines/batchcorr/hvg2000.py ./baselines/batchcorr/hvg2000_ga.yaml /path/to/batchcorr_hvg2000_ga_res.h5ad 0 3 --data_path /path/to/batchcorr_ga.h5t
```

HVG5000:
```bash
python ./baselines/batchcorr/hvg5000.py ./baselines/batchcorr/hvg5000_el.yaml /path/to/batchcorr_hvg5000_el_res.h5ad 0 2 --data_path /path/to/batchcorr_el.h5t
python ./baselines/batchcorr/hvg5000.py ./baselines/batchcorr/hvg5000_ci.yaml /path/to/batchcorr_hvg5000_ci_res.h5ad 0 2 --data_path /path/to/batchcorr_ci.h5t
python ./baselines/batchcorr/hvg5000.py ./baselines/batchcorr/hvg5000_ga.yaml /path/to/batchcorr_hvg5000_ga_res.h5ad 0 3 --data_path /path/to/batchcorr_ga.h5t
```

All genes (no hvg selection):
```bash
python ./baselines/batchcorr/hvgall.py ./baselines/batchcorr/hvgall.yaml /path/to/batchcorr_hvgall_el_res.h5ad 0 2 --data_path /path/to/batchcorr_el.h5t
python ./baselines/batchcorr/hvgall.py ./baselines/batchcorr/hvgall.yaml /path/to/batchcorr_hvgall_ci_res.h5ad 0 2 --data_path /path/to/batchcorr_ci.h5t
python ./baselines/batchcorr/hvgall.py ./baselines/batchcorr/hvgall.yaml /path/to/batchcorr_hvgall_ga_res.h5ad 0 3 --data_path /path/to/batchcorr_ga.h5t
```