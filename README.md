<div align="center">

<img src="https://raw.githubusercontent.com/gdewael/bento-sc/refs/heads/main/assets/bento.svg" align="center" width="450" alt="bento-sc">

<h1></h1>

# Reproducibility

</div>

This repository contains supplementary code and config files to reproduce all experimental results presented in the bento-sc paper. Please refer to:
- The main [`bento-sc` repository](https://github.com/gdewael/bento-sc) for the majority of single-cell language modeling code and scripts.
- The main [documentation page](https://bento-sc.readthedocs.io/en/latest/index.html) for instructions on getting started with `bento-sc` in general.

## Instructions

### Installing and preparing data

For installation and data preparation instructions, please refer to the [Getting started](https://bento-sc.readthedocs.io/en/latest/getting_started.html) of our documentation.

### scLM pre-training and evaluation

All config files to pre-train and evaluate all models are in `./scLM_configs/`.
Config files are categorized per experiment in our original study. For example, config files for the "base" and "final" scLMs in our study are located under: `./scLM_configs/exp_A/binned/` and `./scLM_configs/exp_D/mlm_ctclf_contr/`, respectively.

Here, by example, we provide all steps needed to fully reproduce pre-training and evaluation of these two models.

**Note:** you may need to change values around in the config.yaml files to suit your set-up.
In addition, many of the scripts come with additional flags that are relevant for only some of the scLM models. Before running, make sure you are using the appropriate optional arguments for your model set-up.

#### Base design

```bash
bentosc_pretrain ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ --lr 0.0003 --data_path /path/to/scTab.h5t
```

Evaluation:
```bash
# upscaling on val and test
bentosc_task_upscale ./scLM_configs/exp_A/binned/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_file /path/to/scTab_upsc_val.h5t --clf_output True
bentosc_task_upscale ./scLM_configs/exp_A/binned/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_file /path/to/scTab_upsc_test.h5t --clf_output True

# grn infer on val and test
bentosc_task_grninfer embed ./scLM_configs/exp_A/binned/grn_infer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_val/ --data_path /path/to/scTab_grn_val.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_A/binned/grn_infer.yaml /path/to/embeds_grn_binned_val/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_val.h5t --test_mode val

bentosc_task_grninfer embed ./scLM_configs/exp_A/binned/grn_infer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_test/ --data_path /path/to/scTab_grn_test.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_A/binned/grn_infer.yaml /path/to/embeds_grn_binned_test/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_test.h5t --test_mode test

# post perturbation expression pred
bentosc_task_perturb ./scLM_configs/exp_A/binned/perturb.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_perturb/ --data_path /path/to/perturb.h5t --init_factor 1 --batch_size 32 --lr 0.00007

# celltype id
bentosc_task_celltypeid ./scLM_configs/exp_A/binned/celltypeid.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_celltypeid/ --lr 0.0003 --data_path /path/to/scTab.h5t

# prot conc pred
bentosc_task_protconc ./scLM_configs/exp_A/binned/protconc.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_protconc/ --lr 0.0001 --data_path /path/to/citeseq.h5t

# batch correction on 3 datasets.
bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_el.npz --data_path /path/to/batchcorr_el.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_el.h5t /path/to/embeds_el.npz /path/to/batchcorr_el_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ci.npz --data_path /path/to/batchcorr_ci.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ci.h5t /path/to/embeds_ci.npz /path/to/batchcorr_ci_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ga.npz --data_path /path/to/batchcorr_ga.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ga.h5t /path/to/embeds_ga.npz /path/to/batchcorr_ga_results.h5ad 0 3
```

#### Base design

```bash
bentosc_pretrain ./scLM_configs/exp_A/binned/pretrain.yaml /path/to/logs_binned/ --lr 0.0003 --data_path /path/to/scTab.h5t
```

Evaluation:
```bash
# upscaling on val and test
bentosc_task_upscale ./scLM_configs/exp_D/mlm_ctclf_contr/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_file /path/to/scTab_upsc_val.h5t --clf_output False
bentosc_task_upscale ./scLM_configs/exp_D/mlm_ctclf_contr/upscale.yaml /path/to/logs_binned/ckpt.ckpt --data_file /path/to/scTab_upsc_test.h5t --clf_output False

# grn infer on val and test
bentosc_task_grninfer embed ./scLM_configs/exp_D/mlm_ctclf_contr/grn_infer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_val/ --data_path /path/to/scTab_grn_val.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_D/mlm_ctclf_contr/grn_infer.yaml /path/to/embeds_grn_binned_val/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_val.h5t --test_mode val

bentosc_task_grninfer embed ./scLM_configs/exp_D/mlm_ctclf_contr/grn_infer.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_grn_binned_test/ --data_path /path/to/scTab_grn_test.h5t
bentosc_task_grninfer eval ./scLM_configs/exp_D/mlm_ctclf_contr/grn_infer.yaml /path/to/embeds_grn_binned_test/ /path/to/ext_pertdata.h5ad /path/to/scenicdb.feather --data_path /path/to/scTab_grn_test.h5t --test_mode test

# post perturbation expression pred
bentosc_task_perturb ./scLM_configs/exp_D/mlm_ctclf_contr/perturb.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_perturb/ --data_path /path/to/perturb.h5t --init_factor 1 --batch_size 32 --lr 0.00007

# celltype id
bentosc_task_celltypeid ./scLM_configs/exp_D/mlm_ctclf_contr/celltypeid.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_celltypeid/ --lr 0.0003 --data_path /path/to/scTab.h5t

# prot conc pred
bentosc_task_protconc ./scLM_configs/exp_D/mlm_ctclf_contr/protconc.yaml /path/to/logs_binned/ckpt.ckpt /path/to/logs_binned/logs_protconc/ --lr 0.0001 --data_path /path/to/citeseq.h5t

# batch correction on 3 datasets.
bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_el.npz --data_path /path/to/batchcorr_el.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_el.h5t /path/to/embeds_el.npz /path/to/batchcorr_el_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ci.npz --data_path /path/to/batchcorr_ci.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ci.h5t /path/to/embeds_ci.npz /path/to/batchcorr_ci_results.h5ad 0 2

bentosc_task_batchcorr embed ./scLM_configs/exp_D/mlm_ctclf_contr/pretrain.yaml /path/to/logs_binned/ckpt.ckpt /path/to/embeds_ga.npz --data_path /path/to/batchcorr_ga.h5t
bentosc_task_batchcorr correct /path/to/batchcorr_ga.h5t /path/to/embeds_ga.npz /path/to/batchcorr_ga_results.h5ad 0 3
```

### Baselines