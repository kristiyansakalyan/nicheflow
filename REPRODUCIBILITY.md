# Reproducibility

## Tables

For all experiments described below, you can generate the tables shown in our paper using the [`print_eval_results`](notebooks/print_eval_results.ipynb) notebook after running the corresponding evaluation scripts.

## Main results
To reproduce the main results reported in the paper, you can use the pre-trained checkpoints provided in the `main_ckpts.zip` file and run the evaluation scripts.

Before proceeding, make sure you have followed the instructions in the main [`README`](README.md) and have downloaded both the datasets and the main checkpoints.

Once everything is set up, run:
```bash
python nicheflow/eval_state_dict_ckpts.py
```

## $K$ Regions Ablation Study
To reproduce the results of the $K$ regions ablation study, download the `kregion_ablations_ckpts.zip` from [FigShare](https://figshare.com/articles/software/NicheFlow_-_Data_Checkpoints_and_Results/30426610) and extract the checkpoints into the [`ckpts/kregion_ablations`](ckpts/kregion_ablations/) folder. 

Once everything is set up, run
```bash
python nicheflow/eval_state_dict_ckpts_kregion_ablations.py
```

## $\lambda$ OT Ablation Study
To reproduce the results of the $\lambda$ OT ablation study, download the `ot_ablations_ckpts.zip` file from [FigShare](https://figshare.com/articles/software/NicheFlow_-_Data_Checkpoints_and_Results/30426610) and extract the checkpoints into the [`ckpts/ot_ablations`](ckpts/ot_ablations/) folder. 


Once everything is set up, run:
```bash
python nicheflow/eval_state_dict_ckpts_ot_ablations.py
```
