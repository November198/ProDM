# ProDM: Progress–Density Monitoring

Code for the main Robomimic experiments with flow-matching (FM) and diffusion model (DM) policies. 

## Environment

The policy code builds on [diffusion_policy](https://github.com/real-stanford/diffusion_policy). From the repository root:

```bash
mamba env create -f conda_environment.yaml
conda activate prodm
```

## Prepare models

The four tasks are `square`, `can`, `transport`, and `tool_hang`.  For example, train the FM Square policy and obtain its representations:

```bash
python train.py --config-name=image_square_ph_visual_flow_policy_cnn.yaml \
  training.seed=1103 training.device=cuda:0 \
  hydra.run.dir='data/outputs/${name}_${task_name}'
python save_data.py --config-name=image_square_ph_visual_flow_policy_cnn.yaml \
  training.seed=1103 training.device=cuda:0
```

Train the scoring model:

```bash
cd UQ_baselines/logpZO && python train.py --policy_type flow --type square
```

Use the matching task/policy configuration for other runs.

## Collect and evaluate

Collect ID and OOD rollouts for each task/policy pair:

```bash
cd UQ_test && python eval_together.py --policy_type flow --task_name square --device 0 --modify false --num 2000
cd UQ_test && python eval_together.py --policy_type flow --task_name square --device 0 --modify true --num 2000
```

Evaluate ProDM on all four tasks and both policies using the collected data:

```bash
python UQ_test/prodm_detector.py --data-root data
```

To reproduce the comparison with the original baselines:

```bash
cd UQ_test && python plot_with_CP_band.py
cd UQ_test && python plot_with_CP_band.py --diffusion_policy
```
