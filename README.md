# Mini-Diffuse-Actor

## Highlights

- Achieves **95%** of the performance of [3D-Diffuser-Actor](https://github.com/nickgkan/3d_diffuser_actor) on the RLBench-18 multi-task benchmark, with just **13 hours** of training on a single RTX 4090 GPU, or **1 Day** on an A100. Check our [**Wandb Reports**](https://api.wandb.ai/links/hu2240877635/4r8wa4rt).



- Codebase for RLbench training and headless test. As well as real world training and testing. 

- Checkpoints, training logs and test logs fully available. 

- Training and test also run-able on Cloud platforms. E.g. on Vast.ai, you only needs \< $5 to train your own mini-diffuser.

This code space builds upon [3D-LUTOS](https://github.com/vlc-robot/robot-3dlotus/) — many thanks to the original authors for open-sourcing their container and providing a robust headless training-evaluation-testing pipeline. The idea emerged from comparing [Act3D](https://github.com/nickgkan/3d_diffuser_actor) and [3D diffuser Actor](https://github.com/zhouxian/act3d-chained-diffuser).

🚧 Project in progress

## Installation

Check [INSTALL.md](), for training-only usage, you don't need to install the simulator.

## Training

```
python train/train_diffusion_policy.py
```

for result reproduce, make sure the args aligns with the [wandb logs](https://api.wandb.ai/links/hu2240877635/4r8wa4rt). E.g.

```
python train/train_diffusion_policy.py MODEL.mini_batches=256 TRAIN.learning_rate=3e-4 wandb_name=rlbench18_256 TRAIN_DATASET.num_points=4000 VAL_DATASET.num_points=4000 MODEL.diffusion.total_timesteps=100 TRAIN.num_epochs=800 SEED=2024 TRAIN.lr_sched=cosine TRAIN.num_cosine_cycles=0.6
```


## Testing

you can test your own model or download pretrained checkpoints [Here](https://huggingface.co/datasets/you2who/minidiffuser/tree/main).

make sure the RLBench simulator works well according to [INSTALL.md]().

```
python minidiffuser/evaluation/eval_simple_policy_parrallel.py
```

for headless run
```
xvfb-run -a python minidiffuser/evaluation/eval_simple_policy_parrallel.py
```
for result reproducation, you can do. 
```
bash scripts/locals_policy_peract.sh 
```
Remember to delete the results in the seed folder, otherwise runs with the seeds that already exist will be skipped.



## Realworld Experiemnts

Check this [repo](https://github.com/utomm/fr3_ws) for our real world setup.

we will update how to do:

### Data collection

### Preprocess collected data

### Training

### Test in real world