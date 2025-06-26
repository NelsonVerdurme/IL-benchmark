# Mini-Diffuse-Actor

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/train-a-multi-task-diffusion-policy-on/robot-manipulation-on-rlbench)](https://paperswithcode.com/sota/robot-manipulation-on-rlbench?p=train-a-multi-task-diffusion-policy-on) [![arXiv](https://img.shields.io/badge/arXiv-2505.09430-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.09430)
![image](https://github.com/user-attachments/assets/433c8532-4eb2-49f7-8424-58a9cb29de23)

## Highlights

- Achieves **95%** of the performance of [3D-Diffuser-Actor](https://github.com/nickgkan/3d_diffuser_actor) on the RLBench-18 multi-task benchmark, with just **13 hours** of training on a single RTX 4090 GPU, or **1 day** on an A100. Check our [**Wandb Reports**](https://api.wandb.ai/links/hu2240877635/4r8wa4rt).
- Complete codebase for RLBench training and headless testing, as well as real-world training and evaluation.
- Includes checkpoints, training logs, and test logs.
- Training and testing can be run on cloud platforms. For example, on Vast.ai it costs less than **$5** to train your own Mini-Diffuser.

This codebase builds upon [3D-LUTOS](https://github.com/vlc-robot/robot-3dlotus/) — many thanks to the original authors for open-sourcing their container and providing a robust headless training-evaluation-testing pipeline. The idea was inspired by comparing [3D-Diffuser-Actor](https://github.com/nickgkan/3d_diffuser_actor) and [Act3D](https://github.com/zhouxian/act3d-chained-diffuser).

🚧 Project in progress

## Installation

See [INSTALL.md](https://github.com/utomm/mini-diffuse-actor/blob/master/INSTALL.md).  
For training-only usage, you don’t need to install the simulator.

## Training

Run:

```
python train/train_diffusion_policy.py
```

To reproduce results, ensure arguments match those used in the [wandb logs](https://api.wandb.ai/links/hu2240877635/4r8wa4rt). Example:

```
python train/train_diffusion_policy.py MODEL.mini_batches=256 TRAIN.learning_rate=3e-4 wandb_name=rlbench18_256 TRAIN_DATASET.num_points=4000 VAL_DATASET.num_points=4000 MODEL.diffusion.total_timesteps=100 TRAIN.num_epochs=800 SEED=2024 TRAIN.lr_sched=cosine TRAIN.num_cosine_cycles=0.6
```

## Testing

You can test your own model or download pretrained checkpoints [here](https://huggingface.co/datasets/you2who/minidiffuser/tree/main).

Ensure the RLBench simulator is working properly according to [INSTALL.md](https://github.com/utomm/mini-diffuse-actor/blob/master/INSTALL.md).

Run:

```
python minidiffuser/evaluation/eval_simple_policy_parrallel.py
```

For headless testing:

```
xvfb-run -a python minidiffuser/evaluation/eval_simple_policy_parrallel.py
```

To reproduce benchmark results:

```
bash scripts/locals_policy_peract.sh 
```

⚠️ Remember to delete results in the seed folder first — otherwise, runs with existing seeds will be skipped.

## Real-World Experiments

See our real-world setup in this [repo](https://github.com/utomm/fr3_ws). In short, we seperate the ROS and CUDA learning environment and use websocket to communicate, to aviod python version conflict.

![image](https://github.com/user-attachments/assets/66235340-7ba0-48a4-ab98-38d78156070a)

In that repo we will update with guides on:

### Data Collection

### Preprocessing Collected Data

### Training

### Real-World Testing
