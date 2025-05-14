# Installation Instructions

1. Install general python packages
```bash
conda create -n midi python==3.10

conda activate midi
```

2. manually install cuda and torch. To allow local compiled RLbench co-exist with torch, we have to install torch from pip, to avoid MKL, gcc, ninja issues ...

```
conda install -c nvidia/label/cuda-12.1.0 cuda

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

(Optional, but recommend) If you are using GPUs higher than V100, compile and install flash-attn

```
export CUDA_HOME=$HOME/conda/envs/midi
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

pip install flash_attn==2.5.9.post1

```

3. Install wheels
```
pip install torch_scatter==2.1.2 --find-links https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html
```
4. Install other requirements
```
pip install -r requirements.txt
```

# Install This Package
```
pip install -e .
```

# Peract-18 Datasets in RLbench

Here we borrow the dataset generate by [3D-LUTOS](https://github.com/vlc-robot/robot-3dlotus/), with their pre-generated dataset we don't need to install RLbench for loading data and training.

The RLBench-18task dataset (peract) can be downloaded [here](https://huggingface.co/datasets/rjgpinel/RLBench-18Task/tree/main)

change the `TRAIN_DATASET.data_dir` in `minidiffuser/train/diffusion_ptv3.yaml` to where you store the dataset.


**You can begin to train a mini-diffuser now.**

# For headless RLBench evaluation

1. Install x11 related lib

```
sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev apt-get install -y --no-install-recommends libgl1-mesa-dev xvfb dbus-x11 x11-utils libxkbcommon-x11-0
```

2. Install PyRep and RLbench
```bash
mkdir dependencies
cd dependencies
```

Download CoppeliaSim (see instructions [here](https://github.com/stepjam/PyRep?tab=readme-ov-file#install))
```bash
# change the version if necessary
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Add the following to your ~/.bashrc file:
```bash
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Install Pyrep and RLBench
```bash
git clone https://github.com/cshizhe/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
cd ..

# GemBench modified version of RLBench, fork to fix version.
git clone https://github.com/utomm/RLBench.git
cd RLBench
pip install -r requirements.txt
pip install .
cd ../..
```

4. Running headless

```
export COPPELIASIM_ROOT=${HOME}/mini-diffuse-actor/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT


expr_dir=/home/huser/mini-diffuse-actor/experiments/minidiff
ckpt_step=95200

for seed in 0 1 2
do
xvfb-run -a python minidiffuser/evaluation/eval_simple_policy_parrallel.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 6 \
    --taskvar_file assets/taskvars_peract.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir /home/huser/data/RLBench-18Task/test/microsteps # --record_video False
done
```

or 

```
bash scripts/locals_policy_peract.sh
```

