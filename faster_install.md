conda create -n gembench python=3.10
conda activate gembench


conda install -c nvidia cuda=12.1.0
conda install -c pytorch -c nvidia pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1

conda create -n gembench python=3.10 cuda=12.1.0 pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

conda create -n gembench python=3.10     pytorch torchvision pytorch-cuda=12.1 nvidia/label/cuda-12.1.0::cuda-toolkit     -c pytorch -c conda-forge
conda install -c https://repo.prefix.dev/conda-forge flash-attn
conda install conda-forge::torch-scatter
conda install -c conda-forge numpy scipy matplotlib scikit-learn opencv open3d Flask h5py notebook jupyterlab pyyaml pytest tqdm wandb setuptools -y


pip install lxml==5.3.0 absl-py==2.1.0 accelerate==0.30.1 addict==2.4.0 easydict==1.13 filelock==3.16.1 groq==0.11.0 gym==0.26.2 hydra-core==1.3.2 jsonlines==4.0.0 lmdb==1.4.1 msgpack-numpy==0.4.8 msgpack-python==0.5.6 omegaconf==2.3.0 open-clip-torch==2.24.0 Pillow==10.4.0 Requests==2.32.3 spconv-cu121 tensorboardX==2.6.2.2 termcolor==2.5.0 timm==1.0.9 transformers==4.41.2 typed-argument-parser==1.10.0 yacs==0.1.8 diffusers
