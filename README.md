
<h1 align="center">
Adversarial Environment Design <br> via Regret-Guided Diffusion Models
</h1>

<h3 align="center"><a href="">Paper</a> | <a href="https://rllab-snu.github.io/projects/ADD/">Project Page</a></h3>
<div align="center">
</div>

<p align="center">
Official Github repository for <b>"Adversarial Environment Design via Regret-Guided Diffusion Models"</b>.
<br>
$\color{#00FFFF}{\textsf{spotlighted paper at NeurIPS 2024}}$
<br>
<br>
<br>

This codebase is implemented on the top of [Dual Curriculum Design](https://github.com/facebookresearch/dcd) and [diffusion-human-feedback](https://github.com/tetrzim/diffusion-human-feedback).

## Setup
To install the necessary dependencies, run the following commands:
```
conda env create -f environment.yaml
conda activate add
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install pyglet==1.5.11
```
Ignore error messages regarding dependecies. But you may need to install additional packages (ex. six, xvfb)

You may need to separately install `cudatoolkit` within the virtual environment (especially if the experiment procedure below produces errors related to `from torch._C import *`):

```
conda install cudatoolkit=11.8 -c pytorch -c nvidia
```

## Diffusion pre-training
```
cd diffusion_human_feedback

# for Minigrid
MODEL_FLAGS="--image_size 16 --image_channels 3 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 256 --save_interval 100000"
LOG_DIR="log/minigrid_60_uniform" # The diffusion model will be saved in .pt format within the directory specified by this path.
NUM_GPUS="1" # The number of GPUs used in parallel computing. If larger than 1, adjust the batch_size argument accordingly.

echo $(mpiexec -n $NUM_GPUS python image_train.py --log_dir=$LOG_DIR --data_dir=minigrid_60_uniform --rgb=True --random_flip=False $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS)

# for BipedalWalker
python datasets/bipedal.py
python flat_train.py
```

## Run experiments
Before running the following commands, you must check "log_dir" and "generator_model_path" in the json file first.

```
# for Minigrid
python train_scripts/make_cmd.py --json minigrid/60_blocks_uniform/mg_60b_add --num_trials {number of independent seeds}

# for BipedalWalker
python train_scripts/make_cmd.py --json bipedal/bipedal_add --num_trials {number of independent seeds}

chmod +x run.sh
sh run.sh
```

## Evaluation
```
python -m eval \
--base_path <log_dir> \
--xpid <xpid> \
--model_tar <model>
--benchmark <maze or bipedal> \
--num_episodes <num_episodes> \
```
