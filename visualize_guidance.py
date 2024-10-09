import random
import time
import gym
import gym_minigrid.minigrid as minigrid
import networkx as nx
from networkx import grid_graph
import numpy as np
import torch as th
import gym_minigrid.minigrid as minigrid
import matplotlib.pyplot as plt
import argparse

from envs.multigrid.adversarial import (
    AdversarialEnv,
    GoalLastVariableBlocksAdversarialEnv,
)
from envs.box2d import *
from envs.bipedalwalker import *
from diffusion_human_feedback.predictor import Tutor
from diffusion_human_feedback.generator import EnvGenerator
from util import create_parallel_env, DotDict, str2bool, seed

from arguments import parser
import pyvirtualdisplay


def parse_args():
    parser = argparse.ArgumentParser(description="Eval")

    parser.add_argument(
        "--use_categorical_tutor",
        type=str2bool,
        default=True,
        help="make tutor networks to output logits of the categorical distribution",
    )

    parser.add_argument(
        "--num_bins_tutor",
        type=int,
        default=100,
        help="the number of bins which divide the return domain",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parser.add_argument(
        "--tutor_dir",
        type=str,
        default="../logs/bipedal/add/seed_2/tutors/",
        help="path to the directory where tutor networks are saved",
    )
    # "../logs/minigrid_60/add/seed_3_exponential_cvar_03_guide_5/tutors/"
    parser.add_argument(
        "--tutor_model", type=str, default="model_060000.pt", help="name of model file"
    )
    # "model_030000.pt"

    args = parser.parse_args()

    args.regret_metric = "difficulty_level"
    # args.regret_guidance_weight = 7.0
    # args.use_categorical_tutor = True
    # args.num_bins_tutor = 100
    # args.env_name = "MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7"
    # args.num_processes = 1

    # args.regret_guidance_weight = 30.0
    # args.use_categorical_tutor = True
    # args.num_bins_tutor = 100
    # args.env_name = "CarRacing-Bezier-Adversarial-v0"
    # args.num_processes = 1

    args.regret_guidance_weight = 30.0
    args.use_categorical_tutor = True
    args.num_bins_tutor = 100
    args.env_name = "BipedalWalker-Adversarial-v0"
    args.num_processes = 1
    env = BipedalWalkerAdversarialEnv()

    tutor = Tutor(args)
    ckpt = th.load(args.tutor_dir + args.tutor_model)
    tutor.load_model(ckpt)
    tutor_list = [tutor]

    generator = EnvGenerator(args)
    initial_noise = generator.create_initial_noise()
    print(initial_noise.shape)
    # guided_envs = generator.generate_guided_env(args.num_processes, tutor_list)

    levels = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    for i in range(10):
        level = levels[i]
        guided_envs = generator.generate_guided_env(
            args.num_processes, tutor_list, initial_noise=initial_noise, level=level
        )
        print(guided_envs)
        env.reset_to_generated_img(guided_envs)
        for t in range(2):
            state = env.render(mode="rgb_array")
            plt.imshow(env.viewer.get_array())
            plt.savefig(f"tt/{i}.png", dpi=300)
    # seed(1)
