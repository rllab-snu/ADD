import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import blobfile as bf
from diffusion_human_feedback.guided_diffusion.script_util import (
    classifier_defaults,
    create_classifier,
)
import torch as th


def parse_args():
    parser = argparse.ArgumentParser(description="RL")

    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["dr", "paired", "robust_plr", "accel", "add_random", "add"],
        help="Name of algorithms to analyze",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        default=[],
        help="Name of condition corresponding to each results file.",
    )
    parser.add_argument(
        "--row_prefix",
        type=str,
        default="solved_rate",
        help="Plot rows in results .csv whose metric column matches this prefix.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        default=["Labyrinth", "Maze"],
        help="List of metric names to plot.",
    )
    parser.add_argument(
        "--ylabel", type=str, default="Solved rate", help="Y-axis label."
    )
    parser.add_argument(
        "--savename",
        type=str,
        default="latest",
        help="Filename of saved .pdf of plot, saved to figures/.",
    )
    parser.add_argument(
        "--figsize", type=str, default="(14,2)", help="Dimensions of output figure."
    )

    return parser.parse_args()


def env_pre_processing(env, env_name="maze"):
    if env_name == "maze":
        result = -np.ones((3, 15, 15), dtype=np.float32)
        env = env[:, :, 0]

        result[0][env == 2] = 1
        start_position = np.where(env == 10)
        result[1][start_position] = 1
        result[1][start_position[0] + 1, start_position[1]] = 128 / 127.5 - 1
        result[2][env == 8] = 1

        result = np.pad(
            result, ((0, 0), (0, 1), (0, 1)), "constant", constant_values=-1
        )
        result[0, -1, :] = 1
        result[0, :, -1] = 1
    else:
        result = env
    return result


def _list_numpy_files_recursively(data_dir_list, env_name="maze", batch_size=4):
    files = []
    algo_idxs = [0]
    for data_dir in data_dir_list:
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["npy"]:
                files.append(full_path)
            elif bf.isdir(full_path):
                files.extend(_list_numpy_files_recursively(full_path))
        algo_idxs.append(len(files) * batch_size)

    results = []
    for file in files:
        batched_env = np.load(file)
        for env in batched_env[:batch_size]:
            results.append(env_pre_processing(env, env_name))

    return np.asarray(results), algo_idxs


def to_latent_feature(env_data, tutor_network, batch_size=256):
    num_partition = env_data.shape[0] // batch_size + 1
    result = np.zeros((env_data.shape[0], 256, 2, 2))
    for i in range(num_partition):
        raw_env_batch = env_data[
            i * batch_size : min((i + 1) * batch_size, env_data.shape[0])
        ]
        raw_env_batch = th.from_numpy(raw_env_batch).to("cuda:0")
        latent_env_batch = tutor_network.get_feature_vector(raw_env_batch)
        result[i * batch_size : min((i + 1) * batch_size, env_data.shape[0])] = (
            latent_env_batch.cpu().detach().numpy()
        )

    return result


if __name__ == "__main__":

    args = parse_args()

    # model_dict = classifier_defaults()
    # model_dict["image_size"] = 16
    # model_dict["image_channels"] = 3
    # model_dict["classifier_width"] = 128
    # model_dict["classifier_depth"] = 2
    # model_dict["classifier_attention_resolutions"] = "16, 8, 4"
    # model_dict["output_dim"] = 100
    # tutor_network = create_classifier(**model_dict)
    # tutor_ckpt = th.load('../logs/minigrid_60/add/seed_3_cvar_015/tutors/model_030000.pt')
    # tutor_network.load_state_dict(tutor_ckpt['model'])
    # tutor_network.to("cuda:0")

    # env_dir_list = ["../logs/minigrid_60/dr/seed_1/env_params/",
    # 			 "../logs/minigrid_60/add/seed_3_cvar_015/env_params/",
    # 			 "../logs/minigrid_60/add_random/seed_1_uniform_60/env_params/",
    # 			 "../logs/minigrid_60/paired/seed_1/env_params",
    # 			 "../logs/minigrid_60/robust_plr/seed_5/env_params",
    # 			 "../logs/minigrid_60/accel/seed_1/env_params"]
    env_dir_list = [
        "../logs/bipedal/dr/seed_1/env_params/",
        "../logs/bipedal/add/seed_2/env_params/",
        "../logs/bipedal/add_random/seed_1/env_params/",
        "../logs/bipedal/paired/seed_1/env_params",
        "../logs/bipedal/robust_plr/seed_1/env_params",
        "../logs/bipedal/accel/seed_1/env_params",
    ]
    algo_names = ["DR", "ADD", "ADD w/o guidance", "PAIRED", "PLR$^{\perp}$", "ACCEL"]
    algo_color_map = [
        plt.cm.Greys,
        plt.cm.Blues,
        plt.cm.Blues,
        plt.cm.Reds,
        plt.cm.Oranges,
        plt.cm.Greens,
    ]
    algo_save_names = ["DR", "ADD", "ADD_no_guidance", "PAIRED", "PLR", "ACCEL"]

    env_data, algo_idxs = _list_numpy_files_recursively(
        env_dir_list, env_name="bipedal"
    )
    print(algo_idxs)
    # to_latent_feature(env_data, tutor_network)
    env_data = env_data.reshape((env_data.shape[0], -1))
    print("start tsne")

    tsne = TSNE(random_state=42)
    env_data_tsne = tsne.fit_transform(env_data)

    # fig = plt.figure(figsize=(4, 8))
    # save_algo_idx = [1,3]
    # for j in range(2):
    # 	# plt.figure(figsize=(10,10))
    # 	plt.subplot(2,1,j+1)
    # 	i = save_algo_idx[j]
    # 	plt.title(algo_names[i], fontsize=18)
    # 	plt.xlim(env_data_tsne[:, 0].min(), env_data_tsne[:, 0].max())
    # 	plt.ylim(env_data_tsne[:, 1].min(), env_data_tsne[:, 1].max())
    # 	plt.xticks(fontsize=13)
    # 	plt.yticks(fontsize=13)
    # 	# plt.xticks([], [])
    # 	# plt.yticks([], [])
    # 	plt.scatter(env_data_tsne[algo_idxs[i]:algo_idxs[i+1],0], env_data_tsne[algo_idxs[i]:algo_idxs[i+1], 1],
    # 		  c=np.arange(algo_idxs[i+1] - algo_idxs[i]), cmap=algo_color_map[i], alpha=1.0, edgecolors=None, s=10)
    # plt.subplots_adjust(wspace=0.5, hspace=0.35, top=0.88, bottom=0.08, left=0.2)
    # fig.text(0.5, 0.95, '(d) t-SNE plots', ha='center', fontsize=20)
    # plt.savefig("../logs/minigrid_60/tsne.png")
    for i in range(len(algo_idxs) - 1):
        plt.figure(figsize=(5, 4))
        plt.title(algo_names[i], fontsize=20)
        plt.xlim(env_data_tsne[:, 0].min(), env_data_tsne[:, 0].max())
        plt.ylim(env_data_tsne[:, 1].min(), env_data_tsne[:, 1].max())
        plt.scatter(
            env_data_tsne[algo_idxs[i] : algo_idxs[i + 1], 0],
            env_data_tsne[algo_idxs[i] : algo_idxs[i + 1], 1],
            c=np.linspace(0, 2000000000, algo_idxs[i + 1] - algo_idxs[i]),
            cmap=algo_color_map[i],
            alpha=1.0,
            edgecolors=None,
            s=10,
        )
        cbar = plt.colorbar()
        cbar.set_label("Steps", rotation=270, fontsize=13, labelpad=15)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=13)
        # cbar.ax.set_yticklabels(fontsize=13)
        plt.savefig(algo_save_names[i] + ".png")
