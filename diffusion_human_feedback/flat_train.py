"""
Train a diffusion model on images.
"""

import argparse
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_flat_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    diffusion_defaults,
    create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.unet import UNetModel, SimpleFlowModel
from guided_diffusion.train_util import TrainLoop
import torch as th


def main():
    args = create_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    # model = UNetModel(
    #     image_size = 16,
    #     in_channels = 2,
    #     model_channels = 128,
    #     out_channels = 2,
    #     num_res_blocks = 3,
    #     attention_resolutions=[2, 1],
    #     dropout=False,
    #     channel_mult=(1,2,2,2),
    #     dims=1,
    #     num_classes=None,
    #     use_checkpoint=False,
    #     use_fp16=False,
    #     num_heads=4,
    #     num_head_channels=-1,
    #     num_heads_upsample=-1,
    #     use_scale_shift_norm=True,
    #     resblock_updown=False,
    #     use_new_attention_order=False,
    # )
    model = SimpleFlowModel(data_shape=(8, 1), hidden_dim=256)

    diffusion_dict = diffusion_defaults()
    del diffusion_dict["diffusion_steps"]
    del diffusion_dict["use_ldm"]
    del diffusion_dict["ldm_config_path"]
    diffusion_dict["steps"] = 1000
    diffusion = create_gaussian_diffusion(**diffusion_dict)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_flat_data(
        data_dir=args.data_dir, batch_size=args.batch_size, deterministic=True
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="bipedal_data",
        log_dir="log/bipedal",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=512,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        gpu_idx="1",
    )
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
