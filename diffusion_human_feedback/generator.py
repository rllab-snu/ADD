import numpy as np
import torch as th
import torch.distributed as dist
from functools import partial

from .guided_diffusion import dist_util
from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    diffusion_defaults,
    create_gaussian_diffusion,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
)
from .guided_diffusion.unet import UNetModel, SimpleFlowModel


# from guided_diffusion.transfer_learning import create_pretrained_model
def setup_time_dependent_reward(args):
    kwargs = args_to_dict(args, classifier_defaults().keys())
    kwargs["output_dim"] = 1
    return create_classifier(**kwargs)


class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = None
        self.operation_func = None
        self.optimizer = None  # handle it on string level
        self.lr = None
        self.loss_func = None
        self.backward_steps = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.tv_loss = None
        self.use_forward = False
        self.forward_guidance_wt = 0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None


class EnvGenerator(object):
    """
    generative model which generates adversarial environments for RL agent
    """

    def __init__(self, args):
        self.env_name = args.env_name
        self.generator_model_path = args.generator_model_path
        if self.env_name.startswith("MultiGrid"):
            self.image_channels = 3
            self.image_size = 16
            self.batch_size = 32
            self.clip_denoised = True
            self.original_guidance_wt = args.regret_guidance_weight
            self.num_recurrences = 1
            self.original_guidance = True
            self.sampling_type = "ddim"
            self.use_forward = False
            self.forward_guidance_wt = 1.0
            self.optim_lr = 0.0002
            self.backward_steps = 0

            model_and_diffusion_dict = model_and_diffusion_defaults()
            model_and_diffusion_dict["image_size"] = 16
            model_and_diffusion_dict["image_channels"] = 3
            model_and_diffusion_dict["num_channels"] = 128
            model_and_diffusion_dict["num_res_blocks"] = 3
            model_and_diffusion_dict["diffusion_steps"] = 1000
            model_and_diffusion_dict["noise_schedule"] = "linear"
            model_and_diffusion_dict["timestep_respacing"] = "ddim50"

            self.model, self.diffusion = create_model_and_diffusion(
                **model_and_diffusion_dict
            )

        elif self.env_name.startswith("CarRacing"):
            self.model = SimpleFlowModel(data_shape=(12, 2), hidden_dim=256)

            self.image_channels = 2
            self.image_size = 12
            self.batch_size = 16
            self.clip_denoised = True
            self.original_guidance_wt = args.regret_guidance_weight
            self.num_recurrences = 1
            self.original_guidance = True
            self.sampling_type = "ddim"
            self.use_forward = False
            self.forward_guidance_wt = 1.0
            self.optim_lr = 0.0002
            self.backward_steps = 0

            diffusion_dict = diffusion_defaults()
            del diffusion_dict["diffusion_steps"]
            del diffusion_dict["use_ldm"]
            del diffusion_dict["ldm_config_path"]
            diffusion_dict["steps"] = 1000
            diffusion_dict["timestep_respacing"] = "ddim200"
            self.diffusion = create_gaussian_diffusion(**diffusion_dict)

        elif self.env_name.startswith("Bipedal"):
            self.model = SimpleFlowModel((8, 1), 256)
            # self.model = UNetModel(
            #     image_size = 8,
            #     in_channels = 1,
            #     model_channels = 128,
            #     out_channels = 1,
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

            self.image_channels = 1
            self.image_size = 8
            self.batch_size = 16
            self.clip_denoised = True
            self.original_guidance_wt = args.regret_guidance_weight
            self.num_recurrences = 1
            self.original_guidance = True
            self.sampling_type = "ddim"
            self.use_forward = False
            self.forward_guidance_wt = 1.0
            self.optim_lr = 0.0002
            self.backward_steps = 0

            diffusion_dict = diffusion_defaults()
            del diffusion_dict["diffusion_steps"]
            del diffusion_dict["use_ldm"]
            del diffusion_dict["ldm_config_path"]
            diffusion_dict["steps"] = 1000
            diffusion_dict["timestep_respacing"] = "ddim200"
            self.diffusion = create_gaussian_diffusion(**diffusion_dict)
        else:
            raise NotImplementedError

        self.log_sigmoid = th.nn.LogSigmoid()

        self.model.load_state_dict(
            dist_util.load_state_dict(self.generator_model_path, map_location="cpu")
        )
        dist_util.setup_dist()
        self.model.to(dist_util.dev())
        self.model.eval()

        self.regret_metric = args.regret_metric

        self.initial_noise_schedule = None
        self.step = 0

    def generate_env(self, num_env, tutor_list=None):

        if tutor_list is None:
            return self.generate_random_env(num_env)
        else:
            return self.generate_guided_env(num_env, tutor_list)

    def generate_random_env(self, num_env, initial_noise=None):
        if self.env_name.startswith("MultiGrid"):
            shape = (
                self.batch_size,
                self.image_channels,
                self.image_size,
                self.image_size,
            )
        else:
            shape = (self.batch_size, self.image_size, self.image_channels)

        all_images = []
        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1
        while len(all_images) * self.batch_size < num_env:
            model_kwargs = {}
            sample_fn = self.diffusion.ddim_sample_loop

            sample = sample_fn(
                self.model,
                shape,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            if self.env_name.startswith("MultiGrid"):
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
            # else:
            #     sample = sample.permute(0,2,1)
            #     sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        return arr

    def generate_guided_env(self, num_env, tutor_list, initial_noise=None, level=None):
        if self.env_name.startswith("MultiGrid"):
            shape = (
                self.batch_size,
                self.image_channels,
                self.image_size,
                self.image_size,
            )
        else:
            shape = (self.batch_size, self.image_size, self.image_channels)

        all_images = []

        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1

        while len(all_images) * self.batch_size < num_env:

            model_kwargs = {}
            # classes = th.randint(
            #             low=0, high=NUM_CLASSES, size=(self.batch_size,), device=dist_util.dev()
            #         )
            classes = th.zeros(self.batch_size)
            model_kwargs["y"] = classes

            sample_fn = self.diffusion.ddim_sample_loop
            sample = sample_fn(
                self.model_fn,
                shape,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=partial(self.cond_fn, tutor_list=tutor_list, level=level),
                device=dist_util.dev(),
            )

            if self.env_name.startswith("MultiGrid"):
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
            # else:
            #     sample = sample.permute(0,2,1)
            #     sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        for tutor in tutor_list:
            tutor.set_envs(arr)
        return arr

    def generate_random_env_multigrid(self, num_env, initial_noise=None):
        all_images = []
        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1
        while len(all_images) * self.batch_size < num_env:
            model_kwargs = {}
            sample_fn = self.diffusion.ddim_sample_loop

            sample = sample_fn(
                self.model,
                (
                    self.batch_size,
                    self.image_channels,
                    self.image_size,
                    self.image_size,
                ),
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        return arr

    def generate_guided_env_multigrid(self, num_env, tutor_list, initial_noise=None):

        operation = OptimizerDetails()
        operation.num_recurrences = self.num_recurrences
        operation.operation_func = partial(self.operation_func, tutor_list=tutor_list)
        operation.loss_func = self.loss_func

        # Determines whether we use original guidance
        # (The plain classifier-guidance style guidance)
        operation.original_guidance = self.original_guidance
        operation.sampling_type = self.sampling_type

        # Determines whether we use forward universal guidance
        # (Based on the expectation of the generated x0)
        operation.use_forward = self.use_forward
        operation.forward_guidance_wt = self.forward_guidance_wt

        operation.optimizer = "Adam"
        operation.lr = self.optim_lr
        operation.backward_steps = self.backward_steps
        operation.loss_cutoff = 0.0

        # Other miscellaneous setups
        operation.other_guidance_func = None
        operation.other_criterion = None
        operation.tv_loss = False
        operation.warm_start = False
        operation.print = False
        operation.print_every = 10
        operation.Aug = None

        all_images = []

        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1

        while len(all_images) * self.batch_size < num_env:

            model_kwargs = {}
            # classes = th.randint(
            #             low=0, high=NUM_CLASSES, size=(self.batch_size,), device=dist_util.dev()
            #         )
            classes = th.zeros(self.batch_size)
            model_kwargs["y"] = classes

            sample_fn = self.diffusion.ddim_sample_loop_operation
            sample = sample_fn(
                self.model_fn,
                (
                    self.batch_size,
                    self.image_channels,
                    self.image_size,
                    self.image_size,
                ),
                operated_image=None,
                operation=operation,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=partial(self.cond_fn, tutor_list=tutor_list),
                device=dist_util.dev(),
                progress=False,
            )

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        return arr

    def generate_random_env_carracing(self, num_env, initial_noise=None):
        all_images = []
        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1
        while len(all_images) * self.batch_size < num_env:
            model_kwargs = {}
            sample_fn = self.diffusion.ddim_sample_loop

            sample = sample_fn(
                self.model,
                (self.batch_size, self.image_channels, self.image_size),
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        return arr

    def generate_random_env_bipedal(self, num_env, initial_noise=None):
        all_images = []
        if initial_noise is not None:
            initial_noise = initial_noise.to(dist_util.dev())
        elif self.initial_noise_schedule is not None:
            initial_noise = self.initial_noise_schedule[self.step].clone()
            initial_noise = initial_noise.to(dist_util.dev())
            self.step += 1
        while len(all_images) * self.batch_size < num_env:
            model_kwargs = {}
            sample_fn = self.diffusion.ddim_sample_loop

            sample = sample_fn(
                self.model,
                (self.batch_size, self.image_channels, self.image_size),
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_env]

        return arr

    def create_initial_noise(self):
        if self.env_name.startswith("MultiGrid"):
            return th.randn(
                (self.batch_size, self.image_channels, self.image_size, self.image_size)
            )
        elif self.env_name.startswith("CarRacing"):
            return th.randn((self.batch_size, self.image_size, self.image_channels))
        elif self.env_name.startswith("Bipedal"):
            return th.randn((self.batch_size, self.image_size, self.image_channels))

    def create_initial_noise_schedule(self, args):
        if self.env_name.startswith("MultiGrid"):
            num_episode = (
                int(args.num_env_steps) // args.num_steps // args.num_processes
            )
            self.initial_noise_schedule = th.randn(
                (
                    num_episode,
                    self.batch_size,
                    self.image_channels,
                    self.image_size,
                    self.image_size,
                )
            )
        elif self.env_name.startswith("CarRacing"):
            num_episode = (
                int(args.num_env_steps) // args.num_steps // args.num_processes
            )
            self.initial_noise_schedule = th.randn(
                (num_episode, self.batch_size, self.image_channels, self.image_size)
            )
        if self.env_name.startswith("Bipedal"):
            num_episode = (
                int(args.num_env_steps) // args.num_steps // args.num_processes
            )
            self.initial_noise_schedule = th.randn(
                (num_episode, self.batch_size, self.image_channels, self.image_size)
            )

    def cond_fn(self, x, t, tutor_list, level=None, y=None):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # Computes the sum of log probabilities from each regret (ensembling)
            # NOTE: This is no longer the average as in the previous version.
            #       Updated in order to maintain consistency with the paper.
            if self.regret_metric == "disagreement":
                log_probs = [
                    tutor.get_regret(
                        x_in,
                        t,
                    )
                    for tutor in tutor_list
                ]
                out = th.std(th.stack(log_probs)) * self.original_guidance_wt
            elif self.regret_metric == "difficulty_level":
                log_probs = sum(
                    [tutor.get_regret(x_in, t, level) for tutor in tutor_list]
                )
                out = log_probs.sum() * self.original_guidance_wt
            else:
                log_probs = sum([tutor.get_regret(x_in, t) for tutor in tutor_list])
                out = log_probs.sum() * self.original_guidance_wt

            return th.autograd.grad(out, x_in)[0]

    def model_fn(self, x, t, y=None):
        return self.model(x, t, None)

    def operation_func(self, x, tutor_list, t=None):
        return [tutor(x) if t is None else tutor(x, t) for tutor in tutor_list]

    def loss_func(self, predicted_vals):
        return sum([-self.log_sigmoid(pval) for pval in predicted_vals])
