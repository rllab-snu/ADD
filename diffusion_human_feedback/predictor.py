import argparse
import os
import pickle
import random

import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from itertools import zip_longest
from typing import Iterable

from .guided_diffusion.resample import create_named_schedule_sampler
from .guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    diffusion_defaults,
    create_classifier,
    create_gaussian_diffusion
)
from .guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import gym_minigrid.minigrid as minigrid
from .guided_diffusion.unet import EncoderUNetModel, EncoderLinearModel, EncoderLinearAttentionModel

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

class Tutor(object):
    """
    env generating agent with score-based model style
    """
    def __init__(self, args, device=th.device("cuda:0")):
        
        self.device = th.device(device)
        self.env_name = args.env_name
        self.regret_metric = args.regret_metric
        
        self.lr = args.tutor_lr
        self.weight_decay = args.tutor_weight_decay
        
        self.num_iteration = args.tutor_update_iteration
        self.ema_tau = 0.01
        
        log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
        self.model_dir = os.path.join(log_dir, args.xpid, 'tutors')
        self.buffer_size = args.tutor_buffer_size
        self.batch_size = args.tutor_batch_size
        
        self.buffer_idx = 0
        self.buffer_full = False
        
        self.num_bins = 1
        
        self.categorical = args.use_categorical_tutor
        if args.use_categorical_tutor:
            self.num_processes = args.num_processes
            self.num_bins = args.num_bins_tutor
            self.return_domain = th.arange(0, 1 - 1e-8, 1/self.num_bins).unsqueeze(dim=1).to(self.device)
            self.sm = th.nn.Softmax(dim=1)
            self.logsumexp_temp = args.logsumexp_temperature
            self.cvar_alpha = args.cvar_alpha
            self.added_noise_std = args.return_add_noise_std
            
        if args.env_name == "MultiGrid-GoalLastVariableBlocksAdversarialEnv-v0" or args.env_name == "MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7":     
            model_dict = classifier_defaults()
            model_dict["image_size"] = 16
            model_dict["image_channels"] = 3
            model_dict["classifier_width"] = 128
            model_dict["classifier_depth"] = 2
            model_dict["classifier_attention_resolutions"] = "16, 8, 4"
            model_dict["output_dim"] = 1
            if args.use_categorical_tutor:
                model_dict["output_dim"] = self.num_bins
                self.minimum_return = 0
                self.maximum_return = 1
                self.bin_interval = (self.maximum_return - self.minimum_return) / self.num_bins
                self.two_hot = True
                
            self.model = create_classifier(**model_dict)
            if args.regret_metric == "diff_from_target":
                self.model_target = create_classifier(**model_dict)
            
            diffusion_dict = diffusion_defaults()
            diffusion_dict["steps"] = diffusion_dict.pop("diffusion_steps")
            diffusion_dict.pop("use_ldm")
            diffusion_dict.pop("ldm_config_path")
            self.diffusion = create_gaussian_diffusion(**diffusion_dict)
            
            self.env_buffer = np.zeros((self.buffer_size, 16, 16, 3), dtype=np.float32)
            self.save_interval = 10000
        
        elif self.env_name == "CarRacing-Bezier-Adversarial-v0":
            output_dim = 1
            if args.use_categorical_tutor:
                output_dim = self.num_bins
                self.minimum_return = 0
                self.maximum_return = 1000
                self.bin_interval = (self.maximum_return - self.minimum_return) / self.num_bins
                self.two_hot = True
            self.model = EncoderLinearAttentionModel(2, output_dim, 256)
            # self.model = EncoderUNetModel(
            #     image_size=16,
            #     in_channels=2,
            #     model_channels=128,
            #     out_channels=output_dim,
            #     num_res_blocks=2,
            #     attention_resolutions=[2,1],
            #     dropout=False,
            #     channel_mult=(1,2,2,2,2),
            #     dims=1,
            #     num_head_channels=64,
            #     use_scale_shift_norm=True,
            #     resblock_updown=True,
            #     pool="attention"
            # )
            # if args.regret_metric == "diff_from_target":
            #     self.model_target = EncoderUNetModel(
            #                             image_size=16,
            #                             in_channels=2,
            #                             model_channels=128,
            #                             out_channels=1,
            #                             num_res_blocks=2,
            #                             attention_resolutions=[2,1],
            #                             dropout=False,
            #                             channel_mult=(1,2,2,2),
            #                             dims=1,
            #                             num_head_channels=64,
            #                             use_scale_shift_norm=True,
            #                             resblock_updown=True,
            #                             pool="attention"
            #                         )
            
            diffusion_dict = diffusion_defaults()
            del diffusion_dict["diffusion_steps"]
            del diffusion_dict["use_ldm"]
            del diffusion_dict["ldm_config_path"]
            diffusion_dict["steps"] = 1000
            self.diffusion = create_gaussian_diffusion(**diffusion_dict)

            self.env_buffer = np.zeros((self.buffer_size, 12, 2), dtype=np.float32)
            self.save_interval = 500
        
        elif self.env_name == "BipedalWalker-Adversarial-v0":
            output_dim = 1
            if args.use_categorical_tutor:
                output_dim = self.num_bins
                self.minimum_return = 0
                self.maximum_return = 300
                self.bin_interval = (self.maximum_return - self.minimum_return) / self.num_bins
                self.two_hot = True
            self.model = EncoderLinearModel(8, output_dim, 128)
            # self.model = EncoderUNetModel(
            #     image_size=8,
            #     in_channels=1,
            #     model_channels=128,
            #     out_channels=output_dim,
            #     num_res_blocks=2,
            #     attention_resolutions=[2,1],
            #     dropout=False,
            #     channel_mult=(1,2,2,2),
            #     dims=1,
            #     num_head_channels=64,
            #     use_scale_shift_norm=True,
            #     resblock_updown=True,
            #     pool="attention"
            # )
            # if args.regret_metric == "diff_from_target":
            #     self.model_target = EncoderUNetModel(
            #                             image_size=8,
            #                             in_channels=1,
            #                             model_channels=128,
            #                             out_channels=1,
            #                             num_res_blocks=2,
            #                             attention_resolutions=[2,1],
            #                             dropout=False,
            #                             channel_mult=(1,2,2,2),
            #                             dims=1,
            #                             num_head_channels=64,
            #                             use_scale_shift_norm=True,
            #                             resblock_updown=True,
            #                             pool="attention"
            #                         )
            
            diffusion_dict = diffusion_defaults()
            del diffusion_dict["diffusion_steps"]
            del diffusion_dict["use_ldm"]
            del diffusion_dict["ldm_config_path"]
            diffusion_dict["steps"] = 200
            self.diffusion = create_gaussian_diffusion(**diffusion_dict)
            
            self.env_buffer = np.zeros((self.buffer_size, 8, 1), dtype=np.float32)
            self.save_interval = 20000
        else:
            raise NotImplementedError
        
        self.result_buffer = np.zeros((self.buffer_size, self.num_bins), dtype=np.float32)
        
        self.model.to(self.device)
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
        
        self.regret_metric = args.regret_metric
        if args.regret_metric == "diff_from_target":
            self.model_target.to(self.device)
        
        self.loss_function = th.nn.MSELoss()
        if self.categorical:
            self.loss_function = th.nn.CrossEntropyLoss()
        
        self.optimizer = th.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        self.update_count = 0
        
        self.env_list = None
        
    # def estimate_max(self, distribution):
        
    
    def get_regret(self, x, t, difficulty_level=None):
        if self.regret_metric == "diff_from_target":
            return th.abs(self.model(x, t) - self.model_target(x, t))
        elif self.regret_metric == "fully_adversarial":
            if self.categorical:
                return -th.matmul(self.sm(self.model(x, t)), self.return_domain)
            else:
                return -self.model(x, t)
        elif self.regret_metric == "disagreement":
            return self.model(x, t)
        elif self.regret_metric in ["cvar-mean", "logsumexp-mean", "entropy"]:
            prob = self.sm(self.model(x,t))
            
            if self.regret_metric == "logsumexp-mean":
                # using logsumexp - expectation
                return 1 + self.logsumexp_temp * th.log(th.matmul(
                    prob, th.exp((self.return_domain - 1) / self.logsumexp_temp))) - th.matmul(prob, self.return_domain)
            elif self.regret_metric == "cvar-mean":     
                ## using CVaR - expectation
                
                x = th.cumsum(prob, dim=1)
                x = th.clip(x - (1-self.cvar_alpha), th.zeros_like(prob), prob)

                return th.einsum('bc,bc->b', x, self.return_domain.T + (1 / self.num_bins) * (1 - x / (2 * prob + 1e-8))) / self.cvar_alpha \
                            - th.einsum('bc,bc->b', prob, self.return_domain.T + (1 / self.num_bins) / 2)
            elif self.regret_metric == "entropy":
                return - th.sum(prob * th.log(prob), dim=1)
        elif self.regret_metric == "difficulty_level":
            assert self.categorical
            assert difficulty_level >= 0 and difficulty_level <= 1
            
            level_idx = (int) (self.num_bins * (1 - difficulty_level))
            level_idx = np.clip(level_idx,0, self.num_bins-1)
            prob = self.sm(self.model(x,t))
            return th.log(prob[:, level_idx])
        else:
            raise NotImplementedError
        
    def predict_return(self, x):
        if self.env_name == "MultiGrid-GoalLastVariableBlocksAdversarialEnv-v0" or self.env_name == "MultiGrid-GoalLastVariableBlocksAdversarialEnv-v7":
            x = self.env_to_tensor(x)
            x = self.model(x, th.tensor([0] * x.shape[0]).to(self.device))
        elif self.env_name == "CarRacing-Bezier-Adversarial-v0":
            x = th.from_numpy(x)
            x = x.to(self.device)
            x = self.model(x, th.tensor([0] * x.shape[0]).to(self.device))
        elif self.env_name == "BipedalWalker-Adversarial-v0":
            x = th.from_numpy(x)
            x = x.to(self.device)
            x = self.model(x, th.tensor([0] * x.shape[0]).to(self.device))
        if self.categorical:
            x = th.matmul(self.sm(x), self.return_domain)
        return x
    
    def gaussian_prob(self, bin_idx, scale):
        
        sigma = self.bin_interval / self.added_noise_std
        domain_radius = int(sigma) * 5
        density = np.zeros(2 * domain_radius + 1)
        density[domain_radius] = (1 / (np.sqrt(2 * np.pi) * sigma))
        right = bin_idx
        left = bin_idx
        for i in range(domain_radius):
            right = min(right + 1, self.num_bins - 1)
            left = max(0, left - 1)
            prob = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((i+1)/sigma)**2 / 2)
            density[domain_radius + right - bin_idx] += prob
            density[domain_radius + left - bin_idx] += prob
        density /= sum(density)
        return density[domain_radius + left - bin_idx : domain_radius + right - bin_idx + 1] * scale, left, right + 1
    
    def returns_to_categorical(self, _returns, failures):
        categorical_returns = np.zeros((self.num_processes, self.num_bins), dtype=np.float32)
        assert len(_returns) == len(failures)
        for process_idx in range(len(_returns)):
            _return_list = _returns[process_idx]
            failure_list = failures[process_idx]
            assert len(_return_list) == len(failure_list)
            prob_for_each_return = 1 / len(_return_list)
            for epi_idx in range(len(_return_list)):
                _return = _return_list[epi_idx]
                _return = np.clip(_return, self.minimum_return, self.maximum_return-1e-8)
                bin_idx = int((_return - self.minimum_return) // self.bin_interval)
                failure = failure_list[epi_idx]
                if self.two_hot:
                    remainder = _return - bin_idx * self.bin_interval
                    categorical_returns[process_idx, bin_idx] += prob_for_each_return * (1 - remainder / self.bin_interval)
                    categorical_returns[process_idx, min(bin_idx + 1, self.num_bins - 1)] += prob_for_each_return * (remainder / self.bin_interval)
                else:
                    categorical_returns[process_idx, bin_idx] += prob_for_each_return
                # if failure:
                #     categorical_returns[process_idx, bin_idx] += prob_for_each_return
                # else:
                #     noisy_prob_for_each_return, left_idx, right_idx = self.gaussian_prob(bin_idx, prob_for_each_return)
                #     categorical_returns[process_idx, left_idx:right_idx] += noisy_prob_for_each_return

        return categorical_returns
        
    
    def add_result(self, returns, failures=None):
        if self.categorical:
            returns = self.returns_to_categorical(returns, failures)
        else:
            returns = returns.numpy()
        assert returns.shape[0] == len(self.env_list)
        self.env_buffer[self.buffer_idx:self.buffer_idx + returns.shape[0]] = np.asarray(self.env_list)
        self.result_buffer[self.buffer_idx:self.buffer_idx + returns.shape[0]] = returns.copy()
        self.buffer_idx += returns.shape[0]
        if self.buffer_idx == self.buffer_size:
            self.buffer_full = True
            self.buffer_idx = 0
    
    def sample_from_buffer(self):
        assert self.buffer_full

        # batch_idxs = np.random.randint(0, self.buffer_size, self.batch_size)
        batch_idxs = np.random.choice(self.buffer_size, self.batch_size, replace=False)

        batch_envs = self.env_buffer[batch_idxs]
        batch_results = self.result_buffer[batch_idxs]
        
        return batch_envs, batch_results
        
    def update(self):
        '''
        env_list : list of np env
        returns: np array with shape (num_env, 1)
        '''
        # prepare batched data
        if not self.buffer_full:
            return 2.0
        
        for i in range(self.num_iteration):
            self.optimizer.zero_grad()

            batch_envs_np, batch_returns_np = self.sample_from_buffer()
        
            batch_envs = self.env_to_tensor(batch_envs_np)
            returns = th.from_numpy(batch_returns_np)
            returns = returns.to(self.device)

            t, _ = self.schedule_sampler.sample(batch_envs.shape[0], self.device)
            
            batch_envs = self.diffusion.q_sample(batch_envs, t)
            
            predicted_returns = self.model(batch_envs, t)

            loss = self.loss_function(predicted_returns, returns)

            loss.backward()
            
            self.optimizer.step()
        
        # ema update (polyak averaging)
        if self.regret_metric == "diff_from_target" and self.update_count % 50 == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            # for target_param, param in zip_strict(self.model_target.parameters(), self.model.parameters()):
            #         target_param.data.mul_(1 - self.ema_tau)
            #         th.add(target_param.data, param.data, alpha=self.ema_tau, out=target_param.data)
        # save models
        if (self.update_count % self.save_interval == 0):
            self.save_model(os.path.join(self.model_dir, "model_%06d.pt"%self.update_count))
        self.save_model(os.path.join(self.model_dir, "model.pt"))
        self.update_count += 1
        
        self.env_list = None

    def set_envs(self, env_list):
        self.env_list = list(env_list)
         
    def env_to_tensor(self, env_list):
        batched_env = np.asarray(env_list)
        
        if self.env_name.startswith("MultiGrid"):
            return self.env_to_tensor_multigrid(batched_env)
        elif self.env_name.startswith("CarRacing"):
            return self.env_to_tensor_carracing(batched_env)
        elif self.env_name.startswith("Bipedal"):
            return self.env_to_tensor_bipedal(batched_env)
        return 0
    
    def env_to_tensor_multigrid(self, batched_env):

        batched_env = batched_env / 127.5 - 1
        batched_env = batched_env.astype(np.float32)
        
        batch = th.from_numpy(batched_env)
        batch = batch.permute(0,3,1,2)
        batch = batch.contiguous()
        batch = batch.to(self.device)
            
        return batch

    def env_to_tensor_carracing(self, batched_env):
        batched_env = batched_env.astype(np.float32)
        batch = th.from_numpy(batched_env)
        batch = batch.to(self.device)

        return batch
    
    def env_to_tensor_bipedal(self, batched_env):
        batched_env = batched_env.astype(np.float32)
        batch = th.from_numpy(batched_env)
        batch = batch.to(self.device)

        return batch
    
    def save_model(self, path):
        if self.regret_metric == "diff_from_target":
            th.save({
                'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        else:
            th.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
    
    def load_model(self, ckpt):
        self.model.load_state_dict(ckpt['model'])
        if self.regret_metric == "diff_from_target":
            self.model_target.load_state_dict(ckpt['model_target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
    def train_mode(self):
        self.model.train()
        if self.regret_metric == "diff_from_target":
            self.model_target.train()   
        
    def eval_mode(self):
        self.model.eval()
        if self.regret_metric == "diff_from_target":
            self.model_target.eval()
        
    def to(self, device):
        self.model.to(device)
        if self.regret_metric == "diff_from_target":
            self.model_target.to(device)
                
                
                
# tutor = Tutor("MultiGrid-GoalLastVariableBlocksAdversarialEnv-v0")
# env = np.random.randn(32*3*16*16).astype(np.float32)
# env = env.reshape((32,3,16,16))
# env_returns = np.random.randn(32).astype(np.float32)
# env_returns = env_returns.reshape((32, 1))
# tutor.update(env, env_returns)
