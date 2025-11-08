# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        proprio_obs_dim = kwargs.pop('proprio_obs_dim', None)
        vision_shape = kwargs.pop('vision_shape', None)
        vision_latent_dim = kwargs.pop('vision_latent_dim', 32)
        critic_proprio_obs_dim = kwargs.pop('critic_proprio_obs_dim', None)
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.vision_shape = None
        self.vision_enabled = False
        self.vision_flat_dim = 0

        if vision_shape is not None:
            shape_tuple = tuple(int(x) for x in vision_shape)
            if len(shape_tuple) == 2:
                shape_tuple = (1, *shape_tuple)
            if len(shape_tuple) != 3:
                raise ValueError(f"vision_shape must have length 2 or 3 (C, H, W); got {vision_shape}.")
            self.vision_shape = shape_tuple
            self.vision_flat_dim = int(np.prod(self.vision_shape))
            self.vision_enabled = True

        if proprio_obs_dim is None:
            if self.vision_enabled:
                raise ValueError("proprio_obs_dim must be provided when vision inputs are enabled.")
            self.proprio_obs_dim = num_actor_obs
        else:
            self.proprio_obs_dim = int(proprio_obs_dim)

        if self.proprio_obs_dim + self.vision_flat_dim != num_actor_obs:
            raise ValueError(
                f"Total actor observation size mismatch: proprio {self.proprio_obs_dim} + vision {self.vision_flat_dim} != {num_actor_obs}."
            )

        self.vision_latent_dim = int(vision_latent_dim) if self.vision_enabled else 0

        if self.vision_enabled:
            in_channels = self.vision_shape[0]
            # Lightweight vision encoder to speed up learning passes
            self.vision_backbone = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *self.vision_shape)
                conv_out_dim = self.vision_backbone(dummy).view(1, -1).shape[1]
            self.vision_head = nn.Sequential(
                nn.Linear(conv_out_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.vision_latent_dim),
                nn.ReLU(),
            )
            print(f"Vision backbone: {self.vision_backbone}")
            print(f"Vision head: {self.vision_head}")
        else:
            self.vision_backbone = None
            self.vision_head = None

        if critic_proprio_obs_dim is None:
            if num_critic_obs == num_actor_obs:
                critic_proprio_obs_dim = self.proprio_obs_dim
            else:
                critic_proprio_obs_dim = num_critic_obs
        self.critic_proprio_obs_dim = int(critic_proprio_obs_dim)
        if self.critic_proprio_obs_dim > num_critic_obs:
            raise ValueError(
                f"critic_proprio_obs_dim ({self.critic_proprio_obs_dim}) cannot exceed num_critic_obs ({num_critic_obs})."
            )

        critic_remainder = num_critic_obs - self.critic_proprio_obs_dim
        if self.vision_enabled:
            if critic_remainder == 0:
                self.critic_uses_vision = False
                self.critic_extra_dim = 0
            elif critic_remainder >= self.vision_flat_dim:
                self.critic_uses_vision = True
                self.critic_extra_dim = critic_remainder - self.vision_flat_dim
            else:
                raise ValueError(
                    f"Critic observations do not contain enough depth elements: {critic_remainder} available, "
                    f"need {self.vision_flat_dim}."
                )
        else:
            self.critic_uses_vision = False
            self.critic_extra_dim = critic_remainder

        if self.critic_extra_dim < 0:
            raise ValueError("Critic observation configuration results in negative extra features.")

        mlp_input_dim_a = self.proprio_obs_dim + self.vision_latent_dim
        mlp_input_dim_c = self.critic_proprio_obs_dim + (
            self.vision_latent_dim if self.critic_uses_vision else 0
        ) + self.critic_extra_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        actor_input = self._prepare_actor_input(observations)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_input = self._prepare_actor_input(observations)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_input = self._prepare_critic_input(critic_observations)
        value = self.critic(critic_input)
        return value

    def _encode_vision(self, depth_flat: torch.Tensor) -> torch.Tensor:
        if not self.vision_enabled:
            raise RuntimeError("Vision encoder requested but vision is disabled in this policy.")
        if depth_flat.shape[1] != self.vision_flat_dim:
            raise RuntimeError(
                f"Depth feature width mismatch: expected {self.vision_flat_dim}, got {depth_flat.shape[1]}."
            )
        depth = depth_flat.reshape(depth_flat.shape[0], *self.vision_shape)
        features = self.vision_backbone(depth)
        latent = self.vision_head(features.reshape(features.shape[0], -1))
        return latent

    def _prepare_actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.vision_enabled:
            return observations
        if observations.shape[1] < self.proprio_obs_dim + self.vision_flat_dim:
            raise RuntimeError(
                f"Actor observations too small: expected at least {self.proprio_obs_dim + self.vision_flat_dim}, "
                f"got {observations.shape[1]}."
            )
        proprio = observations[:, :self.proprio_obs_dim]
        depth_flat = observations[:, self.proprio_obs_dim:self.proprio_obs_dim + self.vision_flat_dim]
        latent = self._encode_vision(depth_flat)
        return torch.cat((proprio, latent), dim=-1)

    def _prepare_critic_input(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.vision_enabled:
            return observations
        if observations.shape[1] < self.critic_proprio_obs_dim:
            raise RuntimeError(
                f"Critic observations too small: expected at least {self.critic_proprio_obs_dim}, got {observations.shape[1]}."
            )
        proprio = observations[:, :self.critic_proprio_obs_dim]
        offset = self.critic_proprio_obs_dim
        parts = [proprio]
        if self.critic_uses_vision:
            if observations.shape[1] < offset + self.vision_flat_dim:
                raise RuntimeError(
                    f"Critic observations do not contain enough depth data: need {self.vision_flat_dim} values starting at index {offset}."
                )
            depth_slice = observations[:, offset:offset + self.vision_flat_dim]
            latent = self._encode_vision(depth_slice)
            parts.append(latent)
            offset += self.vision_flat_dim
        if self.critic_extra_dim > 0:
            if observations.shape[1] < offset + self.critic_extra_dim:
                raise RuntimeError(
                    f"Critic observations missing extra features: need {self.critic_extra_dim} values starting at index {offset}."
                )
            extra = observations[:, offset:offset + self.critic_extra_dim]
            parts.append(extra)
        return torch.cat(parts, dim=-1)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
