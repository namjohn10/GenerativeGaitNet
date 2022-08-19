import copy
import psutil
from itertools import combinations
from numpy import linalg as LA
from IPython import embed
from abc import ABCMeta, abstractmethod
from pathlib import Path
from collections import deque
from typing import Dict, List, Callable
import math
import time
import mcmc
from ray_model import MuscleNN, SimulationNN, MarginalNN, PolicyNN, RefNN
from pymss import RayEnvManager
import torch.distributed
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import gym
import numpy as np
import random
from typing import Tuple, Callable, Optional, List
import logging

import ray
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import MultiEnvDict, EnvType, EnvID, MultiAgentDict
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor

from ray._private.services import get_node_ip_address

torch, nn = try_import_torch()


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def extract_from_structured_array(tuples, component, device):
    try:
        numpy_array = np.vstack(tuples[component]).astype(np.float32)
        return torch.from_numpy(numpy_array).to(device)
    except ValueError as e:
        print("tuples: ", tuples)
        raise e


class RefLearner:
    def __init__(self, device, num_paramstate, num_active_dof,
                 buffer_size=30000, learning_rate=1e-4, num_epochs=10, batch_size=128, model=None):
        self.device = device
        self.num_paramstate = num_paramstate
        self.num_active_dof = num_active_dof
        self.num_epochs = num_epochs
        self.ref_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate

        if model:
            self.model = model
        else:
            self.model = RefNN(self.num_paramstate,
                               self.num_active_dof, self.device).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def learn(self, param_all, d_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        assert (len(param_all) == len(d_all))
        idx_all = np.array(range(len(param_all)))
        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.

        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_avg = 0
            for i in range(len(param_all) // self.ref_batch_size):
                mini_batch_idx = idx_all[i *
                                         self.ref_batch_size: (i+1)*self.ref_batch_size]
                param = torch.stack([param_all[idx] for idx in mini_batch_idx])
                d = torch.stack([d_all[idx] for idx in mini_batch_idx])
                d_out = self.model(param)
                loss = ((d - d_out).pow(2)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-0.5, 0.5)

                self.optimizer.step()
                loss_avg += loss.cpu().detach().numpy().tolist()

        loss_ref = loss_avg / (len(param_all) // self.ref_batch_size)
        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': len(param_all),
            'loss_marginal': loss_ref,
            'time': time_stat
        }


class MarginalLearner:
    def __init__(self, device, num_paramstate,
                 buffer_size=30000, learning_rate=1e-4, num_epochs=10, batch_size=128, model=None):
        self.device = device
        self.num_paramstate = num_paramstate
        self.num_epochs = num_epochs
        self.marginal_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate

        if model:
            self.model = model
        else:
            self.model = MarginalNN(
                self.num_paramstate, self.device).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}
        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def learn(self, param_all, v_all) -> Dict:

        converting_time = 0.0
        learning_time = 0.0
        start_time = time.perf_counter()

        assert (len(param_all) == len(v_all))
        idx_all = np.array(range(len(param_all)))
        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.

        for _ in range(self.num_epochs):
            np.random.shuffle(idx_all)
            loss_avg = 0
            for i in range(len(param_all) // self.marginal_batch_size):
                mini_batch_idx = idx_all[i *
                                         self.marginal_batch_size: (i+1)*self.marginal_batch_size]
                param = torch.stack([param_all[idx] for idx in mini_batch_idx])
                v = torch.stack([v_all[idx] for idx in mini_batch_idx])
                v_out = self.model(param)
                loss = ((v - v_out).pow(2)).mean()
                self.optimizer.zero_grad()

                loss.backward()
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-0.5, 0.5)

                self.optimizer.step()
                loss_avg += loss.cpu().detach().numpy().tolist()

        loss_marginal = loss_avg / (len(param_all) // self.marginal_batch_size)
        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': len(param_all),
            'loss_marginal': loss_marginal,
            'time': time_stat
        }


class MuscleLearner:
    def __init__(self, device, num_action, num_muscles, num_muscle_dofs, use_timewarp=0,
                 buffer_size=30000, learning_rate=1e-4, num_epochs=3, batch_size=128, model=None, is_last=False):
        self.device = device

        self.num_action = num_action
        self.num_muscles = num_muscles
        self.num_epochs_muscle = num_epochs
        self.muscle_batch_size = batch_size
        self.default_learning_rate = learning_rate
        self.learning_rate = self.default_learning_rate
        self.use_timewarp = use_timewarp

        if model:
            self.model = model
        else:
            self.model = MuscleNN(num_muscle_dofs, self.num_action,
                                  self.num_muscles, is_last=is_last).to(self.device)

        parameters = self.model.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        for param in parameters:
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))

        self.stats = {}

        self.model.train()

    def get_weights(self) -> Dict:
        return {
            k: v.cpu().detach().numpy()
            for k, v in self.model.state_dict().items()
        }

    def set_weights(self, weights) -> None:
        weights = convert_to_torch_tensor(weights, device=self.device)
        self.model.load_state_dict(weights)

    def get_optimizer_weights(self) -> Dict:
        return self.optimizer.state_dict()

    def set_optimizer_weights(self, weights) -> None:
        self.optimizer.load_state_dict(weights)

    def get_model_weights(self, device=None) -> Dict:
        if device:
            return {k: v.to(device) for k, v in self.model.state_dict().items()}
        else:
            return self.model.state_dict()

    def save(self, name):
        path = Path(name)
        torch.save(self.model.state_dict(), path)
        torch.save(self.optimizer.state_dict(),
                   path.with_suffix(".opt" + path.suffix))

    def load(self, name):
        path = Path(name)
        self.model.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(
            path.with_suffix(".opt" + path.suffix)))

    def learn(self, muscle_transitions: list) -> Dict:
        converting_time = 0.0
        learning_time = 0.0

        start_time = time.perf_counter()
        l = len(muscle_transitions[0])

        idx_all = np.array(range(len(muscle_transitions[0])))
        JtA_all = torch.tensor(muscle_transitions[0], device="cuda")
        tau_des_all = torch.tensor(muscle_transitions[1], device="cuda")
        L_all = torch.tensor(muscle_transitions[2], device="cuda")
        b_all = torch.tensor(muscle_transitions[3], device="cuda")
        prev_out_all = torch.tensor(muscle_transitions[4], device="cuda")
        w_all = torch.tensor(muscle_transitions[5], device="cuda")

        converting_time = (time.perf_counter() - start_time) * 1000
        start_time = time.perf_counter()
        loss_avg = 0.

        for _ in range(self.num_epochs_muscle):
            np.random.shuffle(idx_all)
            loss_avg = 0.
            for i in range(l // self.muscle_batch_size):

                mini_batch_idx = torch.from_numpy(
                    idx_all[i*self.muscle_batch_size: (i+1)*self.muscle_batch_size]).cuda()
                JtA = torch.index_select(JtA_all, 0, mini_batch_idx)
                tau_des = torch.index_select(tau_des_all, 0, mini_batch_idx)
                b = torch.index_select(b_all, 0, mini_batch_idx)
                L = torch.index_select(L_all, 0, mini_batch_idx)
                prev_out = torch.index_select(prev_out_all, 0, mini_batch_idx)
                w = torch.index_select(w_all, 0, mini_batch_idx)

                activation = self.model.forward_with_prev_out(
                    JtA, (tau_des-b), prev_out, w).unsqueeze(2)
                tau = torch.bmm(L, activation).squeeze(-1)
                loss_reg = activation.pow(2).mean()
                loss_target = (((tau - (tau_des-b)) / 100.0).pow(2)).mean()
                loss = 0.01 * loss_reg + loss_target

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad != None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
                loss_avg += loss.cpu().detach().numpy().tolist()

        loss_muscle = loss_avg / (l // self.muscle_batch_size)

        learning_time = (time.perf_counter() - start_time) * 1000

        time_stat = {'converting_time_ms': converting_time,
                     'learning_time_ms': learning_time}
        return {
            'num_tuples': l,
            'loss_muscle': loss_muscle,
            'time': time_stat
        }


class GaitNetInterface:
    @abstractmethod
    def load_muscle_model_weights(self, weights):
        raise NotImplementedError

    @abstractmethod
    def get_muscle_tuples(self):
        raise NotImplementedError


class MyEnv(gym.Env, GaitNetInterface):

    def __init__(self, metadata):

        self.weight = 1.0
        self.minimum_tuple_bound = 480
        self.env = RayEnvManager(metadata)

        self.num_active_dof = self.env.GetNumActiveDof()
        self.use_displacement = self.env.UseDisplacement()
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()

        self.inference_per_sim = self.env.GetInferencePerSim()
        self.metadata = self.env.GetMetadata()
        self.use_timewarp = self.env.UseTimeWarp()
        self.use_adaptive_sampling = self.env.UseAdaptiveSampling()
        self.cascading_type = self.env.GetCascadingType()

        if self.use_muscle:
            self.num_muscles = self.env.GetNumMuscles()
            self.num_muscle_dofs = self.env.GetNumTotalMuscleRelatedDofs()
            self.muscle_tuples = []

        self.num_paramstate = self.env.GetNumParamState()
        self.param_min = self.env.GetMinV()
        self.param_max = self.env.GetMaxV()

        self.sampling_policy = self.env.GetParamSamplingPolicy()

        if self.use_adaptive_sampling:
            self.marginal_tuples = []

        new_state = np.clip(self.CreateInitState(self.param_min, self.param_max, self.sampling_policy) +
                            np.random.normal(0, 0.025, self.num_paramstate), self.param_min, self.param_max)
        self.current_paramstate = new_state
        self.env.SetParamState(self.current_paramstate)
        self.env.Reset()
        self.current_param_tuple = 0

        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_state,))
        self.action_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(
            np.inf), shape=(self.num_action + (self.cascading_type != 0),))
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.use_muscle:
            self.muscle_model = MuscleNN(self.num_muscle_dofs, self.num_active_dof, self.num_muscles, is_last=not (
                self.cascading_type == 0)).to(self.device)
            self.muscle_model.eval()

        if self.use_displacement:
            self.ref_model = RefNN(self.num_paramstate,
                                   self.num_active_dof, self.device)
            self.ref_model.eval()

        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz
        self.obs = None

        self.prev_sim_nn = []
        self.prev_muscle_nn = []
        self.prev_minv = []
        self.prev_maxv = []

        self.prev_proj_states = []
        self.prev_proj_JtPs = []
        self.prev_weights = []

        self.cascading_map = []

        self.use_prev_model = False
        self.stats = {}

        self.displacement_tuples = []
        self.tmp_displacement_tuples = []

        self.num_statediff = self.env.GetStateDiffNum()

    def CreateInitState(self, minv, maxv, sampling_policy):
        assert (len(minv) == len(maxv))
        result = None
        result = np.zeros(len(minv))
        for i in range(len(result)):
            if sampling_policy[i] > 1E-6:
                result[i] = np.random.choice([minv[i], maxv[i]])
            else:
                d = maxv[i] - minv[i]
                result[i] = np.random.choice(
                    [minv[i], minv[i] + 0.25 * d, minv[i] + 0.5 * d, minv[i] + 0.75 * d, maxv[i]])

        return result

    def set_prev_model(self, sim_w, filter, minv, maxv, muscle_w, cascading_type):
        self.use_prev_model = True

        num_state = len(self.env.GetProjState(minv, maxv))

        self.prev_sim_nn.append(PolicyNN(num_state, self.num_action +
                                (cascading_type > 0), sim_w, filter, self.device, cascading_type))
        is_last = False
        # For legacy model
        if len(muscle_w['fc.0.weight'][0]) != (self.num_muscle_dofs + self.num_active_dof):
            is_last = True
        mn = MuscleNN(self.num_muscle_dofs, self.num_active_dof,
                      self.num_muscles, is_last=is_last).to(self.device)
        mn.load_state_dict(muscle_w)
        mn.eval()
        self.prev_muscle_nn.append(mn)

        self.prev_minv.append(minv)
        self.prev_maxv.append(maxv)

    def reset(self):

        if self.current_param_tuple >= self.minimum_tuple_bound:

            if self.use_adaptive_sampling:
                new_state = copy.deepcopy(self.current_paramstate)

            if not self.use_adaptive_sampling:
                self.current_paramstate = np.clip(self.CreateInitState(
                    self.param_min, self.param_max, self.sampling_policy) + np.random.normal(0, 0.025, self.num_paramstate), self.param_min, self.param_max)

            self.env.SetParamState(self.current_paramstate)

            self.current_param_tuple = 0

        self.env.Reset()
        self.obs = self.env.GetState()

        if self.use_displacement:
            self.displacement_tuples += self.tmp_displacement_tuples
            self.tmp_displacement_tuples.clear()

        return self.obs

    def step(self, action):
        self.env.UpdateHeadInfo()

        self.current_param_tuple += 1
        self.prev_weights.clear()
        prev_action = np.zeros(self.num_action)

        self.prev_proj_states.clear()
        cur_axis = -1
        prev_axis = []
        prev_actions = []

        for i in range(len(self.prev_sim_nn)):
            prev_s = self.env.GetProjState(
                self.prev_minv[i], self.prev_maxv[i])
            self.prev_proj_states.append(prev_s)
            prev_a = self.prev_sim_nn[i].get_action(prev_s)

            if (self.prev_sim_nn[i].cascading_type == 0):
                prev_axis.append(-1.0)
                prev_actions.append(prev_a)
                prev_action += prev_a
                self.prev_weights.append(1.0)

            else:
                prev_axis.append(np.clip(0.25 + 0.05 * prev_a[-1], 0.05, 1.0))
                prev_actions.append(prev_a[:-1])
                l = self.cascading_map[i]
                min_norm = 9999.0
                min_w = 1.0
                for l_idx in l:
                    state_diff = self.prev_proj_states[l_idx][:self.num_statediff] - \
                        self.prev_proj_states[i][:self.num_statediff]
                    state_diff[24:50] *= 0.4
                    state_diff[74:100] *= 0.4
                    state_diff[124:150] *= 0.4
                    if LA.norm(state_diff) < min_norm:
                        min_norm = LA.norm(state_diff)
                        min_w = self.env.CalculateWeight(
                            state_diff, prev_axis[i])

                self.prev_weights.append(min_w)
                if min_w > 1E-6:
                    prev_action += min_w * prev_a[:-1]

        min_norm = 9999.0
        cur_axis = -1.0
        self.weight = 1.0

        if self.cascading_type > 0:
            cur_axis = np.clip(0.25 + 0.05 * action[-1], 0.05, 1.0)

            for l_idx in range(len(self.prev_sim_nn)):
                state_diff = self.prev_proj_states[l_idx][:self.num_statediff] - \
                    self.obs[:self.num_statediff]
                state_diff[24:50] *= 0.4
                state_diff[74:100] *= 0.4
                state_diff[124:150] *= 0.4
                if LA.norm(state_diff) < min_norm:
                    min_norm = LA.norm(state_diff)
                    self.weight = self.env.CalculateWeight(
                        state_diff, cur_axis)

            prev_action += self.weight * action[:-1]
        else:
            prev_action += action

        if self.use_displacement:
            d = self.ref_model(Tensor(np.append(
                self.current_paramstate, self.env.GetPhase()))).cpu().detach().numpy()
            for i in range(len(d)):
                action[i] += d[i]

        self.env.SetAction(prev_action)
        selected_mts = None
        done = False

        if self.use_muscle:
            rand_idx = random.choice(
                range(self.num_simulation_per_control//self.inference_per_sim))
            for i in range(self.num_simulation_per_control // self.inference_per_sim):
                mts = self.env.GetMuscleTuple(False)
                if np.any(np.isnan(mts[0])) or np.any(np.isnan(mts[1])) or np.any(np.isnan(mts[2])) or np.any(mts[0] != mts[0]) or np.any(mts[1] != mts[1]) or np.any(mts[2] != mts[2]):
                    print('[DEBUG] Nan Appear')
                    done = True
                    break

                mt = Tensor(mts[0])
                new_dt = Tensor(mts[2] - mts[1])
                prev_unnormalized_activations = []

                idx = 0
                for muscle_nn in self.prev_muscle_nn:
                    if muscle_nn.is_last == True:
                        prev_unnormalized = np.zeros(
                            self.num_muscles, dtype=np.float32)
                        for l_idx in self.cascading_map[idx]:
                            prev_unnormalized += prev_unnormalized_activations[l_idx]
                        prev_unnormalized_activations.append(self.prev_weights[idx] * muscle_nn.forward_with_prev_out_without_filter(
                            mt, new_dt, Tensor(prev_unnormalized), self.prev_weights[idx]).detach().numpy())

                    else:  # Base or Legacy Network
                        prev_unnormalized_activations.append(
                            self.prev_weights[idx] * muscle_nn.forward_without_filter(mt, new_dt).detach().numpy())
                    idx += 1

                prev_out = np.zeros(self.num_muscles, dtype=np.float32)
                for idx in range(len(self.prev_muscle_nn)):
                    prev_out += prev_unnormalized_activations[idx]

                if i == rand_idx:
                    selected_mts = copy.deepcopy(mts)
                    selected_mts.append(prev_out)
                    selected_mts.append([np.float32(self.weight)])

                prev_out = Tensor(prev_out)
                activations = None

                if len(self.prev_muscle_nn) > 0:
                    activations = self.muscle_model.forward_with_prev_out(
                        mt, new_dt, prev_out, self.weight).cpu().detach().numpy()
                else:
                    activations = self.muscle_model.no_grad_forward(
                        mt, new_dt).cpu().detach().numpy()

                self.env.SetMuscleAction(activations)
                for j in range(self.inference_per_sim):
                    self.env.Step()
        else:
            self.env.StepsAtOnce()

        self.obs = self.env.GetState()
        done = np.any(np.isnan(self.obs)) or np.any(np.isnan(action)) or np.any(
            self.obs != self.obs) or np.any(action != action)
        info = {}

        reward = 0
        if not done:
            reward = self.env.GetReward()

        info['end'] = self.env.IsEndOfEpisode()
        done = done or info['end']

        if self.use_muscle:
            self.muscle_tuples.append(selected_mts)

        if (not done) and self.use_displacement:
            self.tmp_displacement_tuples.append([np.append(
                self.env.GetParamState(), self.env.GetPhase()), self.env.GetDisplacement()])

        if self.use_adaptive_sampling:
            self.marginal_tuples.append([self.env.GetParamState(), self.obs])

        return self.obs, reward, done, info

    def load_muscle_model_weights(self, weights):
        assert (self.use_muscle)
        if (type(weights) == dict):
            self.muscle_model.load_state_dict(weights)
        else:
            weights = convert_to_torch_tensor(ray.get(weights))
            self.muscle_model.load_state_dict(weights)

    def load_ref_model_weights(self, weights):
        assert (self.use_displacement)
        weights = convert_to_torch_tensor(ray.get(weights))
        self.ref_model.load_state_dict(weights)

    def load_paramstate(self, paramstate):
        assert (self.use_adaptive_sampling)
        self.current_paramstate = paramstate

    def load_cascading_map(self, cascading_map):
        self.cascading_map = cascading_map

    def get_muscle_tuples(self):
        assert (self.use_muscle)
        mt = np.array(self.muscle_tuples, dtype=object)

        l = len(mt)
        if mt.ndim == 1:
            self.muscle_tuples.clear()
            return [[], [], [], [], [], []]
        JtA_l = len(mt[0, 0])
        tau_des_l = len(mt[0, 2])
        L_l = len(mt[0, 3])
        b_l = len(mt[0, 4])
        prev_out_l = len(mt[0, 5])
        w_l = len(mt[0, 6])

        res = []
        res.append(np.concatenate(mt[:, 0]).reshape(l, JtA_l))
        res.append(np.concatenate(mt[:, 2]).reshape(l, tau_des_l))
        res.append(np.concatenate(mt[:, 3]).reshape(
            l, self.num_active_dof, self.num_muscles))
        res.append(np.concatenate(mt[:, 4]).reshape(l, b_l))
        res.append(np.concatenate(mt[:, 5]).reshape(l, prev_out_l))
        res.append(np.concatenate(mt[:, 6]).reshape(l, w_l))
        self.muscle_tuples.clear()
        return res

    def get_marginal_tuples(self):
        assert (self.use_adaptive_sampling)
        res = copy.deepcopy(self.marginal_tuples)
        self.marginal_tuples.clear()
        return res

    def get_displacement_tuples(self):
        assert (self.use_displacement)
        res = copy.deepcopy(self.displacement_tuples)
        self.displacement_tuples.clear()
        return res
