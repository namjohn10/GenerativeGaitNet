import torch
import torch.nn as nn
import numpy as np
import gym
import pickle5 as pickle
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import torch.nn.functional as F
from typing import Dict
from IPython import embed

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(
    self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class RefNN(nn.Module):
    def __init__(self, num_paramstate, num_active_dof, device):
        super(RefNN, self).__init__()
        self.num_paramstate = num_paramstate

        num_h1 = 256
        num_h2 = 256

        self.fc = nn.Sequential(
            nn.Linear(self.num_paramstate + 1, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, num_active_dof),
        )
        if device != 'cpu' and torch.cuda.is_available():
            self.cuda()
            self.device = device
        else:
            self.device = 'cpu'

        self.fc.apply(weights_init)

    def forward(self, param_state):
        v_out = self.fc.forward(param_state)
        return v_out

    def get_action(self, s):
        s = np.array(s, dtype=np.float32)
        ts = torch.tensor(s)
        p = self.forward(ts)
        return p.detach().numpy()

    def get_displacement(self, param_state):
        with torch.no_grad():
            ts = torch.tensor(param_state).to(self.device)
            v_out = self.forward(ts)
            return v_out.cpu().detach().numpy()[0]

    def load(self, path):
        print('load ref nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save ref nn {}'.format(path))
        torch.save(self.state_dict(), path)


class MarginalNN(nn.Module):
    def __init__(self, num_parameters, device):
        super(MarginalNN, self).__init__()
        self.num_parameters = num_parameters

        num_h1 = 256
        num_h2 = 256

        self.fc = nn.Sequential(
            nn.Linear(self.num_parameters, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, 1),
        )

        if device != 'cpu' and torch.cuda.is_available():
            self.cuda()
            self.device = device
        else:
            self.device = 'cpu'

        self.fc.apply(weights_init)

    def forward(self, param_state):
        v_out = self.fc.forward(param_state)
        return v_out

    def no_grad_forward(self, param_state):
        with torch.no_grad():
            v_out = self.fc.forward(param_state)
            return v_out.cpu().detach().numpy()

    def get_value(self, param_state):
        with torch.no_grad():
            ts = torch.tensor(param_state).to(self.device)
            v_out = self.forward(ts)

            return v_out.cpu().detach().numpy()[0]

    def load(self, path):
        print('load marginal nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save marginal nn {}'.format(path))
        torch.save(self.state_dict(), path)


class MuscleNN(nn.Module):
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles, is_render=False, is_last=False):
        super(MuscleNN, self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        self.is_last = is_last

        num_h1 = 256
        num_h2 = 256
        num_h3 = 256
        input_size = num_total_muscle_related_dofs+num_dofs

        if self.is_last:
            input_size += (num_muscles + 1)

        self.fc = nn.Sequential(
            nn.Linear(input_size, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),


        )

        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
        self.std_tau = torch.zeros(self.num_dofs)

        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0
        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0

        if torch.cuda.is_available() and not is_render:
            self.std_tau = self.std_tau.cuda()
            self.std_muscle_tau = self.std_muscle_tau.cuda()
            self.cuda()

        self.fc.apply(weights_init)

    def forward(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = torch.relu(torch.tanh(self.fc.forward(
            torch.cat([muscle_tau, tau], dim=-1))))
        return out

    def no_grad_forward(self, muscle_tau, tau):
        with torch.no_grad():
            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau
            out = torch.relu(torch.tanh(self.fc.forward(
                torch.cat([muscle_tau, tau], dim=-1))))
            return out

    def forward_with_prev_out_without_filter_render(self, muscle_tau, tau, prev_out, weight=1.0):
        return self.forward_with_prev_out_without_filter(torch.Tensor(muscle_tau.reshape(1, -1)), torch.Tensor(tau.reshape(1, -1)), prev_out, weight).cpu().detach().numpy()[0]

    def forward_with_prev_out_without_filter(self, muscle_tau, tau, prev_out, weight=1.0):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = None
        if type(prev_out) == np.ndarray:
            with torch.no_grad():
                prev_out2 = torch.FloatTensor([prev_out])
                prev_out = torch.FloatTensor(prev_out)
                if self.is_last:
                    if type(weight) == float:
                        weight = torch.FloatTensor([weight])
                    weight = torch.FloatTensor([[weight]])
                    out = self.fc.forward(
                        torch.cat([prev_out2, weight, muscle_tau, tau], dim=-1))
                else:
                    out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            return out
        else:
            if self.is_last:
                if type(weight) == float:
                    weight = torch.Tensor([weight])
                out = self.fc.forward(
                    torch.cat([prev_out, weight, muscle_tau, tau], dim=-1))
            else:
                out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            return out
    ## Include Last Filter

    def forward_with_prev_out(self, muscle_tau, tau, prev_out, weight=1.0):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = None
        if type(prev_out) == np.ndarray:
            with torch.no_grad():
                prev_out2 = torch.FloatTensor([prev_out])
                prev_out = torch.FloatTensor(prev_out)
                if self.is_last:
                    if type(weight) == float:
                        weight = torch.FloatTensor([weight])
                    weight = torch.FloatTensor([[weight]])
                    out = torch.relu(torch.tanh(prev_out + weight * self.fc.forward(
                        torch.cat([prev_out2, weight, muscle_tau, tau], dim=-1))))
                else:
                    out = torch.relu(torch.tanh(
                        prev_out + weight * self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))))
            return out
        else:
            if self.is_last:
                if type(weight) == float:
                    weight = torch.Tensor([weight])
                out = torch.relu(torch.tanh(prev_out + weight * self.fc.forward(
                    torch.cat([prev_out, weight, muscle_tau, tau], dim=-1))))
            else:
                out = torch.relu(torch.tanh(
                    prev_out + weight * self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))))
            return out

    def forward_with_prev_out_render(self, muscle_tau, tau, prev_out, weight):
        return self.forward_with_prev_out(torch.Tensor(muscle_tau.reshape(1, -1)), torch.Tensor(tau.reshape(1, -1)), prev_out, weight).cpu().detach().numpy()[0]

    def forward_without_filter(self, muscle_tau, tau, weight=1.0):
        with torch.no_grad():
            muscle_tau = muscle_tau / self.std_muscle_tau
            tau = tau / self.std_tau
            out = self.fc.forward(torch.cat([muscle_tau, tau], dim=-1))
            return out

    def forward_without_filter_render(self, muscle_tau, tau):
        return self.forward_without_filter(torch.Tensor(muscle_tau.reshape(1, -1)), torch.Tensor(tau.reshape(1, -1))).cpu().detach().numpy()[0]

    def load(self, path):
        print('load muscle nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(torch.Tensor(muscle_tau.reshape(1, -1)),
                           torch.Tensor(tau.reshape(1, -1)))
        return act.cpu().detach().numpy()[0]


class SimulationNN(nn.Module):
    def __init__(self, num_states, num_actions):

        nn.Module.__init__(self)

        self.num_states = num_states
        self.num_actions = num_actions

        self.num_h1 = 512
        self.num_h2 = 512
        self.num_h3 = 512

        self.log_std = torch.zeros(self.num_actions)

        self.p_fc1 = nn.Linear(self.num_states, self.num_h1)
        self.p_fc2 = nn.Linear(self.num_h1, self.num_h2)
        self.p_fc3 = nn.Linear(self.num_h2, self.num_h3)
        self.p_fc4 = nn.Linear(self.num_h3, self.num_actions)

        self.v_fc1 = nn.Linear(self.num_states, self.num_h1)
        self.v_fc2 = nn.Linear(self.num_h1, self.num_h2)
        self.v_fc3 = nn.Linear(self.num_h2, self.num_h3)
        self.v_fc4 = nn.Linear(self.num_h3, 1)

        self.reset()

        if torch.cuda.is_available():
            self.log_std = self.log_std.cuda()
            self.cuda()

    def reset(self):

        self.pi_reset()
        self.vf_reset()

    def pi_reset(self):

        torch.nn.init.xavier_uniform_(self.p_fc1.weight)
        torch.nn.init.xavier_uniform_(self.p_fc2.weight)
        torch.nn.init.xavier_uniform_(self.p_fc3.weight)
        torch.nn.init.xavier_uniform_(self.p_fc4.weight)

        self.p_fc1.bias.data.zero_()
        self.p_fc2.bias.data.zero_()
        self.p_fc3.bias.data.zero_()
        self.p_fc4.bias.data.zero_()

    def vf_reset(self):

        torch.nn.init.xavier_uniform_(self.v_fc1.weight)
        torch.nn.init.xavier_uniform_(self.v_fc2.weight)
        torch.nn.init.xavier_uniform_(self.v_fc3.weight)
        torch.nn.init.xavier_uniform_(self.v_fc4.weight)

        self.v_fc1.bias.data.zero_()
        self.v_fc2.bias.data.zero_()
        self.v_fc3.bias.data.zero_()
        self.v_fc4.bias.data.zero_()

    def forward(self, x):

        p_out = F.relu(self.p_fc1(x))
        p_out = F.relu(self.p_fc2(p_out))
        p_out = F.relu(self.p_fc3(p_out))
        p_out = self.p_fc4(p_out)

        p_out = MultiVariateNormal(p_out, self.log_std.exp())

        v_out = F.relu(self.v_fc1(x))
        v_out = F.relu(self.v_fc2(v_out))
        v_out = F.relu(self.v_fc3(v_out))
        v_out = self.v_fc4(v_out)

        return p_out, v_out

    def soft_load_state_dict(self, _state_dict):
        current_state_dict = self.state_dict()
        for k in _state_dict:
            current_state_dict[k] = _state_dict[k]
        self.load_state_dict(current_state_dict)

    def policy_state_dict(self):
        policy_state_dict = {}
        current_state_dict = self.state_dict()
        for k in current_state_dict:
            if "p_fc" in k:
                policy_state_dict[k] = current_state_dict[k]
        return policy_state_dict

    def value_function_state_dict(self):
        policy_state_dict = {}
        current_state_dict = self.state_dict()
        for k in current_state_dict:
            if "v_fc" in k:
                policy_state_dict[k] = current_state_dict[k]
        return policy_state_dict

    def load(self, path):
        print('load simulation nn {}'.format(path))
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(
                path, map_location=torch.device('cpu')))

    def save(self, path):
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()

    def get_value(self, s):
        ts = torch.tensor(s)
        _, v = self.forward(ts)
        return v.cpu().detach().numpy()

    def get_random_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy()

    def get_noise(self):
        return self.log_std.exp().mean().item()


class SimulationNN_Ray(TorchModelV2, SimulationNN):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        num_states = np.prod(obs_space.shape)
        num_actions = np.prod(action_space.shape)
        SimulationNN.__init__(self, num_states, num_actions)
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, {}, "SimulationNN_Ray")

        num_outputs = 2 * np.prod(action_space.shape)
        self._value = None

    def get_value(self, obs):
        with torch.no_grad():

            _, v = SimulationNN.forward(self, obs)

            return v

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        x = obs.reshape(obs.shape[0], -1)

        action_dist, self._value = SimulationNN.forward(self, x)
        action_tensor = torch.cat([action_dist.loc, action_dist.scale], dim=1)
        return action_tensor, state

    def value_function(self):
        return self._value.squeeze(1)

    def reset(self):
        SimulationNN.reset(self)

    def vf_reset(self):
        SimulationNN.vf_reset(self)

    def pi_reset(self):
        SimulationNN.pi_reset(self)


class PolicyNN:
    def __init__(self, num_states, num_actions, policy_state, filter_state, device, cascading_type=0):
        self.policy = SimulationNN(num_states, num_actions).to(device)
        self.policy.load_state_dict(convert_to_torch_tensor(policy_state))
        self.policy.eval()
        self.filter = filter_state
        self.cascading_type = cascading_type

    def get_filter(self):
        return self.filter.copy()

    def get_value(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        v = self.policy.get_value(obs)
        return v

    def get_value_function_weight(self):
        return self.policy.value_function_state_dict()

    def get_action(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return self.policy.get_action(obs)

    def get_filtered_obs(self, obs):
        obs = self.filter(obs, update=False)
        obs = np.array(obs, dtype=np.float32)
        return obs

    def state_dict(self):
        state = {}
        state["weight"] = (self.policy.state_dict())
        state["filter"] = self.filter
        return state

    def soft_load_state_dict(self, _state_dict):
        self.policy.soft_load_state_dict(_state_dict)


def load_from_checkpoint(checkpoint_file,
                         num_states, num_actions, num_paramstates=0, num_active_dof=0,
                         num_muscles=None, num_total_muscle_related_dofs=None, islast=False,
                         device="cpu"):

    if num_active_dof == 0:
        num_active_dof = num_actions
    state = pickle.load(open(checkpoint_file, "rb"))
    worker_state = pickle.loads(state["worker"])
    policy_state = worker_state["state"]['default_policy']
    filter_state = worker_state["filters"]['default_policy']
    policy_state.pop('_optimizer_variables')
    device = torch.device(device)
    cascading_type = None
    if "cascading_type" in state.keys():
        cascading_type = state["cascading_type"]
    else:
        cascading_type = 0

    if 'weights' in policy_state.keys():
        policy = PolicyNN(num_states, num_actions,
                          policy_state['weights'], filter_state, device, cascading_type)
    else:
        policy = PolicyNN(num_states, num_actions, policy_state,
                          filter_state, device, cascading_type)

    muscle = None
    marginal = None
    ref = None

    if num_muscles:
        muscle_state = convert_to_torch_tensor(state["muscle"])
        islast = True
        if (len(muscle_state['fc.0.weight'][0]) == (num_total_muscle_related_dofs + num_active_dof)):
            islast = False
        muscle = MuscleNN(num_total_muscle_related_dofs, num_active_dof,
                          num_muscles, True, is_last=islast).to(device)
        muscle.load_state_dict(muscle_state)

    if "marginal" in state.keys() and num_paramstates > 0:
        marginal_state = convert_to_torch_tensor(state["marginal"])
        marginal = MarginalNN(num_paramstates, device).to(device)
        marginal.load_state_dict(marginal_state)

    if "ref" in state.keys():
        ref_state = convert_to_torch_tensor(state["ref"])
        ref = RefNN(num_paramstates, num_active_dof, device).to(device)
        ref.load_state_dict(ref_state)

    return policy, muscle, marginal, ref, islast


def metadata_from_checkpoint(checkpoint_file):
    state = pickle.load(open(checkpoint_file, "rb"))
    # print(state["metadata"])
    if "metadata" in state.keys():
        return state["metadata"]
    else:
        return "None"
