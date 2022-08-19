from ray.tune.utils import validate_save_restore
from itertools import combinations
import importlib
from IPython import embed
import mcmc
import argparse
import numpy as np
import json
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict
from ray_model import SimulationNN_Ray, load_from_checkpoint, metadata_from_checkpoint
from ray_env import (MyEnv,
                     MuscleLearner,
                     MarginalLearner,
                     RefLearner)
import math
import ray
import time
import random

from ray import tune
from ray.experimental import dynamic_resources
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer, DDPPOTrainer
from ray_ppo import CustomPPOTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ars import ARSTrainer
from ray.rllib.env.base_env import _VectorEnvToBaseEnv
from ray.rllib.env.remote_vector_env import RemoteVectorEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

torch, nn = try_import_torch()


def with_worker_env(worker, callable):
    if isinstance(worker.async_env, _VectorEnvToBaseEnv):
        envs = worker.async_env.vector_env.get_unwrapped()
        if envs:
            return [callable(env) for env in envs]
        else:
            return [callable(worker.async_env.vector_env)]
    elif isinstance(worker.async_env, RemoteVectorEnv):
        envs = worker.async_env.actors
        return [callable(env) for env in envs]
    else:
        raise RuntimeError(
            f"RolloutWorker has invalid env type f{type(worker.async_env)}!")


def with_multiple_worker_env(workers, callable, block=False):
    res = [worker.apply.remote(lambda worker: with_worker_env(worker, callable))
           for worker in workers]
    if block:
        return ray.get(res)
    else:
        return res


def apply_config_to_worker(worker, callable, weight, filter, minv, maxv, muscle_weight, cascading_type):
    envs = worker.async_env.vector_env.get_unwrapped()
    for env in envs:
        callable(env, weight, filter, minv, maxv,
                 muscle_weight, cascading_type)


def apply_paramstate_to_worker(worker, callable, paramstates):
    envs = worker.async_env.vector_env.get_unwrapped()
    idx = 0
    for env in envs:
        callable(env, paramstates[idx])
        idx += 1


def apply_cascading_map_to_worker(worker, callable, cascading_map):
    envs = worker.async_env.vector_env.get_unwrapped()
    for env in envs:
        callable(env, cascading_map)


# def get_marginal_tuples(worker):
#     v_f = worker.policy_map['default_policy'].model.get_value_function
#     f = worker.filters['default_policy']
#     envs = worker.async_env.vector_env.get_unwrapped()
#     res = []
#     for env in envs:
#         for t in env.get_marginal_tuples():
#             t[1] = v_f(np.float32(f(t[1])))
#             res.append(t)
#     return res


# Create MASSTrainer where cls is PPOTrainer, DDPPOTrainer, IMPALATrainer, etc...
def create_mass_trainer(rl_algorithm: str):
    if rl_algorithm == "CustomPPO":
        RLTrainer = CustomPPOTrainer
    elif rl_algorithm == "PPO":
        RLTrainer = PPOTrainer
    elif rl_algorithm == "DDPPO":
        RLTrainer = DDPPOTrainer
    elif rl_algorithm == "IMPALA":
        RLTrainer = ImpalaTrainer
    elif rl_algorithm == "APPO":
        RLTrainer = APPOTrainer
    elif rl_algorithm == "ARS":
        RLTrainer = ARSTrainer
    else:
        raise RuntimeError(f"Invalid algorithm {rl_algorithm}!")

    class MASSTrainer(RLTrainer):
        def setup(self, config):

            self.max_reward = 0
            self.use_displacement = config["trainer_config"]["use_displacement"]
            self.use_muscle = config["trainer_config"]["use_muscle"]
            self.use_adaptive_sampling = config["trainer_config"]["use_adaptive_sampling"]
            self.use_timewarp = 1 if config["trainer_config"]["use_timewarp"] else 0
            self.trainer_config = config.pop("trainer_config")
            self.metadata = config.pop("metadata")
            self.prev_model = config.pop("prev_model")
            self.cascading_map = config.pop("cascading_map")
            self.num_envs_per_worker = config["num_envs_per_worker"]
            RLTrainer.setup(self, config=config)

            self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.remote_workers = self.workers.remote_workers()
            self.local_worker = self.workers.local_worker()

            self.num_action = config["env_config"]["num_action"]
            self.num_state = config["env_config"]["num_state"]
            self.num_active_dof = config["env_config"]["num_active_dof"]

            self.cascading_type = config["env_config"]["cascading_type"]
            # if self.cascading_type > 0:
            #     self.num_action += 1

            self.num_paramstate = config["env_config"]["num_param_state"]
            self.is_reset = 0
            self.muscle_loss = 1000.0

            if self.use_muscle:
                self.num_muscles = config["env_config"]["num_muscles"]
                self.num_muscle_dofs = config["env_config"]["num_muscle_dofs"]

            if self.use_adaptive_sampling:
                self.min_v = config["env_config"]["min_v"]
                self.max_v = config["env_config"]["max_v"]
                self.initial_paramstate_num = self.trainer_config["initial_state_size"]
                self.initial_param_states = None
                self.marginal_k = self.trainer_config["marginal_k"]

            self.rank = None

            if self.use_muscle:
                self.muscle_learner = MuscleLearner(self.device, self.num_active_dof, self.num_muscles, self.num_muscle_dofs,
                                                    self.use_timewarp,
                                                    # buffer_size=self.trainer_config["muscle_buffer_size"],
                                                    learning_rate=self.trainer_config["muscle_lr"],
                                                    num_epochs=self.trainer_config["muscle_num_epochs"], batch_size=self.trainer_config["muscle_sgd_minibatch_size"], is_last=not (self.cascading_type == 0))

            if self.use_adaptive_sampling:
                self.marginal_learner = MarginalLearner(self.device, self.num_paramstate,
                                                        # buffer_size=self.trainer_config["marginal_buffer_size"],
                                                        learning_rate=self.trainer_config["marginal_lr"],
                                                        num_epochs=self.trainer_config["marginal_num_epochs"], batch_size=self.trainer_config["marginal_sgd_minibatch_size"])

            if self.use_displacement:
                self.ref_learner = RefLearner(self.device, self.num_paramstate, self.num_active_dof,
                                              # buffer_size=self.trainer_config["marginal_buffer_size"],
                                              learning_rate=self.trainer_config["ref_lr"],
                                              num_epochs=self.trainer_config["ref_num_epochs"], batch_size=self.trainer_config["ref_sgd_minibatch_size"])

            #Cascading NN
            if len(self.prev_model) > 0:  # Use Prev Model

                #filter sync
                # prev_filter = self.prev_model['sim_nn'][-1][0]['filter']
                # self.local_worker.filters["default_policy"].sync(prev_filter)
                # for worker in self.remote_workers:
                #     worker.apply.remote(lambda worker: worker.filters['default_policy'].sync(prev_filter))

                # self.muscle_learner.model.load_state_dict(self.prev_model["muscle_nn"])
                # model_weights = self.muscle_learner.get_model_weights(device=torch.device("cpu"))
                # with_multiple_worker_env(self.remote_workers,
                #                          lambda env: env.load_muscle_model_weights(model_weights),
                #                          block=True)
                assert (len(self.prev_model['sim_nn']) == len(
                    self.prev_model['muscle_nn']))

                # for sim_nn, muscle_nn in self.prev_model['sim_nn'], self.prev_model['muscle_nn']:
                for i in range(len(self.prev_model['sim_nn'])):

                    sim_nn = self.prev_model['sim_nn'][i]
                    muscle_nn = self.prev_model['muscle_nn'][i]

                    sim_w = sim_nn[0]['weight']
                    sim_f = sim_nn[0]['filter']
                    min_v = sim_nn[1][0]
                    max_v = sim_nn[1][1]
                    cascading_type = sim_nn[2]

                    muscle_w = muscle_nn

                    for worker in self.remote_workers:
                        worker.apply.remote(lambda worker: apply_config_to_worker(worker, lambda env, sim_w, filter, minv, maxv, muscle_w, cascading_type: env.set_prev_model(
                            sim_w, filter, minv, maxv, muscle_w, cascading_type), sim_w, sim_f, min_v, max_v, muscle_w, cascading_type))

                for worker in self.remote_workers:
                    worker.apply.remote(lambda worker: apply_cascading_map_to_worker(
                        worker, lambda env, cascading_map: env.load_cascading_map(cascading_map), cascading_map))

            if self.use_adaptive_sampling:
                param_set = []
                size = len(self.min_v)
                for i in range(size):
                    for idx_list in list(combinations(range(size), i)):
                        param = copy.deepcopy(self.min_v)
                        for idx in idx_list:
                            param[idx] = self.max_v[idx]
                        param_set.append(param)

                param_set.append(self.max_v)
                self.all_params = torch.tensor(param_set).to(
                    self.marginal_learner.device)

            self.iter_idx = 0

        def step(self):

            #Simulation NN Learning
            result = RLTrainer.step(self)

            # TODO: Run both paths (muscle / adaptive_sampling) simultaneously using ray.remote()
            result["loss"] = {}
            result["num_tuples"] = {}
            result["is_reset"] = self.is_reset
            #Muscle NN Learning
            if self.use_muscle:
                start = time.perf_counter()
                muscle_tuples = ray.get([worker.foreach_env.remote(
                    lambda env: env.get_muscle_tuples()) for worker in self.remote_workers])

                total_muscle_tuples = [[], [], [], [], [], []]
                for worker in muscle_tuples:
                    for env in worker:
                        assert (len(total_muscle_tuples) == len(env))
                        for i in range(len(env)):
                            total_muscle_tuples[i].append(env[i])

                res = []

                for tuples in total_muscle_tuples:
                    while ([] in tuples):
                        tuples.remove([])
                    res.append(np.concatenate(tuples, axis=0))

                loading_time = (time.perf_counter() - start) * 1000
                stats = self.muscle_learner.learn(res)

                distribute_time = time.perf_counter()

                model_weights = ray.put(
                    self.muscle_learner.get_model_weights(device=torch.device("cpu")))

                for worker in self.remote_workers:
                    worker.foreach_env.remote(
                        lambda env: env.load_muscle_model_weights(model_weights))

                distribute_time = (time.perf_counter() -
                                   distribute_time) * 1000
                total_time = (time.perf_counter() - start) * 1000
                result['timers']['muscle_learning'] = stats.pop('time')
                result['num_tuples']['muscle_learning'] = stats.pop(
                    'num_tuples')
                result['timers']['muscle_learning']['distribute_time_ms'] = distribute_time
                result['timers']['muscle_learning']['loading_time_ms'] = loading_time
                result['timers']['muscle_learning']['total_ms'] = total_time
                result["loss"].update(stats)
                self.muscle_loss = result["loss"]["loss_muscle"]

            # Ref NN Learning
            if self.use_displacement:
                start = time.perf_counter()
                displacement_tuples = ray.get([worker.foreach_env.remote(
                    lambda env: env.get_displacement_tuples()) for worker in self.remote_workers])

                # embed()

                total_displacement_tuples = []
                for worker in displacement_tuples:
                    for env in worker:
                        total_displacement_tuples += env

                displacement_tuples = np.array(
                    total_displacement_tuples, dtype=object)

                if (len(displacement_tuples) > 0):
                    param_tuples = torch.from_numpy(
                        np.vstack(displacement_tuples[:, 0]).astype(np.float32)).to(self.device)
                    displacement_tuples = torch.from_numpy(
                        np.vstack(displacement_tuples[:, 1]).astype(np.float32)).to(self.device)

                    loading_time = (time.perf_counter() - start) * 1000
                    stats = self.ref_learner.learn(
                        param_tuples, displacement_tuples)

                    distribute_time = time.perf_counter()

                    model_weights = ray.put(
                        self.ref_learner.get_model_weights(device=torch.device("cpu")))

                    for worker in self.remote_workers:
                        worker.foreach_env.remote(
                            lambda env: env.load_ref_model_weights(model_weights))

                    distribute_time = (time.perf_counter() -
                                       distribute_time) * 1000
                    total_time = (time.perf_counter() - start) * 1000
                    result['timers']['ref_learning'] = stats.pop('time')
                    result['num_tuples']['ref_learning'] = stats.pop(
                        'num_tuples')
                    result['timers']['ref_learning']['distribute_time_ms'] = distribute_time
                    result['timers']['ref_learning']['loading_time_ms'] = loading_time
                    result['timers']['ref_learning']['total_ms'] = total_time
                    result["loss"].update(stats)

            #Marginal NN Learning
            if self.use_adaptive_sampling and self.iter_idx > 0:
                start = time.perf_counter()

                marginal_tuples = with_multiple_worker_env(self.remote_workers,
                                                           lambda env: env.get_marginal_tuples(),
                                                           block=True)
                total_marginal_tuples = []
                for worker in marginal_tuples:
                    for env in worker:
                        total_marginal_tuples += env
                marginal_tuples = np.array(total_marginal_tuples, dtype=object)

                current_filter = self.local_worker.filters['default_policy']
                current_model = self.local_worker.policy_map['default_policy'].model
                current_model.eval()

                filtered_state_tuples = []
                for marginal_tuple in marginal_tuples:
                    filtered_state_tuples.append(
                        current_filter(marginal_tuple[1], update=False))
                filtered_state_tuples = np.array(filtered_state_tuples)

                param_tuples = torch.from_numpy(
                    np.vstack(marginal_tuples[:, 0]).astype(np.float32)).to(self.device)
                filtered_state_tuples = torch.from_numpy(
                    np.vstack(filtered_state_tuples).astype(np.float32)).to(self.device)

                v_tuples = current_model.get_value(filtered_state_tuples)

                loading_time = (time.perf_counter() - start) * 1000

                stats = self.marginal_learner.learn(param_tuples, v_tuples)

                total_time = (time.perf_counter() - start) * 1000
                result['timers']['marginal_learning'] = stats.pop('time')
                result['num_tuples']['marginal_learning'] = stats.pop(
                    'num_tuples')

                result['timers']['marginal_learning']['loading_time_ms'] = loading_time
                result['timers']['marginal_learning']['total_ms'] = total_time
                result["loss"].update(stats)

                current_model.train()

            #Adaptive Sampling And Param State Seed

            if self.use_adaptive_sampling:
                sampling_timer = {}
                start = time.perf_counter()
                self.initial_param_states, stats = self.GenerateInitialStates()
                # sampling_timer['sampling_time_ms'] = (time.perf_counter() - start) * 1000
                sampling_timer.update(stats)

                start = time.perf_counter()
                for worker in self.remote_workers:
                    paramstate = []
                    for i in range(self.num_envs_per_worker):
                        paramstate.append(random.choice(
                            self.initial_param_states))
                    worker.apply.remote(lambda worker: apply_paramstate_to_worker(
                        worker, lambda env, paramstate: env.load_paramstate(paramstate), paramstate))
                    sampling_timer['distribution_time_ms'] = (
                        time.perf_counter() - start) * 1000
                # result['timers'].update(sampling_timer)
                result['timers']['adaptive_sampling'] = sampling_timer

            self.iter_idx += 1

            if self.max_reward < result['episode_reward_mean']:
                self.max_reward = result['episode_reward_mean']
                self.save_max_checkpoint(self._logdir)

            return result

        def __getstate__(self):
            state = RLTrainer.__getstate__(self)
            remote_worker = self.remote_workers[0]
            state["cascading_type"] = self.cascading_type
            if self.use_muscle:
                state["muscle"] = self.muscle_learner.get_weights()
                state["muscle_optimizer"] = self.muscle_learner.get_optimizer_weights()

            if self.use_adaptive_sampling:
                state["marginal"] = self.marginal_learner.get_weights()
                state["marginal_optimizer"] = self.marginal_learner.get_optimizer_weights()

            if self.use_displacement:
                state["ref"] = self.ref_learner.get_weights()
                state["ref_optimizer"] = self.ref_learner.get_optimizer_weights()

            state["metadata"] = self.metadata
            return state

        def __setstate__(self, state):
            RLTrainer.__setstate__(self, state)
            self.cascading_type = state["cascading_type"]
            if self.use_muscle:
                self.muscle_learner.set_weights(state["muscle"])
                if "muscle_optimizer" in state.keys():
                    self.muscle_learner.set_optimizer_weights(
                        state["muscle_optimizer"])

                model_weights = self.muscle_learner.get_model_weights(
                    device=torch.device("cpu"))

                with_multiple_worker_env(self.remote_workers,
                                         lambda env: env.load_muscle_model_weights(
                                             model_weights),
                                         block=True)
            if self.use_displacement:
                self.ref_learner.set_weights(state["ref"])
                if "ref_optimizer" in state.keys():
                    self.ref_learner.set_optimizer_weights(
                        state["ref_optimizer"])
                model_weights = self.ref_learner.get_model_weights(
                    device=torch.device("cpu"))
                with_multiple_worker_env(self.remote_workers,
                                         lambda env: env.load_ref_model_weights(
                                             model_weights),
                                         block=True)

            if self.use_adaptive_sampling:
                self.marginal_learner.set_weights(state["marginal"])
                if "marginal_optimizer" in state.keys():
                    self.marginal_learner.set_optimizer_weights(
                        state["marginal_optimizer"])

        def save_checkpoint(self, checkpoint_path):
            print(f'Saving checkpoint at path {checkpoint_path}')
            RLTrainer.save_checkpoint(self, checkpoint_path)
            with open(Path(checkpoint_path) / "trainer_config.pkl", 'wb') as f:
                pickle.dump(self.trainer_config, f)
            with open(Path(checkpoint_path) / "trainer_config.json", 'w') as f:
                json.dump(self.trainer_config, f)
            return checkpoint_path

        def save_max_checkpoint(self, checkpoint_path) -> str:
            with open(Path(checkpoint_path) / "max_checkpoint", 'wb') as f:
                pickle.dump(self.__getstate__(), f)
            return checkpoint_path

        def load_checkpoint(self, checkpoint_path):
            print(f'Loading checkpoint at path {checkpoint_path}')
            checkpoint_file = list(Path(checkpoint_path).glob("checkpoint-*"))
            if len(checkpoint_file) == 0:
                raise RuntimeError("Missing checkpoint file!")
            RLTrainer.load_checkpoint(self, checkpoint_file[0])
            with open(Path(checkpoint_path) / "trainer_config.pkl", 'rb') as f:
                self.trainer_config = pickle.load(f)

        def GenerateInitialStates(self):
            assert (self.use_adaptive_sampling)

            marginal_model = self.marginal_learner.model
            marginal_model.eval()
            device = self.marginal_learner.device

            print('Generate Initial State......')
            start_time = time.perf_counter()

            min_v = self.min_v
            max_v = self.max_v

            marginal_value_avg = np.average(
                marginal_model.no_grad_forward(self.all_params))
            initializing_time = (time.perf_counter() - start_time) * 1000
            start_time = time.perf_counter()

            # target distribution
            def target_dist(x):
                marginal_value = marginal_model.get_value(x)
                p = math.exp(self.marginal_k *
                             (1. - marginal_value/marginal_value_avg))
                return p

            # proposed distribution
            def proposed_dist(x, min_v, max_v):
                size = x.size
                value = np.array([random.choice([min_v[i], max_v[i]])
                                 for i in range(size)])
                return value

            mcmc_sampler = mcmc.MetropolisHasting(
                self.num_paramstate, min_v, max_v, target_dist, proposed_dist)

            paramstates = mcmc_sampler.get_sample(self.initial_paramstate_num)
            sampling_time = (time.perf_counter() - start_time) * 1000

            stats = {'initializing_time_ms': initializing_time,
                     'sampling_time_ms': sampling_time}

            return paramstates, stats
    return MASSTrainer


def get_config_from_file(filename: str, config: str):
    exec(open(filename).read(), globals())
    config = CONFIG[config]
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action='store_true')
parser.add_argument("--config", type=str, default="ppo")
parser.add_argument("--config-file", type=str,
                    default="../python/ray_config.py")
parser.add_argument("--metadata", type=str, default="../data/metadata.txt")
parser.add_argument("--cascading_nn", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument('-m', '--use-metadata', action='store_true')
parser.add_argument('-n', '--name', type=str)

if __name__ == "__main__":
    metadata = None
    checkpoint_path = None
    args = parser.parse_args()
    print('Argument : ', args)

    # metadata setting
    if (not (args.checkpoint == None)):
        checkpoint_path = Path(args.checkpoint)
        if args.use_metadata == False:
            metadata = metadata_from_checkpoint(str(checkpoint_path))
        else:
            metadata = args.metadata
        checkpoint_path = str(checkpoint_path.parent)
    else:
        metadata = args.metadata

    if args.cluster:
        ray.init(address=os.environ["ip_head"])
    else:
        ray.init()

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    ModelCatalog.register_custom_model("MyModel", SimulationNN_Ray)
    register_env("MyEnv", lambda config: MyEnv(metadata))
    print(f'Loading config {args.config} from config file {args.config_file}.')
    config = get_config_from_file(args.config_file, args.config)

    config["rollout_fragment_length"] = config["train_batch_size"] / \
        (config["num_workers"] * config["num_envs_per_worker"])

    local_dir = "./ray_results"
    algorithm = config["trainer_config"]["algorithm"]

    MASSTrainer = create_mass_trainer(algorithm)

    with MyEnv(metadata) as env:
        config["env_config"]["num_active_dof"] = env.num_active_dof
        config["env_config"]["num_state"] = env.num_state
        config["env_config"]["num_action"] = env.num_action
        config["env_config"]["cascading_type"] = env.env.GetCascadingType()
        config["metadata"] = env.metadata

        config["trainer_config"]["use_muscle"] = env.use_muscle
        config["trainer_config"]["use_adaptive_sampling"] = env.use_adaptive_sampling
        config["trainer_config"]["use_timewarp"] = env.use_timewarp
        config["trainer_config"]["use_displacement"] = env.env.UseDisplacement()

        print(env.metadata)

        config["env_config"]["num_param_state"] = env.num_paramstate

        if env.use_muscle:
            config["env_config"]["num_muscles"] = env.num_muscles
            config["env_config"]["num_muscle_dofs"] = env.num_muscle_dofs

        if env.use_adaptive_sampling:
            config["env_config"]["min_v"] = env.env.GetMinV()
            config["env_config"]["max_v"] = env.env.GetMaxV()

        model_paths = []
        config["prev_model"] = {}

        if not args.cascading_nn == None:
            model_paths = args.cascading_nn.split(',')

            #Load Muscle NN and Value Function Load
            config["prev_model"]["sim_nn"] = []
            config["prev_model"]["muscle_nn"] = []

        # Cascading Learning
        # Order : Old Model -> New Model
        for model_path in model_paths:
            model_cascadingType = 0
            action_diff = 0
            checkpoint = pickle.load(open(model_path, "rb"))

            if "cascading_type" in checkpoint.keys():
                model_cascadingType = checkpoint["cascading_type"]

            if model_cascadingType != 0:
                action_diff = 1

            space = env.env.GetSpace(metadata_from_checkpoint(model_path))
            # if env.env.GetStateType() == 11:
            proj_state_num = len(env.env.GetProjState(space[0], space[1]))

            # embed()
            sim_nn, muscle_nn, _, _, _ = load_from_checkpoint(
                model_path, proj_state_num, env.num_action + action_diff, env.num_paramstate, env.num_active_dof, env.num_muscles, env.num_muscle_dofs)
            # space = env.env.GetSpace(metadata_from_checkpoint(model_path))

            config["prev_model"]["sim_nn"].append(
                (sim_nn.state_dict(), space, sim_nn.cascading_type))
            config["prev_model"]["muscle_nn"].append(muscle_nn.state_dict())
            if model_path == model_paths[-1]:
                config['model']['custom_model_config']['value_function'] = sim_nn.get_value_function_weight()
        print('Cascading Map : ')

        cascading_map = []
        if len(model_paths) > 0:
            for i in range(len(model_paths)):
                cascading_map.append([])
            f = open('../data/cascading_map.txt', 'r')
            while True:
                line = f.readline()
                if not line:
                    break
                idx = line.split()
                cascading_map[int(idx[0])].append(int(idx[1]))
                print(idx)
            f.close()
        config['cascading_map'] = cascading_map

    from ray.tune import CLIReporter

    # Limit the number of rows.
    reporter = CLIReporter(max_report_frequency=10)

    tune.run(MASSTrainer,
             name=args.name,
             config=config,
             local_dir=local_dir,
             restore=checkpoint_path,
             progress_reporter=reporter,
             checkpoint_freq=25)

    ray.shutdown()
