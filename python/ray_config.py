import os
from ray import tune
import copy

CONFIG = dict()

common_config = {
    "env": "MyEnv",
    "trainer_config": { },
    "env_config": { },
    "framework": "torch",
    "extra_python_environs_for_driver": {  },
    "extra_python_environs_for_worker": {  },
    "model": {
        "custom_model": "MyModel",
        "custom_model_config": {
            'value_function' : None
        },
        "max_seq_len": 0    # Placeholder value needed for ray to register model
    },
    "evaluation_config": {  },
}

CONFIG["ppo"] = copy.deepcopy(common_config)
CONFIG["ppo"]["trainer_config"]["algorithm"] = "CustomPPO"
CONFIG["ppo"].update({
    # "horizon": inf,
    "horizon" : 1000,
    "use_critic": True,
    "use_gae": True,
    "lambda": 0.99,
    # "lambda": 1.0,
    "gamma": 0.99,
    "kl_coeff": 0.01,
    "shuffle_sequences": True,
    "num_sgd_iter": 4,
    "lr": 1e-4,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.000,
    "entropy_coeff_schedule": None,
    "clip_param": 0.2,
    "vf_clip_param": 100.0,
    "grad_clip": None,
    "kl_target": 0.01,
    "batch_mode":  "truncate_episodes",
    # "batch_mode":  "complete_episodes",
    "observation_filter": "NoFilter",
    "normalize_actions" : False, 
    "clip_actions": True,
    # "simple_optimizer": False,

    #Device Configuration
    "create_env_on_driver": False,
    "num_cpus_for_driver": 0,
    "num_gpus": 1,
    "num_gpus_per_worker": 0.,
    "num_envs_per_worker": 1,
    "num_cpus_per_worker": 1,
})

# Muscle Configuration
CONFIG["ppo"]["trainer_config"]["muscle_lr"] = 1e-4
CONFIG["ppo"]["trainer_config"]["muscle_num_epochs"] = 4

# Marginal Configuration
CONFIG["ppo"]["trainer_config"]["marginal_lr"] = 1e-4
CONFIG["ppo"]["trainer_config"]["marginal_num_epochs"] = 3
CONFIG["ppo"]["trainer_config"]["initial_state_size"] = 4000
CONFIG["ppo"]["trainer_config"]["marginal_k"] = 3

# Reference Configuration
CONFIG["ppo"]["trainer_config"]["ref_lr"] = 1e-4
CONFIG["ppo"]["trainer_config"]["ref_num_epochs"] = 3


# Medium Set (For a node or a PC)
CONFIG["ppo_medium"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_medium"]["train_batch_size"] =  8192 * 8
CONFIG["ppo_medium"]["sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["muscle_sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["marginal_sgd_minibatch_size"] = 1024
CONFIG["ppo_medium"]["trainer_config"]["ref_sgd_minibatch_size"] = 1024

#===============================Training Configuration For Various Devices=========================================

CONFIG["ppo_medium_server"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_server"]["num_workers"] = 128 * 4

CONFIG["ppo_medium_node"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_node"]["num_workers"] = 128

CONFIG["ppo_medium_pc"] = copy.deepcopy(CONFIG["ppo_medium"])
CONFIG["ppo_medium_pc"]["num_workers"] = 32


# Mini Set (For Test)
CONFIG["ppo_mini"] = copy.deepcopy(CONFIG["ppo"])
CONFIG["ppo_mini"]["train_batch_size"] = 512
CONFIG["ppo_mini"]["sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["trainer_config"]["muscle_sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["trainer_config"]["marginal_sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["trainer_config"]["ref_sgd_minibatch_size"] = 64
CONFIG["ppo_mini"]["num_workers"] = 1