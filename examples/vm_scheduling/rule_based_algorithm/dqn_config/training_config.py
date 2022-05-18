# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k",
        "durations": 666,
        # "start_tick": 100
    },
    "max_episode": 200,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.5,
        "mid": 0.0,
        "end": 0.0
    },
    "training": {
        "min_experiences_to_train": 1024,
        "train_iter": 10,
        # "batch_size": 256,
        "batch_size": 128,
        "prioritized_sampling_by_loss": True
    },
    "group": "vm-dqn",
    "learner_update_trigger": 5,
    "num_actors": 1
}
