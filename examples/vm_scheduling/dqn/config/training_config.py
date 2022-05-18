# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k",
        "durations": 1234,
    },
    "max_episode": 20,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.9,
        "mid": 0.6,
        "end": 0.2
    },
    "training": {
        "min_experiences_to_train": 1024,
        "train_iter": 10,
        "batch_size": 128,
        "prioritized_sampling_by_loss": False
    },
    "group": "vm-dqn5",
    "learner_update_trigger": 5,
    "num_actors": 5
}
