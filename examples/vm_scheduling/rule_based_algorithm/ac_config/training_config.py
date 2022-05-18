# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "vm_scheduling",
        "topology": "azure.2019.10k",
        "durations": 1234,
        "start_tick": 0,
    },
    "max_episode": 200,#?
    "group": "vm-ac",
    "learner_update_trigger": 5,
    "num_actors": 5
}
