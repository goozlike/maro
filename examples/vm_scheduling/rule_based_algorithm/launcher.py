import io
import os
import random
import timeit

import yaml
import importlib

from maro.simulator import Env
from maro.utils import convert_dottable
import pickle
import numpy as np
from maro.simulator.scenarios.vm_scheduling import PostponeAction, AllocateAction

from agent import VMSchedulingAgent


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def get_state(env, event, action):
    frame_index = event.frame_index
    pms_snapshot = env.snapshot_list["pms"]

    cpu = pms_snapshot[frame_index::"cpu_cores_capacity"] - pms_snapshot[frame_index::"cpu_cores_allocated"]
    memory = pms_snapshot[frame_index::"memory_capacity"] - pms_snapshot[frame_index::"memory_allocated"]
    energy = pms_snapshot[frame_index::"energy_consumption"]
    state = {
        'server' : np.array([[[cpu[i], memory[i]], [cpu[i], memory[i]]] for i in range(len(cpu))]),
        'energy' : np.sum(energy),
        'request': {
            'uuid': action.vm_id, 
            'cpu': event.vm_cpu_cores_requirement, 
            'mem': event.vm_memory_requirement, 
            'lifetime': event.lifetime, 
            'time': frame_index, 
            'is_double': 0, 
            'type': 0 if type(action) == AllocateAction else 1
        }, 
        'action': action.pm_id if type(action) == AllocateAction else -1
    }
    return state

CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


if __name__ == "__main__":
    start_time = timeit.default_timer()

    env = Env(
        scenario=config.env.scenario,
        topology=config.env.topology,
        start_tick=config.env.start_tick,
        durations=config.env.durations,
        snapshot_resolution=config.env.resolution
    )

    if config.env.seed is not None:
        env.set_seed(config.env.seed)
        random.seed(config.env.seed)

    metrics, decision_event, is_done = env.step(None)

    time_buffer = decision_event.remaining_buffer_time

    algorithm_class = import_class(config.algorithm.type)
    if config.algorithm.args is None:
        algorithm = algorithm_class(env=env)
    else:
        algorithm = algorithm_class(env=env, **config.algorithm.args)

    agent = VMSchedulingAgent(algorithm)
    states = []
    while not is_done:
        action = agent.choose_action(decision_event, env)
        metrics, decision_event, is_done = env.step(action)
        if decision_event is not None:
            states.append(get_state(env, decision_event, action))

    end_time = timeit.default_timer()
    print(
        f"[{config.algorithm.type.split('.')[1]}] Topology: {config.env.topology}. Total ticks: {config.env.durations}."
        f" Start tick: {config.env.start_tick}."
    )
    print(f"[Timer] {end_time - start_time:.2f} seconds to finish the simulation.")
    print(metrics)
    n_servers = len(states[0]['server'])

    with open(f'{config.algorithm.type}{n_servers}-{time_buffer}.p', 'wb') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)


