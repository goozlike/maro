# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import (
    Actor, ActorProxy, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    Scheduler, OnPolicyLearner, Trajectory
)

from maro.simulator import Env
from maro.utils import set_seeds
from multiprocessing import Process

from config import agent_config, training_config

from maro.simulator.scenarios.vm_scheduling import PostponeAction, AllocateAction
from maro.simulator.scenarios.vm_scheduling.common import Action

from collections import defaultdict

import numpy as np
import torch 

common_config = {
    "pm_attributes" : ["cpu_cores_allocated", "memory_allocated"],
    "action_space" : list(range(5)),
}

class VMTrajectoryForAC(Trajectory):
    def __init__(
        self, env, *, pm_attributes, action_space
    ):
        super().__init__(env)
        self.pm_attributes = pm_attributes
        self.pms_num = env.snapshot_list["pms"][env.frame_index::["cpu_cores_capacity"]].shape[0]
        self.action_space = common_config["action_space"]
        self.look_back = 3
        self.init = False

    def get_state(self, event):
        frame_index, vm_id, vm_cpu_cores_requirement, vm_memory_requirement, valid_pms, lifetime, remaining_buffer_time = event.frame_index, event.vm_id, event.vm_cpu_cores_requirement, event.vm_memory_requirement, event.valid_pms, event.lifetime, event.remaining_buffer_time
        ticks = [max(0, frame_index - rt) for rt in range(self.look_back)]
        pms_snapshot = self.env.snapshot_list["pms"]
        if not self.init:  
            self.init = True
            self.max_cpu_capacity = 32
            self.normilized_cpu_capacity = np.float32(pms_snapshot[ticks::"cpu_cores_capacity"]) / self.max_cpu_capacity

            self.max_memory_capacity = 128
            self.normilized_memory_capacity = np.float32(pms_snapshot[ticks::"memory_capacity"]) / self.max_memory_capacity

        # pms_features = pms_snapshot[ticks::self.pm_attributes]
        # pms_features = pms_snapshot[frame_index::self.pm_attributes]

        # cpu_capacity = self.normilized_cpu_capacity - np.float32(pms_snapshot[ticks::"cpu_cores_allocated"]) / self.max_cpu_capacity
        # memory_capacity = self.normilized_memory_capacity - np.float32(pms_snapshot[ticks::"memory_allocated"]) / self.max_memory_capacity
        cpu_capacity = 1.0 - np.float32(pms_snapshot[ticks::"cpu_cores_allocated"]) / self.max_cpu_capacity
        memory_capacity = 1.0 - np.float32(pms_snapshot[ticks::"memory_allocated"]) / self.max_memory_capacity

        vm_cpu_cores_requirement = vm_cpu_cores_requirement / self.max_cpu_capacity
        vm_memory_requirement = vm_memory_requirement / self.max_memory_capacity
        # is_valid = [1 if x in valid_pms else 0 for x in range(self.pms_num)]
        return np.float32(np.concatenate((cpu_capacity, memory_capacity, [vm_cpu_cores_requirement, vm_memory_requirement, lifetime / 8639])))
    
    def get_action(self, action_by_agent, event):
        vm_id = event.vm_id
        chosen_pm_group = action_by_agent[0]

        if chosen_pm_group in event.valid_pms:
            return {"a":AllocateAction(vm_id=vm_id, pm_id=chosen_pm_group)}

        return {"a":PostponeAction(
            vm_id=vm_id,
            postpone_step=1
        )}

    def get_offline_reward_money(self, event, request_ind):
        if request_ind == 0:
            prev_allocations = 0
            prev_latency_agent = 0
        else:
            prev_allocations = self.trajectory["metrics"][request_ind - 1]["successful_allocation"]
            prev_latency_agent = self.trajectory["latency"][request_ind - 1]["due_to_agent"]

        is_succesfull_allocation = self.trajectory["metrics"][request_ind]["successful_allocation"] - prev_allocations
        is_latency_agent = self.trajectory["latency"][request_ind]["due_to_agent"] - prev_latency_agent
        is_fail = (1 - is_latency_agent) * (1 - is_succesfull_allocation)
        
        frame_index = event.frame_index + 1
        ticks = list(range(frame_index, frame_index + 10))

        pms_snapshot = self.env.snapshot_list["pms"]

        cpu_util = pms_snapshot[ticks::"cpu_cores_allocated"] 
        mem_util = pms_snapshot[ticks::"memory_allocated"]

        energy = pms_snapshot[ticks::"energy_consumption"] * 0.07 * 1.7

        decay_list = [
            0.97 ** i for i in range(10)
            for _ in range(cpu_util.shape[0] // 10)
        ]
    
        money_value = event.lifetime * (event.vm_cpu_cores_requirement * 0.0698 + event.vm_memory_requirement * 0.0078) / 12
        #loss: 
        # if postpone: money / time_buffer_budget 
        # if failed: money * 2 
        # SLA violations 
        latency_loss = is_latency_agent * money_value / 5 + is_fail * money_value * 2
        
        #no future reward for postponed/failed requests
        #иначе будет выгодно откладывать реквесты, чтобы с них лутатать эту часть профита 
        tot_cpu = np.dot(cpu_util, decay_list)
        tot_mem = np.dot(mem_util, decay_list)
        future_profit =  tot_cpu * 0.0698 + tot_mem * 0.0078

        tot_energy_cost =  np.dot(energy, decay_list) * 10

        return np.float32(is_succesfull_allocation * money_value - tot_energy_cost - latency_loss)
        # return np.float32(is_succesfull_allocation * money_value - latency_loss)
        
    def get_offline_reward(self, event, request_ind):
        if request_ind == 0:
            prev_allocations = 0
            prev_latency_agent = 0
        else:
            prev_allocations = self.trajectory["metrics"][request_ind - 1]["successful_allocation"]
            prev_latency_agent = self.trajectory["latency"][request_ind - 1]["due_to_agent"]

        is_succesfull_allocation = self.trajectory["metrics"][request_ind]["successful_allocation"] - prev_allocations
        return 2 * is_succesfull_allocation - 1
      
      
    def on_env_feedback(self, event, state_by_agent, action_by_agent, reward):
        self.trajectory["event"].append(event)
        self.trajectory["state"].append(state_by_agent)
        self.trajectory["action"].append(action_by_agent)
        self.trajectory["metrics"].append(reward.copy())
        self.trajectory["latency"].append(reward["total_latency"].get_latency().copy())

    def on_finish(self):
        training_data = {"args": [[] for _ in range(4)]}
        i = 0
        for event, state, action, metrics in zip(self.trajectory["event"], self.trajectory["state"], self.trajectory["action"], self.trajectory["metrics"]):
            training_data["args"][0].append(state)  # state
            training_data["args"][1].append(action[0])  # action
            training_data["args"][2].append(action[1])  # log_p
            training_data["args"][3].append(self.get_offline_reward_money(event, i))  # reward
            i += 1

        
        training_data["args"] = [
            np.asarray(vals, dtype=np.float32 if i == 3 else None)
            for i, vals in enumerate(training_data["args"])
        ]
        return training_data

def get_ac_agent():
    actor_net = FullyConnectedBlock(**agent_config["model"]["actor"])
    critic_net = FullyConnectedBlock(**agent_config["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net}, optim_option=agent_config["optimization"],
    )
    agent = ActorCritic(ac_model, ActorCriticConfig(**agent_config["hyper_params"]))
    # agent.load_model_from_file("./money/post_25servers_200")

    return agent

def ac_actor():
    env = Env(**training_config["env"])
    agent = get_ac_agent()
    actor = Actor(env, agent, VMTrajectoryForAC, trajectory_kwargs=common_config)
    set_seeds(np.random.randint(0, 12345))
    actor.as_worker(training_config["group"])

def ac_learner():
    agent = get_ac_agent()
    actor = ActorProxy(
        training_config["group"], training_config["num_actors"],
        update_trigger=training_config["learner_update_trigger"],
    )
    learner = OnPolicyLearner(actor, training_config["max_episode"], agent=agent)

    learner.run()

    agent.dump_model_to_file("./money/5-0-simple25")


# Multi-threaded launcher
if __name__ == "__main__":

    actor_processes = [Process(target=ac_actor) for _ in range(training_config["num_actors"])]
    learner_process = Process(target=ac_learner)

    for i, actor_process in enumerate(actor_processes):
        set_seeds(i)  # this is to ensure that the actors explore differently.
        actor_process.start()

    learner_process.start()

    for actor_process in actor_processes:
        actor_process.join()

    learner_process.join()


