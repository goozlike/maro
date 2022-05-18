import numpy as np

from maro.rl import (
    Actor, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    Scheduler, OnPolicyLearner, Trajectory
)
from maro.simulator import Env
from maro.utils import set_seeds

from ac_config import agent_config, training_config
from rule_based_algorithm import RuleBasedAlgorithm

from maro.simulator.scenarios.vm_scheduling import PostponeAction, AllocateAction, DecisionPayload
from maro.simulator.scenarios.vm_scheduling.common import Action

from collections import defaultdict

import numpy as np
import torch 

common_config = {
    "pm_attributes" : ["cpu_cores_allocated", "memory_allocated"],
    "action_space" : list(range(5)),
}

def get_ac_agent():
    actor_net = FullyConnectedBlock(**agent_config["model"]["actor"])
    critic_net = FullyConnectedBlock(**agent_config["model"]["critic"])
    # critic_net.load_state_dict(torch.load("../dqn/lookback_postpone_5servers_model_50_fix"))
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net}, optim_option=agent_config["optimization"],
    )
    agent = ActorCritic(ac_model, ActorCriticConfig(**agent_config["hyper_params"]))
    # agent.load_model_from_file("../ac/25servers_no_postpone_model_250")
    # agent.load_model_from_file("../ac/money/post_25servers_200choose")
    # agent.load_model_from_file("../ac/money/5-0-valid100")
    # agent.load_model_from_file("../ac/money/5-0-simple100")
    # agent.load_model_from_file("../ac/money/5-5-final-250")
    agent.load_model_from_file("../ac/money/25-5-me250")
    return agent

def get_state(env, event):
    frame_index, vm_id, vm_cpu_cores_requirement, vm_memory_requirement, valid_pms, lifetime, remaining_buffer_time = event.frame_index, event.vm_id, event.vm_cpu_cores_requirement, event.vm_memory_requirement, event.valid_pms, event.lifetime, event.remaining_buffer_time
    ticks = [max(0, frame_index - rt) for rt in range(3)]
    
    pms_snapshot = env.snapshot_list["pms"]

    cpu_capacity = 1.0 - np.float32(pms_snapshot[ticks::"cpu_cores_allocated"]) / 32
    memory_capacity = 1.0 - np.float32(pms_snapshot[ticks::"memory_allocated"]) / 128

    vm_cpu_cores_requirement = vm_cpu_cores_requirement / 32
    vm_memory_requirement = vm_memory_requirement / 128
    
    return np.float32(np.concatenate((cpu_capacity, memory_capacity, [vm_cpu_cores_requirement, vm_memory_requirement, lifetime / 8639])))

class ACLauncher(RuleBasedAlgorithm):

    def __init__(self, **kwargs):
        super().__init__()

        self.agent = get_ac_agent()

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        state = get_state(env, decision_event)

        action_by_agent = self.agent.choose_action(state, decision_event.valid_pms, False)[0]
        # action_by_agent = self.agent.choose_action(state, decision_event.valid_pms, True)[0]


        if action_by_agent in decision_event.valid_pms:
            # Take action to allocate on the chose PM.
            return AllocateAction(
                vm_id=decision_event.vm_id,
                pm_id=action_by_agent
            )
        

        return PostponeAction(
            vm_id=decision_event.vm_id,
            postpone_step=1
        )
