from asyncio import events
from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from rule_based_algorithm import RuleBasedAlgorithm
from collections import defaultdict
from multiprocessing import Process
from os import makedirs
from os.path import dirname, join, realpath
from tracemalloc import start

from maro.rl import (
    Actor, ActorProxy, DQN, DQNConfig, FullyConnectedBlock, MultiAgentWrapper, OffPolicyLearner,
    SimpleMultiHeadModel, TwoPhaseLinearParameterScheduler, Trajectory
)
from maro.simulator import Env
from maro.utils import set_seeds

from dqn_config import agent_config
import numpy as np


def get_dqn_agent():
    q_model = SimpleMultiHeadModel(
        FullyConnectedBlock(**agent_config["model"]), optim_option=agent_config["optimization"]
    )
    return DQN(q_model, DQNConfig(**agent_config["hyper_params"]))

def get_state(env, event):
    frame_index, vm_id, vm_cpu_cores_requirement, vm_memory_requirement, valid_pms, lifetime, remaining_buffer_time = event.frame_index, event.vm_id, event.vm_cpu_cores_requirement, event.vm_memory_requirement, event.valid_pms, event.lifetime, event.remaining_buffer_time
    ticks = [max(0, frame_index - rt) for rt in range(3)]
    
    pms_snapshot = env.snapshot_list["pms"]

    cpu_capacity = 1.0 - np.float32(pms_snapshot[ticks::"cpu_cores_allocated"]) / 32
    memory_capacity = 1.0 - np.float32(pms_snapshot[ticks::"memory_allocated"]) / 128

    vm_cpu_cores_requirement = vm_cpu_cores_requirement / 32
    vm_memory_requirement = vm_memory_requirement / 128

    return np.float32(np.concatenate((cpu_capacity, memory_capacity, [vm_cpu_cores_requirement, vm_memory_requirement, lifetime / 8639])))

class DQNLauncher(RuleBasedAlgorithm):

    def __init__(self, **kwargs):
        super().__init__()

        self.agent = get_dqn_agent()
        # self.agent.load_model_from_file("../dqn/money/lookback_postpone_25servers_model_500")
        # self.agent.load_model_from_file("../dqn/money/postpone_25servers_model_200")
        # self.agent.load_model_from_file("../dqn/money/choose_postpone_25servers_model_500")
        self.agent.load_model_from_file("../dqn/money/energy-money-25-5-40")
        # self.agent.load_model_from_file("../dqn/money/money5-0-100")
        # self.agent.load_model_from_file("../dqn/money_util_energy/lookback_postpone_25servers_model_200")

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        state = get_state(env, decision_event)
        action_by_agent = self.agent.choose_action(state, decision_event.valid_pms)

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
