from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from rule_based_algorithm import RuleBasedAlgorithm


class FirstFit(RuleBasedAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        # Use a valid PM based on its order.
        if len(decision_event.valid_pms) == 0:
            return PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_step=1
            )

        return AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=decision_event.valid_pms[0]
        )
