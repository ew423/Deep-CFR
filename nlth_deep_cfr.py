from PokerRL.game.games import NoLimitHoldem  # or any other game

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="SD-CFR_NLTH_TEST",
                                         nn_type="feedforward",
                                         max_buffer_size_adv=3e6,
                                         eval_agent_export_freq=20,  # export API to play against the agent
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
                                         n_batches_avrg_training=2000,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         mini_batch_size_adv=1024,
                                         mini_batch_size_avrg=1024,
                                         init_adv_model="last",
                                         init_avrg_model="last",
                                         use_pre_layers_adv=False,
                                         use_pre_layers_avrg=False,

                                         game_cls=NoLimitHoldem,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
                                         ),

                                         DISTRIBUTED=False,
                                         ),
                  eval_methods={
                      "rl_br": 3,
    },
        n_iterations=None)
    ctrl.run()
