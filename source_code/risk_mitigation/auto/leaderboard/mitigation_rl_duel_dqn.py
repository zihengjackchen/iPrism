import torch
from torch.utils.tensorboard import SummaryWriter
from mitigation_rl_dqn import CarlaLBCDQN
from carla_project.src.duel_dqn_models import DuelQvalueModel

PLOT = False


class CarlaLBCDUELDQN(CarlaLBCDQN):
    def __init__(self):
        super().__init__()
        self.qvalue_net = DuelQvalueModel(self.hparams)
        self.qvalue_net.cuda()
        self.qvalue_net_target = DuelQvalueModel(self.hparams)
        for target_param, param in zip(self.qvalue_net_target.parameters(), self.qvalue_net.parameters()):
            target_param.data.copy_(param.data)
        self.qvalue_net_target.cuda()


def main():
    torch.multiprocessing.set_start_method("spawn")
    duel_dqn_lbc = CarlaLBCDUELDQN()
    # dqn_lbc.inference_step_loop()
    # dqn_lbc.inference_loop_standalone_process(10)
    # dqn_lbc.inference_step_loop_threshold_mitigation()
    # dqn_lbc.training_step_loop_reinforcement_mitigation(episodes=500,
    #                                                     batch_size=32,
    #                                                     start_training_replaybuffer_size=256,
    #                                                     skip_first_seconds=4.5,
    #                                                     skip_mitigation_seconds=7,
    #                                                     save_frequency=50)
    duel_dqn_lbc.training_loop_standalone_process(total_episodes=50,
                                                  batch_size=32,
                                                  start_training_replaybuffer_size=512,
                                                  skip_first_seconds=4,
                                                  skip_mitigation_seconds=5,
                                                  save_frequency=50,
                                                  resume=16,
                                                  target_net_update_freq=5000,
                                                  mitigation_penalty=0.0)


if __name__ == "__main__":
    main()
