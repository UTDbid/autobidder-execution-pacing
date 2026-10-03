import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.dt_baselines.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt_baselines.dt_critics import DT_Critic
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import os
import torch
import random
import argparse
from run.run_evaluate_gas import run_test as run_evaluate_voting

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

def run_dt_baselines(baseline_method='dt_reweight', reweight_w=0.2, use_rtg=False, sparse_data=False, data_path=None):
    writer = SummaryWriter(f"results/{baseline_method}_{formatted_datetime}")
    print(f"results/{baseline_method}_{formatted_datetime}")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    train_model(baseline_method, reweight_w, logger, writer, use_rtg, sparse_data, data_path)


def train_model(baseline_method='dt_reweight', reweight_w=0.1, logger=None, writer=None, use_rtg=False, sparse_data=False, data_path=None):
    logger = logger
    replay_buffer = EpisodeReplayBuffer(state_dim=16, act_dim=1, data_path=data_path, sparse_data=sparse_data)

    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        f"saved_model/{baseline_method}")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DT_Critic(state_dim=16, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std,
                                baseline_method=baseline_method,
                                reweight_w=reweight_w,
                                use_rtg=use_rtg)
    model.to(device)

    # Gradient steps and Batch size
    step_num = 400000
    batch_size = 128
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size

    with open(f'results/{baseline_method}_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs in dataloader:
        # s, a, r, d, rtg, timesteps, mask, ctg, score_t_T
        loss_1, loss_2, loss_value = model.step(states=states, actions=actions, rewards=rewards, dones=dones, rtg=rtg, timesteps=timesteps, attention_mask=attention_mask, ctg=ctg, score_to_go=score_to_go, costs=costs)
        if i % 1000 == 0:
            logger.info(f"Step: {i} loss_critic1: {np.mean(loss_1)}  loss_critic2: {np.mean(loss_2)} loss_value: {np.mean(loss_value)}")
            if i+1 % 10000 == 0:
                model.save_net(f"results/{baseline_method}_{formatted_datetime}/saved_model/{baseline_method}/{i}")
                save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                                    f"results/{baseline_method}_{formatted_datetime}/saved_model/{baseline_method}/{i}")
        i += 1
        model.scheduler.step()

    model.save_net(f"saved_model/{baseline_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training DT Critics...')

    parser.add_argument('--baseline_method', type=str, default='dt_reweight_search_Q', choices=['dt_reweight_search_Q'], help='choose a method to run')
    parser.add_argument('--use_history_rtg', type=bool, default=False, help='whether use Rtg to predict')
    parser.add_argument('--reweight_w', type=float, default=-1, help='for dt_reweight baseline: condition = rtg + w * ctg, -1 for jt_score')
    parser.add_argument('--sparse_data', type=bool, default=True, help='whether use the final stage data of AIGB competition (AuctionNet-sparse)')
    parser.add_argument('--data_path', type=str,default=None, help='path to load the dataset')
    args = parser.parse_args()

    run_dt_baselines(baseline_method=args.baseline_method, reweight_w=args.reweight_w, use_rtg=args.use_history_rtg, sparse_data=args.sparse_data, data_path=args.data_path)
