import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.dt_baselines.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt_baselines.dt_baselines import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import os
import torch
import argparse

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

def run_dt_baselines(baseline_method='dt_reweight', reweight_w=0.2, data_path=None, sparse_data=False):
    writer = SummaryWriter(f"results/{baseline_method}_{formatted_datetime}")
    print(f"results/{baseline_method}_{formatted_datetime}")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    train_model(baseline_method, reweight_w, sparse_data, logger, writer, data_path)


def train_model(baseline_method='dt_reweight', reweight_w=0.1, sparse_data=False, logger=None, writer=None, data_path=None, load_preprocessed_data=False):
    state_dim = 16
    logger = logger
    # Load Dataset
    replay_buffer = EpisodeReplayBuffer(state_dim=state_dim, act_dim=1, data_path=data_path, sparse_data=sparse_data)
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        f"saved_model/{baseline_method}test")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    # setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std,
                                baseline_method=baseline_method,
                                reweight_w=reweight_w)
    model.to(device)

    # Gradient steps and Batch size
    step_num = 400000
    batch_size = 128
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size

    # Record the hyperparameters
    with open(f'results/{baseline_method}_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs in dataloader:
        train_loss = model.step(states=states, actions=actions, rewards=rewards, dones=dones, rtg=rtg, timesteps=timesteps, attention_mask=attention_mask, ctg=ctg, score_to_go=score_to_go, costs=costs)
        if i % 1000 == 0:
            logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
            if i+1 % 10000 == 0:
                model.save_net(f"results/{baseline_method}_{formatted_datetime}/saved_model/{baseline_method}/{i}")
                save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                                    f"results/{baseline_method}_{formatted_datetime}/saved_model/{baseline_method}/{i}")

        writer.add_scalar('Action loss', np.mean(train_loss), i)
        model.scheduler.step()
        i += 1

    model.save_net(f"saved_model/{baseline_method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training dt/cdt baselines...')

    parser.add_argument('--baseline_method', type=str, default='vanilla_dt', choices=['vanilla_dt', 'dt_reweight'], help='choose a method to run')
    parser.add_argument('--reweight_w', type=float, default=0.2, help='for dt_reweight baseline: condition = rtg + w * ctg')
    parser.add_argument('--sparse_data', type=bool, default=False, help='whether train on the AuctionNet-Sparse data')
    parser.add_argument('--data_path', type=str,default=None, help='path to load the dataset')
    args = parser.parse_args()

    run_dt_baselines(baseline_method=args.baseline_method, reweight_w=args.reweight_w, sparse_data=args.sparse_data, data_path=args.data_path)
