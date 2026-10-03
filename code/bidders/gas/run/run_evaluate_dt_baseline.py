import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv

import torch
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def run_test(policy_load_dir=None, policy_method='dt_reweight', reweigth_w=0.2, file_path=None):
    """
    offline evaluation
    """
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv()

    keys, test_dict = data_loader.keys, data_loader.test_dict
    
    overall_score = 0.0
    excced_rate = 0.0
    overall_reward = 0.0
    cpa_ratio = 0.0

    for key in keys:
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category= data_loader.mock_data(key)

        budget = 1. * budget
        agent = PlayerBiddingStrategy(budget=budget, cpa=cpa, load_dir=policy_load_dir, baseline_method=policy_method, reweight_w=reweigth_w)

        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid, alpha = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                        history["historyBids"],
                                        history["historyAuctionResult"], history["historyImpressionResult"],
                                        history["historyLeastWinningCost"],
                                        actual_excuted_action=None)

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                        leastWinningCost)

            # Handling over-cost (a timestep costs more than the remaining budget of the bidding advertiser)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                            leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
        all_reward = np.sum(rewards)
        all_cost = agent.budget - agent.remaining_budget
        cpa_real = np.clip(all_cost / (all_reward + 1e-10), a_min=0, a_max=100.)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        overall_score += score
        overall_reward += all_reward
        if cpa_real > cpa_constraint:
            excced_rate += 1
        cpa_ratio += cpa_real / cpa_constraint

        logger.info(f'Total Reward: {all_reward}')
        logger.info(f'Total Cost: {all_cost}')
        logger.info(f'CPA-real: {cpa_real}')
        logger.info(f'CPA-constraint: {cpa_constraint}')
        logger.info(f'Score: {score}')
    logger.info(f'===========> overall average score: {overall_score/len(keys)}  excced rare: {excced_rate/len(keys)} total reward: {overall_reward/len(keys)} cpa_ratio:{cpa_ratio/len(keys)}')
    return overall_score/len(keys), excced_rate/len(keys), overall_reward/len(keys)

if __name__ == '__main__':
    baseline_method = 'vanilla_dt'  # vanilla_dt  dt_reweight
    policy_load_dir = ['path/to/dir']
    for policy in policy_load_dir:
        print(f'evaluating method: {baseline_method} with ckpt from {policy}')
        file_path = ['/data/final/traffic/period-7.csv']
        for test_file in file_path:
            print(f'\n testing on {test_file}... ===============>')
            avg_score, avg_exceed_rate, avg_reward = run_test(policy_load_dir=policy, policy_method=baseline_method, reweigth_w=0.2, file_path=test_file)

