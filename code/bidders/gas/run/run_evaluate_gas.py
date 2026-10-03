import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy, PlayerBiddingCritic
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import argparse

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Configure logging
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

def run_test(policy_load_dir=None, critic_load_dir=None, policy_method='dt_reweight', critic_method='dt_reweight_search_Q', reweight_w=0.2, multi_act_strategy='multi_rtg', file_path=None, action_sample_num=5, budget_ratio=1.0):
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
        # budget_ratio
        budget = budget_ratio * budget 
        
        # ------------- Load policy agents and critics ------------- #
        agent_ensemble, critic_ensemble = [], []
        if multi_act_strategy == 'multi_scale':  # get multiple action by multiplying random values from Uniform(0.9, 1.1)
            assert len(policy_load_dir) == 1
            agent = PlayerBiddingStrategy(budget=budget, cpa=cpa, load_dir=policy_load_dir[0], baseline_method=policy_method, reweight_w=reweight_w)  
            agent_ensemble.append(agent)

        # load critics
        for load_dir in critic_load_dir:
            critic_ensemble.append(PlayerBiddingCritic(budget=budget, cpa=cpa, load_dir=load_dir, baseline_method=critic_method, reweight_w=reweight_w))
        # ---------------------------------------------------------- #

        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }
        actual_excuted_action = None
        for timeStep_index in range(num_timeStepIndex):
            # logger.info(f'Timestep Index: {timeStep_index + 1} Begin')

            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                # get an ensemble of actions here
                bids, alpha = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"],
                                    actual_excuted_action)
                random_numbers = np.random.uniform(0.9, 1.1, action_sample_num).astype(np.float32)
                random_numbers[-1] = 1.
                action_ensemble = random_numbers * alpha

                action_proposals = action_ensemble.reshape(-1, 1)
            
                action_values = []
                for action in action_proposals:
                    voting_value = []
                    for critic in critic_ensemble:
                        value_1, value_2  = critic.access_value(action, timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                            history["historyBids"],
                                            history["historyAuctionResult"], history["historyImpressionResult"],
                                            history["historyLeastWinningCost"])

                        q1 = value_1[0,-1,0].cpu().item()
                        q2 = value_2[0,-1,0].cpu().item()
                        q_value = min(q1, q2)
                        voting_value.append(q_value)
                        
                    action_values.append(voting_value)

                action_values = np.array(action_values).T
                for critic_i in range(len(critic_ensemble)):
                    critic_i_ranking = action_values[critic_i]
                    action_values[critic_i] = (critic_i_ranking - critic_i_ranking.min()) / (critic_i_ranking.max() - critic_i_ranking.min())
                action_values = action_values.T
                action_values = action_values.mean(-1)
                # pick out the largest-value action
                max_value = max(action_values)
                action_values = action_values.tolist()
                max_index = action_values.index(max_value)
                actual_excuted_action = action_proposals[max_index]

                bid = actual_excuted_action * pValue

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
        if all_cost ==0:
            cpa_real = 0
        else:
            cpa_real = all_cost / (all_reward)
        cpa_constraint = agent.cpa
        score = getScore_nips(all_reward, cpa_real, cpa_constraint)
        overall_score += score
        overall_reward += all_reward
        if cpa_real > cpa_constraint:
            excced_rate += 1
        cpa_ratio += cpa_real/cpa_constraint
        logger.info(f'Total Reward: {all_reward}')
        logger.info(f'Total Cost: {all_cost}')
        logger.info(f'CPA-real: {cpa_real}')
        logger.info(f'CPA-constraint: {cpa_constraint}')
        logger.info(f'Score: {score}')
    logger.info(f'===========> overall average score: {overall_score/len(keys)}  exceed rate: {excced_rate/len(keys)} total reward: {overall_reward/len(keys)} cpa_ratio:{cpa_ratio/len(keys)}')
    return overall_score/len(keys), excced_rate/len(keys), overall_reward/len(keys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='running evaluations...')
    parser.add_argument('--policy_method', type=str, default='dt_reweight', help='base actor method')
    parser.add_argument('--Q_num', type=int, default=3, help='the first N Q will be used to do voting')
    parser.add_argument('--action_num', type=int, default=5, help='how many actions to sample')
    parser.add_argument('--baseline_method', type=str, default='dt_reweight_search_Q', choices=['dt_reweight_search_Q'], help='choose a method to run')
    parser.add_argument('--reweight_w', type=float, default=0.2, help='for dt_reweight baseline: condition = rtg + w * ctg')
    parser.add_argument('--budget_ratio', type=float, default=1., choices=[0.5, 0.75, 1.0, 1.25, 1.5])
    args = parser.parse_args()

    policy_load_dir = ['path/to/dir']
    print(f'evaluating policy with ckpt at {policy_load_dir}')

    critic_load_dir = ['path/to/critic_0/dir',
                        'path/to/critic_1/dir',
                        'path/to/critic_2/dir']

    critic_load_dir = critic_load_dir[:args.Q_num]
    critic_method = 'dt_reweight_search_Q_voting'
    multi_act_strategy = 'multi_scale'

    file_path = ['./data/final/traffic/period-7.csv']
    
    for test_file in file_path:
        print(f'\n testing on {test_file}... ===============>')
        avg_score, avg_exceed_rate, avg_reward = run_test(policy_load_dir, critic_load_dir, args.policy_method, critic_method, reweight_w=0.2, multi_act_strategy=multi_act_strategy, file_path=test_file, action_sample_num=args.action_num, budget_ratio=args.budget_ratio)
