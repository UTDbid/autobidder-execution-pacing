import numpy as np
import os
from bidding_train_env.baseline.dt_baselines.dt_baselines import DecisionTransformer
from bidding_train_env.baseline.dt_baselines.dt_critics import DT_Critic
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle


class DtBiddingStrategy(BaseBiddingStrategy):
    """
        Decision-Transformer-PlayerStrategy
    """
    def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1, load_dir=None, baseline_method = 'dt_reweight', reweight_w=0.2):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        if load_dir is None:
            model_path = os.path.join(dir_name, "saved_model", "CDTtest", "dt.pt")
            picklePath = os.path.join(dir_name, "saved_model", "CDTtest", "normalize_dict.pkl")
        else:
            model_path = os.path.join(load_dir, "dt.pt")
            picklePath = os.path.join(load_dir, "normalize_dict.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        
        if 'dt_reweight' in baseline_method:
            self.model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                            state_std=normalize_dict["state_std"],
                                            target_return=1. + reweight_w, target_ctg=1.,
                                            baseline_method = baseline_method)
        else:
            self.model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                    state_std=normalize_dict["state_std"],
                                    target_return=1., target_ctg=1.,
                                    baseline_method = baseline_method)                                
        self.model.load_net(model_path)
        self.model.to(device)
        self.test_state_old = np.zeros(16)
        self.cpa = cpa
        self.budget =  budget
        self.category = category
        self.remaining_budget_last =self.budget
        self.reweight_w = self.model.target_return

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                actual_excuted_action=None):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """

        # if timeStepIndex == 0:
        #     if (self.category, self.cpa) in self.day_counter:
        #         self.day_counter[(self.category, self.cpa)] += 1
        #     else:
        #         self.day_counter[(self.category, self.cpa)] = 0
        # day = self.day_counter[(self.category, self.cpa)]
        

        self.cost_cur = self.remaining_budget_last -  self.remaining_budget

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()
            self.test_state_old = np.zeros(16)

        self.test_state = test_state
        if len(history_conversion) != 0:
            self.cost_constraint = self.cost_cur
        else:
            self.cost_constraint = None

        alpha = self.model.take_actions(self.test_state, actual_excuted_action,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None,
                                        pre_cost=self.cost_constraint if len(history_conversion) != 0 else None,
                                        cpa_constrain=self.cpa
                                        )
        self.remaining_budget_last = self.remaining_budget

        bids = alpha * pValues
        return bids, alpha



class DtBiddingCritic(BaseBiddingStrategy):
    """
    Decision-Transformer Critic
    """
    def __init__(self, budget=100, name="Decision-Transformer-Critic", cpa=2, category=1, load_dir=None, baseline_method = 'dt_reweight_search_greedy', reweight_w=0.2):
        super().__init__(budget, name, cpa, category)

        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        if load_dir is None:
            model_path = os.path.join(dir_name, "saved_model", "CDTtest", "dt.pt")
            picklePath = os.path.join(dir_name, "saved_model", "CDTtest", "normalize_dict.pkl")
        else:
            model_path = os.path.join(load_dir, "dt.pt")
            picklePath = os.path.join(load_dir, "normalize_dict.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)
        
        if 'dt_reweight' in baseline_method:
            self.critic = DT_Critic(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                            state_std=normalize_dict["state_std"],
                                            target_return=1. + reweight_w, target_ctg=1.,
                                            baseline_method = baseline_method,
                                            use_rtg=False)
                
        self.critic.load_net(model_path)
        self.critic.to(device)
        self.test_state_old = np.zeros(16)
        self.cpa = cpa
        self.budget =  budget
        self.category = category
        self.remaining_budget_last =self.budget

        self.last_timestep = 0

    def reset(self):
        self.remaining_budget = self.budget

    def access_value(self, action, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        self.cost_cur = self.remaining_budget_last -  self.remaining_budget

        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.critic.init_eval()
            self.test_state_old = np.zeros(16)
            self.last_timestep = 0
            whether_concat_new = True

        if self.last_timestep == timeStepIndex and timeStepIndex!=0:
            whether_concat_new = False
        if self.last_timestep < timeStepIndex:
            self.last_timestep = timeStepIndex
            whether_concat_new = True

        self.test_state = test_state
        if len(history_conversion) != 0:
            self.cost_constraint = self.cost_cur
        else:
            self.cost_constraint = None

        assece_values  = self.critic.take_critics(self.test_state, action, whether_concat_new,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None,
                                        pre_cost=self.cost_constraint if len(history_conversion) != 0 else None,
                                        cpa_constrain=self.cpa
                                        )
        self.remaining_budget_last = self.remaining_budget

        return assece_values

