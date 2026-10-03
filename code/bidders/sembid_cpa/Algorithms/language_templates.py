"""
Language Templates for Bidding Auction Domain with Task Description
生成拍卖出价相关的语言指导（History + Strategy + Task）
"""
import numpy as np
import random


class BiddingLanguageGeneratorWithTask:
    """生成拍卖出价相关的语言指导（History + Strategy + Task）"""

    def __init__(self, roi_thresholds=(0.5, 1.0, 1.5), cvr_change_thresholds=(0.001, 0.002), cpa_change_thresholds=(5, 10)):
        """
        参数:
            roi_thresholds: ROI阈值 (low, moderate, good)
            cvr_change_thresholds: CVR变化阈值 (small, large)
            cpa_change_thresholds: CPA变化阈值 (small, large)
        """
        self.roi_low, self.roi_moderate, self.roi_good = roi_thresholds
        self.cvr_small, self.cvr_large = cvr_change_thresholds
        self.cpa_small, self.cpa_large = cpa_change_thresholds

        # Task Description 模板
        self.task_templates = [
            "Balance conversion volume and cost efficiency with target CPA {cpa:.1f}.",
            "Maximize conversions while maintaining CPA below {cpa:.1f}.",
            "Optimize bidding to achieve target CPA of {cpa:.1f}.",
            "Control cost per acquisition to stay within {cpa:.1f}.",
        ]

        # History 模板（描述过去的行为和结果）
        self.history_templates = {
            'roi_low': [
                "The ROI was low after the last bid.",
                "Previous bid resulted in low return on investment.",
                "Last action led to poor ROI performance.",
                "The bid was not profitable.",
                "ROI dropped below expectations."
            ],
            'roi_moderate': [
                "The ROI was moderate after the last bid.",
                "Previous bid achieved acceptable ROI.",
                "Last action resulted in moderate returns.",
                "The bid showed moderate profitability.",
                "ROI was within acceptable range."
            ],
            'roi_good': [
                "The ROI was good after the last bid.",
                "Previous bid achieved high return on investment.",
                "Last action resulted in excellent ROI.",
                "The bid was highly profitable.",
                "ROI exceeded expectations."
            ],
            'cvr_increase': [
                "Conversion rate increased after the bid.",
                "CVR improved following the action.",
                "The bid led to higher conversion rate.",
                "Conversion performance improved.",
                "CVR showed positive trend."
            ],
            'cvr_decrease': [
                "Conversion rate decreased after the bid.",
                "CVR dropped following the action.",
                "The bid led to lower conversion rate.",
                "Conversion performance declined.",
                "CVR showed negative trend."
            ],
            'cpa_increase': [
                "Cost per acquisition increased.",
                "CPA rose after the bid.",
                "Acquisition cost went up.",
                "CPA showed upward trend.",
                "Cost efficiency decreased."
            ],
            'cpa_decrease': [
                "Cost per acquisition decreased.",
                "CPA dropped after the bid.",
                "Acquisition cost went down.",
                "CPA showed downward trend.",
                "Cost efficiency improved."
            ]
        }

        # Strategy 模板（提供未来行动的指导）
        self.strategy_templates = {
            'conservative': [
                "Consider bidding conservatively to maintain budget.",
                "Use a conservative bid to preserve resources.",
                "A moderate bid would be safer now.",
                "Keep the bid conservative to avoid overspending.",
                "Maintain conservative bidding strategy."
            ],
            'moderate': [
                "A moderate bid would be appropriate.",
                "Consider a balanced bidding approach.",
                "Use a moderate bid to balance risk and reward.",
                "Moderate bid would help maintain stability.",
                "Try a moderate bidding strategy."
            ],
            'aggressive': [
                "Consider increasing the bid to capture more opportunities.",
                "An aggressive bid might capture better conversions.",
                "Raise the bid to compete more effectively.",
                "Higher bid could improve win rate.",
                "Consider bidding more aggressively."
            ],
            'high_pvalue': [
                "High pValue suggests good conversion potential. Consider higher bid.",
                "The pValue is favorable, you can bid higher.",
                "Strong pValue indicates good opportunity.",
                "High pValue means better conversion probability.",
                "Favorable pValue supports higher bidding."
            ],
            'low_pvalue': [
                "Low pValue suggests lower conversion potential. Consider conservative bid.",
                "The pValue is low, bid conservatively.",
                "Weak pValue indicates limited opportunity.",
                "Low pValue means lower conversion probability.",
                "Unfavorable pValue suggests lower bidding."
            ],
            'budget_low': [
                "Remaining budget is low. Bid conservatively.",
                "Limited budget remaining, use cautious bidding.",
                "Low budget requires careful bid management.",
                "Conserve budget with moderate bids.",
                "Budget constraint suggests conservative approach."
            ],
            'budget_high': [
                "Remaining budget is sufficient. You can bid more aggressively.",
                "Adequate budget allows for higher bids.",
                "Good budget availability supports competitive bidding.",
                "Sufficient budget enables flexible bidding strategy.",
                "Budget is healthy, consider more aggressive bids."
            ]
        }

    def _extract_features(self, state):
        """从状态中提取特征"""
        if isinstance(state, np.ndarray):
            state = state.flatten()
        else:
            state = np.array(state).flatten()

        # 假设state是16维，提取关键特征
        if len(state) >= 16:
            return {
                'budget': state[3] if len(state) > 3 else 0,
                'remaining_budget': state[6] if len(state) > 6 else 0,
                'pvalue': state[8] if len(state) > 8 else 0,
                'cost': state[13] if len(state) > 13 else 0,
                'conversion': state[15] if len(state) > 15 else 0,
            }
        else:
            return {
                'budget': 0,
                'remaining_budget': 0,
                'pvalue': 0,
                'cost': 0,
                'conversion': 0,
            }

    def _calculate_roi(self, prev_state, reward, current_state):
        """计算ROI"""
        prev_features = self._extract_features(prev_state)
        curr_features = self._extract_features(current_state)

        cost = curr_features.get('cost', 0) - prev_features.get('cost', 0)
        conversion = curr_features.get('conversion', 0) - prev_features.get('conversion', 0)

        if cost > 0:
            roi = (conversion * 10.0 - cost) / cost  # 简化计算
        else:
            roi = 0

        return roi

    def generate_task_description(self, target_cpa):
        """生成Task Description"""
        return random.choice(self.task_templates).format(cpa=target_cpa)

    def generate_history(self, prev_state, prev_action, reward, current_state):
        """生成 History 语言（描述过去的行为和结果）"""
        if prev_state is None or current_state is None:
            return ""

        texts = []

        # ROI分析
        roi = self._calculate_roi(prev_state, reward, current_state)
        if roi < self.roi_low:
            texts.append(random.choice(self.history_templates['roi_low']))
        elif roi < self.roi_good:
            texts.append(random.choice(self.history_templates['roi_moderate']))
        else:
            texts.append(random.choice(self.history_templates['roi_good']))

        # CVR变化（简化：基于reward）
        if reward > 0.1:
            texts.append(random.choice(self.history_templates['cvr_increase']))
        elif reward < -0.1:
            texts.append(random.choice(self.history_templates['cvr_decrease']))

        # CPA变化（简化：基于cost）
        prev_features = self._extract_features(prev_state)
        curr_features = self._extract_features(current_state)
        cost_change = curr_features.get('cost', 0) - prev_features.get('cost', 0)

        if cost_change > self.cpa_large:
            texts.append(random.choice(self.history_templates['cpa_increase']))
        elif cost_change < -self.cpa_small and cost_change < 0:
            texts.append(random.choice(self.history_templates['cpa_decrease']))

        return " ".join(texts) if texts else "Previous bid was executed."

    def generate_strategy(self, current_state, suggested_bid):
        """生成 Strategy 语言（提供未来行动的指导）"""
        if current_state is None:
            return "Start bidding conservatively."

        features = self._extract_features(current_state)
        texts = []

        # 基于pValue
        pvalue = features.get('pvalue', 0)
        if pvalue > 0.01:
            texts.append(random.choice(self.strategy_templates['high_pvalue']))
        elif pvalue < 0.001:
            texts.append(random.choice(self.strategy_templates['low_pvalue']))

        # 基于剩余预算
        remaining_budget = features.get('remaining_budget', 0)
        budget = features.get('budget', 1)
        budget_ratio = remaining_budget / budget if budget > 0 else 0

        if budget_ratio < 0.2:
            texts.append(random.choice(self.strategy_templates['budget_low']))
        elif budget_ratio > 0.7:
            texts.append(random.choice(self.strategy_templates['budget_high']))

        # 基于建议出价
        if suggested_bid < 10:
            texts.append(random.choice(self.strategy_templates['conservative']))
        elif suggested_bid > 50:
            texts.append(random.choice(self.strategy_templates['aggressive']))
        else:
            texts.append(random.choice(self.strategy_templates['moderate']))

        return " ".join(texts) if texts else f"Consider bidding around {suggested_bid:.2f}."


    def generate_all(self, prev_state, prev_action, reward, current_state, suggested_bid, target_cpa):
        """生成 Task + History + Strategy 组合语言"""
        task = self.generate_task_description(target_cpa)
        history = self.generate_history(prev_state, prev_action, reward, current_state)
        strategy = self.generate_strategy(current_state, suggested_bid)
        return task, history, strategy
