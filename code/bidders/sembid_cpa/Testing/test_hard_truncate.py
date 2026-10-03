#!/usr/bin/env python3
"""
硬性预算截断测试脚本
- 保证成本永远不超过预算
- 注意：预算不超标并不等价于 CPA 不超标；CPA 仍按 cost / conversions 判断
- 按 timestep 逐个截断：超出预算的 PV 直接丢弃
- 卡着预算线花，不多花一分
"""
import sys
import torch
import numpy as np
import math
import logging
import json
import pickle
import time
import argparse
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.join(code_dir, "Algorithms"))
sys.path.insert(0, code_dir)

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 直接复用原始测试脚本的策略类
sys.path.insert(0, current_dir)
from test_exp23_standard import Exp23BiddingStrategy, getScore_nips


def run_test_hard_truncate(model_dir, test_file, budget_ratio=1.0, language_emb_dim=2048, embedding_lookup=None):
    """硬性预算截断测试：保证成本 <= 预算，但 CPA 可能仍超过约束。"""
    logger.info("="*80)
    logger.info("硬性预算截断测试 (budget hard truncate)")
    logger.info(f"模型: {os.path.basename(model_dir)}")
    logger.info(f"测试数据: {os.path.basename(test_file)}")
    logger.info("="*80)

    data_loader = TestDataLoader(file_path=test_file)
    env = OfflineEnv()
    keys = data_loader.keys

    total_score = 0.0
    total_reward = 0.0
    total_cost = 0.0
    total_budget = 0.0
    exceed_count = 0
    budget_exceed_count = 0
    results_per_agent = []

    for idx, key in enumerate(keys):
        num_steps, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)
        adjusted_budget = budget * budget_ratio
        total_budget += adjusted_budget

        # 预计算统计特征
        group = data_loader.test_dict[key].sort_values('timeStepIndex')
        agg = group.groupby('timeStepIndex').agg({'pValue': 'mean', 'leastWinningCost': 'mean'})
        volume = group.groupby('timeStepIndex').size()
        pvalue_mean = np.zeros(num_steps, dtype=np.float32)
        lwc_mean = np.zeros(num_steps, dtype=np.float32)
        vol_arr = np.zeros(num_steps, dtype=np.float32)
        for t, row in agg.iterrows():
            tt = int(t)
            if tt < num_steps:
                pvalue_mean[tt] = float(row['pValue'])
                lwc_mean[tt] = float(row['leastWinningCost'])
        for t, v in volume.items():
            if int(t) < num_steps:
                vol_arr[int(t)] = float(v)
        historical_volume = np.cumsum(vol_arr) - vol_arr
        last3_volume = np.array([float(np.sum(vol_arr[max(0, t-3):t])) for t in range(num_steps)])

        agent = Exp23BiddingStrategy(
            model_dir=model_dir, budget=adjusted_budget, cpa=cpa,
            category=category, language_emb_dim=language_emb_dim, embedding_lookup=embedding_lookup
        )
        agent.precomputed_stats = {
            "pvalue_mean": pvalue_mean, "lwc_mean": lwc_mean,
            "xi_mean": np.zeros(num_steps, dtype=np.float32),
            "volume": vol_arr, "historical_volume": historical_volume, "last3_volume": last3_volume
        }

        history = {'historyBids': [], 'historyAuctionResult': [], 'historyImpressionResult': []}
        rewards = np.zeros(num_steps)
        actual_cost = 0.0  # 跟踪实际花费

        for t in range(num_steps):
            pValue = pValues[t]
            pValueSigma = pValueSigmas[t]
            leastWinningCost = leastWinningCosts[t]
            remaining = adjusted_budget - actual_cost

            # 预算花完就不出价
            if remaining <= 0:
                bid = np.zeros(pValue.shape[0])
            else:
                bid = agent.bidding(
                    t, pValue, pValueSigma,
                    history['historyBids'],
                    history['historyAuctionResult'],
                    history['historyImpressionResult'],
                    [leastWinningCost]
                )

            # 环境模拟
            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                pValue, pValueSigma, bid, leastWinningCost)

            # 硬性截断：卡着预算线，超出部分全部丢弃
            tick_total_cost = np.sum(tick_cost)
            if tick_total_cost > remaining:
                # 找到所有竞价成功的PV，按成本升序排列，贪心保留
                win_idx = np.where(tick_status == 1)[0]
                if len(win_idx) > 0:
                    sorted_idx = win_idx[np.argsort(tick_cost[win_idx])]
                    cum_cost = 0.0
                    for i, si in enumerate(sorted_idx):
                        if cum_cost + tick_cost[si] <= remaining:
                            cum_cost += tick_cost[si]
                        else:
                            # 从这个开始全部丢弃
                            drop_idx = sorted_idx[i:]
                            tick_status[drop_idx] = 0
                            tick_cost[drop_idx] = 0.0
                            tick_conversion[drop_idx] = 0.0
                            tick_value[drop_idx] = 0.0
                            break

            step_cost = float(np.sum(tick_cost))
            actual_cost += step_cost
            rewards[t] = np.sum(tick_conversion)
            agent.last_reward = rewards[t]
            agent.remaining_budget = adjusted_budget - actual_cost
            agent.bid_mean_hist.append(float(np.mean(bid)) if len(bid) > 0 else 0.0)
            agent.conv_mean_hist.append(float(np.mean(tick_conversion)) if len(tick_conversion) > 0 else 0.0)

            history['historyBids'].append(bid)
            history['historyAuctionResult'].append(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(len(tick_status))])
            history['historyImpressionResult'].append(
                [(tick_conversion[i], tick_conversion[i]) for i in range(len(tick_conversion))])

        # 计算指标
        all_reward = np.sum(rewards)
        cpa_real = np.clip(actual_cost / (all_reward + 1e-10), 0, 100)
        score = getScore_nips(all_reward, cpa_real, cpa)
        cpa_exceeded = bool(cpa_real > cpa)
        budget_exceeded = bool(actual_cost > adjusted_budget + 1e-6)

        total_score += score
        total_reward += all_reward
        total_cost += actual_cost
        if cpa_exceeded:
            exceed_count += 1
        if budget_exceeded:
            budget_exceed_count += 1

        budget_usage = actual_cost / adjusted_budget * 100 if adjusted_budget > 0 else 0
        logger.info(f"  [{idx+1}/{len(keys)}] 预算: {adjusted_budget:.2f} | 实际花费: {actual_cost:.2f} | 使用率: {budget_usage:.1f}%")
        logger.info(f"    转化数: {all_reward:.0f} | CPA: {cpa_real:.2f} (限制:{cpa:.0f}) | 分数: {score:.2f} | CPA超标:{cpa_exceeded}")

        results_per_agent.append({
            'key': list(key), 'budget': float(adjusted_budget), 'actual_cost': float(actual_cost),
            'reward': float(all_reward), 'cpa_real': float(cpa_real), 'cpa_limit': float(cpa),
            'score': float(score), 'budget_usage': float(budget_usage),
            'cpa_exceeded': cpa_exceeded, 'budget_exceeded': budget_exceeded
        })

    num_agents = len(keys)
    avg_score = total_score / num_agents
    avg_cpa = total_cost / (total_reward + 1e-10)
    budget_usage_total = total_cost / total_budget * 100 if total_budget > 0 else 0
    exceed_rate = exceed_count / num_agents * 100 if num_agents else 0.0
    budget_exceed_rate = budget_exceed_count / num_agents * 100 if num_agents else 0.0

    logger.info("\n" + "="*80)
    logger.info("硬性截断测试最终结果")
    logger.info("="*80)
    logger.info(f"平均分数: {avg_score:.2f}")
    logger.info(f"总转化数: {total_reward:.0f}")
    logger.info(f"总成本/总预算: {total_cost:.2f}/{total_budget:.2f} (使用率:{budget_usage_total:.1f}%)")
    logger.info(f"平均CPA: {avg_cpa:.2f}")
    logger.info(f"CPA超标率: {exceed_rate:.1f}% ({exceed_count}/{num_agents})")
    logger.info(f"预算超标率: {budget_exceed_rate:.1f}% ({budget_exceed_count}/{num_agents})")
    logger.info("="*80)

    return {
        'avg_score': float(avg_score),
        'total_reward': float(total_reward),
        'total_cost': float(total_cost),
        'total_budget': float(total_budget),
        'budget_usage': float(budget_usage_total),
        'avg_cpa': float(avg_cpa),
        'exceed_rate': float(exceed_rate),
        'exceed_count': exceed_count,
        'budget_exceed_rate': float(budget_exceed_rate),
        'budget_exceed_count': budget_exceed_count,
        'num_advertisers': num_agents,
        'per_agent': results_per_agent
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='硬性预算截断测试')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--budget_ratio', type=float, default=1.0)
    parser.add_argument('--language_emb_dim', type=int, default=2048)
    parser.add_argument('--embedding_lookup', type=str, default=None)
    parser.add_argument('--output', type=str, default=None, help='结果输出路径')
    args = parser.parse_args()

    embedding_lookup = None
    if args.embedding_lookup and os.path.exists(args.embedding_lookup):
        logger.info(f"加载embedding查找表: {args.embedding_lookup}")
        with open(args.embedding_lookup, 'rb') as f:
            embedding_lookup = pickle.load(f)

    start_time = time.time()
    results = run_test_hard_truncate(
        args.model_dir, args.test_file, args.budget_ratio,
        args.language_emb_dim, embedding_lookup
    )
    elapsed = time.time() - start_time
    logger.info(f"测试耗时: {elapsed:.1f}秒")

    # 保存结果
    output_path = args.output or os.path.join(args.model_dir, 'test_results_hard_truncate.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_path}")
