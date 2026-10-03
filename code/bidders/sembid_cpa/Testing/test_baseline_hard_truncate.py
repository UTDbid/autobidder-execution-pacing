#!/usr/bin/env python3
"""
Baseline models test (DT/GAS/GAVE/CQL/IQL/BCQ) — original + hard_truncate modes.
Reuses existing strategy classes from the project, only adds budget truncation logic.

hard_truncate guarantees cost <= budget, but does not guarantee CPA <= cpa_constraint.
"""
import sys, os, torch, numpy as np, math, logging, json, pickle, time, argparse, random

# ─── Paths: use existing strategy classes ───
STRATEGY_ENV = '/home/wangmeiyi/AuctionNet/AuctionNet/strategy_train_env'
GAS_CODE = '/home/wangmeiyi/AuctionNet/GAS'
DT_TEST_DIR = '/home/wangmeiyi/AuctionNet/DT+Teachable+bid/Experiments/Exp_01_LanguageDT_Baseline_Comparison/Code/Testing'
# Remove any conflicting paths that shadow bidding_train_env
_conflict_dirs = ['Code', 'Code/Algorithms']
sys.path = [p for p in sys.path if not any(p.endswith(c) or ('/sembid_cpa/' + c) in p for c in _conflict_dirs)]
sys.path.insert(0, STRATEGY_ENV)
sys.path.insert(0, GAS_CODE)
sys.path.insert(0, DT_TEST_DIR)
os.chdir(STRATEGY_ENV)

from bidding_train_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
from bidding_train_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from bidding_train_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2; penalty = 1
    if cpa > cpa_constraint:
        penalty = pow(cpa_constraint / (cpa + 1e-10), beta)
    return penalty * reward

def create_baseline_agent(model_type, model_dir, budget, cpa, category, **kwargs):
    """Create agent using existing strategy classes."""
    if model_type == "dt":
        from test_baseline_dt_gas_style import BaselineDTBiddingStrategy
        return BaselineDTBiddingStrategy(model_dir=model_dir, budget=budget, cpa=cpa, category=category)
    elif model_type == "gas":
        from bidding_train_env.strategy import PlayerBiddingStrategy
        os.chdir('/home/wangmeiyi/AuctionNet/GAS')
        return PlayerBiddingStrategy(budget=budget, cpa=cpa, category=category,
            load_dir=model_dir, baseline_method='dt_reweight', reweight_w=0.2)
    elif model_type == "gave":
        from bidding_train_env.strategy import PlayerBiddingStrategy
        os.chdir('/home/wangmeiyi/AuctionNet/GAS')
        mp = {"device":"cpu","save_dir":model_dir,"budget_rate":1.0,"hidden_size":512,
              "learning_rate":1e-4,"time_dim":8,
              "block_config":{"n_ctx":1024,"n_embd":512,"n_layer":8,"n_head":4,
                              "attn_pdrop":0.1,"resid_pdrop":0.1,"n_inner":1024},
              "expectile":0.5}
        return PlayerBiddingStrategy(budget=budget, cpa=cpa, category=category,
            load_dir=model_dir, baseline_method='vanilla_dt', **mp)
    elif model_type == "cql":
        os.environ["CQL_MODEL_DIR"] = model_dir
        return CqlBiddingStrategy(budget=budget, cpa=cpa, category=category)
    elif model_type == "iql":
        return IqlBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                  model_base_dir=model_dir, checkpoint=kwargs.get("checkpoint"))
    elif model_type == "bcq":
        return BcqBiddingStrategy(budget=budget, cpa=cpa, category=category,
                                  model_base_dir=model_dir, checkpoint=kwargs.get("checkpoint"))

def run_test(model_type, model_dir, test_file, mode="original",
             budget_ratio=1.0, **kwargs):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data_loader = TestDataLoader(file_path=test_file)
    env = OfflineEnv()
    keys = data_loader.keys
    total_score = total_reward = total_cost = total_budget = 0.0
    exceed_count = 0
    budget_exceed_count = 0
    results_per_agent = []

    for idx, key in enumerate(keys):
        num_steps, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)
        adj_budget = budget * budget_ratio
        total_budget += adj_budget
        agent = create_baseline_agent(model_type, model_dir, adj_budget, cpa, category, **kwargs)
        history = {"historyBids":[],"historyAuctionResult":[],"historyImpressionResult":[],
                   "historyLeastWinningCost":[],"historyPValueInfo":[]}
        rewards = np.zeros(num_steps)
        actual_cost = 0.0

        for t in range(num_steps):
            pValue, pValueSigma, lwc = pValues[t], pValueSigmas[t], leastWinningCosts[t]
            if mode == "hard_truncate":
                remaining = adj_budget - actual_cost
                if remaining <= 0:
                    bid = np.zeros(pValue.shape[0])
                else:
                    bid = agent.bidding(t, pValue, pValueSigma, history["historyPValueInfo"],
                        history["historyBids"], history["historyAuctionResult"],
                        history["historyImpressionResult"], history["historyLeastWinningCost"])
                tv, tc, ts, tconv = env.simulate_ad_bidding(pValue, pValueSigma, bid, lwc)
                tick_total = np.sum(tc)
                if tick_total > remaining:
                    win = np.where(ts == 1)[0]
                    if len(win) > 0:
                        si = win[np.argsort(tc[win])]
                        cum = 0.0
                        for i, s in enumerate(si):
                            if cum + tc[s] <= remaining: cum += tc[s]
                            else:
                                d = si[i:]
                                ts[d]=0; tc[d]=0; tconv[d]=0; tv[d]=0; break
                step_cost = float(np.sum(tc))
                actual_cost += step_cost
                rewards[t] = np.sum(tconv)
                agent.remaining_budget = adj_budget - actual_cost
            else:  # original GAS-style
                if agent.remaining_budget < env.min_remaining_budget:
                    bid = np.zeros(pValue.shape[0])
                else:
                    bid = agent.bidding(t, pValue, pValueSigma, history["historyPValueInfo"],
                        history["historyBids"], history["historyAuctionResult"],
                        history["historyImpressionResult"], history["historyLeastWinningCost"])
                tv, tc, ts, tconv = env.simulate_ad_bidding(pValue, pValueSigma, bid, lwc)
                over = max((np.sum(tc) - agent.remaining_budget) / (np.sum(tc) + 1e-4), 0)
                while over > 0:
                    pv_idx = np.where(ts == 1)[0]
                    if len(pv_idx) == 0: break
                    dropped = np.random.choice(pv_idx, int(math.ceil(len(pv_idx) * over)), replace=False)
                    bid[dropped] = 0
                    tv, tc, ts, tconv = env.simulate_ad_bidding(pValue, pValueSigma, bid, lwc)
                    over = max((np.sum(tc) - agent.remaining_budget) / (np.sum(tc) + 1e-4), 0)
                agent.remaining_budget -= np.sum(tc)
                rewards[t] = np.sum(tconv)

            history["historyPValueInfo"].append(np.array([(pValue[i],pValueSigma[i]) for i in range(len(pValue))]))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(lwc)
            history["historyAuctionResult"].append(np.array([(ts[i],ts[i],tc[i]) for i in range(len(ts))]))
            history["historyImpressionResult"].append(np.array([(tconv[i],tconv[i]) for i in range(len(tconv))]))

        all_cost = actual_cost if mode == "hard_truncate" else adj_budget - agent.remaining_budget
        all_reward = np.sum(rewards)
        cpa_real = np.clip(all_cost / (all_reward + 1e-10), 0, 100)
        score = getScore_nips(all_reward, cpa_real, cpa)
        total_score += score; total_reward += all_reward; total_cost += all_cost
        cpa_exceeded = bool(cpa_real > cpa)
        budget_exceeded = bool(all_cost > adj_budget + 1e-6)
        if cpa_exceeded: exceed_count += 1
        if budget_exceeded: budget_exceed_count += 1
        bu = all_cost / adj_budget * 100 if adj_budget > 0 else 0
        logger.info(f"  [{idx+1}/{len(keys)}] cost:{all_cost:.1f}/{adj_budget:.1f} conv:{all_reward:.0f} cpa:{cpa_real:.2f}/{cpa:.0f} score:{score:.2f} cpa_exceeded:{cpa_exceeded}")
        results_per_agent.append({"key":list(key),"budget":float(adj_budget),"actual_cost":float(all_cost),
            "reward":float(all_reward),"cpa_real":float(cpa_real),"cpa_limit":float(cpa),"score":float(score),"budget_usage":float(bu),
            "cpa_exceeded":cpa_exceeded,"budget_exceeded":budget_exceeded})

    n = len(keys)
    return {"avg_score":float(total_score/n if n else 0), "total_reward":float(total_reward), "total_cost":float(total_cost),
        "total_budget":float(total_budget), "budget_usage":float(total_cost/total_budget*100 if total_budget else 0),
        "avg_cpa":float(total_cost/(total_reward+1e-10)),
        "exceed_rate": float(exceed_count/n*100 if n else 0),
        "exceed_count":exceed_count, "budget_exceed_rate":float(budget_exceed_count/n*100 if n else 0),
        "budget_exceed_count":budget_exceed_count, "num_advertisers":n, "per_agent":results_per_agent,
        "model_type":model_type, "test_mode":mode}

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_type", required=True, choices=["dt","gas","gave","cql","iql","bcq"])
    pa.add_argument("--model_dir", required=True)
    pa.add_argument("--test_file", required=True)
    pa.add_argument("--mode", default="original", choices=["original","hard_truncate"])
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--output", default=None)
    a = pa.parse_args()
    kw = {}
    if a.checkpoint: kw["checkpoint"] = a.checkpoint
    t0 = time.time()
    res = run_test(a.model_type, a.model_dir, a.test_file, a.mode, **kw)
    res["elapsed"] = time.time() - t0
    out = a.output or os.path.join(a.model_dir, f"test_{a.mode}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f: json.dump(res, f, indent=2, ensure_ascii=False)
    logger.info(f"Done {a.model_type}/{a.mode}: score={res['avg_score']:.2f} exceed={res['exceed_rate']:.1f}% -> {out}")
