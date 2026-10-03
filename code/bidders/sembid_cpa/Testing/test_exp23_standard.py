#!/usr/bin/env python3
"""
Exp23 标准测试脚本 - 使用GAS/GAVE标准预算约束处理
Qwen-0.5B + 2048维 LanguageDT测试 (H/F before State)
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

# 添加本地路径（Exp23内置依赖）
current_dir = os.path.dirname(os.path.abspath(__file__))
exp23_code_dir = os.path.abspath(os.path.join(current_dir, ".."))
exp23_algo_dir = os.path.join(exp23_code_dir, "Algorithms")
sys.path.insert(0, exp23_algo_dir)
sys.path.insert(0, exp23_code_dir)

# 导入环境和数据加载器
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

# 导入模型和编码器
from language_dt_with_task_flexible_hf_first import LanguageGuidedDTWithTaskFlexible
from language_templates import BiddingLanguageGeneratorWithTask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class QwenEncoder2048:
    """Qwen-0.5B编码器 - 输出2048维"""

    def __init__(self, output_dim=2048, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", max_length=256):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device
        self.output_dim = output_dim
        self.max_length = max_length

        logger.info(f"初始化Qwen-0.5B编码器 ({output_dim}维)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": device}
        )
        self.model.eval()

        # Qwen-0.5B原生维度896，投影到2048
        self.projection = torch.nn.Linear(896, output_dim).to(device).half()
        torch.nn.init.xavier_uniform_(self.projection.weight)
        logger.info(f"Qwen编码器加载完成 (896 → {output_dim})")

    def encode(self, text, convert_to_tensor=True, device=None):
        """编码单个文本"""
        if device is None:
            device = self.device

        if not text or text.strip() == "":
            if convert_to_tensor:
                return torch.zeros(self.output_dim, dtype=torch.float32, device=device)
            return np.zeros(self.output_dim, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0)  # [896]
            embedding = self.projection(embedding.half())  # [2048]

        if convert_to_tensor:
            return embedding.float().to(device)
        return embedding.float().cpu().numpy()


class Exp23BiddingStrategy:
    """Exp23: Qwen-0.5B + 2048维策略 - 标准版本"""

    def __init__(self, model_dir, budget, cpa, category, device='cuda',
                 shared_encoder=None, language_emb_dim=2048, embedding_lookup=None):
        self.budget = budget
        self.remaining_budget = budget
        self.cpa = cpa
        self.category = category
        self.device = device
        self.model_dir = model_dir
        self.language_emb_dim = language_emb_dim
        self.embedding_lookup = embedding_lookup
        self.use_precomputed = embedding_lookup is not None
        # 推理期可调参数（通过环境变量控制）
        self.action_clip = float(os.getenv("EXP17B_ACTION_CLIP", "300"))
        self.action_scale = float(os.getenv("EXP17B_ACTION_SCALE", "1.0"))
        self.rtg_mode = os.getenv("EXP17B_RTG_MODE", "fixed")  # fixed / budget_cpa
        self.rtg_value = float(os.getenv("EXP17B_RTG_VALUE", "8"))
        # 评测期动态统计（用于构造与训练一致的16维state）
        self.bid_mean_hist = []
        self.conv_mean_hist = []
        self.precomputed_stats = None

        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'use_language': True,
                'language_type': 'both',
                'state_dim': 16,
                'act_dim': 1,
                'max_length': 20,
                'max_ep_len': 96,
                'language_emb_dim': language_emb_dim
            }

        # 加载归一化数据
        norm_path = os.path.join(model_dir, 'normalize_dict.pkl')
        if os.path.isdir(norm_path):
            norm_path = os.path.join(norm_path, 'normalize_dict.pkl')
        if os.path.exists(norm_path) and os.path.isfile(norm_path):
            with open(norm_path, 'rb') as f:
                normalize_dict = pickle.load(f)
            self.state_mean = normalize_dict['state_mean']
            self.state_std = normalize_dict['state_std']
        else:
            logger.warning(f"Normalize dict not found, using zeros")
            self.state_mean = np.zeros(self.config['state_dim'])
            self.state_std = np.ones(self.config['state_dim'])

        # 初始化模型
        self.model = LanguageGuidedDTWithTaskFlexible(
            state_dim=self.config['state_dim'],
            act_dim=self.config['act_dim'],
            state_mean=self.state_mean,
            state_std=self.state_std,
            use_language=self.config['use_language'],
            K=self.config['max_length'],
            max_ep_len=self.config['max_ep_len'],
            language_emb_dim=language_emb_dim,
            use_precomputed_embeddings=True
        )

        # 加载模型权重
        model_path = os.path.join(model_dir, 'language_dt.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'language_dt_2048.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'pytorch_model.bin')

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        # Qwen编码器（仅在不使用查找表时需要）
        if self.use_precomputed:
            self.encoder = None
            logger.info("使用预计算的embedding查找表 (2048维)")
        elif shared_encoder:
            self.encoder = shared_encoder
            logger.info("使用共享的Qwen-0.5B编码器 (2048维)")
        else:
            self.encoder = QwenEncoder2048(output_dim=language_emb_dim, device=device)

        # 语言生成器
        self.language_generator = BiddingLanguageGeneratorWithTask()

        # 历史记录缓冲区
        self.max_length = self.config['max_length']
        self.reset_buffer()
        self.emb_cache = {}

    def reset_buffer(self):
        """重置历史缓冲区"""
        self.states = []
        self.actions = []
        self.timesteps = []
        self.lang_task_desc = []  # 原lang_task
        self.lang_history = []    # 原lang_h
        self.lang_strategy = []   # 原lang_f
        self.last_state = None
        self.last_action = None
        self.last_reward = 0

    def _encode_text_cached(self, text, text_type='task'):
        """带缓存的文本编码"""
        if not text or text.strip() == "":
            return torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)

        if text in self.emb_cache:
            return self.emb_cache[text]

        if len(self.emb_cache) > 1000:
            self.emb_cache.clear()

        # 使用查找表或实时编码
        if self.use_precomputed:
            lookup = self.embedding_lookup.get(text_type, {})
            if text in lookup:
                emb = torch.from_numpy(lookup[text]).float().to(self.device)
            else:
                emb = torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)
        else:
            emb = self.encoder.encode(text, convert_to_tensor=True, device=self.device)

        self.emb_cache[text] = emb
        return emb

    def _get_language_embeddings(self, current_state, suggested_bid=1.0):
        """生成语言嵌入"""
        # Task Description
        task_text = self.language_generator.generate_task_description(self.cpa)
        lang_task_desc_emb = self._encode_text_cached(task_text, text_type='task')

        # History (原Hindsight)
        lang_history_emb = torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)
        if self.last_state is not None:
            history_text = self.language_generator.generate_history(
                prev_state=self.last_state,
                prev_action=self.last_action,
                reward=self.last_reward,
                current_state=current_state
            )
            if history_text:
                lang_history_emb = self._encode_text_cached(history_text, text_type='history')

        # Strategy (原Foresight)
        strategy_text = self.language_generator.generate_strategy(current_state, suggested_bid=suggested_bid)
        lang_strategy_emb = self._encode_text_cached(strategy_text, text_type='strategy') if strategy_text else torch.zeros(self.language_emb_dim, dtype=torch.float32, device=self.device)

        return lang_task_desc_emb, lang_history_emb, lang_strategy_emb

    def bidding(self, timeStep_index, pValue, pValueSigma, historyBids,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """出价决策"""
        # 构建状态（优先使用与训练一致的16维特征）
        if self.precomputed_stats is not None and self.config.get('state_dim', 16) == 16:
            stats = self.precomputed_stats
            t = int(timeStep_index)
            timeStepIndexNum = 48
            timeleft = (timeStepIndexNum - t) / timeStepIndexNum
            bgtleft = self.remaining_budget / self.budget if self.budget > 0 else 0.0

            avg_bid_all = float(np.mean(self.bid_mean_hist)) if self.bid_mean_hist else 0.0
            avg_bid_last_3 = float(np.mean(self.bid_mean_hist[-3:])) if self.bid_mean_hist else 0.0

            pvalue_mean = stats["pvalue_mean"]
            lwc_mean = stats["lwc_mean"]
            xi_mean = stats["xi_mean"]
            volume = stats["volume"]
            hist_volume = stats["historical_volume"]
            last3_volume = stats["last3_volume"]

            def _mean_slice(arr, end, k=3):
                if end <= 0:
                    return 0.0
                start = max(0, end - k)
                return float(np.mean(arr[start:end]))

            avg_lwc_all = float(np.mean(lwc_mean[:t])) if t > 0 else 0.0
            avg_lwc_last_3 = _mean_slice(lwc_mean, t, 3)
            avg_pvalue_all = float(np.mean(pvalue_mean[:t])) if t > 0 else 0.0
            avg_pvalue_last_3 = _mean_slice(pvalue_mean, t, 3)
            avg_conv_all = float(np.mean(self.conv_mean_hist)) if self.conv_mean_hist else 0.0
            avg_conv_last_3 = float(np.mean(self.conv_mean_hist[-3:])) if self.conv_mean_hist else 0.0
            avg_xi_all = float(np.mean(xi_mean[:t])) if t > 0 else 0.0
            avg_xi_last_3 = _mean_slice(xi_mean, t, 3)

            pvalue_agg = float(pvalue_mean[t]) if t < len(pvalue_mean) else 0.0
            timeStepIndex_volume_agg = float(volume[t]) if t < len(volume) else 0.0
            last_3_timeStepIndexs_volume = float(last3_volume[t]) if t < len(last3_volume) else 0.0
            historical_volume = float(hist_volume[t]) if t < len(hist_volume) else 0.0

            current_state = np.array([
                timeleft, bgtleft,
                avg_bid_all, avg_bid_last_3,
                avg_lwc_all, avg_pvalue_all, avg_conv_all, avg_xi_all,
                avg_lwc_last_3, avg_pvalue_last_3, avg_conv_last_3, avg_xi_last_3,
                pvalue_agg, timeStepIndex_volume_agg, last_3_timeStepIndexs_volume, historical_volume
            ], dtype=np.float32)
        else:
            # 回退：简化8维特征
            budget_usage = 1.0 - (self.remaining_budget / self.budget)

            if len(historyBids) > 0:
                avg_bid = np.mean(historyBids[-1])
                win_rate = np.mean([r[0] for r in historyAuctionResult[-1]])
                cvr = np.mean([r[0] for r in historyImpressionResult[-1]])
            else:
                avg_bid = 0.0
                win_rate = 0.0
                cvr = 0.0

            current_state = np.zeros(self.config['state_dim'])
            current_state[0] = budget_usage
            current_state[1] = self.remaining_budget
            current_state[2] = self.budget
            current_state[3] = timeStep_index
            current_state[4] = self.cpa
            current_state[5] = avg_bid
            current_state[6] = win_rate
            current_state[7] = cvr

        # 归一化
        state_norm = (current_state - self.state_mean) / (self.state_std + 1e-8)
        state_tensor = torch.from_numpy(state_norm).float().to(self.device)

        # 生成语言嵌入
        if len(historyBids) > 0:
            suggested_bid = float(np.mean(historyBids[-1]))
        else:
            suggested_bid = 10.0
        lang_task_desc_emb, lang_history_emb, lang_strategy_emb = self._get_language_embeddings(current_state, suggested_bid)

        # 更新历史缓冲区
        self.states.append(state_tensor)
        self.timesteps.append(timeStep_index)
        self.lang_task_desc.append(lang_task_desc_emb)
        self.lang_history.append(lang_history_emb)
        self.lang_strategy.append(lang_strategy_emb)

        # 保持最大长度
        if len(self.states) > self.max_length:
            self.states = self.states[-self.max_length:]
            self.timesteps = self.timesteps[-self.max_length:]
            self.lang_task_desc = self.lang_task_desc[-self.max_length:]
            self.lang_history = self.lang_history[-self.max_length:]
            self.lang_strategy = self.lang_strategy[-self.max_length:]
            if len(self.actions) > self.max_length - 1:
                self.actions = self.actions[-(self.max_length-1):]

        # 准备模型输入
        seq_len = len(self.states)
        states_seq = torch.stack(self.states).unsqueeze(0)
        timesteps_seq = torch.tensor([self.timesteps], dtype=torch.long, device=self.device)
        rtg_seq = torch.ones((1, seq_len, 1), device=self.device) * self.rtg_value

        if len(self.actions) > 0:
            actions_seq = torch.stack(self.actions).unsqueeze(0)
            if actions_seq.shape[1] < seq_len:
                pad_len = seq_len - actions_seq.shape[1]
                actions_seq = torch.cat([
                    actions_seq,
                    torch.zeros((1, pad_len, self.config['act_dim']), device=self.device)
                ], dim=1)
        else:
            actions_seq = torch.zeros((1, seq_len, self.config['act_dim']), device=self.device)

        rewards_seq = torch.zeros((1, seq_len, 1), device=self.device)
        lang_task_desc_seq = torch.stack(self.lang_task_desc).unsqueeze(0)
        lang_history_seq = torch.stack(self.lang_history).unsqueeze(0)
        lang_strategy_seq = torch.stack(self.lang_strategy).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            action_preds = self.model.forward(
                states=states_seq,
                actions=actions_seq,
                rewards=rewards_seq,
                returns_to_go=rtg_seq,
                timesteps=timesteps_seq,
                language_task=lang_task_desc_seq,
                language_history=lang_history_seq,
                language_strategy=lang_strategy_seq
            )
            action = action_preds[0, -1, 0].item()

        # 后处理
        action = action * self.action_scale
        action = np.clip(action, 0, self.action_clip)
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.float32)
        self.actions.append(action_tensor)

        self.last_state = current_state
        self.last_action = action

        # 计算出价
        bid = action * pValue
        return bid


def getScore_nips(reward, cpa, cpa_constraint):
    """NIPS评分函数"""
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def run_test(model_dir, test_file, budget_ratio=1.0, language_emb_dim=2048, embedding_lookup=None):
    """运行Exp23测试 - 使用GAS/GAVE标准方法"""
    logger.info("="*80)
    logger.info("Exp23: Qwen-0.5B + 2048维 测试 (标准版, H/F before State)")
    logger.info(f"模型: {os.path.basename(model_dir)}")
    logger.info(f"测试数据: {os.path.basename(test_file)}")
    logger.info("="*80)

    # 加载测试数据
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

    # 共享Qwen编码器（仅在不使用查找表时需要）
    shared_encoder = None
    if embedding_lookup is None:
        logger.info("初始化共享Qwen-0.5B编码器 (2048维)...")
        shared_encoder = QwenEncoder2048(output_dim=language_emb_dim)
        logger.info("共享编码器初始化完成")
    else:
        logger.info("使用预计算的embedding查找表")

    for idx, key in enumerate(keys):
        logger.info(f"\n[{idx+1}/{len(keys)}] 测试Agent: {key}")

        # 获取测试数据
        num_steps, pValues, pValueSigmas, leastWinningCosts, budget, cpa, category = data_loader.mock_data(key)
        adjusted_budget = budget * budget_ratio
        total_budget += adjusted_budget

        # 预计算与训练一致的统计特征（基于测试数据本身）
        group = data_loader.test_dict[key].sort_values('timeStepIndex')
        agg_cols = {'pValue': 'mean', 'leastWinningCost': 'mean'}
        if 'xi' in group.columns:
            agg_cols['xi'] = 'mean'
        agg = group.groupby('timeStepIndex').agg(agg_cols)
        volume = group.groupby('timeStepIndex').size()

        pvalue_mean = np.zeros(num_steps, dtype=np.float32)
        lwc_mean = np.zeros(num_steps, dtype=np.float32)
        xi_mean = np.zeros(num_steps, dtype=np.float32)
        vol_arr = np.zeros(num_steps, dtype=np.float32)

        for t, row in agg.iterrows():
            tt = int(t)
            if tt >= num_steps:
                continue
            pvalue_mean[tt] = float(row['pValue'])
            lwc_mean[tt] = float(row['leastWinningCost'])
            if 'xi' in row:
                xi_mean[tt] = float(row['xi'])
        for t, v in volume.items():
            tt = int(t)
            if tt < num_steps:
                vol_arr[tt] = float(v)

        historical_volume = np.zeros(num_steps, dtype=np.float32)
        last3_volume = np.zeros(num_steps, dtype=np.float32)
        cum = 0.0
        for t in range(num_steps):
            historical_volume[t] = cum
            last3_volume[t] = float(np.sum(vol_arr[max(0, t-3):t]))
            cum += vol_arr[t]

        # 初始化策略
        agent = Exp23BiddingStrategy(
            model_dir=model_dir,
            budget=adjusted_budget,
            cpa=cpa,
            category=category,
            shared_encoder=shared_encoder,
            language_emb_dim=language_emb_dim,
            embedding_lookup=embedding_lookup
        )
        if getattr(agent, "rtg_mode", "fixed") == "budget_cpa":
            agent.rtg_value = adjusted_budget / (cpa + 1e-10)
        agent.precomputed_stats = {
            "pvalue_mean": pvalue_mean,
            "lwc_mean": lwc_mean,
            "xi_mean": xi_mean,
            "volume": vol_arr,
            "historical_volume": historical_volume,
            "last3_volume": last3_volume
        }

        # 历史记录
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': []
        }

        rewards = np.zeros(num_steps)

        for t in range(num_steps):
            pValue = pValues[t]
            pValueSigma = pValueSigmas[t]
            leastWinningCost = leastWinningCosts[t]

            # Agent出价决策
            if agent.remaining_budget < env.min_remaining_budget:
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

            # 预算约束处理 - GAS/GAVE标准方法
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                if len(pv_index) == 0:
                    break
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
                    pValue, pValueSigma, bid, leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            # 更新预算和统计
            agent.remaining_budget -= np.sum(tick_cost)
            rewards[t] = np.sum(tick_conversion)
            agent.last_reward = rewards[t]
            # 更新历史统计（用于构造state）
            agent.bid_mean_hist.append(float(np.mean(bid)) if len(bid) > 0 else 0.0)
            agent.conv_mean_hist.append(float(np.mean(tick_conversion)) if len(tick_conversion) > 0 else 0.0)

            # 记录历史
            history['historyBids'].append(bid)
            history['historyAuctionResult'].append(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(len(tick_status))])
            history['historyImpressionResult'].append(
                [(tick_conversion[i], tick_conversion[i]) for i in range(len(tick_conversion))])

        # 计算指标
        all_reward = np.sum(rewards)
        all_cost = adjusted_budget - agent.remaining_budget
        cpa_real = np.clip(all_cost / (all_reward + 1e-10), 0, 100)
        score = getScore_nips(all_reward, cpa_real, cpa)
        cpa_exceeded = bool(cpa_real > cpa)
        budget_exceeded = bool(all_cost > adjusted_budget + 1e-6)
        budget_usage = all_cost / adjusted_budget * 100 if adjusted_budget > 0 else 0

        # 统计
        total_score += score
        total_reward += all_reward
        total_cost += all_cost
        if cpa_exceeded:
            exceed_count += 1
        if budget_exceeded:
            budget_exceed_count += 1

        logger.info(f"  预算: {adjusted_budget:.2f} | CPA限制: {cpa:.2f}")
        logger.info(f"  转化数: {all_reward:.2f} | 成本: {all_cost:.2f}")
        logger.info(f"  CPA实际: {cpa_real:.2f} | 分数: {score:.2f} | CPA超标:{cpa_exceeded}")
        results_per_agent.append({
            'key': list(key),
            'budget': float(adjusted_budget),
            'actual_cost': float(all_cost),
            'reward': float(all_reward),
            'cpa_real': float(cpa_real),
            'cpa_limit': float(cpa),
            'score': float(score),
            'budget_usage': float(budget_usage),
            'cpa_exceeded': cpa_exceeded,
            'budget_exceeded': budget_exceeded,
        })

    # 最终结果
    num_agents = len(keys)
    avg_score = total_score / num_agents
    avg_cpa = total_cost / (total_reward + 1e-10)
    exceed_rate = exceed_count / num_agents * 100
    budget_exceed_rate = budget_exceed_count / num_agents * 100
    budget_usage_total = total_cost / total_budget * 100 if total_budget > 0 else 0

    logger.info("\n" + "="*80)
    logger.info("Exp23 测试最终结果")
    logger.info("="*80)
    logger.info(f"平均分数: {avg_score:.2f}")
    logger.info(f"总转化数: {total_reward:.2f}")
    logger.info(f"总成本/总预算: {total_cost:.2f}/{total_budget:.2f} (使用率:{budget_usage_total:.1f}%)")
    logger.info(f"平均CPA: {avg_cpa:.2f}")
    logger.info(f"CPA超标率: {exceed_rate:.1f}% ({exceed_count}/{num_agents})")
    logger.info(f"预算超标率: {budget_exceed_rate:.1f}% ({budget_exceed_count}/{num_agents})")
    logger.info("="*80)

    return {
        'avg_score': avg_score,
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
        'per_agent': results_per_agent,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp23 Qwen-0.5B + 2048维测试 (标准版)')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录路径')
    parser.add_argument('--budget_ratio', type=float, default=1.0, help='预算比例')
    parser.add_argument('--result_suffix', type=str, default="",
                        help='结果文件名后缀（用于多预算测试，避免覆盖）')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出路径；提供后不写入模型checkpoint目录')
    parser.add_argument('--test_file', type=str,
                       default='/home/wangmeiyi/AuctionNet/data/test/alimama/high_conv_sampled/period-7.csv',
                       help='测试数据文件')
    parser.add_argument('--language_emb_dim', type=int, default=2048, help='语言嵌入维度')
    parser.add_argument('--embedding_lookup', type=str, default=None, help='Embedding查找表路径')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Exp23 Qwen-0.5B + 2048维 测试开始 (标准版)")
    logger.info("="*80)

    # 加载embedding查找表（如果提供）
    embedding_lookup = None
    if args.embedding_lookup and os.path.exists(args.embedding_lookup):
        logger.info(f"加载embedding查找表: {args.embedding_lookup}")
        with open(args.embedding_lookup, 'rb') as f:
            embedding_lookup = pickle.load(f)
        logger.info("查找表加载完成")

    start_time = time.time()
    results = run_test(args.model_dir, args.test_file, args.budget_ratio, args.language_emb_dim, embedding_lookup)
    elapsed = time.time() - start_time

    logger.info(f"\n测试总耗时: {elapsed:.2f}秒")
    logger.info("测试完成！")

    # 保存结果
    if args.output:
        result_file = args.output
        os.makedirs(os.path.dirname(result_file) or '.', exist_ok=True)
    else:
        suffix = args.result_suffix.strip()
        if suffix:
            result_file = os.path.join(args.model_dir, f'test_results_exp17b_standard_{suffix}.json')
        else:
            result_file = os.path.join(args.model_dir, 'test_results_exp17b_standard.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存到: {result_file}")
