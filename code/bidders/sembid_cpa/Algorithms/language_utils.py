"""
Language Utils for Exp16
完全复用Exp01的数据处理逻辑，添加Task Description支持
"""
import numpy as np
import pandas as pd
import torch
import pickle
import sys
from torch.utils.data import Dataset
from language_templates import BiddingLanguageGeneratorWithTask

# Numpy 2.0 pickles may reference numpy._core.*; provide a shim for numpy 1.x.
try:
    import numpy.core as _np_core
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = _np_core
    if 'numpy._core.numeric' not in sys.modules:
        sys.modules['numpy._core.numeric'] = _np_core.numeric
except Exception:
    pass


class LanguageAugmentedReplayBufferWithTask(Dataset):
    """
    与Exp01的LanguageAugmentedReplayBuffer完全一致，添加Task Description
    """
    def __init__(self, state_dim, act_dim, data_path, K=20, scale=3000, 
                 use_language=True, language_type='both', use_precomputed_embeddings=False):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.K = K
        self.scale = scale
        self.use_language = use_language
        self.language_type = language_type
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        # 初始化语言生成器
        if self.use_language and not use_precomputed_embeddings:
            self.language_generator = BiddingLanguageGeneratorWithTask()
        
        # 加载数据
        print(f"Loading data from {data_path}...")
        if data_path.endswith('.pkl') and use_precomputed_embeddings:
            # 加载预计算的embeddings（新格式）
            with open(data_path, 'rb') as f:
                precomputed_data = pickle.load(f)
            
            # 检查是否是DataFrame格式（新的预处理格式）
            if hasattr(precomputed_data, 'columns') and 'task_description_embeddings' in precomputed_data.columns:
                print(f"✅ Loading precomputed embeddings from DataFrame format...")
                # 直接从DataFrame构建轨迹
                self.trajectories = self._load_from_csv_df(precomputed_data, use_precomputed=True)
                print(f"✅ Loaded {len(self.trajectories)} trajectories with embeddings")
            elif isinstance(precomputed_data, dict) and 'task_embeddings' in precomputed_data:
                # 旧的字典格式
                print(f"✅ Loading precomputed embeddings format...")
                csv_path = data_path.replace('_exp16_with_task_0.5gb.pkl', '_0.5gb.csv')
                print(f"📂 Loading raw trajectories from {csv_path}...")
                self.trajectories = self._load_from_csv(csv_path, use_precomputed=False)
                
                # 将预计算的embeddings分配到每个轨迹
                print(f"🔗 Assigning embeddings to trajectories...")
                task_embs = precomputed_data['task_embeddings']
                history_embs = precomputed_data['history_embeddings']
                strategy_embs = precomputed_data['strategy_embeddings']
                
                row_idx = 0
                for traj in self.trajectories:
                    traj_len = len(traj['states'])
                    traj['task_embeddings'] = task_embs[row_idx:row_idx+traj_len]
                    traj['history_embeddings'] = history_embs[row_idx:row_idx+traj_len]
                    traj['strategy_embeddings'] = strategy_embs[row_idx:row_idx+traj_len]
                    row_idx += traj_len
                
                print(f"✅ Assigned embeddings to {len(self.trajectories)} trajectories")
            else:
                # 旧格式：直接是轨迹列表
                self.trajectories = precomputed_data
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.trajectories = pickle.load(f)
        else:
            self.trajectories = self._load_from_csv(data_path, use_precomputed=False)
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        
        # 计算归一化参数
        all_states = np.concatenate([traj['states'] for traj in self.trajectories], axis=0)
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-6
        
        # 采样权重（与Exp01一致）
        self.p_sample = self._compute_sampling_weights()
    
    def _load_from_csv_df(self, df, use_precomputed=False):
        """从DataFrame加载数据（支持预计算的embeddings）"""
        # 解析state列（如果需要）
        if isinstance(df['state'].iloc[0], str):
            df['state'] = df['state'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        trajectories = []
        # 按 (advertiserNumber, deliveryPeriodIndex) 分组，每个组合是一条轨迹
        grouped = df.groupby(['advertiserNumber', 'deliveryPeriodIndex'])

        for (adv_num, period_idx), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)

            states = np.array([row['state'] for _, row in group.iterrows()], dtype=np.float32)
            actions = group['action'].values.astype(np.float32)

            # Reward处理
            if 'reward_continuous' in group.columns:
                rewards = group['reward_continuous'].values.astype(np.float32)
            elif 'reward' in group.columns:
                rewards = group['reward'].values.astype(np.float32)
            else:
                rewards = np.zeros(len(group), dtype=np.float32)

            # 计算RTG
            rtgs = np.cumsum(rewards[::-1])[::-1]
            timesteps = np.clip(np.arange(len(states)), 0, 95)

            # 提取CPA（用于Task Description）
            cpa_constraint = group['CPAConstraint'].iloc[0] if 'CPAConstraint' in group.columns else 10.0

            traj_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'rtgs': rtgs,
                'timesteps': timesteps,
                'cpa_constraint': cpa_constraint
            }

            # 如果有预计算的embeddings，提取它们
            if use_precomputed:
                task_embs = np.array([row['task_description_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                history_embs = np.array([row['history_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                strategy_embs = np.array([row['strategy_embeddings'] for _, row in group.iterrows()], dtype=np.float32)

                traj_data['task_embeddings'] = task_embs
                traj_data['history_embeddings'] = history_embs
                traj_data['strategy_embeddings'] = strategy_embs

            trajectories.append(traj_data)

        return trajectories

    def _load_from_csv(self, csv_path, use_precomputed=False):
        """从CSV或PKL加载数据（与Exp01一致，支持预处理的pkl）"""
        # 检查是否是pkl文件
        if csv_path.endswith('.pkl'):
            print(f"Loading preprocessed data from {csv_path}...")
            df = pd.read_pickle(csv_path)
            use_precomputed = 'task_description_embeddings' in df.columns
        else:
            df = pd.read_csv(csv_path)
            use_precomputed = False

            # 解析state列
            if 'state' in df.columns and isinstance(df['state'].iloc[0], str):
                df['state'] = df['state'].apply(lambda x: eval(x) if isinstance(x, str) else x)

        trajectories = []
        # 按 (advertiserNumber, deliveryPeriodIndex) 分组
        grouped = df.groupby(['advertiserNumber', 'deliveryPeriodIndex'])

        for (adv_num, period_idx), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)
            
            states = np.array([row['state'] for _, row in group.iterrows()], dtype=np.float32)
            actions = group['action'].values.astype(np.float32)
            
            # Reward处理
            if 'reward_continuous' in group.columns:
                rewards = group['reward_continuous'].values.astype(np.float32)
            elif 'reward' in group.columns:
                rewards = group['reward'].values.astype(np.float32)
            else:
                rewards = np.zeros(len(group), dtype=np.float32)
            
            # 计算RTG
            rtgs = np.cumsum(rewards[::-1])[::-1]
            timesteps = np.clip(np.arange(len(states)), 0, 95)
            
            # 提取CPA（用于Task Description）
            cpa_constraint = group['CPAConstraint'].iloc[0] if 'CPAConstraint' in group.columns else 10.0
            
            traj_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'rtgs': rtgs,
                'timesteps': timesteps,
                'cpa_constraint': cpa_constraint
            }
            
            # 如果有预计算的embeddings，保存它们
            if use_precomputed:
                traj_data['task_embeddings'] = np.array([row['task_description_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                traj_data['history_embeddings'] = np.array([row['history_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
                traj_data['strategy_embeddings'] = np.array([row['strategy_embeddings'] for _, row in group.iterrows()], dtype=np.float32)
            
            trajectories.append(traj_data)
        
        return trajectories
    
    def _compute_sampling_weights(self):
        """计算采样权重（与Exp01一致）"""
        # 基于轨迹长度的采样权重
        traj_lens = np.array([len(traj['states']) for traj in self.trajectories])
        p_sample = traj_lens / np.sum(traj_lens)
        return p_sample
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """获取一个样本（与Exp01一致，添加Task）"""
        traj = self.trajectories[idx]
        traj_len = len(traj['states'])
        
        # 随机选择起始点
        if traj_len > self.K:
            start_idx = np.random.randint(0, traj_len - self.K + 1)
            end_idx = start_idx + self.K
        else:
            start_idx = 0
            end_idx = traj_len
        
        # 提取序列
        states = traj['states'][start_idx:end_idx]
        actions = traj['actions'][start_idx:end_idx]
        rewards = traj['rewards'][start_idx:end_idx]
        rtgs = traj['rtgs'][start_idx:end_idx]
        timesteps = traj['timesteps'][start_idx:end_idx]
        cpa_constraint = traj['cpa_constraint']
        
        seq_len = end_idx - start_idx
        
        # Padding
        if seq_len < self.K:
            pad_len = self.K - seq_len
            states = np.concatenate([np.zeros((pad_len, self.state_dim)), states], axis=0)
            actions = np.concatenate([np.zeros(pad_len), actions], axis=0)
            rewards = np.concatenate([np.zeros(pad_len), rewards], axis=0)
            rtgs = np.concatenate([np.zeros(pad_len), rtgs], axis=0)
            timesteps = np.concatenate([np.zeros(pad_len), timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(pad_len), np.ones(seq_len)], axis=0)
        else:
            pad_len = 0  # 没有padding
            attention_mask = np.ones(self.K)
        
        # 归一化状态
        states = (states - self.state_mean) / self.state_std
        
        # 归一化RTG
        rtgs = rtgs / self.scale
        
        # 检查是否有预计算的embeddings
        has_precomputed = 'task_embeddings' in traj
        
        # 生成或使用预计算的语言embeddings
        if self.use_language:
            if has_precomputed:
                # 使用预计算的embeddings (速度快，固定)
                task_embs = traj['task_embeddings'][start_idx:end_idx]
                h_embs = traj['history_embeddings'][start_idx:end_idx]
                f_embs = traj['strategy_embeddings'][start_idx:end_idx]
                
                # Padding embeddings
                if seq_len < self.K:
                    zero_emb = np.zeros((pad_len, 768), dtype=np.float32)
                    task_embs = np.concatenate([zero_emb, task_embs], axis=0)
                    h_embs = np.concatenate([zero_emb, h_embs], axis=0)
                    f_embs = np.concatenate([zero_emb, f_embs], axis=0)
                
                return {
                    'states': torch.tensor(states, dtype=torch.float32),
                    'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                    'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                    'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                    'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                    'timesteps': torch.tensor(timesteps, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'lang_task_embs': torch.tensor(task_embs, dtype=torch.float32),
                    'lang_h_embs': torch.tensor(h_embs, dtype=torch.float32),
                    'lang_f_embs': torch.tensor(f_embs, dtype=torch.float32)
                }
            else:
                # 实时生成（保持多样性）
                lang_task_texts = []
                lang_h_texts = []
                lang_f_texts = []
                
                for t in range(self.K):
                    if t < pad_len:
                        # Padding部分使用空字符串
                        lang_task_texts.append("")
                        lang_h_texts.append("")
                        lang_f_texts.append("")
                    else:
                        actual_t = t - pad_len
                        # Task Description (每个轨迹固定)
                        task_text = self.language_generator.generate_task_description(cpa_constraint)
                        
                        # Hindsight
                        if actual_t > 0:
                            prev_state = traj['states'][start_idx + actual_t - 1]
                            prev_action = traj['actions'][start_idx + actual_t - 1]
                            prev_reward = traj['rewards'][start_idx + actual_t - 1]
                            curr_state = traj['states'][start_idx + actual_t]
                            h_text = self.language_generator.generate_history(
                                prev_state, prev_action, prev_reward, curr_state
                            )
                        else:
                            h_text = ""
                        
                        # Foresight
                        curr_state = traj['states'][start_idx + actual_t]
                        suggested_bid = traj['actions'][start_idx + actual_t - 1] if actual_t > 0 else 25.0
                        f_text = self.language_generator.generate_strategy(curr_state, suggested_bid)
                        
                        lang_task_texts.append(task_text)
                        lang_h_texts.append(h_text)
                        lang_f_texts.append(f_text)
                
                return {
                    'states': torch.tensor(states, dtype=torch.float32),
                    'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                    'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                    'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                    'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                    'timesteps': torch.tensor(timesteps, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'lang_task_texts': lang_task_texts,
                    'lang_h_texts': lang_h_texts,
                    'lang_f_texts': lang_f_texts
                }
        else:
            return {
                'states': torch.tensor(states, dtype=torch.float32),
                'actions': torch.tensor(actions, dtype=torch.float32).unsqueeze(-1),
                'rewards': torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                'dones': torch.zeros(self.K, dtype=torch.float32).unsqueeze(-1),
                'rtgs': torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1),
                'timesteps': torch.tensor(timesteps, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'lang_task_texts': None,
                'lang_h_texts': None,
                'lang_f_texts': None
            }


def language_collate_fn_with_task(batch):
    """
    Collate函数（与Exp01一致，添加Task，支持预计算embeddings）
    """
    states = torch.stack([item['states'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    rewards = torch.stack([item['rewards'] for item in batch])
    dones = torch.stack([item['dones'] for item in batch])
    rtgs = torch.stack([item['rtgs'] for item in batch])
    timesteps = torch.stack([item['timesteps'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # 检查是否使用预计算的embeddings
    if 'lang_task_embs' in batch[0]:
        # 预计算的embeddings (直接返回tensor)
        lang_task_embs = torch.stack([item['lang_task_embs'] for item in batch])
        lang_h_embs = torch.stack([item['lang_h_embs'] for item in batch])
        lang_f_embs = torch.stack([item['lang_f_embs'] for item in batch])
        return states, actions, rewards, dones, rtgs, timesteps, attention_mask, lang_task_embs, lang_h_embs, lang_f_embs
    else:
        # 文本（需要实时编码）
        lang_task_texts = [item['lang_task_texts'] for item in batch] if batch[0]['lang_task_texts'] is not None else None
        lang_h_texts = [item['lang_h_texts'] for item in batch] if batch[0]['lang_h_texts'] is not None else None
        lang_f_texts = [item['lang_f_texts'] for item in batch] if batch[0]['lang_f_texts'] is not None else None
        return states, actions, rewards, dones, rtgs, timesteps, attention_mask, lang_task_texts, lang_h_texts, lang_f_texts
