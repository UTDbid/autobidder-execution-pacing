"""
Exp23: Language-Guided Decision Transformer with Flexible Embedding Dimension
支持灵活的语言嵌入维度：768, 896, 1024, 1536, 2048等
架构顺序: [Task, RTG, Hindsight, Foresight, State, Action]
"""
import torch
import torch.nn as nn
import os
import sys

# 导入本地组件
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from dt import Block

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")
    SentenceTransformer = None

class LanguageGuidedDTWithTaskFlexible(nn.Module):
    def __init__(self, state_dim, act_dim, state_mean, state_std, 
                 use_language=True, language_emb_dim=768,  # 🔧 新增：可配置维度
                 language_model_name="paraphrase-TinyBERT-L6-v2",
                 action_tanh=False, K=20, max_ep_len=96, scale=3000, target_return=8,
                 use_precomputed_embeddings=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_language = use_language
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.language_emb_dim = language_emb_dim  # 🔧 保存嵌入维度
        self.hidden_size = 128
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device) if not isinstance(state_mean, torch.Tensor) else state_mean.to(self.device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device) if not isinstance(state_std, torch.Tensor) else state_std.to(self.device)
        
        self.max_length = K
        self.max_ep_len = max_ep_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.length_times = 6 if use_language else 3

        # 配置
        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 0.0001

        # Transformer blocks配置
        block_config = {
            "n_ctx": 1024,
            "n_embd": 128,
            "n_layer": 6,
            "n_head": 4,
            "n_inner": 1024,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "block_size": 1024
        }
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])
        
        # Embeddings
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_reward = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_size)
        
        # 🆕 语言相关组件（支持灵活维度）
        if self.use_language:
            # 只在不使用预计算embeddings时加载语言编码器
            if not use_precomputed_embeddings:
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers is required for language mode")
                
                print(f"Loading language encoder: {language_model_name}...")
                self.language_encoder = SentenceTransformer(f"sentence-transformers/{language_model_name}")
                self.language_encoder.eval()
                for param in self.language_encoder.parameters():
                    param.requires_grad = False
            else:
                print(f"Using precomputed embeddings ({language_emb_dim}D), skipping language encoder loading...")
                self.language_encoder = None
            
            # 🔧 核心改动：使用可配置的language_emb_dim
            # Task + Hindsight + Foresight
            print(f"🔧 初始化语言嵌入层: {language_emb_dim} → {self.hidden_size}")
            self.embed_language_task = nn.Linear(language_emb_dim, self.hidden_size)
            self.embed_language_h = nn.Linear(language_emb_dim, self.hidden_size)
            self.embed_language_f = nn.Linear(language_emb_dim, self.hidden_size)
            
            # 空语言embedding (用于padding)
            if not use_precomputed_embeddings and self.language_encoder is not None:
                with torch.no_grad():
                    empty_emb = self.language_encoder.encode("", convert_to_tensor=True, device=self.device)
                    if isinstance(empty_emb, torch.Tensor):
                        self.empty_language_emb = empty_emb.to(self.device)
                    else:
                        self.empty_language_emb = torch.tensor(empty_emb, dtype=torch.float32, device=self.device)
            else:
                # 使用预计算embeddings时，使用零向量作为空embedding
                self.empty_language_emb = torch.zeros(language_emb_dim, dtype=torch.float32, device=self.device)
            
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(self.hidden_size, 1)
        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)
        
        # 优化器配置
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / self.warmup_steps, 1)
        )
        
        self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, timesteps, 
                language_task=None, language_history=None, language_strategy=None, attention_mask=None):
        """
        前向传播
        Token顺序: [Task, RTG, Hindsight, Foresight, State, Action]
        从 State token (Index 4) 预测 Action
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # 在训练时，rtg会是[:, :-1]，所以长度可能不同
        rtg_seq_length = returns_to_go.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, rtg_seq_length), dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask[:, :rtg_seq_length]
        
        # 裁剪所有输入到相同长度
        states = states[:, :rtg_seq_length, :]
        actions = actions[:, :rtg_seq_length, :]
        rewards = rewards[:, :rtg_seq_length, :]
        timesteps = timesteps[:, :rtg_seq_length]
        
        # 原始嵌入
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # 语言嵌入 (如果使用)
        if self.use_language:
            # Task Description embedding
            if language_task is not None and language_task.numel() > 0:
                if language_task.dim() == 2:
                    language_task = language_task.unsqueeze(1)  # [batch, 1, emb_dim]
                # 裁剪到正确长度（训练时rtg会是[:, :-1]，所以需要裁剪language inputs）
                language_task = language_task[:, :rtg_seq_length, :]
                task_embeddings = self.embed_language_task(language_task)
                task_embeddings = task_embeddings + time_embeddings
            else:
                task_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # Hindsight embedding
            if language_history is not None and language_history.numel() > 0:
                if language_history.dim() == 2:
                    language_history = language_history.unsqueeze(1)  # [batch, 1, emb_dim]
                # 裁剪到正确长度
                language_history = language_history[:, :rtg_seq_length, :]
                history_embeddings = self.embed_language_h(language_history)
                history_embeddings = history_embeddings + time_embeddings
            else:
                history_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # Foresight embedding
            if language_strategy is not None and language_strategy.numel() > 0:
                if language_strategy.dim() == 2:
                    language_strategy = language_strategy.unsqueeze(1)  # [batch, 1, emb_dim]
                # 裁剪到正确长度
                language_strategy = language_strategy[:, :rtg_seq_length, :]
                strategy_embeddings = self.embed_language_f(language_strategy)
                strategy_embeddings = strategy_embeddings + time_embeddings
            else:
                strategy_embeddings = torch.zeros((batch_size, rtg_seq_length, self.hidden_size), device=self.device)
            
            # 🔄 Token顺序: [Task, RTG, Hindsight, Foresight, State, Action]
            stacked_inputs = torch.stack(
                (task_embeddings, returns_embeddings, history_embeddings,
                 strategy_embeddings, state_embeddings, action_embeddings),
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 6*rtg_seq_length, self.hidden_size)
            
            # 扩展attention mask (6个token)
            stacked_attention_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask, 
                 attention_mask, attention_mask, attention_mask], 
                dim=1
            ).permute(0, 2, 1).reshape(batch_size, 6*rtg_seq_length)
        else:
            # 无语言: [RTG, State, Action]
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), 
                dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*rtg_seq_length, self.hidden_size)
            
            stacked_attention_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask], 
                dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*rtg_seq_length)
        
        # Transformer forward
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # ✅ 使用正确的attention mask（与原始代码一致）
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, rtg_seq_length), device=self.device)
        
        attention_mask = attention_mask.to(self.device)
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * rtg_seq_length)
        stacked_attention_mask = stacked_attention_mask.to(device=stacked_inputs.device, dtype=stacked_inputs.dtype)
        
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)
        
        # 从State token (Index 4) 预测Action
        if self.use_language:
            x = x.reshape(batch_size, rtg_seq_length, 6, self.hidden_size).permute(0, 2, 1, 3)
            state_preds = x[:, 4]  # State token
        else:
            x = x.reshape(batch_size, rtg_seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
            state_preds = x[:, 1]  # State token in non-language mode
        
        # 预测action
        action_preds = self.predict_action(state_preds)
        
        return action_preds

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask,
             language_task=None, language_history=None, language_strategy=None):
        """训练步骤：计算损失并更新参数"""
        # 预测下一个action (使用[:-1]的inputs预测[1:]的targets)
        action_target = actions[:, 1:]
        action_preds = self.forward(
            states, actions, rewards, rtg[:, :-1], timesteps,
            language_task=language_task,
            language_history=language_history,
            language_strategy=language_strategy,
            attention_mask=attention_mask
        )
        
        # 计算损失
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask[:, 1:].reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask[:, 1:].reshape(-1) > 0]
        
        loss = torch.mean((action_preds - action_target) ** 2)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.detach().cpu().item()

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   language_task=None, language_history=None, language_strategy=None):
        """推理：获取下一个action"""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        rewards = rewards.reshape(1, -1, 1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]

        action_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps,
            language_task=language_task,
            language_history=language_history,
            language_strategy=language_strategy
        )
        
        return action_preds[0, -1].cpu().numpy()

    def init_eval(self):
        """初始化评估模式的缓存"""
        self.eval_states = None
        self.eval_actions = None
        self.eval_rewards = None
        self.eval_returns_to_go = None
        self.eval_timesteps = None
        self.eval_language_task = None
        self.eval_language_h = None
        self.eval_language_f = None
