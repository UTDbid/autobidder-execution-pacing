#!/usr/bin/env python3
"""
Exp23 Training Script - 使用2048维Qwen embeddings的Language-Guided DT
H/F 在 State 之前，保证同一步信息可见
"""
import os
import sys
import numpy as np
import torch
import logging
import time
import argparse
from datetime import datetime

# 添加本地路径（Exp23内置依赖）
current_dir = os.path.dirname(os.path.abspath(__file__))
exp23_code_dir = os.path.abspath(os.path.join(current_dir, ".."))
exp23_algo_dir = os.path.join(exp23_code_dir, "Algorithms")
sys.path.insert(0, exp23_algo_dir)
sys.path.insert(0, exp23_code_dir)

from bidding_train_env.common.utils import save_normalize_dict
from language_utils import LanguageAugmentedReplayBufferWithTask, language_collate_fn_with_task
from language_dt_with_task_flexible_hf_first import LanguageGuidedDTWithTaskFlexible  # Exp23版本
from torch.utils.data import DataLoader, WeightedRandomSampler

# 配置日志
log_dir = "/home/wangmeiyi/AuctionNet/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/train_exp23_2048.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Exp23: 2048维 Language-Guided DT (H/F before State)')

    # 语言配置
    parser.add_argument('--use_language', action='store_true', default=True,
                        help='是否使用语言指导')
    parser.add_argument('--language_type', type=str, default='both',
                        choices=['history', 'strategy', 'both', 'none'],
                        help='语言类型: history(原history), strategy(原strategy), both, none')
    parser.add_argument('--language_emb_dim', type=int, default=2048,
                        help='语言嵌入维度')

    # 数据配置
    parser.add_argument('--data_file', type=str,
                        default='/home/wangmeiyi/AuctionNet/data/train/alimama_trajectories/high_conversion_sampled/trajectory_data_exp17b_dim2048_0.5gb.pkl',
                        help='训练数据路径（2048维预处理后的pkl文件）')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_steps', type=int, default=800000,
                        help='训练步数')
    parser.add_argument('--save_interval', type=int, default=50000,
                        help='保存间隔')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认自动生成）')
    parser.add_argument('--exp_name', type=str, default='',
                        help='实验名称后缀')

    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("="*70)
    logger.info("🤖 Exp23: 2048维 Language-Guided DT Training (H/F before State)")
    logger.info("="*70)

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"Exp23_Qwen_DT_2048_{timestamp}{args.exp_name}"
        output_dir = f"/home/wangmeiyi/AuctionNet/models/LanguageDT/{exp_name}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info(f"📏 语言嵌入维度: {args.language_emb_dim}")

    # 加载数据
    logger.info(f"📊 加载数据: {args.data_file}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")

    # 创建数据集
    dataset = LanguageAugmentedReplayBufferWithTask(
        state_dim=16,
        act_dim=1,
        data_path=args.data_file,
        K=20,
        scale=3000,
        use_language=args.use_language,
        language_type=args.language_type,
        use_precomputed_embeddings=True
    )

    # 创建数据加载器
    sampler = WeightedRandomSampler(
        weights=torch.ones(len(dataset)),
        num_samples=len(dataset),
        replacement=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=language_collate_fn_with_task,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型
    logger.info("🏗️ 初始化模型...")
    model = LanguageGuidedDTWithTaskFlexible(
        state_dim=16,
        act_dim=1,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        use_language=args.use_language,
        language_emb_dim=args.language_emb_dim,  # 🔧 关键：2048维
        K=20,
        max_ep_len=96,
        scale=3000,
        use_precomputed_embeddings=True
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"🎯 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=10000
    )

    # 训练循环
    logger.info(f"🚀 开始训练... 总步数: {args.num_steps}")
    model.train()

    # 创建持久的数据迭代器
    data_iter = iter(dataloader)

    for step in range(args.num_steps):
        start_time = time.time()

        # 获取批次数据
        try:
            batch = next(data_iter)
            states, actions, rewards, dones, rtg, timesteps, mask, task_desc_emb, history_emb, strategy_emb = batch
        except StopIteration:
            # 重新创建迭代器
            data_iter = iter(dataloader)
            batch = next(data_iter)
            states, actions, rewards, dones, rtg, timesteps, mask, task_desc_emb, history_emb, strategy_emb = batch
        except Exception as e:
            logger.error(f"数据加载错误: {e}")
            continue

        # 移动到设备
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device)
        mask = mask.to(device)
        task_desc_emb = task_desc_emb.to(device)
        history_emb = history_emb.to(device)
        strategy_emb = strategy_emb.to(device)

        # 前向传播并计算损失
        loss = model.step(states, actions, rewards, dones, rtg, timesteps, mask,
                         language_task=task_desc_emb, language_history=history_emb, language_strategy=strategy_emb)

        # 日志
        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Step {step}/{args.num_steps} | Loss: {loss:.4f} | Time: {elapsed_time:.3f}s")

        # 保存模型
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{step+1}')
            os.makedirs(checkpoint_path, exist_ok=True)

            # 保存模型
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'language_dt_2048.pt'))

            # 保存配置
            config = {
                'state_dim': 16,
                'act_dim': 1,
                'max_length': 20,
                'max_ep_len': 96,
                'use_language': args.use_language,
                'language_type': args.language_type,
                'language_emb_dim': args.language_emb_dim,  # 🔧 保存2048
                'step': step + 1
            }

            import json
            with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)

            # 保存标准化参数
            if hasattr(dataset, 'state_mean') and hasattr(dataset, 'state_std'):
                normalize_dict = {
                    'state_mean': dataset.state_mean,
                    'state_std': dataset.state_std
                }
                save_normalize_dict(normalize_dict, os.path.join(checkpoint_path, 'normalize_dict.pkl'))

            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    logger.info("✅ 训练完成!")
    logger.info(f"🏁 最终模型保存在: {output_dir}")

if __name__ == "__main__":
    main()
