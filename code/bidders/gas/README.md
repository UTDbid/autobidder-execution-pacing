<h1 align="center">
  <img src="figs/kuaishou_tech.png" alt="Kuaishou Logo" width="150" height="40"><br>
  GAS: Generative Auto-Bidding with Post-Training Search
</h1>


<p align="center">
  <a href="https://arxiv.org/pdf/2412.17018"><img src="https://img.shields.io/badge/📖_Paper-WWW'25-red" alt="Paper WWW'25"></a>
  &nbsp;
  <a href="./figs/GAS-Poster.pdf"><img src="https://img.shields.io/badge/🖼_Poster-GAS-green" alt="Poster GAS"></a>
  &nbsp;
  <a href="https://tianchi.aliyun.com/competition/entrance/532236/rankingList"><img src="https://img.shields.io/badge/🏆_Competition-NeurIPS'24_Auction_Competition_Winner-blue" alt="Competition GAS"></a>
</p>

## 📝 Introduction
We propose a flexible and practical Generative Auto-bidding scheme
using post-training Search, termed GAS, to refine a base policy
model’s output and adapt to various advertisers' preferences. Our online A/B test on the **Kuaishou advertising platform** demonstrate the effectiveness of GAS, achieving
significant improvements, e.g., **4.60% increment of target cost**.🎉🎉🎉

<p align="center">
    <img src="./figs/www_main_figure_page.jpg" alt="method" width="500" height="400">
   <img src="./figs/online_system5_page.jpg" alt="method" width="300" height="400">
</p>


## 📢 Updates
- 2025-05-27: We release the code implementation for GAS.



## 💾 Installation

### Python Environment
```
conda create -n gas_env python=3.9.12 pip=23.0.1
conda activate gas_env
pip install -r requirements.txt 
```

### Prepare the Datasets
The datasets could be downloaded from the [NeurIPS 2024 Competition Auto-Bidding in Large-Scale Auctions](https://tianchi.aliyun.com/competition/entrance/532236/rankingList).
We express our utmost respect for their tremendous contributions to the auto-bidding and computational advertising community!
#### 1) AuctionNet Dataset
```
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_7-8.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_9-10.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_11-12.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_period_13.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_1.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/autoBidding_aigb_track_data_trajectory_data_extended_2.zip
```

#### 2) AuctionNet-sparse Dataset
```
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_period_7-8.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_period_9-10.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_period_11-12.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_period_13.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_1.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_2.zip
https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/final/autoBidding_aigb_track_final_data_trajectory_data_3.zip
```
After download, you should concat them to a full dataset local file.

## 🚀 Get Started

#### Step 1 Train a base policy model
We choose the decision transformer as our base policy model.
You could simply run by 

`python run/train_dt_baselines.py --baseline_method 'vanilla_dt' --data_path path/to/local/dataset`

#### Step 2 Train multiple QTs (Transformer-based Q-value critics) 
We could train multiple Q-value critics by randomly run multiple times and save different models.
You could run simply run by

`python run/train_dt_critics.py --baseline_method 'dt_reweight_search_Q' --reweight_w 0.2 --data_path path/to/local/dataset`

where reweight could simulate different preference (value_reward + reweight_w * cpa_reward, a higher reweight_w means more preference on the cpa.)


#### Step 3 Evaluation
The evaluation procedure is based on the AuctionNet simulator.

For evaluating the DT baselines, please run

`python run/run_evaluate_dt_baselines.py`

For evaluating the GAS method, please run 

`python run/run_evaluate_gas.py`

## 📚 Citation
If you find our work useful, please consider citing us!
```
@inproceedings{li2025gas,
  title={GAS: Generative Auto-bidding with Post-training Search},
  author={Li, Yewen and Mao, Shuai and Gao, Jingtong and Jiang, Nan and Xu, Yunjian and Cai, Qingpeng and Pan, Fei and Jiang, Peng and An, Bo},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={315--324},
  year={2025}
}
```
