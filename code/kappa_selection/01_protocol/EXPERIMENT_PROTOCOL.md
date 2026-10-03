# Experiment Protocol

## Research question

Can calibration evidence identify where a platform should stop increasing execution width for a deployed bidder–reference pair?

## Packages

### A. Retrospective screen

- 已消费 later-period environment；
- 比较 R1–R4 与 Fixed、Global-Cal、Reference-Cal；
- 只作科研 GO/STOP；
- 服务器覆盖审计后，仅补 later-period 主菜单缺口：Traffic-aware 的
  0.3、0.5、1.0，以及 PID/Dual 的 1.0。五个 learned bidders、八个
  CRN seeds，共 200 个 config-seed evaluations；其余节点继承既有结果。

### B. Fresh challenge

- 只保留 A 中最有希望的 1–2 rules；
- rule、guardrail、metric、seed 与 success gate 全部哈希冻结；
- 真正新 temporal block 优先；
- 若没有真正未读取的新时间段，则使用冻结后的新 CRN seeds做 stochastic
  transport，并明确不将其表述为 temporal OOS。主 grid 600 evaluations。

### C. Conditional extension

- fresh T9Sim seeds 检验 information view 是否改变 selection regret；
- 轻量 Macro follow-up 检验 rule-selected κ 的市场后果；
- 只有 B 成功才运行。

## Primary scientific distinction

Package A 即使表现很好，也只能叫 retrospective/post-confirmatory screening。只有 Package B 满足 split integrity 后，才能支持 prospective/OOS deployment-rule claim。
