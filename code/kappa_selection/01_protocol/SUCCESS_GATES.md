# Success Gates

阈值为投稿分叉所需的预先科研标准，不是理论常数。

## Primary regime

Incremental underdelivery cap：10 pp。5 pp 与 20 pp 是敏感性。

## G0 Integrity

- fresh evaluation 在 outcome access 前冻结；
- 已查看 periods 不得标为 untouched；
- technical retries保留，结果导向 rerun 禁止。

## G1 Regret

- 相对 Global-Cal/Reference-Cal 中较强者，budget-normalized regret reduction ≥15%；
- 20% 是目标量级；
- paired 95% bootstrap interval 对 improvement 排除 0。

## G2 Breadth

- 至少 10/15 pairs 不劣；
- 至少 2/3 references 平均不恶化；
- leave-one-bidder/reference-out 后方向保留。

## G3 Delivery

- violation rate 不高于 strongest pooled baseline；
- 若做 non-inferiority，最大 margin 为5 percentage points；
- mean violation severity 不增加；
- absolute underdelivery 同时披露。

## G4 Guardrail robustness

- 10 pp 必须通过；
- 5 pp 与20 pp 至少一个同方向；
- 另一个不能出现明显反向或 feasibility collapse。

## G5 Economic proximity

- oracle gap capture ≥50%，或 ε-optimal rate ≥70%；
- exact κ accuracy 不是成功门槛。

## Interpretation

- G0–G5：Strong rule。
- PID/Dual 等明确子集通过：Architecture-specific rule。
- 只有 retrospective 通过：Patch-only positive。
- regret改善但G3失败：False positive。
- 不超过 Reference-Cal：No selection value，停止扩展。
