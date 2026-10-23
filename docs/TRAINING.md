# Training settings

Settings used for the five learned autobidders, as reported in the paper's
E-Companion (`ec:tab:impl-offline-rl` and the DT/GAS/SemBid implementation
text).

## CQL

| Setting | Value |
| --- | --- |
| Actor | 16 → 256 → 256 → 1, ReLU; learned state-independent log standard deviation |
| Value estimation | Twin Q networks, 17-dimensional input, two 256-unit Mish layers; target networks |
| Discount factor | γ = 1.0 |
| Target-update coefficient | τ = 0.001 |
| Learning rates | Actor, critics, entropy temperature: 1e-4 |
| Objective | CQL temperature 1.0; conservative weight 1.0 (`min_q_weight`) |
| Normalization | Training-derived min-max scaling on state coordinates 13–15 |
| Evaluation action | Deterministic Gaussian-policy mean |

## IQL

| Setting | Value |
| --- | --- |
| Actor | 16 → 128 → 64; separate mean and log-standard-deviation heads |
| Value estimation | Value network 16 → 128 → 64 → 32 → 1; twin Q networks with separate 64-dimensional state/action embeddings and a 128–64–64 trunk |
| Discount factor | γ = 0.99 |
| Target-update coefficient | τ = 0.01 |
| Learning rates | Actor, value network, critics: 1e-4 |
| Objective | Expectile 0.7; advantage temperature 3.0 |
| Normalization | Same statistics and coordinates as CQL |
| Evaluation action | Deterministic actor mean, clipped below at zero |

## DT

- Return-conditioned sequence policy with a 20-step context, 16 state dimensions, and one action dimension.
- Transformer: hidden size 128, six layers, four attention heads, a 1,024-unit feed-forward sublayer, dropout 0.1.
- Maximum episode length 96; campaign evaluation uses 48 steps.
- AdamW with learning rate 1e-4, weight decay 1e-4, and 10,000 warm-up steps.
- Return scale 3,000; evaluation target return 8.

## GAS

- One policy and three independently initialized DT critics.
- 400,000 training steps, batch size 128.
- Policy learning rate 1e-5; critic learning rate 1e-4; reweighting coefficient 0.2.
- Transformer: hidden size 512, six layers, eight heads, 10-step context, maximum episode length 48, weight decay 1e-4.
- Inference: five candidates scaled within [0.9, 1.1]; each critic ranks candidates by its lower value estimate; the highest mean normalized rank becomes the native action.

## SemBid

- Numerical model: 20-step context, hidden size 128, six layers, four heads, a 1,024-unit feed-forward sublayer, dropout 0.1.
- Learning rate 1e-4, weight decay 1e-4, 10,000 warm-up steps, return scale 3,000.
- Semantic representations from frozen Qwen2.5-0.5B-Instruct weights, projected from 896 to 2,048 dimensions.
- Task text encodes the CPA constraint; history text summarizes the preceding state, executed action, settled reward, and current state; strategy text uses the current state and preceding mean executed bid.
