# Rules Lock — To Be Frozen Before Fresh Evaluation

当前状态：`FROZEN FOR RETROSPECTIVE SCREEN — FRESH RULE SUBSET PENDING`

## Frozen candidates

- R1 Constrained Frontier
- R2 Conservative Near-Best 95%
- R3 Sequential Stability（3/4 calibration seeds）
- R4 Structural-Kink-Aware

Package B 最多锁定其中 1–2 个；选择依据只能是 Package A 预设 screening metrics。

## Frozen constants

- Main κ menu：0.0、0.3、0.5、0.8、1.0
- Stress width：1.2
- Endpoint：Raw
- Primary guardrail：10 pp incremental underdelivery
- Sensitivity：5 pp、20 pp
- Near-best threshold：95%
- Sequential sign gate：3/4
- Tie-break：smaller κ

## Lock record

| Item | File | SHA256 | Frozen time | Status |
|---|---|---|---|---|
| Rules | `01_protocol/RULES_LOCK.md` | computed on server | 2026-09-04 | FROZEN FOR SCREEN |
| Success gates | `01_protocol/SUCCESS_GATES.md` | TBD | TBD | NOT FROZEN |
| Split | `01_protocol/DATA_SPLIT.md` | TBD | TBD | NOT FROZEN |
| Configs | `02_configs/*.yaml` | TBD | TBD | NOT FROZEN |
| Code | `03_code/` | computed on server | 2026-09-04 | IMPLEMENTED FOR SCREEN |
