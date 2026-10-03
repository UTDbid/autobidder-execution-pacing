# Data Split and Contamination Ledger

状态：`PRE-AUDIT DRAFT`

| Evidence block | Known status | Allowed role in Challenge |
|---|---|---|
| AuctionNet reference-development period | consumed | calibration / retrospective only |
| AuctionNet later-period temporal evaluation | consumed | Package A retrospective screen only |
| Platform formal market periods | consumed | existing macro evidence / safety overlay |
| Platform future-period holdout | consumed once | never reuse as fresh validation |
| T9Sim existing official seeds | consumed | calibration / mechanism only |
| iPinYou existing campaign test logs | consumed | logged-support plausibility only |
| New AuctionNet temporal block | unknown | preferred Package B evaluation if truly unopened |
| New AuctionNet opportunity block | unknown | second-best Package B evaluation if frozen before run |
| New CRN seeds on known traffic | available in principle | stochastic transport only, not temporal OOS |
| New T9Sim generator seeds | available in principle | Package C secondary fresh validation |

执行前的 KS-A00 只能检查 metadata、路径、访问日志、缓存与哈希；对候选新 split 不得读取 outcome columns。
