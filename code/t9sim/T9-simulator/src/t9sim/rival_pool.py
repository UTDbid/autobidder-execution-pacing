"""v10 Private-Rival Market (reason-1c fidelity). A pool of K rival archetypes
whose PRIVATE valuations feed the competing bid LU7 = max_{k in A_i} b_ik, replacing
v7's single max(gaming, nongaming) iid draw.

log b_ik = log base_e(i)
         + rho * [ w_k . phi(LU_u)            # rival k's own LTV model on the user (DSP-invisible)
                 + beta_R * R[seg(u), k]       # retargeting per observable segment (SSP-recoverable via cells)
                 + pace_k(day_i) ]             # AR(1) budget pacing / flights (time footprint)
         + sigma_k * eps_ik                    # idiosyncratic

Participation Z_ik ~ Bernoulli( pi_k(exchange) . F_k(day) . g(pace_k(day)) ); a
generalist archetype (k=0) is always-on so A_i != empty and LU7 > 0 always. N_i =
|A_i| is emitted as H9 (SSP-only competition density, endogenous).

rho (`rho_v10`) is the headline dial: share of log-bid variation from the structured
private components vs iid noise. rho=0 => max-of-K pure-noise competing bids (the
SSP-null baseline; NOT byte-identical v7 whose max is over 2 branches — byte-identity
is the *flag-OFF* guarantee, handled in auctions.py, not this module).

Design notes:
- All draws use a dedicated / spawned RNG (never the main auction stream), so flag-OFF
  stays byte-identical v7 and flag-ON is chunk-size invariant.
- Variance is NOT perfectly raked (max-of-K changes distribution shape); autocal
  (k_global) + the 6 validation gates are the backstop, per the v10 proposal.
- Non-gaming archetypes load w_k on the general-value slice of phi only (not
  game-specific relevance) - implemented as smaller loadings on z.
"""
import numpy as np


class RivalPool:
    def __init__(self, cfg, n_days, exch_labels, users):
        bm = cfg.benchmarks
        # params live under bn.rival_pool (declared in benchmarks.yaml, beside the other
        # edge-group params); the legacy top-level key is still honoured as a fallback
        rp = ((bm.get("bn") or {}).get("rival_pool")
              or bm.get("rival_pool") or {})
        self.K = int(rp.get("K", 8))
        self.n_gaming = int(rp.get("n_gaming", 3))          # k in [0, n_gaming) are gaming DSPs
        self.beta_R = float(rp.get("beta_R", 0.5))
        self.n_days = int(n_days)
        self.exch_labels = list(exch_labels)
        E = len(self.exch_labels)
        self._exch_idx = {b: i for i, b in enumerate(self.exch_labels)}
        rng = cfg.rng("rival_pool")                          # dedicated deterministic stream

        # --- observable retargeting segments: os x device_type (persistent) ---
        os_v = users["os"].astype(str).to_numpy()
        dv_v = users["device_type"].astype(str).to_numpy()
        combos = sorted(set(zip(os_v.tolist(), dv_v.tolist())))
        self._seg_map = {c: i for i, c in enumerate(combos)}
        self.n_seg = len(combos)

        # --- per-rival private LTV-model loadings on phi=z (gaming strong, non-gaming weak) ---
        self.w_k = np.empty(self.K)
        self.w_k[:self.n_gaming] = rng.uniform(0.35, 0.75, self.n_gaming)
        self.w_k[self.n_gaming:] = rng.uniform(0.00, 0.20, self.K - self.n_gaming)

        # --- retargeting matrix R[seg, k] ~ N(0,1), persistent all window ---
        self.R = rng.standard_normal((self.n_seg, self.K))

        # --- per-rival idiosyncratic scale ---
        self.sigma_k = rng.uniform(0.30, 0.55, self.K)

        # --- participation propensity per (rival, exchange); generalist k=0 always-on ---
        self.pi_ke = rng.uniform(0.15, 0.60, (self.K, E))
        self.pi_ke[0, :] = 1.0

        # --- AR(1) pacing p_k(d), stationary mean 0 ---
        phi_p = float(rp.get("pacing_ar", 0.85))
        sig_p = float(rp.get("pacing_sigma", 0.30))
        p = np.zeros((self.K, self.n_days))
        p[:, 0] = sig_p * rng.standard_normal(self.K)
        for d in range(1, self.n_days):
            p[:, d] = phi_p * p[:, d - 1] + sig_p * rng.standard_normal(self.K)
        self.pace = p                                        # (K, n_days)

        # --- flight windows: ~half of non-generalist rivals dark for a day-block ---
        F = np.ones((self.K, self.n_days))
        span = max(1, self.n_days - 5)
        for k in range(1, self.K):
            if rng.random() < 0.5:
                s = int(rng.integers(0, span)); L = int(rng.integers(2, 6))
                F[k, s:s + L] = 0.0
        self.flight = F

    def seg_index(self, os_arr, dev_arr):
        default = 0
        return np.array([self._seg_map.get((str(o), str(d)), default)
                         for o, d in zip(os_arr, dev_arr)], dtype="int64")

    def competing_bids(self, df, base_e, z, rng, rho):
        """Return (lu7, n_rivals) per row. Uses a SPAWNED rng only (main stream
        untouched -> flag-OFF byte-identity + chunk-size invariance)."""
        n = len(base_e)
        crng = np.random.default_rng(rng.bit_generator._seed_seq.spawn(1)[0])

        exch_idx = np.array([self._exch_idx.get(b, 0)
                             for b in df["ad_exchange"].astype(str).to_numpy()],
                            dtype="int64")
        day_idx = np.clip((df["timestamp"].to_numpy() - df["timestamp"].to_numpy().min())
                          // 86400, 0, self.n_days - 1).astype("int64")
        seg_idx = self.seg_index(df["os"].astype(str).to_numpy(),
                                 df["device_type"].astype(str).to_numpy())

        log_base = np.log(base_e)
        logb = np.full((n, self.K), -np.inf)
        participating = np.zeros((n, self.K), dtype=bool)
        for k in range(self.K):
            pace_k = self.pace[k, day_idx]                    # (n,)
            gate = 0.5 + 1.0 / (1.0 + np.exp(-pace_k))        # (0.5, 1.5)
            p_part = self.pi_ke[k, exch_idx] * self.flight[k, day_idx] * gate
            part = crng.random(n) < np.clip(p_part, 0.0, 1.0)
            if k == 0:
                part[:] = True                                # always-on generalist
            struct = self.w_k[k] * z + self.beta_R * self.R[seg_idx, k] + pace_k
            eps = crng.standard_normal(n)
            lb = log_base + rho * struct + self.sigma_k[k] * eps
            logb[part, k] = lb[part]
            participating[:, k] = part

        lu7 = np.exp(logb.max(axis=1))                        # generalist guarantees >-inf
        n_rivals = participating.sum(axis=1).astype("int64")
        return lu7, n_rivals
