"""Stage 2 - auction & outcome engine (v6).

Per auction: sample one user + one app + one campaign from the frozen pools,
draw per-auction context (B3-B6, D1-D3), compute the ground-truth funnel
probabilities and EV, then run the value-aware first-price auction:

  LU7 = max( base_eCPM(B6) * exp(gamma * z(logEV) + eps_g),     # gaming demand
             base_eCPM(B6) * exp(eps_x) )                       # non-gaming
  H2  = k_global * k_cpa(C2) * v_slot * ease(B2) * E[ltv|B2] * shade * explore
  H3  = 1[H2 >= max(LU7, H1)]
  H4  = H2 if won | LU7 if lost & LU7 >= H1 | NaN if unsold     (v5 piecewise)

Funnel outcomes (E, F, G) are drawn on ALL rows (v6 MMP censoring correction):
lost-row outcomes are the declared counterfactual "what the attribution chain
would have recorded had our ad been served". The master is never NaN-censored
(only H4 on unsold rows); won/lost visibility is the censoring layer's job
(censor.py). None of the outcome formulas reads `won`. One bid per auction.
All draws vectorised; generation is chunked.

Calibrated win-rate pinning (decision 10 Jun 2026): k_global is solved by
bisection on a warm-up sample so the aggregate win rate hits the profile
target; per-auction outcomes stay fully generative.

EV = 0 rows (inactive users, LU5 = 0): the gaming term is exactly 0 - the
impression is worthless to value-aware gaming competitors - so LU7 reduces to
the non-gaming noise term; z-stats are computed on EV > 0 rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_loader import split_shares

_FORMAT_CODE = {"banner": 1, "interstitial": 2, "rewarded": 3}


class AuctionEngine:

    def __init__(self, cfg, users, apps, campaigns):
        self.cfg = cfg
        self.users = users.reset_index(drop=True)
        self.apps = apps.reset_index(drop=True)
        self.campaigns = campaigns.reset_index(drop=True)
        bm = cfg.benchmarks

        # categorical samplers
        self.exch_labels, self.exch_p = split_shares(bm["ad_exchanges"])
        self.fmt_labels, self.fmt_p = split_shares(bm["slot_format_shares"])
        self.sizes = {f: split_shares(bm["slot_sizes"][f])
                      for f in self.fmt_labels}
        self.hourdow_vals, self.hourdow_p = cfg.hourdow_pmf()

        # price shapes, median-normalised, rescaled to USD format targets
        pay_vals, pay_p = cfg.pmf(bm["prices"]["paying_shape_csv"])
        floor_vals, floor_p = cfg.pmf(bm["prices"]["floor_shape_csv"])
        med = cfg.paying_median()
        self.pay_shape = (pay_vals.astype("float64") / med, pay_p)
        self.floor_shape = (floor_vals.astype("float64") / med, floor_p)
        self.ecpm_target = bm["ecpm_targets_usd"]

        # per-category constants
        cats = bm["app_categories"]
        sigma = bm["ltv"]["lognormal_sigma"]
        self.mu_cat = {c: bm["ltv"]["lognormal_mu"] + np.log(
            cats[c]["ltv_multiplier"]) for c in cats}
        self.ease = {c: cats[c]["install_ease"] for c in cats}
        # observable per-category mean spend (no latents): DSP's E[ltv | B2]
        self.eltv_b2 = {c: float(np.exp(self.mu_cat[c] + sigma**2 / 2))
                        for c in cats}
        self.k_cpa = {s: bm["advertiser_scale"][s]["k_cpa"]
                      for s in bm["advertiser_scale"]}

        self.window_start = int(pd.Timestamp(
            bm["time"]["window_start_utc"], tz="UTC").timestamp())
        self.n_weeks = bm["time"]["window_days"] // 7

        # solved on warm-up (calibrate()); defaults let tests run uncalibrated
        self.k_global = bm["auction"]["k_global"]
        self.z_mu, self.z_sigma = 0.0, 1.0

        # BN dependency layer (v7 funnel + v8 price): per-edge flags + CPTs.
        # COPY the dict (autocal builds many engines off one cfg; the v8 auto-force
        # below must not mutate the shared cfg.bn_edges for later instances).
        self.bn_edges = dict(getattr(cfg, "bn_edges", {}))
        # v8 invariant: ANY price edge ON => EV_market (genre-marginal) ON, so the
        # focal ad_genre/campaign value channel is routed OUT of LU7 (C1-C4 out).
        _PRICE_EDGES = ("competition", "app_quality_price", "exchange_price",
                        "app_price", "time_competition")
        if any(self.bn_edges.get(f) for f in _PRICE_EDGES):
            self.bn_edges["ev_market"] = True
        self.bn_cpts = getattr(cfg, "bn_cpts", None)

        # reason-1c probe: hidden persistent rival-segment factor in LU7. rho
        # REALLOCATES the competing-bid noise (gaming eps_g and nongaming eps_x)
        # from pure-iid to a common per-segment factor R[seg] (variance-preserving,
        # so win-rate is held fixed). rival_seg is SSP-exposed (a C3/C4-only Tier-2
        # feature); the DSP (C1/C2) never sees it. rho=0 => byte-identical v7 LU7.
        self.rho_rival = float(bm["auction"].get("rho", 0.0))
        if self.bn_edges.get("rival_hidden"):
            self._n_rival_seg = int(bm["auction"].get("n_rival_seg", 200))
            # persistent standardized rival-demand per segment, fixed across chunks/splits
            _rr = np.random.default_rng(int(cfg.profile["seed"]) + 987654321)
            self._rival_R = _rr.standard_normal(self._n_rival_seg)
        if self.bn_edges.get("rival_pool"):        # v10: Private-Rival Market
            from .rival_pool import RivalPool
            self._rival_pool = RivalPool(
                cfg, bm["time"]["window_days"], self.exch_labels, self.users)
        if self.bn_edges.get("pairing"):
            self._setup_pairing(cfg)
        if self.bn_edges.get("os_spend"):        # edge #3: iOS ARPU multiplier
            self._rho = float(cfg.benchmarks["bn"]["ios_ltv_multiplier"])
            ios = float(cfg.benchmarks["bn"]["os_split_ios"])
            self._eplat = ios * self._rho + (1.0 - ios)   # E_os[plat], mean-preserve
        if self.bn_edges.get("payer_timing"):    # edge #4: t_pay(hour,dow)
            pt = cfg.benchmarks["bn"]["payer_timing"]
            self._hour_mult = np.asarray(pt["hour_mult"], dtype=float)
            self._dow_mult = np.asarray(pt["dow_mult"], dtype=float)
            _dw, _hr = self.hourdow_vals[:, 0], self.hourdow_vals[:, 1]
            self._tpay_rake = float(np.sum(             # E_pop[t_pay] -> rake to mean 1
                self.hourdow_p * self._hour_mult[_hr] * self._dow_mult[_dw]))
        if self.bn_edges.get("hour"):            # edge #5: archetype-conditioned hour
            _dows = self.hourdow_vals[:, 0]
            self._dow_marg = np.array([self.hourdow_p[_dows == d].sum()
                                       for d in range(7)])
            self._dow_marg /= self._dow_marg.sum()
            res = self.bn_cpts["resolved"]["hour_of_day"]
            self._hour_cpt = {}
            for a in res["cpt"]:
                v = np.asarray(res["cpt"][a], dtype=float)
                self._hour_cpt[a] = v / v.sum()   # renormalise (yaml rounding)
            self._arch_labels = list(cfg.archetypes["shares"])
        if self.bn_edges.get("exposure"):        # edge #2: value -> campaign exposure
            self._setup_exposure(cfg)
        if self.bn_edges.get("competition"):     # v8 #6: bid_density N_t + order stat
            self._setup_competition(cfg)
        if self.bn_edges.get("ev_market"):       # v8: genre-marginal EV for LU7
            self._setup_ev_market(cfg)
        if any(self.bn_edges.get(f) for f in ("app_quality_price", "exchange_price",
                                              "app_price", "time_competition")):
            self._setup_price_multipliers(cfg)   # v8 #7-#10

    # ────────────────────────────────────────────────────────────────
    def _setup_pairing(self, cfg):
        """Edge #1: precompute the IPF-coupled (archetype x app) joint M and the
        per-archetype user index groups. M is raked to row-sums pi (archetype
        marginal) and col-sums pop=1/A (uniform app popularity), so BOTH marginals
        are preserved; only P(archetype | app) becomes non-uniform, making app_id
        a proxy for the user's archetype -> LU2-6 -> value."""
        arch_labels = list(cfg.archetypes["shares"])
        pi = np.array([float(cfg.archetypes["shares"][a]) for a in arch_labels])
        pi = pi / pi.sum()
        A = len(self.apps)
        pop = np.full(A, 1.0 / A)
        la2 = self.apps[[f"la2_{a}" for a in arch_labels]].to_numpy()   # (A, K)
        s = float(cfg.benchmarks["bn"]["pairing_strength_s"])
        M = (pi[:, None] * pop[None, :]) * (1.0 + s * (la2.T / pi[:, None] - 1.0))
        M = np.clip(M, 1e-12, None)
        for _ in range(500):                       # IPF rake: rows->pi, cols->pop
            M *= (pi / M.sum(1))[:, None]
            M *= (pop / M.sum(0))[None, :]
        self._pair_flat = (M / M.sum()).ravel()
        self._n_arch, self._n_app = len(arch_labels), A
        arch = self.users["lu1_archetype"].to_numpy()
        self._users_by_arch = [np.where(arch == a)[0] for a in arch_labels]

    def _setup_exposure(self, cfg):
        """Edge #2: value-optimised campaign exposure. Users are binned by value
        g = z(log(LU4*LU5)); per value-bin the campaign distribution is tilted
        toward high-LC2 campaigns (weight ~ budget * exp(beta * g * z(log LC2)))
        and IPF-raked so the campaign marginal (hence ad_genre_mix) is PRESERVED.
        Footprint shows only in MMP-uncensored payer/whale outcomes; no C4/genre
        signal. Separate flag (highest confound risk)."""
        beta = float(cfg.benchmarks["bn"]["exposure_beta_vo"])
        lu4 = self.users["lu4_payer_prob"].to_numpy()
        lu5 = self.users["lu5_ltv_mult"].to_numpy()
        val = lu4 * lu5
        g = np.empty(len(val))
        pos = val > 0
        lg = np.log(val[pos])
        g[pos] = (lg - lg.mean()) / (lg.std() + 1e-9)
        g[~pos] = (g[pos].min() - 1.0) if pos.any() else 0.0
        vb = pd.qcut(g, 10, labels=False, duplicates="drop").astype(int)
        self._user_vbin = vb
        self._n_vbin = int(vb.max()) + 1
        binpop = np.array([(vb == b).mean() for b in range(self._n_vbin)])
        gbar = np.array([g[vb == b].mean() for b in range(self._n_vbin)])
        lc2 = self.campaigns["lc2_game_quality"].to_numpy()
        zlc2 = (np.log(lc2) - np.log(lc2).mean()) / (np.log(lc2).std() + 1e-9)
        sw = self.campaigns["sample_weight"].to_numpy()
        P = sw[None, :] * np.exp(beta * gbar[:, None] * zlc2[None, :])
        P /= P.sum(1, keepdims=True)
        for _ in range(300):                     # IPF: campaign marginal -> sw
            marg = binpop @ P
            P *= (sw / np.maximum(marg, 1e-12))[None, :]
            P /= P.sum(1, keepdims=True)
        self._camp_pmf = P

    def _setup_competition(self, cfg):
        """v8 edge #6: competition density. N_t = 1 + Poisson(lambda) rivals;
        LU7_gaming = max over N_t iid gaming draws (an order statistic). More
        rivals -> a higher max -> higher clearing (faithful first-price economics).
        A GLOBAL mean-1 rake (single scalar comp_single / E_N[c_N]) preserves the
        UNCONDITIONAL E[gaming]=v7 while KEEPING E[gaming|N_t] increasing in N_t,
        so bid_density (=N_t, SSP-only) stays predictive of LU7 (the per-row rake
        the design first proposed cancels exactly that channel -> probe dies)."""
        from scipy.stats import norm
        bm = cfg.benchmarks
        cp = bm["bn"]["competition"]
        sg = float(bm["auction"]["sigma_g"])
        self._lamN = float(np.exp(cp["lambda_b0"]))     # constant rate (MVP)
        self._comp_nmax = int(cp.get("nmax", 64))
        Nmax = self._comp_nmax
        srng = cfg.rng("comp_setup")                    # dedicated stream
        u = np.clip(srng.random(120000), 1e-12, 1.0 - 1e-12)
        # c_N = E[exp(sg * max of N iid N(0,1))]; c_1 analytic, N>=2 by MC.
        cN = np.empty(Nmax + 1)
        cN[0] = cN[1] = float(np.exp(sg * sg / 2.0))    # max of 1 == single draw
        Ns = np.arange(2, Nmax + 1)[:, None]            # (Nmax-1, 1)
        M = norm.ppf(u[None, :] ** (1.0 / Ns))          # (Nmax-1, samples)
        cN[2:] = np.mean(np.exp(sg * M), axis=1)
        self._comp_cN = cN
        self._comp_single = float(np.exp(sg * sg / 2.0))
        # global rake = comp_single / E over the marginal N_t of c_N
        Nsamp = np.minimum(1 + srng.poisson(self._lamN, 200000), Nmax)
        self._comp_rake = self._comp_single / float(np.mean(cN[Nsamp]))

    def _setup_price_multipliers(self, cfg):
        """v8 edges #7-#10: deterministic mean-1-raked LU7 price multipliers, each an
        app/exchange/category/time PROPERTY (common to all rivals -> factors out of the
        order-statistic max) and a deterministic function of an OBSERVABLE feature, so
        the win model in EVERY condition can learn it -> raises ABSOLUTE auc_win (the
        combined-schema p(win) gain). NOT SSP-exclusive (so they don't move C3-C1). Each
        rake preserves E[LU7] over the relevant marginal (recalibrate k_global after ON).
        Rakes for app/cat assume ~uniform app sampling (holds with pairing OFF)."""
        bm = cfg.benchmarks
        pb = bm["bn"].get("price", {})
        if self.bn_edges.get("app_quality_price"):           # #7 m_app (uses la1)
            self._kappa_q = float(pb.get("kappa_q", 2.0))
            la1 = self.apps["la1_app_quality"].to_numpy()
            self._m_app_rake = float(np.mean(la1 ** self._kappa_q))
        if self.bn_edges.get("exchange_price"):              # #8 m_exch
            ep = pb.get("exch_price", {}) or {}
            self._a_price = np.array([float(ep.get(b, 1.0)) for b in self.exch_labels])
            self._m_exch_rake = float(np.sum(self.exch_p * self._a_price))
        if self.bn_edges.get("app_price"):                   # #9 m_cat
            cp = pb.get("cat_price", {}) or {}
            self._cat_price = {c: float(cp.get(c, 1.0)) for c in cfg.categories}
            cat_arr = self.apps["app_category"].to_numpy()
            self._m_cat_rake = float(np.mean([self._cat_price[c] for c in cat_arr]))
        if self.bn_edges.get("time_competition"):            # #10 c_time (daytime-peaked)
            self._hour_price = np.asarray(pb.get("hour_price_mult", [1.0] * 24), float)
            self._dow_price = np.asarray(pb.get("dow_price_mult", [1.0] * 7), float)
            _dw, _hr = self.hourdow_vals[:, 0], self.hourdow_vals[:, 1]
            self._ctime_rake = float(np.sum(self.hourdow_p
                               * self._hour_price[_hr] * self._dow_price[_dw]))

    def _setup_ev_market(self, cfg):
        """v8: EV_market = the genre-MARGINAL, supply+user EV that drives LU7 in
        place of the focal ev_truth. Routes the focal ad_genre (C4) and the focal
        campaign latents lc1/lc2 (C3) OUT of LU7 -> the negative-control / C1-C4-out
        invariant. Genre weights align to cfg.categories (= lu6_ column order)."""
        gm = cfg.benchmarks["ad_genre_mix"]
        cats = cfg.categories
        def _val(x):
            return float(x["value"]) if isinstance(x, dict) else float(x)
        w = np.array([_val(gm[c]) for c in cats], dtype=float)
        assert len(w) == len(cats), "ad_genre_mix / categories misaligned"
        self._genre_w = w / w.sum()

    def _join_pools(self, n, rng):
        """Sample the (user, app, campaign) join for n auctions.

        BN seam (BN_Build_Plan.md). bn_edges OFF (baseline) = independent draw.
        Edge #1 (pairing): IPF-coupled user<->app joint (app_id proxies archetype).
        Edge #2 (exposure): value-tilted campaign draw per user. All marginal-
        preserving; OFF path is verbatim -> byte-identical output.
        """
        if self.bn_edges.get("pairing"):
            K, A = self._n_arch, self._n_app
            idx = rng.choice(K * A, size=n, p=self._pair_flat)
            arch_i, app_i = idx // A, idx % A
            u_rows = np.empty(n, dtype="int64")
            for k in range(K):
                mk = arch_i == k
                mm = int(mk.sum())
                if mm:
                    u_rows[mk] = rng.choice(self._users_by_arch[k], size=mm)
            a = self.apps.iloc[app_i].reset_index(drop=True)
        else:
            u_rows = rng.integers(0, len(self.users), n)
            a = self.apps.iloc[rng.integers(0, len(self.apps), n)] \
                .reset_index(drop=True)
        u = self.users.iloc[u_rows].reset_index(drop=True)
        if self.bn_edges.get("exposure"):
            vb = self._user_vbin[u_rows]
            c_idx = np.empty(n, dtype="int64")
            for b in range(self._n_vbin):
                mb = vb == b
                mm = int(mb.sum())
                if mm:
                    c_idx[mb] = rng.choice(len(self.campaigns), size=mm,
                                           p=self._camp_pmf[b])
        else:
            c_idx = rng.choice(len(self.campaigns), size=n,
                               p=self.campaigns["sample_weight"].to_numpy())
        c = self.campaigns.iloc[c_idx].reset_index(drop=True) \
            .drop(columns=["sample_weight"])
        return u, a, c

    def _draw_context(self, n, rng):
        """Per-auction pool joins + context columns, before any pricing."""
        u, a, c = self._join_pools(n, rng)
        df = pd.concat([u, a, c], axis=1)

        df["ad_exchange"] = rng.choice(self.exch_labels, size=n,
                                       p=self.exch_p)
        fmt = rng.choice(self.fmt_labels, size=n, p=self.fmt_p)
        df["slot_format"] = pd.Series(fmt).map(_FORMAT_CODE).to_numpy()
        df["_format"] = fmt
        size = np.empty(n, dtype=object)
        for f in self.fmt_labels:
            mask = fmt == f
            labels, p = self.sizes[f]
            size[mask] = rng.choice(labels, size=int(mask.sum()), p=p)
        wh = np.array([s.split("x") for s in size], dtype="int64")
        df["slot_width"], df["slot_height"] = wh[:, 0], wh[:, 1]
        df["_size"] = size

        # D1-D3: hour x dow shape, uniform week + second-within-hour
        if self.bn_edges.get("hour"):            # edge #5: archetype -> hour CPT
            dow = rng.choice(7, size=n, p=self._dow_marg)
            arch = df["lu1_archetype"].to_numpy()
            hour = np.empty(n, dtype="int64")
            for lab in self._arch_labels:
                mk = arch == lab
                mm = int(mk.sum())
                if mm:
                    hour[mk] = rng.choice(24, size=mm, p=self._hour_cpt[lab])
        else:
            hd = self.hourdow_vals[rng.choice(len(self.hourdow_vals), size=n,
                                              p=self.hourdow_p)]
            dow, hour = hd[:, 0], hd[:, 1]
        week = rng.integers(0, self.n_weeks, n)
        df["timestamp"] = (self.window_start
                           + ((week * 7 + dow) * 24 + hour) * 3600
                           + rng.integers(0, 3600, n))
        df["hour_of_day"], df["day_of_week"] = hour, dow
        return df

    def _truth(self, df):
        """Ground-truth funnel probabilities, E[ltv|payer], EV (no draws)."""
        bm = self.cfg.benchmarks
        w = bm["funnel"]["w_stage"]
        sq = bm["slot_quality"]

        cats = self.cfg.categories
        lu6 = df[[f"lu6_{c}" for c in cats]].to_numpy()
        gidx = pd.Series(df["ad_genre"]).map(
            {c: j for j, c in enumerate(cats)}).to_numpy()
        r = lu6[np.arange(len(df)), gidx]
        m = {s: (1 - w[s]) + w[s] * r for s in ("click", "install", "pay")}
        v_slot = (pd.Series(df["_format"]).map(sq["format_weight"]).to_numpy()
                  * pd.Series(df["_size"]).map(sq["size_weight"]).to_numpy())

        ease = pd.Series(df["app_category"]).map(self.ease).to_numpy()
        mu_cat = pd.Series(df["app_category"]).map(self.mu_cat).to_numpy()
        sigma = bm["ltv"]["lognormal_sigma"]

        p_click = np.clip(bm["funnel"]["base_ctr"] * df["lu2_click_prop"]
                          * v_slot * m["click"] * df["la1_app_quality"]
                          * df["lc1_creative_appeal"], 0, 1)
        p_install = np.clip(bm["funnel"]["base_ir"] * df["lu3_install_prop"]
                            * ease * m["install"] * df["la1_app_quality"],
                            0, 1)
        p_payer = np.clip(bm["funnel"]["base_payer"] * df["lu4_payer_prob"]
                          * m["pay"], 0, 1)
        # BN edge #4 (D2/D3 -> payer-timing): raked t_pay(hour,dow) on p_payer
        # (mean 1 over the population hour x dow joint => install->payer preserved).
        if self.bn_edges.get("payer_timing"):
            hr = df["hour_of_day"].to_numpy()
            dw = df["day_of_week"].to_numpy()
            t_pay = self._hour_mult[hr] * self._dow_mult[dw] / self._tpay_rake
            p_payer = np.clip(p_payer * t_pay, 0, 1)
        # BN edge #3 (os->spend): iOS ARPU ~rho x Android; re-centre mu_cat so the
        # LTV MEAN is preserved (median shifts ~4%). OFF => plat=1, mu unchanged.
        if self.bn_edges.get("os_spend"):
            plat = np.where(df["os"].to_numpy() == "iOS", self._rho, 1.0)
            mu_cat = mu_cat - np.log(self._eplat)
        else:
            plat = np.ones(len(df))

        e_ltv = (np.exp(mu_cat + sigma**2 / 2) * df["lu5_ltv_mult"]
                 * df["lc2_game_quality"] * plat)
        ev = p_click * p_install * p_payer * e_ltv

        # v8 EV_market: the genre-MARGINAL, supply+user EV that drives LU7 instead
        # of the focal ev_truth. r_mkt marginalises the focal ad_genre (C4); the
        # focal campaign latents lc1/lc2 are dropped (C3) -> C1-C4 OUT of LU7. The
        # z-standardisation (calibrate) makes the absolute scale irrelevant.
        if self.bn_edges.get("ev_market"):
            r_mkt = lu6 @ self._genre_w
            mm = {s: (1 - w[s]) + w[s] * r_mkt for s in ("click", "install", "pay")}
            pc_m = np.clip(bm["funnel"]["base_ctr"] * df["lu2_click_prop"]
                           * v_slot * mm["click"] * df["la1_app_quality"], 0, 1)
            pi_m = np.clip(bm["funnel"]["base_ir"] * df["lu3_install_prop"]
                           * ease * mm["install"] * df["la1_app_quality"], 0, 1)
            pp_m = np.clip(bm["funnel"]["base_payer"] * df["lu4_payer_prob"]
                           * mm["pay"], 0, 1)
            eltv_m = np.exp(mu_cat + sigma**2 / 2) * df["lu5_ltv_mult"] * plat
            df["_ev_lu7"] = (pc_m * pi_m * pp_m * eltv_m).to_numpy()

        df["p_click"], df["p_install"] = p_click, p_install
        df["p_payer"], df["e_ltv"], df["ev_truth"] = p_payer, e_ltv, ev
        df["_mu_cat"], df["_v_slot"], df["_ease"] = mu_cat, v_slot, ease
        df["_plat"] = plat
        return df

    def _prices(self, df, rng):
        """H1 floor, LU7 competing bid, H2 our bid (uses warm-up stats)."""
        au = self.cfg.benchmarks["auction"]
        n = len(df)
        target = pd.Series(df["_format"]).map(self.ecpm_target).to_numpy()

        fv, fp = self.floor_shape
        df["floor_price"] = fv[rng.choice(len(fv), size=n, p=fp)] * target

        pv, pp = self.pay_shape
        base_e = pv[rng.choice(len(pv), size=n, p=pp)] * target
        base_e = np.maximum(base_e, 0.01 * target)   # degenerate-zero guard

        ev = (df["_ev_lu7"] if self.bn_edges.get("ev_market")
              else df["ev_truth"]).to_numpy()
        z = np.zeros(n)
        pos = ev > 0
        z[pos] = (np.log(ev[pos]) - self.z_mu) / self.z_sigma

        # M_price: deterministic mean-1-raked LU7 multipliers (v8 #7-#10), each
        # flag-gated; all-OFF => ones => identity (x*1.0 is exact in IEEE-754, and
        # base_e * ones * exp(.) keeps v7's grouping). No RNG consumed (deterministic).
        M_price = np.ones(n)
        if self.bn_edges.get("app_quality_price"):           # #7
            la1 = df["la1_app_quality"].to_numpy()
            M_price = M_price * (la1 ** self._kappa_q) / self._m_app_rake
        if self.bn_edges.get("exchange_price"):              # #8
            emap = {b: i for i, b in enumerate(self.exch_labels)}
            aidx = pd.Series(df["ad_exchange"]).map(emap).to_numpy()
            M_price = M_price * self._a_price[aidx] / self._m_exch_rake
        if self.bn_edges.get("app_price"):                   # #9
            cat = df["app_category"].to_numpy()
            M_price = M_price * np.array([self._cat_price[c] for c in cat]) / self._m_cat_rake
        if self.bn_edges.get("time_competition"):            # #10
            hr = df["hour_of_day"].to_numpy()
            dw = df["day_of_week"].to_numpy()
            M_price = M_price * (self._hour_price[hr] * self._dow_price[dw]) / self._ctime_rake

        if self.bn_edges.get("rival_pool"):        # v10: Private-Rival Market
            # LU7 = max over K participating private rivals (spawned rng inside ->
            # main stream untouched there). Consume+discard the v7 gaming+nongaming
            # normals so bid_price AND the whole funnel/LTV realisation stay IDENTICAL
            # to v7 (ONLY LU7 differs) -> a clean controlled v7-vs-v10 comparison.
            rng.normal(0, au["sigma_g"], n)
            rng.normal(au["mu_x"], au["sigma_x"], n)
            lu7, n_riv = self._rival_pool.competing_bids(
                df, base_e, z, rng, self.rho_rival)
            df["lu7_competing_bid"] = lu7
            df["bid_density"] = n_riv               # H9 (endogenous competition density)
        elif self.bn_edges.get("rival_hidden"):    # reason-1c: hidden rival factor
            # spawn (does NOT consume main stream) -> segment assignment; then the
            # main rng draws exactly 2 standard-normal blocks (gaming, nongaming) in
            # the SAME order as the v7 path, so rho=0 => byte-identical v7 LU7.
            rrng = np.random.default_rng(rng.bit_generator._seed_seq.spawn(1)[0])
            seg = rrng.integers(0, self._n_rival_seg, n)
            df["rival_seg"] = seg.astype("int64")   # SSP-exposed (C3/C4-only feature)
            Rseg = self._rival_R[seg]               # common per-segment rival demand
            if self.bn_edges.get("rival_user_corr"):
                # pilot-2 (Ken's remedy ingredient): the hidden factor is the rival
                # group's PRIVATE VALUATION OF THIS USER = group loading x latent user
                # value z. Correlated with the user, invisible to DSP features (z is
                # latent). Var(R*z)~=1 on ev>0 rows; z==0 rows lose the hidden share
                # (pilot caveat - not re-raked).
                Rseg = Rseg * z
            a = self.rho_rival; b = float(np.sqrt(max(1.0 - a * a, 0.0)))
            eg = au["sigma_g"] * (a * Rseg + b * rng.standard_normal(n))
            gaming = np.where(pos, base_e * M_price * np.exp(
                au["gamma"] * z + eg), 0.0)
            ex = au["mu_x"] + au["sigma_x"] * (a * Rseg + b * rng.standard_normal(n))
            nongaming = base_e * np.exp(ex)
        elif self.bn_edges.get("competition"):     # v8 #6: N_t order statistic
            from scipy.stats import norm
            # consume+discard the v7 gaming normal so the MAIN rng stays aligned
            # with the v7/OFF stream: nongaming, bid_price AND the downstream funnel
            # realisation stay IDENTICAL across OFF/ON -> a CONTROLLED v7-vs-v8
            # comparison where ONLY LU7's structure differs (no RNG-realignment confound).
            rng.normal(0, au["sigma_g"], n)
            # chunk-distinct, reproducible competition stream spawned off the passed
            # (already chunk-keyed "auctions-{idx}") rng. spawn() does NOT consume the
            # main stream -> OFF stays byte-identical AND ON is chunk_size-invariant
            # (a fresh cfg.rng("competition") per chunk would tile the same draws).
            crng = np.random.default_rng(rng.bit_generator._seed_seq.spawn(1)[0])
            Nt = np.minimum(1 + crng.poisson(self._lamN, n), self._comp_nmax)
            u = crng.random(n)
            u = u * (1.0 - 2.0**-53) + 2.0**-53    # guard u==0 -> ppf(-inf)
            maxeps = au["sigma_g"] * norm.ppf(u ** (1.0 / Nt))   # max of N_t iid eps
            gaming = np.where(pos, base_e * M_price * np.exp(
                au["gamma"] * z + maxeps) * self._comp_rake, 0.0)
            df["bid_density"] = Nt.astype("int64")             # H5 (emitted iff ON)
            nongaming = base_e * np.exp(rng.normal(au["mu_x"], au["sigma_x"], n))
        else:                                       # v7 path (single gaming draw)
            gaming = np.where(pos, base_e * M_price * np.exp(
                au["gamma"] * z + rng.normal(0, au["sigma_g"], n)), 0.0)
            nongaming = base_e * np.exp(rng.normal(au["mu_x"], au["sigma_x"], n))
        if not self.bn_edges.get("rival_pool"):     # rival_pool sets LU7 in-branch
            df["lu7_competing_bid"] = np.maximum(gaming, nongaming)

        k_cpa = pd.Series(df["advertiser_scale"]).map(self.k_cpa).to_numpy()
        eltv = pd.Series(df["app_category"]).map(self.eltv_b2).to_numpy()
        df["bid_price"] = (self.k_global * k_cpa * df["_v_slot"] * df["_ease"]
                           * eltv * au["shade"]
                           * rng.lognormal(0, au["sigma_explore"], n))
        if self.bn_edges.get("explore_traffic"):   # v11 B2: exploration slice
            # spawned rng AFTER all other spawns in _prices -> OFF byte-identical;
            # ON changes only H2 on the slice (LU7/funnel draws untouched)
            ex_cfg = self.cfg.benchmarks.get("explore_traffic") or {}
            share = float(ex_cfg.get("share", 0.05))
            sigma_w = float(ex_cfg.get("sigma_wide", 0.8))
            er = np.random.default_rng(rng.bit_generator._seed_seq.spawn(1)[0])
            ex = er.random(n) < share
            df["bid_price"] = df["bid_price"] * np.where(
                ex, er.lognormal(0.0, sigma_w, n), 1.0)
        return df

    def _settle_and_funnel(self, df, rng):
        """H3 won, piecewise H4, funnel outcomes on ALL rows (the master is
        uncensored; won/lost visibility is censor.py's job)."""
        bm = self.cfg.benchmarks
        h2 = df["bid_price"].to_numpy()
        lu7 = df["lu7_competing_bid"].to_numpy()
        h1 = df["floor_price"].to_numpy()
        n = len(df)

        won = h2 >= np.maximum(lu7, h1)
        sold_lost = ~won & (lu7 >= h1)
        df["won"] = won.astype("int8")
        h4 = np.full(n, np.nan)
        h4[won] = h2[won]
        h4[sold_lost] = lu7[sold_lost]            # NaN where unsold
        df["clearing_price"] = h4

        if self.bn_edges.get("min_bid_to_win"):   # v11 B1: H10 (no RNG -> OFF byte-identical)
            h10 = np.full(n, np.nan)
            h10[won] = np.maximum(lu7[won], h1[won])   # price needed to win, revealed on wins
            df["min_bid_to_win"] = h10

        # funnel on EVERY row - lost rows are the declared counterfactual
        click = rng.random(n) < df["p_click"].to_numpy()
        install = click & (rng.random(n) < df["p_install"].to_numpy())
        payer = install & (rng.random(n) < df["p_payer"].to_numpy())

        sigma = bm["ltv"]["lognormal_sigma"]
        spend = np.where(payer, rng.lognormal(df["_mu_cat"], sigma)
                         * df["lu5_ltv_mult"] * df["lc2_game_quality"]
                         * df["_plat"], 0.0)
        # (v11 B3 mmp_txcount REMOVED 9 Jul 2026 — an LU5->count coupling would be a
        #  designed-in, unanchored signal; transaction counts belong to the Track-C
        #  IAP bundle process where they arise organically. See schema v10 doc.)

        ts = df["timestamp"].to_numpy()
        e2 = np.where(click, np.floor(ts + rng.exponential(
            bm["funnel"]["click_delay_mean_s"], n)), -1.0)
        f2 = np.where(install, np.floor(e2 + rng.lognormal(
            bm["funnel"]["install_delay"]["mu"],
            bm["funnel"]["install_delay"]["sigma"], n)), -1.0)

        df["click"] = click.astype("int8")
        df["click_timestamp"] = e2
        df["install"] = install.astype("int8")
        df["install_timestamp"] = f2
        df["is_payer"] = payer.astype("int8")
        df["ltv_value"] = spend
        df["ltv_7d"] = spend * bm["ltv"]["decay_d7"]
        df["ltv_30d"] = spend * bm["ltv"]["decay_d30"]
        return df

    def funnel_sample(self, n, rng):
        """Funnel-only fast path for the autocal loop: context + truth +
        outcome draws, skipping the auction (in v6 the funnel never reads
        won/prices, so this is exact)."""
        df = self._draw_context(n, rng)
        df = self._truth(df)
        click = rng.random(n) < df["p_click"].to_numpy()
        install = click & (rng.random(n) < df["p_install"].to_numpy())
        payer = install & (rng.random(n) < df["p_payer"].to_numpy())
        sigma = self.cfg.benchmarks["ltv"]["lognormal_sigma"]
        spend = np.where(payer, rng.lognormal(df["_mu_cat"], sigma)
                         * df["lu5_ltv_mult"] * df["lc2_game_quality"]
                         * df["_plat"], 0.0)
        return pd.DataFrame({"click": click, "install": install,
                             "payer": payer, "spend": spend})

    # ────────────────────────────────────────────────────────────────
    def chunk(self, n, rng):
        """One fully-settled chunk of n auctions as a DataFrame."""
        df = self._draw_context(n, rng)
        df = self._truth(df)
        df = self._prices(df, rng)
        df = self._settle_and_funnel(df, rng)
        return df.drop(columns=[c for c in df.columns if c.startswith("_")])

    def calibrate(self):
        """Warm-up: fix z-stats of log EV, then bisect k_global to the
        win-rate target (calibrated pinning). Returns a summary dict."""
        n = int(self.cfg.win_rate["warmup_auctions"])
        target = float(self.cfg.win_rate["target"])
        rng = self.cfg.rng("warmup")

        df = self._draw_context(n, rng)
        df = self._truth(df)
        evcol = "_ev_lu7" if self.bn_edges.get("ev_market") else "ev_truth"
        ev = df[evcol].to_numpy()
        logev = np.log(ev[ev > 0])
        self.z_mu = float(logev.mean())
        self.z_sigma = float(max(logev.std(), 1e-6))

        self.k_global = 1.0
        df = self._prices(df, rng)                # H2 at k_global = 1
        h2_base = df["bid_price"].to_numpy()
        hurdle = np.maximum(df["lu7_competing_bid"].to_numpy(),
                            df["floor_price"].to_numpy())

        lo, hi = 1e-4, 1e4
        for _ in range(60):
            k = np.sqrt(lo * hi)
            if float(np.mean(k * h2_base >= hurdle)) > target:
                hi = k
            else:
                lo = k
        self.k_global = float(np.sqrt(lo * hi))
        achieved = float(np.mean(self.k_global * h2_base >= hurdle))
        return {"z_mu": self.z_mu, "z_sigma": self.z_sigma,
                "k_global": self.k_global, "warmup_win_rate": achieved}
