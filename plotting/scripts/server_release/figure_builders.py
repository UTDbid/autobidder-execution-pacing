from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

from plot_style import *


BIDDERS = ["cql", "iql", "dt", "gas", "sembid"]
BIDDER_LABEL = {"cql": "CQL", "iql": "IQL", "dt": "DT", "gas": "GAS", "sembid": "SemBid", "pid": "PID bidder", "constant5": "Constant5"}
REFS = ["traffic_aware", "pid", "dual"]
REFS_MACRO = ["traffic", "pid", "dual"]
WIDTH_X = {"0": 0.0, "0.3": 0.3, "0.5": 0.5, "0.8": 0.8, "1.0": 1.0, "1.2": 1.2, "Raw": 1.48}
EC_BAR_LIGHT_BLUE = "#D9E9F5"
EC_BAR_DARK_BLUE = "#6E96BB"


def rd(path: str) -> pd.DataFrame:
    return pd.read_csv(DERIVED / path)


def negative_cell_lines(ax, row: int, col: int, count: int = 4,
                        linewidth: float = 0.75) -> None:
    """Distinguish negative heatmap cells in grayscale print."""
    # Reserve the center of each cell for its numeric label.
    offsets = [-0.36, -0.27, 0.27, 0.36] if count == 4 else np.linspace(-0.36, 0.36, count)
    for y in row + np.asarray(offsets):
        ax.plot([col - 0.46, col + 0.46], [y, y], color="#858585",
                linewidth=linewidth, alpha=0.82, solid_capstyle="butt", zorder=3)


def negative_cell_legend(fig) -> None:
    """Explain the grayscale-safe sign cue used by the heatmaps."""
    fig.legend(
        handles=[Patch(facecolor="#F0F3F6", edgecolor="#858585", hatch="---",
                       label="Negative cell (gray horizontal lines)")],
        loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False,
        fontsize=9.3, handlelength=1.8,
    )


def main_fig1() -> None:
    configure()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    def box(x, y, w, h, text, face, edge="#71808D", fs=8.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor=face, edgecolor=edge, linewidth=0.9)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=MUTED))

    box(0.35,4.6,3.0,1.15,"Campaign pacing reference\nTraffic-aware · PID · Dual","#E8F0F8")
    box(8.65,4.6,3.0,1.15,"Autobidder\nCQL · IQL · DT · GAS · SemBid","#E6F2EE")
    box(4.55,4.35,2.9,1.65,"Execution boundary  $\\kappa$\npermissible bidder deviation","#F7EBDD",edge="#B97A35",fs=8.5)
    arrow(3.35,5.15,4.55,5.15); arrow(8.65,5.15,7.45,5.15)
    box(4.7,2.7,2.6,0.9,"Executed auction action","#F3F5F6")
    arrow(6.0,4.35,6.0,3.62)
    for x, label, face in [(0.45,"Purchase set\n& advertiser value","#E8F0F8"),(3.45,"Budget delivery\n& CPA","#F7EBDD"),(6.45,"Competitive bids\n& clearing prices","#E6F2EE"),(9.45,"Current platform\nauction revenue","#F3E9EF")]:
        box(x,0.75,2.1,1.05,label,face,fs=7.5); arrow(6.0,2.7,x+1.05,1.8)
    ax.text(0.4,0.15,"AuctionNet: campaign and market evidence",fontsize=6.8,color=MUTED)
    ax.text(4.45,0.15,"T9Sim: truth-level purchase reallocation",fontsize=6.8,color=MUTED)
    ax.text(8.55,0.15,"iPinYou: logged budget-pressure boundary",fontsize=6.8,color=MUTED)
    save(fig,"fig1_operating_architecture","main")


def main_fig2() -> None:
    configure(); f=rd("main/figure2a_campaign_frontier.csv"); t=rd("main/figure2b_temporal_scatter.csv")
    fig, (a,b)=plt.subplots(1,2,figsize=(7.2,3.05),gridspec_kw={"width_ratios":[1.25,1]}); fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.88,wspace=.31)
    for ref in REFS:
        d=f[f.reference.eq(ref)].copy(); d["xord"]=d.width.map(WIDTH_X); d=d.sort_values("xord")
        a.errorbar(d.mean_underdelivery_delta_pp,d.mean_relative_value_delta_pct,xerr=d.sd_underdelivery_delta_pp,yerr=d.sd_relative_value_delta_pct,fmt="none",ecolor=REF_COLORS[ref],elinewidth=.5,capsize=1.6,alpha=.28)
        a.plot(d.mean_underdelivery_delta_pp,d.mean_relative_value_delta_pct,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
        kink=d[d.width.eq("1.0")]
        a.scatter(kink.mean_underdelivery_delta_pp,kink.mean_relative_value_delta_pct,s=25,facecolor=REF_COLORS[ref],edgecolor="white",linewidth=.5,zorder=5)
    a.axhline(0,color=INK,lw=.6);a.axvline(0,color=INK,lw=.6);a.set_xlabel("Change in underdelivery (percentage points)");a.set_ylabel("Relative advertiser value change (%)");clean(a);panel(a,"a");a.legend(loc="upper left")
    bidder_markers={"cql":"o","iql":"s","dt":"^","gas":"D","sembid":"P"}
    for ref in REFS:
        d=t[t.reference.eq(ref)]
        for r in d.itertuples(): b.scatter(r.underdelivery_delta_pp,r.value_delta/1000,color=REF_COLORS[ref],marker=bidder_markers[r.bidder],facecolors="white",s=24)
    b.axhline(0,color=INK,lw=.6);b.axvline(0,color=INK,lw=.6);b.set_xlabel("Change in underdelivery (percentage points)");b.set_ylabel(r"Advertiser value change ($\times 10^3$)");clean(b);panel(b,"b")
    b.legend(handles=[plt.Line2D([],[],color=INK,marker=bidder_markers[x],mfc="white",ls="none",label=BIDDER_LABEL[x]) for x in BIDDERS],loc="lower right",ncol=2,columnspacing=.7,handletextpad=.3)
    save(fig,"fig2_campaign_frontier","main")


def main_fig3() -> None:
    configure(); u=rd("main/figure3ab_uniform_width.csv"); r=rd("main/figure3c_rollout_paths.csv"); adoption=rd("main/figure3d_adoption_frontier.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.55));fig.subplots_adjust(left=.085,right=.985,bottom=.10,top=.95,wspace=.28,hspace=.34)
    a,b,c,d=axes.flat
    for ref in REFS_MACRO:
        q=u[u.reference.eq(ref)].copy();q["x"]=q.width.map(WIDTH_X);q=q.sort_values("x")
        a.plot(q.x,q.mean_relative_total_base_expected_value_pct,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
        b.plot(q.x,q.mean_relative_platform_revenue_pct,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white")
    for ax,ylabel in [(a,"Advertiser value change (%)"),(b,"Current auction revenue change (%)")]:
        ax.axhline(0,color=INK,lw=.6);ax.axvline(1.0,color=MUTED,lw=.7,ls=":");ax.set_xticks([0,.3,.5,.8,1,1.2,1.48]);ax.set_xticklabels(["0.0","0.3","0.5","0.8","1.0","1.2","Raw"]);ax.set_xlabel("Uniform market execution width");ax.set_ylabel(ylabel);clean(ax)
    panel(a,"a");panel(b,"b");a.legend(loc="best")
    path_mark={"0_to_0.3":"o","0.3_to_0.5":"s","0.3_to_0.8":"D","0_to_0.8":"^"}; path_lab={"0_to_0.3":"0.0→0.3","0.3_to_0.5":"0.3→0.5","0.3_to_0.8":"0.3→0.8","0_to_0.8":"0.0→0.8"}
    q=r[np.isclose(r.alpha,.5)]
    for path,m in path_mark.items():
        z=q[q.rollout_path.eq(path)]
        for rr in REFS_MACRO:
            x=z[z.reference.eq(rr)]
            c.scatter(x.mean_relative_platform_revenue_pct,x.mean_relative_total_base_expected_value_pct,color=REF_COLORS[rr],marker=m,s=32,edgecolor="white",linewidth=.4)
    c.axhline(0,color=INK,lw=.6);c.axvline(0,color=INK,lw=.6);c.set_xlabel("Current auction revenue change (%)");c.set_ylabel("Advertiser value change (%)");clean(c);panel(c,"c")
    handles=[plt.Line2D([],[],color=INK,marker=m,ls="none",label=path_lab[p]) for p,m in path_mark.items()]
    c.legend(handles=handles,ncol=2,loc="best",handletextpad=.3,columnspacing=.8)
    for ref in REFS_MACRO:
        z=adoption[adoption.reference.eq(ref)].sort_values("alpha");x=np.r_[0,z.revenue_delta_pct];y=np.r_[0,z.advertiser_value_delta_pct];d.plot(x,y,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
        for rr in z.itertuples():
            if np.isclose(rr.alpha,1/48) or np.isclose(rr.alpha,.5) or np.isclose(rr.alpha,1): d.annotate("1/48" if np.isclose(rr.alpha,1/48) else f"{rr.alpha:.0%}",(rr.revenue_delta_pct,rr.advertiser_value_delta_pct),xytext=(3,3),textcoords="offset points",fontsize=5.7,color=REF_COLORS[ref])
    d.axhline(0,color=INK,lw=.6);d.axvline(0,color=INK,lw=.6);d.set_xlabel("Current auction revenue change (%)");d.set_ylabel("Advertiser value change (%)");clean(d);panel(d,"d");d.legend(loc="best")
    save(fig,"fig3_market_width_and_rollout","main")


def main_fig4() -> None:
    configure(); direction=rd("main/figure4a_directional.csv"); tv=rd("main/figure4b_t9_true_value.csv"); eff=rd("main/figure4c_t9_purchase_efficiency.csv")
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.9));fig.subplots_adjust(left=.075,right=.99,bottom=.20,top=.88,wspace=.34);a,b,c=axes
    conds=["down_only","up_only","symmetric"];x=np.arange(3);w=.23
    for i,ref in enumerate(REFS):
        q=direction[direction.reference.replace({"traffic":"traffic_aware"}).eq(ref)].set_index("condition").reindex(conds)
        a.bar(x+(i-1)*w,q.value_delta/1000,width=w,color=REF_COLORS[ref],alpha=.88,label=REF_LABELS[ref])
    a.axhline(0,color=INK,lw=.6);a.set_xticks(x);a.set_xticklabels(["Down only","Up only","Symmetric"],rotation=20,ha="right");a.set_ylabel(r"Advertiser value change ($\times 10^3$)");clean(a);panel(a,"a")
    views=["C1","C2","C3","C4"]
    for ax,data,col,ylabel in [(b,tv,"true_value_delta",r"True expected-value change ($\times 10^3$)"),(c,eff,"exclusive_efficiency_delta","Wide-only minus tight-only\ntrue efficiency")]:
        for ref in REFS:
            q=data[data.reference.eq(ref)].set_index("information_view").reindex(views);y=q[col]/1000 if col=="true_value_delta" else q[col]
            ax.plot(views,y,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
        ax.axhline(0,color=INK,lw=.6);ax.set_xlabel("Information view");ax.set_ylabel(ylabel);clean(ax)
    panel(b,"b");panel(c,"c");b.legend(loc="best")
    save(fig,"fig4_purchase_reallocation","main")


def main_fig5() -> None:
    configure(); s=rd("main/figure5a_auctionnet_supply.csv"); i=rd("main/figure5b_ipinyou_pressure.csv")
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0));fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.88,wspace=.30)
    order=["500","2000","8000","full"]
    for ref in REFS:
        q=s[s.reference.eq(ref)].copy();q["ord"]=q.opportunity_supply.astype(str).map({v:k for k,v in enumerate(order)});q=q.sort_values("ord")
        a.plot(range(4),q.value_delta/1000,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
    a.axhline(0,color=INK,lw=.6);a.set_xticks(range(4));a.set_xticklabels(["500","2,000","8,000","Full"]);a.set_xlabel("Available opportunities");a.set_ylabel(r"Advertiser value change ($\times 10^3$)");clean(a);panel(a,"a");a.legend(loc="best")
    for ref in REFS:
        q=i[i.reference.eq(ref)].sort_values("budget_fraction");b.plot(100*q.budget_fraction,q.clicks_delta,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
    b.axhline(0,color=INK,lw=.6);b.set_xscale("log",base=2);b.set_xticks([3.125,12.5,50]);b.set_xticklabels(["3.125%","12.5%","50%"]);b.set_xlabel("Budget / logged spend");b.set_ylabel("Mean click change per campaign");clean(b);panel(b,"b")
    save(fig,"fig5_room_for_selection","main")


def ec_r1() -> None:
    configure()
    f = rd("ec/figure_r1_full_bidder_frontiers.csv")
    widths = ["0.0", "0.3", "0.5", "0.8", "1.0", "1.2", "Raw"]
    for bidder in BIDDERS:
        for ref in REFS:
            observed = set(f.loc[f.bidder.eq(bidder) & f.reference.eq(ref), "width"])
            assert observed == set(widths), (bidder, ref, observed)
    assert len(f) == len(BIDDERS) * len(REFS) * len(widths)

    fig, axes = plt.subplots(5, 3, figsize=(7.5, 9.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.21, right=.985, bottom=.10, top=.90, wspace=.30, hspace=.40)
    color = "#356EA8"
    markers = {
        "0.0": ("o", 3.4, "#D8E1E9"),
        "0.3": ("o", 4.0, "white"),
        "0.5": ("s", 4.0, "white"),
        "0.8": ("o", 6.2, color),
        "1.0": ("^", 4.9, "white"),
        "1.2": ("v", 4.9, "white"),
        "Raw": ("D", 4.7, color),
    }
    for row, bidder in enumerate(BIDDERS):
        for col, ref in enumerate(REFS):
            ax = axes[row, col]
            q = f.loc[f.bidder.eq(bidder) & f.reference.eq(ref)].set_index("width").reindex(widths).dropna(subset=["relative_value_delta_pct"])
            ax.plot(q.underdelivery_delta_pp, q.relative_value_delta_pct,
                    color=color, linewidth=1.2, zorder=2)
            for width, point in q.iterrows():
                marker, size, face = markers[width]
                ax.plot(point.underdelivery_delta_pp, point.relative_value_delta_pct,
                        marker=marker, markersize=size, markerfacecolor=face,
                        markeredgecolor=color, markeredgewidth=.9,
                        linestyle="none", zorder=3)
            ax.set_xlim(-2, 46)
            ax.set_ylim(-5, 72)
            ax.set_xticks([0, 20, 40])
            ax.set_yticks([0, 20, 40, 60])
            ax.axhline(0, color=MUTED, linewidth=.55)
            ax.axvline(0, color=MUTED, linewidth=.55)
            ax.tick_params(axis="both", labelsize=8.1, length=2, pad=2)
            clean(ax)
            if row == 0:
                ax.set_title(REF_LABELS[ref], fontsize=10.2, pad=7)
        fig.text(.12, (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1)/2,
                 BIDDER_LABEL[bidder], ha="right", va="center", fontsize=10.5,
                 fontweight="semibold", color=INK)
    fig.supxlabel("Δ underdelivery (pp)", y=.045, fontsize=10.5)
    fig.supylabel("Δ advertiser value (%)", x=.025, fontsize=10.5)
    labels = {"0.0": "Base", "0.3": "0.3", "0.5": "0.5",
              "0.8": "Wide (0.8)", "1.0": "1.0", "1.2": "1.2", "Raw": "Raw"}
    handles = [plt.Line2D([], [], color=color, linestyle="none", marker=markers[w][0],
                          markersize=markers[w][1] + 1, markerfacecolor=markers[w][2],
                          markeredgecolor=color, markeredgewidth=.9, label=labels[w])
               for w in widths]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(.60, .975), fontsize=8.7,
               columnspacing=.85, handletextpad=.25)
    save(fig,"fig_r1_full_bidder_frontiers","ec")


def ec_r2() -> None:
    configure();plt.rcParams.update({"xtick.labelsize":8.8,"ytick.labelsize":8.8,"axes.labelsize":9.0});d=rd("ec/figure_r2_architecture_heatmap.csv");p=d.pivot(index="bidder",columns="reference",values="value_delta")/1000
    rows=[x for x in BIDDERS+["pid","constant5"] if x in p.index];cols=[x for x in REFS+["smoothed_controller","response_aware"] if x in p.columns];p=p.reindex(index=rows,columns=cols)
    fig,ax=plt.subplots(figsize=(7.2,3.8));lim=np.nanmax(np.abs(p.values));im=ax.imshow(p,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    ax.set_xticks(range(len(cols)));ax.set_xticklabels([REF_LABELS[x] for x in cols],rotation=25,ha="right");ax.set_yticks(range(len(rows)));ax.set_yticklabels([BIDDER_LABEL[x] for x in rows])
    for i in range(len(rows)):
        for j in range(len(cols)):
            v=p.iloc[i,j]
            if v < 0: negative_cell_lines(ax,i,j)
            ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=10.2,
                    color="white" if abs(v)>.58*lim else INK,zorder=4)
    ax.axhline(4.5,color="white",lw=2);ax.axvline(2.5,color="white",lw=2);c=fig.colorbar(im,ax=ax,pad=.02);c.set_label(r"Value change ($\times 10^3$)");c.ax.tick_params(labelsize=8.5)
    negative_cell_legend(fig)
    save(fig,"fig_r2_reference_bidder_heatmap","ec")


def ec_r3() -> None:
    configure();plt.rcParams["hatch.linewidth"] = .65;d=rd("ec/figure_r3_directional.csv");refs=["traffic","pid","dual","smoothed","response"];conds=["down_only","up_only","symmetric"]
    assert set(d.reference) == set(refs)
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.2));fig.subplots_adjust(left=.10,right=.965,bottom=.22,top=.80,wspace=.26);x=np.arange(len(refs));w=.24
    condition_colors=[EC_BAR_LIGHT_BLUE,EC_BAR_LIGHT_BLUE,EC_BAR_DARK_BLUE]
    condition_hatches=["////","||||","----"]
    for j,cnd in enumerate(conds):
        q=d[d.condition.eq(cnd)].set_index("reference").reindex(refs)
        style=dict(color=condition_colors[j],hatch=condition_hatches[j],edgecolor=INK,linewidth=.35)
        a.bar(x+(j-1)*w,q.value_delta/1000,w,label=cnd.replace("_"," "),**style)
        b.bar(x+(j-1)*w,q.underdelivery_delta_pp,w,label=cnd.replace("_"," "),**style)
    for ax,y in [(a,r"Δ value ($\times 10^3$)"),(b,"Δ underdelivery (pp)")]:ax.axhline(0,color=INK,lw=.6);ax.set_xticks(x);ax.set_xticklabels([REF_LABELS.get(v,v) for v in refs],rotation=25,ha="right");ax.set_ylabel(y);clean(ax)
    panel(a,"a","Advertiser value");panel(b,"b","Underdelivery");handles,labels=a.get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.99));save(fig,"fig_r3_directional_decomposition","ec")


def ec_r4() -> None:
    configure();plt.rcParams.update({"xtick.labelsize":11.0,"ytick.labelsize":11.0});d=rd("ec/figure_r4_opportunity_slack.csv");fig,axes=plt.subplots(1,3,figsize=(7.4,3.1),sharey=True);fig.subplots_adjust(left=.085,right=.985,bottom=.22,top=.78,wspace=.22);sup=["500","2000","8000","full"];bud=[.5,1,1.5]
    for idx,(ax,ref) in enumerate(zip(axes,REFS)):
        q=d[d.reference.eq(ref)].copy();q.opportunity_supply=q.opportunity_supply.astype(str);p=q.pivot(index="budget_multiplier",columns="opportunity_supply",values="value_delta").reindex(index=bud,columns=sup)/1000;lim=max(1,np.nanmax(np.abs(p.values)));im=ax.imshow(p,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
        ax.set_xticks(range(4));ax.set_xticklabels(["500","2k","8k","Full"]);ax.set_yticks(range(3));ax.set_yticklabels(["0.5","1.0","1.5"]);ax.set_xlabel("Opportunities");panel(ax,chr(ord("a")+idx),REF_LABELS[ref])
        for i in range(3):
            for j in range(4):
                value=p.iloc[i,j]
                if value < 0: negative_cell_lines(ax,i,j)
                ax.text(j,i,f"{value:+.1f}",ha="center",va="center",fontsize=10.3,
                        color="white" if abs(value)>.58*lim else INK,zorder=4)
    axes[0].set_ylabel("Budget multiplier");negative_cell_legend(fig);save(fig,"fig_r4_opportunity_slack","ec")


def ec_r5() -> None:
    configure();plt.rcParams["hatch.linewidth"] = .65;d=rd("ec/figure_r5_sparse_feedback.csv");models=[m for m in BIDDERS if m in set(d.model)];fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.3));fig.subplots_adjust(left=.10,right=.965,bottom=.22,top=.77,wspace=.26);x=np.arange(len(models));w=.18
    for i,(ref,block) in enumerate([("pid","sparse_block_1"),("pid","sparse_block_2"),("dual","sparse_block_1"),("dual","sparse_block_2")]):
        q=d[(d.reference.eq(ref))&(d.temporal_block.eq(block))].set_index("model").reindex(models)
        style=dict(color=EC_BAR_LIGHT_BLUE if ref=="pid" else EC_BAR_DARK_BLUE,
                   hatch="---" if block.endswith("2") else "",
                   edgecolor=INK,linewidth=.35)
        a.bar(x+(i-1.5)*w,q.mean_delta_total_expected_value/1000,w,
              label=f"{REF_LABELS[ref]} · {block[-1]}",**style)
        b.bar(x+(i-1.5)*w,100*q.mean_delta_underdelivery,w,**style)
    for ax,y in [(a,r"Δ value ($\times 10^3$)"),(b,"Δ underdelivery (pp)")]:ax.axhline(0,color=INK,lw=.6);ax.set_xticks(x);ax.set_xticklabels([BIDDER_LABEL[v] for v in models],rotation=20,ha="right");ax.set_ylabel(y);clean(ax)
    panel(a,"a","Advertiser value");panel(b,"b","Underdelivery");handles,labels=a.get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.5,.99));save(fig,"fig_r5_sparse_feedback","ec")


def ec_r6() -> None:
    configure()
    d = rd("ec/figure_r6_complete_rollouts.csv")
    paths = ["0_to_0.3", "0.3_to_0.5", "0.3_to_0.8", "0_to_0.8"]
    labels = {"0_to_0.3": "0.0→0.3", "0.3_to_0.5": "0.3→0.5",
              "0.3_to_0.8": "0.3→0.8", "0_to_0.8": "0.0→0.8"}
    measures = ["mean_relative_total_base_expected_value_pct",
                "mean_relative_platform_revenue_pct"]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.8), sharex="col", sharey="row")
    fig.subplots_adjust(left=.115, right=.985, bottom=.14, top=.80,
                        wspace=.16, hspace=.36)
    for row, measure in enumerate(measures):
        for col, path in enumerate(paths):
            ax = axes[row, col]
            for ref in REFS_MACRO:
                q = d[(d.rollout_path.eq(path)) & (d.reference.eq(ref))].sort_values("alpha")
                ax.plot(100 * q.alpha, q[measure], color=REF_COLORS[ref],
                        ls=REF_STYLES[ref], marker=REF_MARKERS[ref],
                        mfc="white", lw=1.35, ms=4.2)
            ax.axhline(0, color=INK, lw=.65)
            ax.set_xlim(20, 105)
            ax.set_xticks([25, 50, 75, 100])
            if row == 1:
                ax.set_xlabel("Adoption rate (%)", fontsize=9.2)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            clean(ax)
            panel(ax, chr(ord("a") + row * 4 + col))
    for col, path in enumerate(paths):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .89, labels[path],
                 ha="center", va="bottom", fontsize=10.6, color=INK)
    for ax in axes[0]:
        ax.set_ylim(0, 250)
        ax.set_yticks([0, 50, 100, 150, 200, 250])
    for ax in axes[1]:
        ax.set_ylim(-40, 5)
        ax.set_yticks([-40, -30, -20, -10, 0])
    axes[0, 0].set_ylabel("Advertiser value\nchange (%)")
    axes[1, 0].set_ylabel("Current auction revenue\nchange (%)")
    refs = [plt.Line2D([], [], color=REF_COLORS[ref], marker=REF_MARKERS[ref],
                       ls=REF_STYLES[ref], mfc="white", label=REF_LABELS[ref])
            for ref in REFS_MACRO]
    fig.legend(handles=refs, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .99))
    save(fig,"fig_r6_complete_market_rollouts","ec")


def ec_r7() -> None:
    # ORIGINAL: ...;fig,ax=plt.subplots(figsize=(5.5,3.7));...
    # FIX (2026-09-28): taller figure (height 3.7 -> 4.4) so the rotated y-label's "×10³" superscript
    #   (added by the /1000 unit fix below) is not clipped at the top edge by bbox_inches="tight".
    configure();plt.rcParams.update({"axes.labelsize":12.0,"xtick.labelsize":11.0,"ytick.labelsize":11.0,"legend.fontsize":12.0});d=rd("ec/figure_r7_targeted_rollout.csv");fig,ax=plt.subplots(figsize=(5.5,4.4));mark={"private_gain":"^","gain_to_footprint":"s"};mark_s={"private_gain":110,"gain_to_footprint":70}
    for ref in REFS_MACRO:
        q=d[d.reference.eq(ref)]
        for r in q.itertuples():
            # ORIGINAL (bug): y was `...total_base_expected_value` WITHOUT /1000 while x IS /1000,
            # so advertiser value plotted ~1000x too large vs table ec:tab:g1 (value column is ×10³).
            #   ax.scatter(r.mean_targeted_minus_random_platform_revenue/1000,r.mean_targeted_minus_random_total_base_expected_value,color=REF_COLORS[ref],marker=mark[r.heuristic],s=45,edgecolor="white",linewidth=.5)
            # FIX (2026-09-28): divide y by 1000 to match the x-axis and table G1 units.
            ax.scatter(r.mean_targeted_minus_random_platform_revenue/1000,r.mean_targeted_minus_random_total_base_expected_value/1000,color=REF_COLORS[ref],marker=mark[r.heuristic],s=mark_s[r.heuristic],edgecolor="white",linewidth=.5)
    # ORIGINAL: ax.set_ylabel("Targeted minus random advertiser value")  # no ×10³ unit
    # FIX (2026-09-28): label now carries (×10³) to match the /1000 scaling added above.
    ax.axhline(0,color=INK,lw=.6);ax.axvline(0,color=INK,lw=.6);ax.set_xlabel(r"Targeted minus random current auction revenue ($\times 10^3$)");ax.set_ylabel(r"Targeted minus random advertiser value ($\times 10^3$)");clean(ax)
    # SIZE (2026-09-29): bigger markers (scatter 45 -> 70/110 by shape) and legend (font 10.8->12.0, markersize 8).
    # Triangle (private gain) gets a larger s than the square (gain to footprint) so the two shapes look the same visual size.
    color_handles=[plt.Line2D([],[],color=REF_COLORS[ref],marker="o",ls="none",markersize=8,label=REF_LABELS[ref]) for ref in REFS_MACRO]
    shape_handles=[plt.Line2D([],[],color=INK,marker=mark[h],ls="none",markersize=8,label=h.replace("_"," ")) for h in ("private_gain","gain_to_footprint")]
    # Reorder: matplotlib packs legend entries column-major with ncol=3, so interleave [c0,s0,c1,s1,c2] to render
    # row 1 = 3 references (colors), row 2 = 2 heuristics (shapes).
    handles=[color_handles[0],shape_handles[0],color_handles[1],shape_handles[1],color_handles[2]]
    ax.legend(handles=handles,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.18),columnspacing=.9,handletextpad=.35)
    save(fig,"fig_r7_targeted_rollout","ec")


def ec_r8() -> None:
    configure();plt.rcParams.update({"xtick.labelsize":11.0,"ytick.labelsize":11.0});d=rd("ec/figure_r8_ipinyou_heterogeneity.csv");fig,axes=plt.subplots(1,3,figsize=(7.4,4.8),sharey=True);fig.subplots_adjust(left=.085,right=.875,bottom=.17,top=.86,wspace=.22);bud=[.03125,.125,.5]
    norm=TwoSlopeNorm(vmin=-180,vcenter=0,vmax=60)
    assert d.delta_clicks.between(-180,60).all()
    cmap=plt.get_cmap("RdBu_r")
    campaigns=sorted(d.campaign.astype(str).unique())
    for idx,(ax,ref) in enumerate(zip(axes,REFS)):
        q=d[d.reference.eq(ref)].copy();q.campaign=q.campaign.astype(str);p=q.pivot(index="campaign",columns="budget_fraction",values="delta_clicks").reindex(index=campaigns,columns=bud);im=ax.imshow(p,cmap=cmap,norm=norm,aspect="auto");ax.set_xticks(range(3));ax.set_xticklabels(["3.125%","12.5%","50%"],rotation=20,ha="right");ax.set_yticks(range(len(campaigns)));ax.set_yticklabels(campaigns);panel(ax,chr(ord("a")+idx),REF_LABELS[ref])
        for i in range(len(campaigns)):
            for j in range(3):
                value=p.iloc[i,j]
                if value < 0: negative_cell_lines(ax,i,j,count=4,linewidth=0.6)
                red,green,blue,_=cmap(norm(value))
                luminance=.2126*red+.7152*green+.0722*blue
                ax.text(j,i,f"{value:+.0f}",ha="center",va="center",fontsize=10.2,
                        color="white" if luminance<.52 else INK,zorder=4)
    cax=fig.add_axes([.905,.21,.018,.61])
    colorbar=fig.colorbar(im,cax=cax,ticks=[-180,-120,-60,0,60])
    colorbar.set_label("Click change (shared scale)",fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Campaign");negative_cell_legend(fig);save(fig,"fig_r8_ipinyou_campaign_heterogeneity","ec")


def ec_r9() -> None:
    configure();d=rd("ec/figure_r9_t9sim_truth.csv");fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),sharey=True)
    for ax,budget in zip(axes,[.5,1,1.5]):
        for ref in REFS:
            q=d[(d.reference.eq(ref))&np.isclose(d.budget_multiplier,budget)].sort_values("information_view");ax.plot(q.information_view,q.true_value_delta/1000,color=REF_COLORS[ref],ls=REF_STYLES[ref],marker=REF_MARKERS[ref],mfc="white",label=REF_LABELS[ref])
        ax.axhline(0,color=INK,lw=.6);ax.set_xlabel("Information view");ax.text(.03,.95,f"Budget ×{budget:.1f}",transform=ax.transAxes,va="top",fontweight="bold");clean(ax)
    axes[0].set_ylabel(r"True value change ($\times 10^3$)");axes[0].legend(loc="best");save(fig,"fig_r9_t9sim_truth_mechanism","ec")


def ec_r10() -> None:
    configure();d=rd("ec/figure_r10_kappa_selection.csv");rules=["B_fixed_0","B_fixed_0.3","B_fixed_0.8","B_fixed_1","B_global_cal","B_reference_cal","R1_frontier","R2_near_best_95","R3_sequential","R4_kink_aware"]
    labels=["Fixed 0.0","Fixed 0.3","Fixed 0.8","Fixed 1.0","Global cal.","Reference cal.","Frontier","Near-best","Sequential","Kink-aware"]
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.4));x=np.arange(len(rules));w=.25;colors=["#9BA5AE","#6F7E8C","#405466"]
    for i,cap in enumerate([5,10,20]):
        q=d[np.isclose(d.cap_pp,cap)].set_index("rule").reindex(rules);a.bar(x+(i-1)*w,q.mean_normalized_regret,w,color=colors[i],label=f"{cap} pp cap");b.bar(x+(i-1)*w,100*q.violation_rate,w,color=colors[i])
    for ax,y in [(a,"Normalized value regret"),(b,"Delivery-violation rate (%)")]:ax.set_xticks(x);ax.set_xticklabels(labels,rotation=52,ha="right");ax.set_ylabel(y);clean(ax)
    # ORIGINAL: ...;b.text(.98,.96,"STOP",transform=b.transAxes,ha="right",va="top",fontsize=9,fontweight="bold",color=NEG)
    # FIX (2026-09-28): remove the leftover "STOP" placeholder text drawn in panel (b)'s top-right corner.
    panel(a,"a");panel(b,"b");a.legend(loc="best")
    save(fig,"fig_r10_kappa_selection_challenge","ec")
