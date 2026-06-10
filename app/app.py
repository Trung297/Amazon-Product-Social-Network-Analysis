"""
Community Detection on Amazon0302 Dataset
Streamlit App — Đồ án Phân tích Mạng Xã hội
Run: streamlit run app.py  (from project root)
"""

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_root(start):
    candidate = start
    for _ in range(4):
        if os.path.isdir(os.path.join(candidate, "data")):
            return candidate
        candidate = os.path.dirname(candidate)
    return start

ROOT = _find_root(SCRIPT_DIR)

def R(path):
    return os.path.join(ROOT, path)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Community Detection",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #07090f; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #07090f 100%);
    border-right: 1px solid #1a2540;
}
section[data-testid="stSidebar"] * { color: #8899b8 !important; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1520 0%, #182035 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 14px 18px; position: relative; overflow: hidden;
    box-shadow: 0 0 18px rgba(59,130,246,0.07);
}
[data-testid="metric-container"]::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
[data-testid="metric-container"] label {
    color: #56688a !important; font-size: 0.68rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.5rem !important; font-weight: 700; }

.sec-header {
    background: linear-gradient(90deg, rgba(59,130,246,0.13) 0%, transparent 100%);
    border-left: 3px solid #3b82f6; padding: 8px 14px;
    border-radius: 0 8px 8px 0; margin: 20px 0 10px;
    font-size: 0.85rem; font-weight: 600; color: #e2e8f0;
}
.insight-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.07) 0%, rgba(6,182,212,0.03) 100%);
    border: 1px solid rgba(16,185,129,0.2); border-radius: 10px;
    padding: 12px 16px; margin: 10px 0;
    color: #8899b8; font-size: 0.85rem; line-height: 1.65;
}
.insight-box b { color: #6ee7b7; }

.stTabs [data-baseweb="tab-list"] {
    background: #0f1520; border-radius: 10px;
    padding: 4px; gap: 3px; border: 1px solid #1e2840;
}
.stTabs [data-baseweb="tab"] {
    color: #56688a; border-radius: 7px;
    padding: 7px 22px; font-weight: 500; font-size: 0.86rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #0891b2 100%) !important;
    color: #fff !important;
}

h1 { color: #f1f5f9 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { color: #e2e8f0 !important; font-weight: 600 !important; }
p, li { color: #8899b8 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
DARK = dict(
    paper_bgcolor="#0d1120", plot_bgcolor="#0f1520",
    font=dict(color="#8899b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=13),
    legend=dict(bgcolor="rgba(15,21,32,0.85)", bordercolor="#1e2840",
                borderwidth=1, font=dict(color="#8899b8", size=10)),
    xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540", color="#56688a"),
    yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540", color="#56688a"),
)
PAL = ["#3b82f6","#06b6d4","#10b981","#f59e0b","#f43f5e",
       "#8b5cf6","#ec4899","#14b8a6","#f97316","#6366f1"]
HUB_C  = {"Normal":"#3b82f6","Hub":"#f59e0b","Super Hub":"#f43f5e"}
PROD_C = {"Book":"#3b82f6","DVD":"#f43f5e","Music":"#10b981",
           "Software":"#8b5cf6","Video":"#f59e0b","Toy":"#ec4899",
           "Video Games":"#06b6d4","Unknown":"#64748b"}

def df(fig, h=380):
    fig.update_layout(**DARK, height=h, margin=dict(l=10,r=10,t=42,b=10))
    return fig

def sec(icon, title):
    st.markdown(f"<div class='sec-header'>{icon} {title}</div>", unsafe_allow_html=True)

def insight(html):
    st.markdown(f"<div class='insight-box'>{html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load():
    d = {}
    files = {
        "ca":  "data/community_analysis/df_community_analysis.csv",
        "cp":  "data/community_analysis/cluster_profiling_summary.csv",
        "gi":  "data/graph_information/final_merged_data.csv",
        "rh":  "data/graph_information/real_hub.csv",
        "sh":  "data/graph_information/superhub.csv",
    }
    for k, rel in files.items():
        p = R(rel)
        if os.path.exists(p):
            try:
                tmp = pd.read_csv(p)
                d[k] = tmp.loc[:, ~tmp.columns.str.startswith("Unnamed")]
            except Exception as e:
                st.sidebar.warning(f"⚠️ {k}: {e}")
    return d

D = load()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 6px; text-align:center;'>
        <div style='font-size:1.8rem;'>🛒</div>
        <div style='font-size:0.95rem; font-weight:700; color:#e2e8f0; margin-top:4px;'>Amazon0302</div>
        <div style='font-size:0.68rem; color:#56688a; letter-spacing:0.06em; text-transform:uppercase;'>Community Detection</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2540; margin:8px 0;'>", unsafe_allow_html=True)
    tab = st.radio("", ["🏘️ Phân tích Community", "🔍 Phân tích Clustering"],
                   label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a2540; margin:8px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.60rem; color:#374151; word-break:break-all;'>📁 {ROOT}</div>",
                unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1a2540; margin:6px 0;'>", unsafe_allow_html=True)
    for k, label in {"ca":"Community Analysis","cp":"Cluster Profiling",
                     "gi":"Graph Info","rh":"Real Hubs","sh":"Super Hubs"}.items():
        ok = k in D
        c  = "#10b981" if ok else "#f43f5e"
        r  = f" ({len(D[k]):,} rows)" if ok else " — missing"
        st.markdown(f"<div style='font-size:0.72rem; color:{c}; padding:2px 0;'>{'●' if ok else '○'} {label}{r}</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 1 — PHÂN TÍCH COMMUNITY
# ══════════════════════════════════════════════════════════
if tab == "🏘️ Phân tích Community":
    ca = D.get("ca")
    gi = D.get("gi")
    sh = D.get("sh")
    rh = D.get("rh")

    st.markdown("""
    <div style='padding:18px 0 12px;'>
      <div style='font-size:0.65rem; color:#3b82f6; text-transform:uppercase;
                  letter-spacing:0.14em; font-weight:600; margin-bottom:6px;'>Amazon0302 · Stanford SNAP</div>
      <h1 style='font-size:1.9rem; margin:0 0 8px;
                 background:linear-gradient(90deg,#e2e8f0,#93c5fd);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        Phân tích Cấu trúc Cộng đồng</h1>
      <p style='font-size:0.88rem; color:#56688a; margin:0;'>
        Cấu trúc nội tại: kích thước, mật độ, thành phần sản phẩm, phân phối hub nodes.</p>
    </div>""", unsafe_allow_html=True)

    if ca is None:
        st.error("Không tìm thấy df_community_analysis.csv")
        st.stop()

    top    = "total_product" if "total_product" in ca.columns else None
    tcols  = [c for c in ca.columns if c.startswith("count_")
              and c not in ["count_Hub","count_Normal","count_Super Hub"]]
    hcols  = [c for c in ["count_Hub","count_Normal","count_Super Hub"] if c in ca.columns]
    fcols  = [c for c in ["avg_density","avg_indegree","avg_outdegree","avg_totaldegree",
                           "avg_salesrank","avg_rating","avg_totalreviews","avg_similar",
                           "avg_categorytypecount","avg_downloaded"] if c in ca.columns]

    # ── KPIs ──
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Nodes",       f"{gi.shape[0]:,}" if gi is not None else "~262K")
    c2.metric("Communities", f"{ca['Community_ID'].nunique():,}")
    c3.metric("Avg Size",    f"{ca[top].mean():.0f}" if top else "N/A")
    c4.metric("Max Size",    f"{int(ca[top].max()):,}" if top else "N/A")
    c5.metric("Avg Density", f"{ca['avg_density'].mean():.4f}" if "avg_density" in ca.columns else "N/A")
    c6.metric("Super Hubs",  f"{len(sh):,}" if sh is not None else "N/A")
    c7.metric("Real Hubs",   f"{len(rh):,}" if rh is not None else "N/A")
    st.markdown("<hr style='border-color:#1a2540; margin:18px 0 6px;'>", unsafe_allow_html=True)

    # ════════════════════════════
    # ROW 1 — Size & Density
    # ════════════════════════════
    sec("📐", "Phân phối Kích thước & Mật độ Cộng đồng")
    r1a, r1b, r1c = st.columns(3)

    with r1a:
        if top:
            fig = px.histogram(ca, x=top, nbins=80, log_y=True,
                               color_discrete_sequence=["#3b82f6"],
                               title="Size Distribution (log Y)",
                               labels={top:"Số sản phẩm"})
            df(fig, 300); st.plotly_chart(fig, use_container_width=True)

    with r1b:
        if "avg_density" in ca.columns:
            fig = px.histogram(ca, x="avg_density", nbins=60,
                               color_discrete_sequence=["#06b6d4"],
                               title="Density Distribution",
                               labels={"avg_density":"Mật độ"})
            df(fig, 300); st.plotly_chart(fig, use_container_width=True)

    with r1c:
        if top:
            ss = np.sort(ca[top].values)
            cdf_y = np.arange(1, len(ss)+1) / len(ss)
            fig = go.Figure(go.Scatter(x=ss, y=cdf_y, mode="lines",
                                       line=dict(color="#10b981", width=2),
                                       fill="tozeroy", fillcolor="rgba(16,185,129,0.07)"))
            fig.update_layout(**DARK, height=300, margin=dict(l=10,r=10,t=42,b=10),
                              title="CDF — Community Sizes")
            fig.update_xaxes(title="Size"); fig.update_yaxes(title="Cumul. %")
            st.plotly_chart(fig, use_container_width=True)

    r1d, r1e = st.columns(2)
    with r1d:
        if top and "avg_density" in ca.columns:
            fig = px.scatter(ca, x=top, y="avg_density",
                             color="KMeans_Cluster" if "KMeans_Cluster" in ca.columns else None,
                             color_discrete_sequence=PAL, opacity=0.6, size_max=10,
                             title="Size vs Density — mỗi điểm = 1 cộng đồng",
                             hover_data=["Community_ID"],
                             labels={top:"Size","avg_density":"Density"})
            df(fig, 340); st.plotly_chart(fig, use_container_width=True)

    with r1e:
        if top:
            top60 = ca.nlargest(60, top)
            color_col = "avg_density" if "avg_density" in ca.columns else top
            fig = px.treemap(top60, path=["Community_ID"], values=top,
                             color=color_col,
                             color_continuous_scale=["#1e3a5f","#3b82f6","#06b6d4"],
                             title="Treemap — Top 60 cộng đồng (màu = density)")
            df(fig, 340); fig.update_traces(textinfo="label+value")
            st.plotly_chart(fig, use_container_width=True)

    insight("<b>Size:</b> Phân phối power-law — đa số cộng đồng rất nhỏ, một số ít cộng đồng cực lớn (hub communities). "
            "<b>Density:</b> Cộng đồng nhỏ thường có density cao hơn — sản phẩm niche được mua cùng nhau chặt chẽ hơn.")

    # ════════════════════════════
    # ROW 2 — Thành phần sản phẩm
    # ════════════════════════════
    if tcols:
        sec("📦", "Thành phần Loại Sản phẩm")
        r2a, r2b = st.columns(2)

        with r2a:
            totals = ca[tcols].sum().reset_index()
            totals.columns = ["type","count"]
            totals["type"] = totals["type"].str.replace("count_","", regex=False)
            totals = totals.sort_values("count", ascending=False)
            fig = px.pie(totals, names="type", values="count", hole=0.52,
                         color="type",
                         color_discrete_map={k.replace("count_",""):v for k,v in PROD_C.items()},
                         title="Tỷ lệ loại sản phẩm — toàn network")
            df(fig, 360)
            fig.update_traces(textinfo="percent", textposition="inside",
                              insidetextorientation="radial", textfont_size=11,
                              pull=[0.03]*len(totals))
            fig.update_layout(legend=dict(font=dict(size=10), x=1.0, y=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with r2b:
            if top:
                dom = ca[["Community_ID"] + tcols + [top]].copy()
                dom["dominant"] = dom[tcols].idxmax(axis=1).str.replace("count_","", regex=False)
                dom_cnt = dom["dominant"].value_counts().reset_index()
                dom_cnt.columns = ["type","communities"]
                fig = px.bar(dom_cnt, x="type", y="communities",
                             color="type",
                             color_discrete_map={k.replace("count_",""):v for k,v in PROD_C.items()},
                             title="Loại sản phẩm chủ đạo — số cộng đồng",
                             text_auto=True)
                df(fig, 360).update_layout(showlegend=False)
                fig.update_traces(textposition="outside", textfont_size=11)
                st.plotly_chart(fig, use_container_width=True)

        insight("<b>Book chiếm ~72%</b> tổng sản phẩm trong network. "
                "Hầu hết cộng đồng có Book là sản phẩm chủ đạo — phản ánh danh mục lớn nhất của Amazon thời kỳ đầu.")

    # ════════════════════════════
    # ROW 3 — Hub Nodes
    # ════════════════════════════
    if hcols:
        sec("⭐", "Phân tích Hub Nodes")
        r3a, r3b = st.columns(2)

        with r3a:
            hub_tot = ca[hcols].sum().reset_index()
            hub_tot.columns = ["type","count"]
            hub_tot["type"] = hub_tot["type"].str.replace("count_","", regex=False)
            fig = px.pie(hub_tot, names="type", values="count", hole=0.52,
                         color="type", color_discrete_map=HUB_C,
                         title="Normal / Hub / Super Hub — tổng toàn network")
            df(fig, 340)
            fig.update_traces(textinfo="percent+label", textposition="inside",
                              insidetextorientation="radial", textfont_size=11)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with r3b:
            if top and "KMeans_Cluster" in ca.columns:
                hub_cl = ca.groupby("KMeans_Cluster")[hcols].mean().reset_index()
                hub_m  = hub_cl.melt(id_vars="KMeans_Cluster", var_name="type", value_name="avg")
                hub_m["type"] = hub_m["type"].str.replace("count_","", regex=False)
                fig = px.bar(hub_m, x="KMeans_Cluster", y="avg",
                             color="type", color_discrete_map=HUB_C,
                             barmode="group",
                             title="Avg Normal / Hub / Super Hub per Cluster",
                             labels={"KMeans_Cluster":"Cluster","avg":"Avg count"},
                             text_auto=".1f")
                df(fig, 340)
                fig.update_traces(textposition="outside", textfont_size=9)
                st.plotly_chart(fig, use_container_width=True)
            else:
                hub_avg = ca[hcols].mean().reset_index()
                hub_avg.columns = ["type","avg"]
                hub_avg["type"] = hub_avg["type"].str.replace("count_","", regex=False)
                fig = px.bar(hub_avg, x="type", y="avg",
                             color="type", color_discrete_map=HUB_C,
                             title="Avg Hub / Normal / Super Hub per Community",
                             text_auto=".2f")
                df(fig, 340).update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # In/Out degree từ graph_info nếu có
        if gi is not None and "node_type" in gi.columns and "total_degree" in gi.columns:
            r3c, r3d = st.columns(2)
            with r3c:
                fig = px.box(gi, x="node_type", y="total_degree",
                             color="node_type", color_discrete_map=HUB_C,
                             log_y=True, points=False,
                             title="Total Degree by Node Type")
                df(fig, 300).update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with r3d:
                if "avg_rating" in gi.columns:
                    fig = px.violin(gi.dropna(subset=["avg_rating"]),
                                    x="node_type", y="avg_rating",
                                    color="node_type", color_discrete_map=HUB_C,
                                    box=True, points=False,
                                    title="Rating Distribution by Node Type")
                    df(fig, 300).update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        insight("<b>Super Hub</b> rất hiếm nhưng có degree cực cao — đây là các sản phẩm bestseller được co-purchase nhiều nhất. "
                "Cluster nào có avg Hub cao hơn → cộng đồng đó có cấu trúc tập trung hơn.")

        # Real Hub table
        if rh is not None:
            sec("🔗", f"Real Hub Nodes — {len(rh):,} nodes")
            if gi is not None and "product_id" in gi.columns:
                rh["_k"] = rh[rh.columns[0]].astype(str)
                gi["_k"] = gi["product_id"].astype(str)
                rh_d = rh.merge(gi, on="_k", how="left").drop(columns=["_k"])
                gi.drop(columns=["_k"], inplace=True, errors="ignore")
            else:
                rh_d = rh
            st.dataframe(rh_d, use_container_width=True, height=220)

    # ════════════════════════════
    # ROW 4 — Correlation heatmap
    # ════════════════════════════
    if len(fcols) >= 4:
        sec("🔥", "Correlation Heatmap — Đặc trưng Community")
        corr = ca[fcols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}", textfont_size=9,
        ))
        fig.update_layout(**DARK, height=460, margin=dict(l=10,r=10,t=42,b=10),
                          title="Feature Correlation — community level")
        st.plotly_chart(fig, use_container_width=True)
        insight("<b>Màu đỏ</b> = tương quan âm, <b>màu xanh</b> = tương quan dương. "
                "Avg degree và avg density tương quan ngược chiều với kích thước — cộng đồng lớn thường thưa hơn.")


# ══════════════════════════════════════════════════════════
# TAB 2 — PHÂN TÍCH CLUSTERING
# ══════════════════════════════════════════════════════════
elif tab == "🔍 Phân tích Clustering":
    ca = D.get("ca")
    cp = D.get("cp")

    st.markdown("""
    <div style='padding:18px 0 12px;'>
      <div style='font-size:0.65rem; color:#10b981; text-transform:uppercase;
                  letter-spacing:0.14em; font-weight:600; margin-bottom:6px;'>K-Means · Community Grouping</div>
      <h1 style='font-size:1.9rem; margin:0 0 8px;
                 background:linear-gradient(90deg,#e2e8f0,#6ee7b7);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        Phân tích K-Means Clustering</h1>
      <p style='font-size:0.88rem; color:#56688a; margin:0;'>
        Nhóm các cộng đồng có đặc trưng tương đồng → khám phá cấu trúc macro của network.</p>
    </div>""", unsafe_allow_html=True)

    if ca is None:
        st.error("Không tìm thấy df_community_analysis.csv")
        st.stop()

    if "KMeans_Cluster" not in ca.columns:
        st.error("Không có cột KMeans_Cluster trong df_community_analysis.csv")
        st.stop()

    top   = "total_product" if "total_product" in ca.columns else None
    fcols = [c for c in ["avg_density","avg_indegree","avg_outdegree","avg_totaldegree",
                          "avg_salesrank","avg_rating","avg_totalreviews","avg_similar",
                          "avg_categorytypecount","avg_downloaded"] if c in ca.columns]
    tcols = [c for c in ca.columns if c.startswith("count_")
             and c not in ["count_Hub","count_Normal","count_Super Hub"]]
    hcols = [c for c in ["count_Hub","count_Normal","count_Super Hub"] if c in ca.columns]
    n_cl  = ca["KMeans_Cluster"].nunique()

    # ── KPIs ──
    cnt_cl  = ca.groupby("KMeans_Cluster")["Community_ID"].count()
    prod_cl = ca.groupby("KMeans_Cluster")[top].sum() if top else None
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("K Clusters",        str(n_cl))
    c2.metric("Total Communities", f"{ca['Community_ID'].nunique():,}")
    c3.metric("Largest Cluster",   f"C{cnt_cl.idxmax()} ({cnt_cl.max():,})")
    c4.metric("Smallest Cluster",  f"C{cnt_cl.idxmin()} ({cnt_cl.min():,})")
    c5.metric("Most Products",     f"C{prod_cl.idxmax()} ({prod_cl.max():,})" if prod_cl is not None else "N/A")
    st.markdown("<hr style='border-color:#1a2540; margin:18px 0 6px;'>", unsafe_allow_html=True)

    # ════════════════════════════
    # ROW 1 — Cluster overview
    # ════════════════════════════
    sec("🎯", "Tổng quan phân phối Cluster")
    r1a, r1b, r1c = st.columns(3)

    with r1a:
        cnt = ca["KMeans_Cluster"].value_counts().reset_index()
        cnt.columns = ["Cluster","Count"]
        cnt["Cluster"] = "C" + cnt["Cluster"].astype(str)
        fig = px.pie(cnt, names="Cluster", values="Count", hole=0.52,
                     color_discrete_sequence=PAL,
                     title="Số cộng đồng per Cluster")
        df(fig, 320)
        fig.update_traces(textinfo="percent+label", textposition="inside",
                          insidetextorientation="radial", textfont_size=11)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        if top:
            tp = ca.groupby("KMeans_Cluster")[top].sum().reset_index()
            tp.columns = ["Cluster","Total_Products"]
            tp["Cluster"] = "C" + tp["Cluster"].astype(str)
            fig = px.bar(tp, x="Cluster", y="Total_Products",
                         color="Cluster", color_discrete_sequence=PAL,
                         title="Tổng sản phẩm per Cluster", text_auto=",")
            df(fig, 320).update_layout(showlegend=False)
            fig.update_traces(textposition="outside", textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)

    with r1c:
        if top and "avg_density" in ca.columns:
            fig = px.scatter(ca, x=top, y="avg_density",
                             color="KMeans_Cluster",
                             color_discrete_sequence=PAL,
                             opacity=0.65, size_max=10,
                             title="Clusters trong feature space",
                             hover_data=["Community_ID"],
                             labels={top:"Size","avg_density":"Density",
                                     "KMeans_Cluster":"Cluster"})
            df(fig, 320); st.plotly_chart(fig, use_container_width=True)

    insight("Clusters phân tách rõ trong không gian Size × Density — "
            "xác nhận K-Means đã capture được sự khác biệt cấu trúc giữa các nhóm cộng đồng.")

    # ════════════════════════════
    # ROW 2 — Cluster Profiles (Radar + Heatmap)
    # ════════════════════════════
    sec("🕸️", "Cluster Profiles")

    if cp is not None:
        cl_col  = "Communities" if "Communities" in cp.columns else cp.columns[0]
        avg_cp  = [c for c in cp.columns if c.startswith("avg_")]
        type_cp = [c for c in cp.columns if c.startswith("count_")
                   and c not in ["count_Hub","count_Normal","count_Super Hub"]]
        hub_cp  = [c for c in ["count_Hub","count_Normal","count_Super Hub"] if c in cp.columns]

        r2a, r2b = st.columns([3, 2])

        with r2a:
            # Radar
            radar_dims = avg_cp[:min(8, len(avg_cp))]
            if len(radar_dims) >= 3:
                mins = cp[radar_dims].min()
                maxs = cp[radar_dims].max()
                fig_r = go.Figure()
                for i, row in cp.iterrows():
                    norm = [(row[c]-mins[c])/(maxs[c]-mins[c]+1e-9) for c in radar_dims]
                    norm += [norm[0]]
                    cats  = [c.replace("avg_","") for c in radar_dims] + [radar_dims[0].replace("avg_","")]
                    fig_r.add_trace(go.Scatterpolar(
                        r=norm, theta=cats, fill="toself",
                        name=f"Cluster {row[cl_col]}",
                        line_color=PAL[i % len(PAL)], opacity=0.8,
                    ))
                fig_r.update_layout(
                    polar=dict(bgcolor="#0f1520",
                               radialaxis=dict(color="#56688a", gridcolor="#1a2540", range=[0,1]),
                               angularaxis=dict(color="#8899b8")),
                    **DARK, height=400, margin=dict(l=50,r=50,t=50,b=50),
                    title="Normalised Cluster Radar (0=min, 1=max)",
                )
                st.plotly_chart(fig_r, use_container_width=True)

        with r2b:
            # Summary table styled
            show_cols = [cl_col] + avg_cp[:6]
            show_cols = [c for c in show_cols if c in cp.columns]
            st.markdown("**Cluster Summary Table**")
            st.dataframe(
                cp[show_cols].style.format({c:"{:.3f}" for c in avg_cp if c in show_cols}),
                use_container_width=True, height=400,
            )

        # Heatmap normalized
        if avg_cp:
            heat = cp.set_index(cl_col)[avg_cp]
            heat_n = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
            fig = go.Figure(go.Heatmap(
                z=heat_n.values,
                x=[c.replace("avg_","") for c in avg_cp],
                y=[f"Cluster {v}" for v in heat_n.index],
                colorscale="Viridis",
                text=np.round(heat.values, 3),
                texttemplate="%{text}", textfont_size=9,
            ))
            fig.update_layout(**DARK, height=max(280, n_cl*55+80),
                              margin=dict(l=10,r=10,t=42,b=10),
                              title="Feature Heatmap per Cluster (normalized, raw values shown)")
            st.plotly_chart(fig, use_container_width=True)

        insight("Radar cho thấy <b>profile riêng biệt</b> của từng cluster. "
                "Cluster nào có đỉnh cao ở avg_totaldegree + avg_density → cộng đồng nhỏ, chặt, kết nối cao.")

        # Product & Hub mix per cluster
        r3a, r3b = st.columns(2)
        with r3a:
            if type_cp:
                sec("📦", "Product Mix per Cluster")
                fig = go.Figure()
                for tc in type_cp:
                    pt = tc.replace("count_","")
                    fig.add_trace(go.Bar(
                        name=pt, x=cp[cl_col].astype(str), y=cp[tc],
                        marker_color=PROD_C.get(pt, "#aaa"),
                    ))
                fig.update_layout(barmode="stack", **DARK, height=340,
                                  margin=dict(l=10,r=10,t=42,b=10),
                                  title="Tổng sản phẩm theo loại per Cluster")
                st.plotly_chart(fig, use_container_width=True)
        with r3b:
            if hub_cp:
                sec("⭐", "Hub Composition per Cluster")
                fig = go.Figure()
                for hc in hub_cp:
                    fig.add_trace(go.Bar(
                        name=hc.replace("count_",""),
                        x=cp[cl_col].astype(str), y=cp[hc],
                        marker_color=HUB_C.get(hc.replace("count_",""),"#aaa"),
                    ))
                fig.update_layout(barmode="group", **DARK, height=340,
                                  margin=dict(l=10,r=10,t=42,b=10),
                                  title="Hub / Normal / Super Hub per Cluster")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Không có cluster_profiling_summary.csv — hiển thị từ df_community_analysis.")

    # ════════════════════════════
    # ROW 3 — So sánh feature distributions
    # ════════════════════════════
    if fcols:
        sec("📊", "So sánh Feature Distributions per Cluster")
        r4a, r4b = st.columns(2)

        with r4a:
            sel_box = st.selectbox("Boxplot — chọn feature", fcols, key="bx_sel")
            fig = px.box(ca, x="KMeans_Cluster", y=sel_box,
                         color="KMeans_Cluster", color_discrete_sequence=PAL,
                         points=False,
                         title=f"{sel_box} distribution per Cluster")
            df(fig, 360).update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with r4b:
            sel_viol = st.selectbox("Violin — chọn feature", fcols,
                                    index=min(1, len(fcols)-1), key="vl_sel")
            fig = px.violin(ca, x="KMeans_Cluster", y=sel_viol,
                            color="KMeans_Cluster", color_discrete_sequence=PAL,
                            box=True, points=False,
                            title=f"{sel_viol} violin per Cluster")
            df(fig, 360).update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Multi-feature scatter explorer
        sec("🔍", "Scatter Explorer — 2D feature space per Cluster")
        ea, eb = st.columns(2)
        xc = ea.selectbox("X", fcols, 0, key="sc_x")
        yc = eb.selectbox("Y", fcols, min(1,len(fcols)-1), key="sc_y")
        fig = px.scatter(ca, x=xc, y=yc,
                         color="KMeans_Cluster", color_discrete_sequence=PAL,
                         size=top if top else None, size_max=20, opacity=0.7,
                         hover_data=["Community_ID"],
                         title=f"{xc.replace('avg_','')} vs {yc.replace('avg_','')} — colored by Cluster")
        df(fig, 420); st.plotly_chart(fig, use_container_width=True)

        insight("Boxplot và violin cho thấy sự phân tách của từng cluster theo từng feature. "
                "Scatter explorer giúp xác nhận clusters có ranh giới rõ ràng trong không gian 2D.")

    # ════════════════════════════
    # ROW 4 — Browse + Download
    # ════════════════════════════
    sec("📋", "Browse Cộng đồng theo Cluster")
    cl_list = sorted(ca["KMeans_Cluster"].unique())
    sel_cl  = st.selectbox("Chọn Cluster", cl_list)
    df_cl   = ca[ca["KMeans_Cluster"] == sel_cl].sort_values(
        top if top else "Community_ID", ascending=False)
    st.caption(f"Cluster {sel_cl}: {len(df_cl):,} cộng đồng")
    st.dataframe(df_cl, use_container_width=True, height=340)
    st.download_button(
        f"⬇️ Download Cluster {sel_cl} CSV",
        df_cl.to_csv(index=False).encode(),
        file_name=f"cluster_{sel_cl}.csv", mime="text/csv",
    )