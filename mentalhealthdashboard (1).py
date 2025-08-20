

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------- PAGE / THEME --------------------
st.set_page_config(page_title="Cancer Card — Mock Mental Health Dashboard",
                   page_icon="🧠", layout="wide")

# Colors inspired by the reference image (royal blue, red, green, gold)
COLORS = {
    "blue_dark": "#1F4E8C",
    "blue":      "#2B6CB0",
    "blue_lt":   "#5FA8FF",
    "red":       "#E15866",
    "green":     "#22A06B",
    "gold":      "#F4A82A",
    "gray":      "#E5E7EB",
    "text":      "#111827",
    "bg":        "#FFFFFF",
    "card_border": "#D1D5DB",
}

# Global plot settings
TEMPLATE = "plotly_white"
FIG_BG   = COLORS["bg"]
GRID     = "#EDF2F7"
FONTCLR  = COLORS["text"]
CONFIG   = {"displayModeBar": False, "responsive": True}

# Compact, single-page heights
H_KPI   = 90
H_GAUGE = 170
H_CHART = 300

# -------------------- CSS (white, drop-shadows, no overlap) --------------------
st.markdown(
    '''
    <style>
      :root, .stApp { background-color: #FFFFFF !important; color: #111827 !important; }
      .block-container { padding-top: .6rem; padding-bottom: .6rem; }
      [data-testid="stHeader"] { background: #FFFFFF !important; border-bottom:none !important; box-shadow:none !important; }

      /* Left sidebar clean white with black text */
      [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #D1D5DB;
      }
      [data-testid="stSidebar"] * { color: #111827 !important; }

      /* KPI & Viz cards with subtle borders + drop shadow */
      .card {
        background:#fff; border:1px solid #D1D5DB; border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,.08);
        padding: 10px 12px; margin-bottom: 10px;
      }
      .kpi-title { font-size:.80rem; color:#6b7280; margin-bottom:6px; }
      .kpi-value { font-weight:900; font-size:1.35rem; }

      .viz-card { background:#fff; border:1px solid #D1D5DB; border-radius: 12px;
                   box-shadow: 0 10px 22px rgba(0,0,0,.12); padding: 10px 12px; margin-bottom: 12px; }
      .viz-title { font-weight:900; font-size:1.05rem; margin: 0 0 6px 2px; }

      /* Apply drop shadow on Plotly canvas to pop from the page */
      .viz-card [data-testid="stPlotlyChart"] > div, .viz-card .js-plotly-plot {
        filter: drop-shadow(0 8px 16px rgba(0,0,0,.15));
        border-radius: 8px;
        background: #fff;
      }

      /* Big page title */
      .page-title { text-align:center; font-weight:900; font-size: 1.8rem; letter-spacing:.02em; margin:.2rem 0 .6rem 0; }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown('<div class="page-title">MOCK MENTAL HEALTH DASHBOARD</div>', unsafe_allow_html=True)

# -------------------- DATA --------------------
CANDIDATES = [
    r"C:\Users\akans\Documents\Mental Health\CancerCard_MockMentalHealth.csv",
    "./CancerCard_MockMentalHealth.csv",
    "./data/CancerCard_MockMentalHealth.csv",
    "/mnt/data/CancerCard_MockMentalHealth.csv",
]
CSV_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    st.error("CSV 'CancerCard_MockMentalHealth.csv' not found. Put it in ./, ./data/, /mnt/data/, or update CANDIDATES.")
    st.stop()

df = pd.read_csv(CSV_PATH)

def std_cols(cols): return [c.strip().lower().replace(" ", "_").replace("&","and") for c in cols]
df.columns = std_cols(df.columns)
for c in df.columns:
    if df[c].dtype == "O":
        df[c] = df[c].astype(str).str.strip()

LIKERT_03  = {"not at all":0, "several days":1, "more than half the days":2, "nearly every day":3, "nearly everyday":3}
SAT_5      = {"very dissatisfied":1, "dissatisfied":2, "neutral":3, "satisfied":4, "very satisfied":5}
ENG_MAP    = {"first time":0, "monthly or less":1, "2-3x/month":2, "weekly":3, "several times/week":4}

def map03(s):   return s.str.lower().map(LIKERT_03)  if s.dtype == "O" else pd.to_numeric(s, errors="coerce")
def map5s(s):   return s.str.lower().map(SAT_5)      if s.dtype == "O" else pd.to_numeric(s, errors="coerce")
def mapeng(s):  return s.str.lower().map(ENG_MAP)    if s.dtype == "O" else pd.to_numeric(s, errors="coerce")

for col in ["anxiety","worry_control","pleasure","hopeless"]:
    if col in df.columns: df[col+"_score"] = map03(df[col])

if "stress_level" in df.columns:         df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")
if "satisfaction" in df.columns:         df["satisfaction_score"] = map5s(df["satisfaction"])
if "usage_frequency" in df.columns:      df["engagement_score"] = mapeng(df["usage_frequency"])

df["gad2_total"]    = df["anxiety_score"].fillna(0) + df["worry_control_score"].fillna(0) if all(c in df.columns for c in ["anxiety_score","worry_control_score"]) else np.nan
df["phq2_total"]    = df["pleasure_score"].fillna(0) + df["hopeless_score"].fillna(0)     if all(c in df.columns for c in ["pleasure_score","hopeless_score"]) else np.nan
df["gad2_flag_ge3"] = df["gad2_total"] >= 3
df["phq2_flag_ge3"] = df["phq2_total"] >= 3
if "stress_level" in df.columns: df["high_stress"] = df["stress_level"] >= 7

# Sidebar filters (simple, non-intrusive)
st.sidebar.header("Filters")
def multiselect_filter(col, label=None):
    if col not in df.columns: return None
    opts = sorted([v for v in df[col].dropna().unique()])
    sel = st.sidebar.multiselect(label or col.replace("_"," ").title(), opts, default=opts)
    return sel

view = df.copy()
for col in ["age_group","cancer_stage","gender"]:
    sel = multiselect_filter(col)
    if sel:
        view = view[view[col].isin(sel)]

# -------------------- KPI ROW (4 tiles) --------------------
k1, k2, k3, k4 = st.columns(4)
def kpi(col, name, value, accent):
    with col:
        st.markdown(f'''
          <div class="card" style="height:{H_KPI}px">
            <div class="kpi-title">{name}</div>
            <div class="kpi-value" style="color:{accent}">{value}</div>
          </div>
        ''', unsafe_allow_html=True)

kpi(k1, "Total Responses", f"{len(view):,}", COLORS["blue"])
kpi(k2, "High Stress (≥7)", f"{(view.get('high_stress', pd.Series(dtype=bool))==True).mean()*100:.1f}%", COLORS["gold"])
kpi(k3, "Anxiety (GAD-2 ≥3)", f"{(view.get('gad2_flag_ge3', pd.Series(dtype=bool))==True).mean()*100:.1f}%", COLORS["red"])
kpi(k4, "Depression (PHQ-2 ≥3)", f"{(view.get('phq2_flag_ge3', pd.Series(dtype=bool))==True).mean()*100:.1f}%", COLORS["green"])

# -------------------- RING “GAUGE” ROW (4 donuts) --------------------
def ring_gauge(value_pct: float, color: str, title: str):
    v = 0 if pd.isna(value_pct) else float(value_pct)
    v = max(0, min(100, v))
    dfp = pd.DataFrame({"label": [title, "rest"], "value": [v, 100-v]})
    fig = px.pie(dfp, values="value", names="label", hole=0.7,
                 color="label",
                 color_discrete_map={title: color, "rest": COLORS["gray"]})
    fig.update_traces(textinfo="none", sort=False, showlegend=False)
    fig.update_layout(template=TEMPLATE, paper_bgcolor=FIG_BG, plot_bgcolor=FIG_BG,
                      margin=dict(t=18, r=10, b=10, l=10), height=H_GAUGE)
    fig.add_annotation(text=f"<b>{v:.0f}%</b>", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=22, color=FONTCLR), xanchor="center", yanchor="middle")
    return fig

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.markdown(f'<div class="viz-card"><div class="viz-title">Anxiety %</div>', unsafe_allow_html=True)
    val = (view.get("gad2_flag_ge3", pd.Series(dtype=bool))==True).mean()*100
    st.plotly_chart(ring_gauge(val, COLORS["red"], "Anxiety"), use_container_width=True, config=CONFIG, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)
with g2:
    st.markdown(f'<div class="viz-card"><div class="viz-title">Depression %</div>', unsafe_allow_html=True)
    val = (view.get("phq2_flag_ge3", pd.Series(dtype=bool))==True).mean()*100
    st.plotly_chart(ring_gauge(val, COLORS["blue"], "Depression"), use_container_width=True, config=CONFIG, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)
with g3:
    st.markdown(f'<div class="viz-card"><div class="viz-title">High stress %</div>', unsafe_allow_html=True)
    val = (view.get("high_stress", pd.Series(dtype=bool))==True).mean()*100
    st.plotly_chart(ring_gauge(val, COLORS["green"], "Stress"), use_container_width=True, config=CONFIG, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)
with g4:
    st.markdown(f'<div class="viz-card"><div class="viz-title">Satisfaction %</div>', unsafe_allow_html=True)
    sat = view.get("satisfaction_score", pd.Series(dtype=float)).mean()
    sat_pct = np.nan if pd.isna(sat) else (sat-1)/(5-1)*100
    st.plotly_chart(ring_gauge(0 if np.isnan(sat_pct) else sat_pct, COLORS["gold"], "Satisfaction"),
                    use_container_width=True, config=CONFIG, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- CHART HELPERS --------------------
def style_fig(fig, height=H_CHART, legend_top=True):
    fig.update_layout(
        template=TEMPLATE, height=height,
        plot_bgcolor=FIG_BG, paper_bgcolor=FIG_BG,
        margin=dict(t=30, r=10, b=10, l=10),
        font=dict(color=FONTCLR, size=13),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right") if legend_top else dict(orientation="v"),
        legend_title_text=""
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=FONTCLR, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=FONTCLR, automargin=True)
    return fig

def viz_card(title:str, fig:go.Figure, height=H_CHART):
    st.markdown(f'<div class="viz-card"><div class="viz-title">{title}</div>', unsafe_allow_html=True)
    fig = style_fig(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, config=CONFIG, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- BOTTOM ROW: 2 CHARTS (bar + line) --------------------
c1, c2 = st.columns(2)

# Bar (stacked): Stress distribution by stage — FORCE our royal blues explicitly
with c1:
    if all(c in view.columns for c in ["stress_level","cancer_stage"]):
        data = view.dropna(subset=["stress_level","cancer_stage"]).copy()
        def band(x):
            try: x = float(x)
            except: return np.nan
            if x <= 3: return "Low (1–3)"
            if x <= 6: return "Medium (4–6)"
            return "High (7–10)"
        data["Band"] = data["stress_level"].apply(band)
        tab = pd.crosstab(data["cancer_stage"], data["Band"], normalize="index").mul(100).round(1)
        order_cols = ["Low (1–3)", "Medium (4–6)", "High (7–10)"]
        tab = tab[[c for c in order_cols if c in tab.columns]]
        tidy = tab.reset_index().melt(id_vars="cancer_stage", var_name="Band", value_name="Percent")
        fig = px.bar(
            tidy, x="cancer_stage", y="Percent", color="Band", barmode="stack",
            category_orders={"Band": order_cols},
            color_discrete_map={"Low (1–3)": COLORS["blue_lt"],
                                "Medium (4–6)": COLORS["blue"],
                                "High (7–10)": COLORS["blue_dark"]}
        )
        fig.update_layout(xaxis_title="Cancer stage", yaxis_title="Percent")
        viz_card("Stress distribution by cancer stage", fig)
    else:
        viz_card("Stress distribution by cancer stage", go.Figure())

# Line: Usage frequency by age group — single royal blue line (no theme override)
with c2:
    if "age_group" in view.columns and "engagement_score" in view.columns:
        def age_key(a):
            s = str(a)
            nums = [int(t) for t in "".join(ch if ch.isdigit() else " " for ch in s).split() or [999]]
            return nums[0] if nums else 999
        gp = (view.groupby("age_group")["engagement_score"].mean()
                    .dropna().reset_index()
                    .sort_values(by="age_group", key=lambda s: s.map(age_key)))
        fig = px.line(gp, x="age_group", y="engagement_score", markers=True)
        fig.update_traces(line=dict(width=3, color=COLORS["blue"]), marker=dict(color=COLORS["blue"]))
        fig.update_layout(xaxis_title="Age group", yaxis_title="Avg usage frequency (0–4)")
        viz_card("Usage frequency by age group", fig)
    else:
        viz_card("Usage frequency by age group", go.Figure())

# -------------------- EXTRA ROW (2 CHARTS): Anxiety/Depression + Helpful resources --------------
c3, c4 = st.columns(2)

with c3:
    if all(c in view.columns for c in ["cancer_stage","gad2_flag_ge3","phq2_flag_ge3"]):
        gp = view.groupby("cancer_stage").agg(
            Anxiety=("gad2_flag_ge3", lambda s: np.mean(s==True)*100),
            Depression=("phq2_flag_ge3", lambda s: np.mean(s==True)*100)
        ).reset_index().rename(columns={"cancer_stage":"Stage"})
        tidy = gp.melt(id_vars="Stage", var_name="Metric", value_name="Percent")
        fig = px.bar(
            tidy, x="Percent", y="Stage", color="Metric", orientation="h", barmode="group",
            color_discrete_map={"Anxiety": COLORS["red"], "Depression": COLORS["green"]}
        )
        fig.update_layout(xaxis_title="Percent", yaxis_title="")
        viz_card("Anxiety vs Depression by stage", fig, height=H_CHART)
    else:
        viz_card("Anxiety vs Depression by stage", go.Figure())

with c4:
    if "most_helpful" in view.columns:
        vc = (view["most_helpful"].fillna("Unknown").astype(str).str.strip()
              .replace("", "Unknown").value_counts().head(7).reset_index())
        vc.columns = ["Resource","Count"]
        vc = vc.sort_values("Count")
        fig = px.bar(vc, x="Count", y="Resource", orientation="h")
        # FORCE royal blue bars (no gradients)
        fig.update_traces(marker=dict(color=COLORS["blue"]))
        fig.update_layout(xaxis_title="Count", yaxis_title="")
        viz_card("Top helpful resources", fig, height=H_CHART)
    else:
        viz_card("Top helpful resources", go.Figure())
