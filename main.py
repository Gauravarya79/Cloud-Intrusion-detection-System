"""
main.py — Cloud Intrusion Detection System Dashboard
Run with: streamlit run main.py
"""

import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# ─── Local Modules ────────────────────────────────────────────────────────────
from simulator import generate_traffic_batch, generate_initial_history
from detector import run_detection, block_ip, flag_ip, filter_blocked_traffic
from utils import (
    alerts_to_csv, logs_to_csv, filter_alerts, filter_logs,
    compute_summary, severity_badge, attack_icon, SEVERITY_COLORS
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cloud IDS — Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg-dark:    #0a0e1a;
    --bg-card:    #111827;
    --bg-panel:   #1a2035;
    --accent:     #00d4ff;
    --accent2:    #7c3aed;
    --danger:     #ff4b4b;
    --warning:    #ffa500;
    --success:    #00c49a;
    --text:       #e2e8f0;
    --text-muted: #64748b;
    --border:     rgba(0,212,255,0.15);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; }

body, .stApp {
    background-color: var(--bg-dark) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 0 20px rgba(0,212,255,0.05);
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace; font-size: 2rem; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; color: var(--text) !important; }
h1 { color: var(--accent) !important; letter-spacing: 0.05em; }

.alert-card {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    font-family: 'Rajdhani', sans-serif;
}
.alert-high   { border-color: #ff4b4b; }
.alert-medium { border-color: #ffa500; }
.alert-low    { border-color: #00c49a; }
.alert-title  { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem; }
.alert-detail { font-size: 0.9rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
.alert-rec    { font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; }

[data-testid="stDataFrame"] { background: var(--bg-card); border-radius: 10px; }
.stDataFrame th { background: var(--bg-panel) !important; color: var(--accent) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
.stDataFrame td { color: var(--text) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.4; }
}
.status-active   { background: var(--success); }
.status-inactive { background: var(--text-muted); animation: none; }

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.js-plotly-plot { background: transparent !important; }

.alert-scroll {
    max-height: 480px;
    overflow-y: auto;
    padding-right: 4px;
}

.ml-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(124,58,237,0.08);
}
.ml-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #00d4ff);
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    margin-right: 6px;
}

.dataset-card {
    background: var(--bg-panel);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.dataset-badge {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #00d4ff);
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    margin-right: 6px;
}

.warn-card {
    background: rgba(255,165,0,0.08);
    border: 1px solid rgba(255,165,0,0.3);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #ffa500;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_ROWS = 15_000   # max rows kept in memory after sampling

def _smart_load(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV upload with class-aware sampling so time-ordered datasets
    (e.g. CICIDS Wednesday: all BENIGN in the first N rows) don't produce
    a single-class training set.

    Strategy
    --------
    1. Read the whole file into a BytesIO buffer (avoids re-upload on rerun).
    2. Peek at the first 500 rows to sniff the label column name.
    3. If the file has > SAMPLE_ROWS rows → stratified sample by label column
       so every class is proportionally represented.
    4. If no label column found → plain random sample (safe fallback).
    """
    raw = uploaded_file.read()          # read once; file pointer now at EOF
    buf = io.BytesIO(raw)

    # ── quick peek to find total rows & label column ──────────────────────
    peek = pd.read_csv(io.BytesIO(raw), nrows=500)
    total_rows = sum(1 for _ in io.BytesIO(raw)) - 1   # subtract header

    LABEL_CANDIDATES = ["label", " label", "Label", " Label",
                        "class", "attack", "attack_type", "category", "target"]
    label_col = next((c for c in peek.columns if c.strip().lower() in
                      [x.strip().lower() for x in LABEL_CANDIDATES]), None)

    if total_rows <= SAMPLE_ROWS:
        # Small file — just read everything
        return pd.read_csv(buf)

    # Large file — need to sample
    if label_col:
        # Stratified sample: read full file then sample per class
        df_full = pd.read_csv(io.BytesIO(raw), low_memory=False)
        # Each class contributes proportionally, but at least 1 row
        df_sampled = (
            df_full
            .groupby(label_col, group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, int(SAMPLE_ROWS * len(g) / len(df_full))),
                random_state=42
            ))
        )
        # Top-up or trim to exactly SAMPLE_ROWS
        if len(df_sampled) < SAMPLE_ROWS:
            extra = df_full.drop(df_sampled.index).sample(
                n=SAMPLE_ROWS - len(df_sampled), random_state=42
            )
            df_sampled = pd.concat([df_sampled, extra])
        return df_sampled.head(SAMPLE_ROWS).reset_index(drop=True)
    else:
        # No label column detected — plain random sample
        df_full = pd.read_csv(io.BytesIO(raw), low_memory=False)
        return df_full.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)


# ─── Session State Init ───────────────────────────────────────────────────────

def init_state():
    defaults = {
        "monitoring":    False,
        "logs":          generate_initial_history(40),
        "alerts":        [],
        "blocked_ips":   set(),
        "flagged_ips":   set(),
        "tick":          0,
        "ml_result":     None,
        "real_dataset":  None,
        "dataset_name":  None,
        "dataset_info":  None,   # stores class distribution info for display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Cloud IDS")
    st.markdown("**Intrusion Detection System**")
    st.markdown("---")

    mon_label = "⏹ Stop Monitoring" if st.session_state.monitoring else "▶ Start Monitoring"
    if st.button(mon_label, use_container_width=True):
        st.session_state.monitoring = not st.session_state.monitoring

    if st.session_state.monitoring:
        st.markdown('<p><span class="status-dot status-active"></span> <b>MONITORING ACTIVE</b></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p><span class="status-dot status-inactive"></span> <b>MONITORING STOPPED</b></p>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Simulation Settings")
    attack_ratio      = st.slider("Attack Traffic Ratio", 0.0, 1.0, 0.25, 0.05)
    batch_size        = st.slider("Batch Size (logs/tick)", 5, 30, 10)
    refresh_interval  = st.slider("Refresh Interval (sec)", 1, 10, 3)

    st.markdown("---")

    st.markdown("### 🔍 Filter Alerts")
    ip_filter   = st.text_input("Filter by IP", placeholder="e.g. 10.0.0")
    type_filter = st.selectbox("Attack Type", ["All", "DDoS", "Restricted Access", "Brute Force", "Anomaly"])
    sev_filter  = st.selectbox("Severity", ["All", "High", "Medium", "Low"])

    st.markdown("---")

    st.markdown("###  Export Data")
    alert_csv = alerts_to_csv(st.session_state.alerts)
    st.download_button("Download Alerts CSV", alert_csv, "ids_alerts.csv", "text/csv", use_container_width=True)
    log_csv = logs_to_csv(st.session_state.logs)
    st.download_button("Download Logs CSV", log_csv, "ids_logs.csv", "text/csv", use_container_width=True)

    st.markdown("---")

    if st.button(" Reset Dashboard", use_container_width=True):
        for k in ["logs", "alerts", "blocked_ips", "flagged_ips", "tick",
                  "ml_result", "real_dataset", "dataset_name", "dataset_info"]:
            del st.session_state[k]
        init_state()
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("# CLOUD INTRUSION DETECTION SYSTEM")
st.markdown(
    f"<span style='color:#64748b; font-family:JetBrains Mono; font-size:0.85rem;'>"
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─── Monitoring Tick ──────────────────────────────────────────────────────────

if st.session_state.monitoring:
    new_logs = generate_traffic_batch(n=batch_size, attack_ratio=attack_ratio)
    new_df   = pd.DataFrame(new_logs)
    new_df   = filter_blocked_traffic(new_df, st.session_state.blocked_ips)
    st.session_state.logs = pd.concat(
        [st.session_state.logs, new_df], ignore_index=True
    ).tail(500)
    new_alerts = run_detection(new_df)
    st.session_state.alerts = (new_alerts + st.session_state.alerts)[:200]
    st.session_state.tick += 1


# ─── Summary Cards ────────────────────────────────────────────────────────────

summary = compute_summary(st.session_state.logs, st.session_state.alerts, st.session_state.blocked_ips)
c1, c2, c3, c4 = st.columns(4)
for col, (label, val) in zip([c1, c2, c3, c4], summary.items()):
    col.metric(label, f"{val:,}")

st.markdown("---")

# ─── Main Layout ──────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1.4, 1], gap="large")

with left_col:

    st.markdown('<div class="section-header">📡 Live Traffic Monitor</div>', unsafe_allow_html=True)
    logs_df      = st.session_state.logs.copy()
    display_cols = ["timestamp", "ip", "endpoint", "method", "status_code", "request_count"]
    filtered_logs = filter_logs(logs_df, ip_filter, "")
    st.dataframe(
        filtered_logs[display_cols].tail(15).iloc[::-1].reset_index(drop=True),
        use_container_width=True, height=230,
    )

    st.markdown('<div class="section-header"> Requests Over Time (by IP)</div>', unsafe_allow_html=True)
    if not logs_df.empty:
        top_ips = logs_df.groupby("ip")["request_count"].sum().nlargest(6).index.tolist()
        top_df  = logs_df[logs_df["ip"].isin(top_ips)].copy()
        top_df["timestamp"] = pd.to_datetime(top_df["timestamp"])
        top_df = top_df.sort_values("timestamp")
        fig_line = px.line(
            top_df, x="timestamp", y="request_count", color="ip",
            color_discrete_sequence=["#00d4ff","#7c3aed","#ff4b4b","#ffa500","#00c49a","#f472b6"],
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.9)",
            font_color="#e2e8f0", font_family="JetBrains Mono",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0), height=240,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown('<div class="section-header">⚔️ Attack Frequency by Type</div>', unsafe_allow_html=True)
    if st.session_state.alerts:
        alert_df = pd.DataFrame(st.session_state.alerts)
        freq = alert_df.groupby(["attack_type","severity"]).size().reset_index(name="count")
        fig_bar = px.bar(
            freq, x="attack_type", y="count", color="severity",
            color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00c49a"},
            barmode="stack",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.9)",
            font_color="#e2e8f0", font_family="JetBrains Mono",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0), height=220,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No attack data yet. Start monitoring to see charts.")

    st.markdown('<div class="section-header"> Top Traffic Sources</div>', unsafe_allow_html=True)
    if not logs_df.empty:
        ip_totals = logs_df.groupby("ip")["request_count"].sum().nlargest(8).reset_index()
        fig_pie = px.pie(
            ip_totals, names="ip", values="request_count",
            color_discrete_sequence=px.colors.sequential.Plasma, hole=0.5,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            font_family="JetBrains Mono", legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0), height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)


with right_col:

    filtered_alerts = filter_alerts(st.session_state.alerts, ip_filter, type_filter, sev_filter)
    st.markdown(
        f'<div class="section-header"> Alerts '
        f'<span style="color:#64748b; font-size:0.85rem;">({len(filtered_alerts)} shown)</span></div>',
        unsafe_allow_html=True
    )

    if not filtered_alerts:
        if st.session_state.monitoring:
            st.success("✅ No intrusions detected. System clean.")
        else:
            st.info("▶ Press **Start Monitoring** to begin detection.")
    else:
        st.markdown('<div class="alert-scroll">', unsafe_allow_html=True)
        for alert in filtered_alerts[:30]:
            sev   = alert["severity"]
            atype = alert["attack_type"]
            css   = {"High":"alert-high","Medium":"alert-medium","Low":"alert-low"}[sev]
            icon  = attack_icon(atype)
            badge = severity_badge(sev)
            recs  = "<br>".join(alert["recommendations"][:2])
            blocked_tag = " <b>BLOCKED</b> · " if alert.get("blocked") else ""
            st.markdown(f"""
            <div class="alert-card {css}">
                <div class="alert-title">{icon} {atype} &nbsp;|&nbsp; {badge} {sev}</div>
                <div class="alert-detail">
                    {blocked_tag} {alert['timestamp']}<br>
                     IP: <b>{alert['ip']}</b><br>
                      {alert['detail']}
                </div>
                <div class="alert-rec">{recs}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-header"> Prevention Actions</div>', unsafe_allow_html=True)
    alert_ips = list({a["ip"] for a in st.session_state.alerts})
    if alert_ips:
        selected_ip = st.selectbox("Select IP for action", alert_ips)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(" Block IP", use_container_width=True):
                msg = block_ip(selected_ip, st.session_state.blocked_ips)
                for a in st.session_state.alerts:
                    if a["ip"] == selected_ip:
                        a["blocked"] = True
                st.success(msg)
        with btn_col2:
            if st.button(" Flag IP", use_container_width=True):
                msg = flag_ip(selected_ip, st.session_state.flagged_ips)
                st.warning(msg)
    else:
        st.info("No IPs to act on yet.")

    if st.session_state.blocked_ips or st.session_state.flagged_ips:
        st.markdown("**Blocked IPs:**")
        for ip in sorted(st.session_state.blocked_ips):
            st.markdown(f"<span style='color:#ff4b4b; font-family:JetBrains Mono;'>🚫 {ip}</span>", unsafe_allow_html=True)
        st.markdown("**Flagged IPs:**")
        for ip in sorted(st.session_state.flagged_ips):
            st.markdown(f"<span style='color:#ffa500; font-family:JetBrains Mono;'>🚩 {ip}</span>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-header">💡 Security Recommendations</div>', unsafe_allow_html=True)
    attack_types_seen = list({a["attack_type"] for a in st.session_state.alerts})
    if attack_types_seen:
        tabs = st.tabs(attack_types_seen[:4])
        for tab, atype in zip(tabs, attack_types_seen[:4]):
            with tab:
                recs = next((a["recommendations"] for a in st.session_state.alerts if a["attack_type"] == atype), [])
                for rec in recs:
                    st.markdown(f"- {rec}")
    else:
        st.markdown("""
        -  Keep all services patched and updated
        -  Enable firewall rules for all cloud resources
        -  Set up centralized log monitoring
        -  Enforce least-privilege IAM policies
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ML ANALYSIS SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🤖 ML Analysis — Hybrid Feature Selection + Random Forest Classifier")
st.markdown(
    "<span style='color:#64748b; font-size:0.9rem; font-family:JetBrains Mono;'>"
    "Filter (Chi²) + Wrapper (RFE) hybrid feature selection → Random Forest classifier. "
    "Supports simulated data and real-world datasets (KDD99 / CICIDS2017 / Custom CSV).</span>",
    unsafe_allow_html=True
)
st.markdown("")

ml_col1, ml_col2 = st.columns([1, 2], gap="large")

with ml_col1:
    st.markdown('<div class="section-header">⚙️ ML Controls</div>', unsafe_allow_html=True)

    data_source = st.radio(
        " Data Source",
        [" Simulated Data (Live)", " Real Dataset (Upload CSV)"],
        help="Choose between live simulated data or upload a real dataset"
    )
    use_real = "Real Dataset" in data_source

    if use_real:
        st.markdown("---")
        st.markdown("###  Upload Dataset")
        st.markdown(
            "<span style='color:#64748b; font-size:0.82rem;'>"
            "Supported: KDD Cup 99, CICIDS 2017, or any CSV with a label column.<br>"
            "Large files are <b>stratified-sampled</b> to 15,000 rows so all attack "
            "classes are preserved.</span>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload KDD99, CICIDS2017, or any CSV with attack labels"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("📂 Loading & sampling dataset…"):
                    # ── KEY FIX: stratified sample instead of nrows=15000 ──
                    df_uploaded = _smart_load(uploaded_file)

                st.session_state.real_dataset  = df_uploaded
                st.session_state.dataset_name  = uploaded_file.name

                # ── Show class distribution so user can verify sampling ────
                LABEL_CANDIDATES = ["label", " label", "Label", " Label",
                                    "class", "attack", "attack_type", "category", "target"]
                lbl_col = next(
                    (c for c in df_uploaded.columns
                     if c.strip().lower() in [x.strip().lower() for x in LABEL_CANDIDATES]),
                    None
                )
                st.session_state.dataset_info = {
                    "label_col": lbl_col,
                    "class_dist": df_uploaded[lbl_col].value_counts().to_dict() if lbl_col else {}
                }

                st.success(f"✅ Loaded: **{uploaded_file.name}**")

                # Dataset card
                col_preview = ", ".join(df_uploaded.columns[:5].tolist())
                if len(df_uploaded.columns) > 5:
                    col_preview += "…"
                st.markdown(f"""
                <div class='dataset-card'>
                    <span class='dataset-badge'>DATASET</span><br><br>
                     <b>File:</b> {uploaded_file.name}<br>
                     <b>Rows (sampled):</b> {len(df_uploaded):,}<br>
                     <b>Columns:</b> {len(df_uploaded.columns)}<br>
                     <b>Columns:</b> {col_preview}
                </div>
                """, unsafe_allow_html=True)

                # Class distribution
                if lbl_col and st.session_state.dataset_info["class_dist"]:
                    n_classes = len(st.session_state.dataset_info["class_dist"])
                    if n_classes < 2:
                        st.markdown(
                            "<div class='warn-card'>⚠️ Only 1 class found after sampling — "
                            "try a different file or switch dataset type to <b>custom</b>.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        with st.expander(f" Class distribution ({n_classes} classes)"):
                            for cls, cnt in sorted(
                                st.session_state.dataset_info["class_dist"].items(),
                                key=lambda x: -x[1]
                            ):
                                pct = cnt / len(df_uploaded) * 100
                                st.markdown(
                                    f"<span style='font-family:JetBrains Mono; font-size:0.82rem;'>"
                                    f"<b>{str(cls)[:30]}</b>: {cnt:,} ({pct:.1f}%)</span>",
                                    unsafe_allow_html=True
                                )

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

        # Dataset type selector
        if st.session_state.real_dataset is not None:
            dtype_hint = st.selectbox(
                "Dataset Type (auto-detect if unsure)",
                ["auto", "kdd99", "cicids", "custom"],
                help="auto = automatic detection"
            )
        else:
            dtype_hint = "auto"
            st.info("👆 Upload a CSV file to proceed")

        with st.expander(" Where to get datasets?"):
            st.markdown("""
            **KDD Cup 99:**
            - [UCI Repository](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
            - File: `kddcup.data_10_percent.gz`

            **CICIDS 2017:**
            - [UNB Website](https://www.unb.ca/cic/datasets/ids-2017.html)
            - Download individual day CSV files (Monday–Friday)

            **Custom CSV requirements:**
            - Must have a column named: `label`, `class`, `attack`, or `target`
            - Remaining columns should be numeric features
            """)

    else:
        total_logs = len(st.session_state.logs)
        st.metric("Logs Available for Training", f"{total_logs:,}")
        if total_logs < 20:
            st.warning(" Need at least 20 logs. Start monitoring first!")
        else:
            st.success(f"✅ {total_logs} logs ready for training")
        dtype_hint = "auto"

    st.markdown("---")

    # ── Run Button ────────────────────────────────────────────────────────────
    btn_disabled = use_real and st.session_state.real_dataset is None
    run_ml = st.button(
        " Run ML Classifier",
        use_container_width=True,
        disabled=btn_disabled
    )
    if btn_disabled:
        st.warning("Upload a CSV dataset first!")

    if run_ml:
        from ml import train_model, train_on_real_dataset

        with st.spinner(" Hybrid Feature Selection + Training Random Forest…"):
            if use_real and st.session_state.real_dataset is not None:
                result = train_on_real_dataset(
                    st.session_state.real_dataset,
                    dataset_type=dtype_hint
                )
                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                    result = None
            else:
                result = train_model(st.session_state.logs)

        if result is None:
            st.error("❌ Not enough varied data! Run monitoring longer.")
            st.session_state.ml_result = None
        else:
            st.session_state.ml_result = result
            st.success("✅ Model trained successfully!")

    # ── Feature Selection Steps ───────────────────────────────────────────────
    if st.session_state.ml_result:
        result = st.session_state.ml_result
        st.markdown("---")
        st.markdown("### 🔍 Hybrid Feature Selection")

        ds_type = result.get("dataset_type", "simulated")
        ds_badge_color = "#059669" if ds_type != "simulated" else "#7c3aed"
        st.markdown(
            f"<span style='background:{ds_badge_color}; color:white; border-radius:6px; "
            f"padding:2px 10px; font-size:0.78rem; font-family:JetBrains Mono;'>"
            f"{'🟢 REAL: ' + ds_type.upper() if ds_type != 'simulated' else '🔵 SIMULATED DATA'}"
            f"</span> &nbsp; "
            f"<span style='color:#64748b; font-size:0.82rem;'>"
            f"{result.get('total_samples', 0):,} samples</span>",
            unsafe_allow_html=True
        )
        st.markdown("")

        st.markdown("**Step 1 — Filter (Chi²):**")
        for f in result["filter_features"]:
            st.markdown(f"<span class='ml-badge'>✓</span> `{f}`", unsafe_allow_html=True)

        st.markdown("**Step 2 — Wrapper (RFE):**")
        for f in result["selected_features"]:
            st.markdown(f"<span class='ml-badge'>★</span> `{f}`", unsafe_allow_html=True)

        st.markdown(
            f"<span style='color:#64748b; font-size:0.82rem;'>"
            f"{len(result['filter_features'])} → {len(result['selected_features'])} features selected</span>",
            unsafe_allow_html=True
        )


with ml_col2:
    if st.session_state.ml_result:
        result  = st.session_state.ml_result
        acc_pct = result["accuracy"] * 100
        acc_color = "#00c49a" if acc_pct >= 80 else "#ffa500" if acc_pct >= 60 else "#ff4b4b"
        ds_type   = result.get("dataset_type", "simulated")

        accuracy_note = (
            "<span style='color:#64748b; font-size:0.8rem; margin-left:0.5rem;'>"
            "Real-world dataset — realistic accuracy</span>"
            if ds_type != "simulated" else
            "<span style='color:#ffa500; font-size:0.8rem; margin-left:0.5rem;'>"
            "⚠️ Simulated data — may be inflated</span>"
        )

        st.markdown(
            f'<div class="ml-card">'
            f'<span style="font-size:1rem; color:#64748b; font-family:JetBrains Mono;">MODEL ACCURACY</span><br>'
            f'<span style="font-size:3rem; font-weight:700; color:{acc_color}; font-family:JetBrains Mono;">'
            f'{acc_pct:.2f}%</span>'
            f'<span style="color:#64748b; font-size:0.85rem; margin-left:1rem;">'
            f'Random Forest · 100 trees · max_depth=8</span>'
            f'<br>{accuracy_note}'
            f'</div>',
            unsafe_allow_html=True
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("####  Feature Importance")
            fig_imp = px.bar(
                result["feature_importance"],
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal",
            )
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.9)",
                font_color="#e2e8f0", font_family="JetBrains Mono",
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with chart_col2:
            st.markdown("####  Confusion Matrix")
            label_map     = result["label_map"]
            unique_labels = result["unique_labels"]
            cm            = result["confusion_matrix"]
            cm_labels     = [label_map.get(i, f"Class {i}") for i in unique_labels]

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=cm_labels, y=cm_labels,
                color_continuous_scale="Purples",
                text_auto=True,
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.9)",
                font_color="#e2e8f0", font_family="JetBrains Mono",
                height=250, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("####  Classification Report")
        report_df = pd.DataFrame(result["report"]).transpose().round(3)
        st.dataframe(report_df, use_container_width=True, height=200)

    else:
        st.markdown(
            '<div class="ml-card" style="text-align:center; padding: 3rem;">'
            '<span style="font-size:3rem;">🤖</span><br><br>'
            '<span style="color:#64748b; font-family:JetBrains Mono;">'
            'Press "Run ML Classifier" to train the model<br>'
            'and see results here.</span>'
            '</div>',
            unsafe_allow_html=True
        )


# ─── Auto-Refresh ─────────────────────────────────────────────────────────────

if st.session_state.monitoring:
    time.sleep(refresh_interval)
    st.rerun()