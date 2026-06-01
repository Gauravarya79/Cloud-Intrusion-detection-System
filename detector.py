"""
detector.py — Intrusion Detection Engine for Cloud IDS
Implements rule-based and basic anomaly detection algorithms.
"""

import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─── Detection Thresholds ─────────────────────────────────────────────────────

DDOS_THRESHOLD = 50           # requests from same IP → DDoS alert
BRUTE_FORCE_THRESHOLD = 5     # failed logins from same IP → brute-force alert
RESTRICTED_THRESHOLD = 1      # any access to restricted endpoint → alert
ANOMALY_ZSCORE_THRESHOLD = 2  # z-score for anomaly detection

RESTRICTED_ENDPOINTS = {"/admin", "/admin/panel", "/root", "/.env",
                         "/config", "/api/secret", "/db"}
LOGIN_ENDPOINT = "/login"
FAILED_STATUS_CODES = {401, 403, 500}

# ─── Severity Helpers ─────────────────────────────────────────────────────────

def _ddos_severity(count: int) -> str:
    if count >= 150:
        return "High"
    elif count >= 80:
        return "Medium"
    return "Low"


def _brute_force_severity(count: int) -> str:
    if count >= 20:
        return "High"
    elif count >= 10:
        return "Medium"
    return "Low"


# ─── Recommendations Map ──────────────────────────────────────────────────────

RECOMMENDATIONS = {
    "DDoS": [
        " Enable rate limiting on your API gateway",
        " Configure a Web Application Firewall (WAF)",
        " Use a CDN with DDoS protection (e.g., Cloudflare)",
        " Set up auto-scaling to absorb traffic spikes",
    ],
    "Restricted Access": [
        " Move admin endpoints behind VPN",
        " Return 404 (not 401/403) for hidden admin routes",
        " Implement IP allowlisting for sensitive routes",
        " Set up intrusion alerts for repeated access attempts",
    ],
    "Brute Force": [
        " Enforce multi-factor authentication (MFA)",
        " Implement account lockout after N failed attempts",
        " Add CAPTCHA to login endpoints",
        " Use geo-blocking for anomalous login origins",
    ],
    "Anomaly": [
        " Review recent traffic patterns manually",
        " Correlate with server logs for false positives",
        " Train an ML model on labeled traffic data",
    ],
}

# ─── Core Detection Functions ─────────────────────────────────────────────────

def detect_ddos(df: pd.DataFrame) -> list[dict]:
    """
    Detect DDoS-like behavior: too many requests from a single IP.

    Args:
        df: DataFrame of recent log entries.

    Returns:
        List of alert dicts.
    """
    alerts = []
    ip_counts = df.groupby("ip")["request_count"].sum()

    for ip, total in ip_counts.items():
        if total >= DDOS_THRESHOLD:
            severity = _ddos_severity(total)
            alerts.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ip": ip,
                "attack_type": "DDoS",
                "severity": severity,
                "detail": f"{total} requests detected (threshold: {DDOS_THRESHOLD})",
                "recommendations": RECOMMENDATIONS["DDoS"],
                "blocked": False,
            })
    return alerts


def detect_restricted_access(df: pd.DataFrame) -> list[dict]:
    """
    Detect access attempts to restricted/sensitive endpoints.

    Args:
        df: DataFrame of recent log entries.

    Returns:
        List of alert dicts.
    """
    alerts = []
    restricted_df = df[df["endpoint"].isin(RESTRICTED_ENDPOINTS)]

    for _, row in restricted_df.iterrows():
        alerts.append({
            "timestamp": row["timestamp"],
            "ip": row["ip"],
            "attack_type": "Restricted Access",
            "severity": "High",
            "detail": f"Access to {row['endpoint']} (status: {row['status_code']})",
            "recommendations": RECOMMENDATIONS["Restricted Access"],
            "blocked": False,
        })
    return alerts


def detect_brute_force(df: pd.DataFrame) -> list[dict]:
    """
    Detect brute-force login attempts: repeated failed logins from same IP.

    Args:
        df: DataFrame of recent log entries.

    Returns:
        List of alert dicts.
    """
    alerts = []
    login_df = df[
        (df["endpoint"] == LOGIN_ENDPOINT) &
        (df["status_code"].isin(FAILED_STATUS_CODES))
    ]
    ip_counts = login_df.groupby("ip").size()

    for ip, count in ip_counts.items():
        if count >= BRUTE_FORCE_THRESHOLD:
            severity = _brute_force_severity(count)
            alerts.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ip": ip,
                "attack_type": "Brute Force",
                "severity": severity,
                "detail": f"{count} failed login attempts detected",
                "recommendations": RECOMMENDATIONS["Brute Force"],
                "blocked": False,
            })
    return alerts


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """
    Basic anomaly detection using Z-score on request counts per IP.

    Args:
        df: DataFrame of recent log entries.

    Returns:
        List of alert dicts for statistically anomalous IPs.
    """
    alerts = []
    if len(df) < 5:
        return alerts  # Not enough data

    ip_counts = df.groupby("ip")["request_count"].sum()
    mean = ip_counts.mean()
    std = ip_counts.std()

    if std == 0:
        return alerts

    for ip, count in ip_counts.items():
        zscore = (count - mean) / std
        if zscore > ANOMALY_ZSCORE_THRESHOLD:
            alerts.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ip": ip,
                "attack_type": "Anomaly",
                "severity": "Medium",
                "detail": f"Unusual traffic volume (z-score: {zscore:.2f})",
                "recommendations": RECOMMENDATIONS["Anomaly"],
                "blocked": False,
            })
    return alerts


# ─── Master Detector ──────────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame) -> list[dict]:
    """
    Run all detection modules on a batch of logs.

    Args:
        df: DataFrame of log entries to analyze.

    Returns:
        Combined list of unique alerts sorted by severity.
    """
    if df.empty:
        return []

    all_alerts = []
    all_alerts.extend(detect_ddos(df))
    all_alerts.extend(detect_restricted_access(df))
    all_alerts.extend(detect_brute_force(df))
    all_alerts.extend(detect_anomalies(df))

    # Deduplicate: same IP + attack type within same second
    seen = set()
    unique_alerts = []
    for alert in all_alerts:
        key = (alert["ip"], alert["attack_type"])
        if key not in seen:
            seen.add(key)
            unique_alerts.append(alert)

    # Sort: High → Medium → Low
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    unique_alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))

    return unique_alerts


# ─── Prevention Actions ───────────────────────────────────────────────────────

def block_ip(ip: str, blocked_ips: set) -> str:
    """Simulate blocking an IP address."""
    blocked_ips.add(ip)
    return f" IP {ip} has been blocked."


def flag_ip(ip: str, flagged_ips: set) -> str:
    """Simulate flagging a suspicious IP for review."""
    flagged_ips.add(ip)
    return f" IP {ip} has been flagged for review."


def filter_blocked_traffic(df: pd.DataFrame, blocked_ips: set) -> pd.DataFrame:
    """Remove rows from blocked IPs from a DataFrame."""
    if not blocked_ips:
        return df
    return df[~df["ip"].isin(blocked_ips)]
