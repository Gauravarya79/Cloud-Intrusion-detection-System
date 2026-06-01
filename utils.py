"""
utils.py — Helper Utilities for Cloud IDS
Shared functions for data export, formatting, and state management.
"""

import pandas as pd
import io
from datetime import datetime


# ─── Severity Styling ─────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "High":   "#FF4B4B",
    "Medium": "#FFA500",
    "Low":    "#00C49A",
}

SEVERITY_ICONS = {
    "High":   "",
    "Medium": "",
    "Low":    "",
}

ATTACK_ICONS = {
    "DDoS":              "",
    "Restricted Access": "",
    "Brute Force":       "",
    "Anomaly":           "",
}


def severity_badge(severity: str) -> str:
    """Return an emoji badge for a severity level."""
    return SEVERITY_ICONS.get(severity, "")


def attack_icon(attack_type: str) -> str:
    """Return an emoji icon for an attack type."""
    return ATTACK_ICONS.get(attack_type, "")


# ─── Data Export ──────────────────────────────────────────────────────────────

def alerts_to_csv(alerts: list[dict]) -> bytes:
    """
    Convert a list of alert dicts to a CSV byte string for download.

    Args:
        alerts: List of alert dictionaries.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    if not alerts:
        return b"No alerts to export."

    # Flatten recommendations list to a string
    flat = []
    for a in alerts:
        row = {k: v for k, v in a.items() if k != "recommendations"}
        row["recommendations"] = " | ".join(a.get("recommendations", []))
        flat.append(row)

    df = pd.DataFrame(flat)
    return df.to_csv(index=False).encode("utf-8")


def logs_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a traffic log DataFrame to CSV bytes.

    Args:
        df: Traffic log DataFrame.

    Returns:
        UTF-8 encoded CSV bytes.
    """
    if df is None or df.empty:
        return b"No logs to export."
    return df.to_csv(index=False).encode("utf-8")


# ─── Filtering ────────────────────────────────────────────────────────────────

def filter_alerts(
    alerts: list[dict],
    ip_filter: str = "",
    type_filter: str = "All",
    severity_filter: str = "All",
) -> list[dict]:
    """
    Filter alerts by IP, type, and severity.

    Args:
        alerts: Full list of alert dicts.
        ip_filter: Partial or full IP string to match.
        type_filter: Attack type to filter on, or "All".
        severity_filter: Severity level to filter on, or "All".

    Returns:
        Filtered list of alerts.
    """
    result = alerts

    if ip_filter.strip():
        result = [a for a in result if ip_filter.strip() in a["ip"]]

    if type_filter != "All":
        result = [a for a in result if a["attack_type"] == type_filter]

    if severity_filter != "All":
        result = [a for a in result if a["severity"] == severity_filter]

    return result


def filter_logs(
    df: pd.DataFrame,
    ip_filter: str = "",
    endpoint_filter: str = "",
) -> pd.DataFrame:
    """
    Filter traffic logs by IP or endpoint.

    Args:
        df: Traffic log DataFrame.
        ip_filter: Partial IP string.
        endpoint_filter: Partial endpoint string.

    Returns:
        Filtered DataFrame.
    """
    if df is None or df.empty:
        return df

    if ip_filter.strip():
        df = df[df["ip"].str.contains(ip_filter.strip(), na=False)]

    if endpoint_filter.strip():
        df = df[df["endpoint"].str.contains(endpoint_filter.strip(), na=False)]

    return df


# ─── Summary Stats ────────────────────────────────────────────────────────────

def compute_summary(
    logs: pd.DataFrame,
    alerts: list[dict],
    blocked_ips: set,
) -> dict:
    """
    Compute summary statistics for the dashboard header cards.

    Args:
        logs: Traffic log DataFrame.
        alerts: List of alert dicts.
        blocked_ips: Set of blocked IP strings.

    Returns:
        Dict of metric name → value.
    """
    total_requests = int(logs["request_count"].sum()) if not logs.empty else 0
    unique_ips = int(logs["ip"].nunique()) if not logs.empty else 0
    high_alerts = sum(1 for a in alerts if a["severity"] == "High")
    blocked_count = len(blocked_ips)

    return {
        "Total Requests": total_requests,
        "Unique IPs":     unique_ips,
        "High Alerts":    high_alerts,
        "Blocked IPs":    blocked_count,
    }


# ─── Timestamp Utility ────────────────────────────────────────────────────────

def now_str() -> str:
    """Return current time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
