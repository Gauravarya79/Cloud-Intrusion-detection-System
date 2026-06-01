"""
simulator.py — Traffic Generator for Cloud IDS
Generates realistic and attack-simulated network traffic logs.
"""

import random
import time
from datetime import datetime
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

NORMAL_IPS = [f"192.168.1.{i}" for i in range(10, 50)]
ATTACKER_IPS = [f"10.0.0.{i}" for i in range(1, 20)]

NORMAL_ENDPOINTS = ["/home", "/about", "/products", "/contact", "/api/data", "/search"]
RESTRICTED_ENDPOINTS = ["/admin", "/admin/panel", "/root", "/.env", "/config", "/api/secret", "/db"]
LOGIN_ENDPOINT = "/login"

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE"]
STATUS_CODES_NORMAL = [200, 200, 200, 301, 304, 404]
STATUS_CODES_FAILED = [401, 403, 500]

# ─── Single Log Entry Generator ───────────────────────────────────────────────

def generate_log_entry(
    ip: str = None,
    force_attack: str = None
) -> dict:
    """
    Generate a single network log entry.

    Args:
        ip: Specific IP to use (random if None).
        force_attack: One of 'ddos', 'restricted', 'brute_force', or None for normal.

    Returns:
        A dict representing one log entry.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if force_attack == "ddos":
        ip = ip or random.choice(ATTACKER_IPS)
        endpoint = random.choice(NORMAL_ENDPOINTS)
        method = "GET"
        status = 200
        request_count = random.randint(80, 200)

    elif force_attack == "restricted":
        ip = ip or random.choice(ATTACKER_IPS)
        endpoint = random.choice(RESTRICTED_ENDPOINTS)
        method = random.choice(["GET", "POST"])
        status = random.choice([401, 403])
        request_count = random.randint(1, 5)

    elif force_attack == "brute_force":
        ip = ip or random.choice(ATTACKER_IPS)
        endpoint = LOGIN_ENDPOINT
        method = "POST"
        status = random.choice([401, 403])
        request_count = random.randint(10, 50)

    else:
        # Normal traffic
        ip = ip or random.choice(NORMAL_IPS)
        endpoint = random.choice(NORMAL_ENDPOINTS)
        method = random.choice(HTTP_METHODS)
        status = random.choice(STATUS_CODES_NORMAL)
        request_count = random.randint(1, 15)

    return {
        "timestamp": timestamp,
        "ip": ip,
        "endpoint": endpoint,
        "method": method,
        "status_code": status,
        "request_count": request_count,
        "bytes_sent": random.randint(200, 50000),
        "response_time_ms": random.randint(10, 3000),
    }


# ─── Batch Generator ──────────────────────────────────────────────────────────

def generate_traffic_batch(n: int = 10, attack_ratio: float = 0.2) -> list[dict]:
    """
    Generate a batch of mixed normal + attack traffic entries.

    Args:
        n: Number of entries to generate.
        attack_ratio: Fraction of entries that are attacks (0.0–1.0).

    Returns:
        List of log entry dicts.
    """
    entries = []
    attack_types = ["ddos", "restricted", "brute_force"]

    for _ in range(n):
        if random.random() < attack_ratio:
            attack = random.choice(attack_types)
            entries.append(generate_log_entry(force_attack=attack))
        else:
            entries.append(generate_log_entry())

    return entries


def generate_initial_history(n: int = 50) -> pd.DataFrame:
    """
    Generate a starting batch of logs to pre-populate the dashboard.

    Returns:
        DataFrame of log entries.
    """
    logs = generate_traffic_batch(n=n, attack_ratio=0.15)
    return pd.DataFrame(logs)
