"""
ml.py — ML Classifier with Hybrid Feature Selection for Cloud IDS
Hybrid Approach: Filter Method (Chi2) + Wrapper Method (RFE)
Classifier: Random Forest
Supports: Simulated data + Real datasets (KDD99, CICIDS2017, Custom CSV)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─── Label Encoding ───────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame):
    """Encode categorical features for ML model."""
    df = df.copy()
    encoders = {}
    for col in ["ip", "endpoint", "method"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


# ─── Label Generator ──────────────────────────────────────────────────────────

def assign_labels(df: pd.DataFrame) -> pd.Series:
    """
    Assign attack labels based on traffic patterns.
    0 = Normal, 1 = DDoS, 2 = Brute Force, 3 = Restricted Access
    """
    restricted = {"/admin", "/admin/panel", "/root",
                  "/.env", "/config", "/api/secret", "/db"}
    labels = []
    for _, row in df.iterrows():
        if row["request_count"] >= 50:
            labels.append(1)   # DDoS
        elif row.get("endpoint", "") == "/login" and row["status_code"] in [401, 403]:
            labels.append(2)   # Brute Force
        elif row.get("endpoint", "") in restricted:
            labels.append(3)   # Restricted Access
        else:
            labels.append(0)   # Normal
    return pd.Series(labels)


# ─── Hybrid Feature Selection ─────────────────────────────────────────────────

def hybrid_feature_selection(X: pd.DataFrame, y: pd.Series):
    """
    Hybrid Feature Selection:
      Step 1 — Filter Method  : SelectKBest with Chi2
      Step 2 — Wrapper Method : RFE with RandomForest
    """
    feature_names = X.columns.tolist()

    # Step 1: Filter — SelectKBest (Chi2)
    k = min(5, X.shape[1])
    selector_filter = SelectKBest(chi2, k=k)
    selector_filter.fit_transform(np.abs(X.values), y)
    filter_mask = selector_filter.get_support()
    filter_features = [f for f, m in zip(feature_names, filter_mask) if m]

    # Step 2: Wrapper — RFE with RandomForest
    rfe_model = RandomForestClassifier(n_estimators=50, random_state=42)
    n_features = min(4, len(filter_features))
    rfe = RFE(rfe_model, n_features_to_select=n_features)
    rfe.fit(X[filter_features], y)
    rfe_mask = rfe.support_
    final_features = [f for f, m in zip(filter_features, rfe_mask) if m]

    return final_features, filter_features


# ─── KDD99 Dataset Loader ─────────────────────────────────────────────────────

KDD99_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

KDD99_LABEL_MAP = {
    "normal": 0,
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "mailbomb": 1,
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3,
    "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "worm": 3,
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "httptunnel": 4, "ps": 4, "sqlattack": 4, "xterm": 4,
}

CICIDS_LABEL_MAP = {
    "benign": 0,
    "normal": 0,
    "dos hulk": 1, "dos goldeneye": 1, "dos slowloris": 1,
    "dos slowhttptest": 1, "ddos": 1,
    "portscan": 2, "ftp-patator": 3, "ssh-patator": 3,
    "web attack \u2013 brute force": 3, "brute force": 3,
    "web attack \u2013 xss": 4, "web attack \u2013 sql injection": 4,
    "infiltration": 4, "heartbleed": 4, "botnet": 4,
}


def load_kdd99(df: pd.DataFrame):
    """Process KDD99 dataset into standard format."""
    if len(df.columns) == 42:
        df.columns = KDD99_COLUMNS
    elif len(df.columns) == 43:
        df.columns = KDD99_COLUMNS + ["extra"]

    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    df["label_encoded"] = df["label"].str.lower().map(KDD99_LABEL_MAP).fillna(0).astype(int)

    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    numeric_features = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "rerror_rate", "same_srv_rate", "dst_host_count",
        "dst_host_srv_count", "protocol_type", "service", "flag"
    ]
    available = [f for f in numeric_features if f in df.columns]
    return df[available], df["label_encoded"], available


def load_cicids(df: pd.DataFrame):
    """Process CICIDS2017 dataset into standard format."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    label_col = None
    for c in ["label", "attack_type", "class"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise ValueError("No label column found in CICIDS dataset")

    # Strip leading/trailing whitespace BEFORE mapping — CICIDS CSVs often have
    # values like " DoS Hulk" or " BENIGN" with a leading space that silently
    # fall through .map() and become 0 (Benign), producing a single-class dataset.
    df["label_encoded"] = (
        df[label_col].astype(str).str.strip().str.lower()
        .map(CICIDS_LABEL_MAP).fillna(0).astype(int)
    )

    drop_cols = [label_col, "label_encoded", "flow_id", "source_ip",
                 "destination_ip", "timestamp", "src_ip", "dst_ip"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, df["label_encoded"], X.columns.tolist()


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Auto-detect dataset type from columns."""
    cols = [c.lower().strip() for c in df.columns]
    if len(df.columns) in [41, 42, 43] and "duration" in cols:
        return "kdd99"
    if any("flow" in c for c in cols) or "flow_duration" in cols:
        return "cicids"
    if "request_count" in cols:
        return "simulated"
    return "custom"


def load_custom_dataset(df: pd.DataFrame):
    """
    Load any custom CSV dataset.
    Tries to find label column and numeric features automatically.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    label_col = None
    for c in ["label", "class", "attack", "attack_type", "category", "target"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        raise ValueError("No label column found! CSV must have a column named: label, class, attack, or target")

    le_label = LabelEncoder()
    y = pd.Series(le_label.fit_transform(df[label_col].astype(str)))

    drop_cols = [label_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.empty:
        raise ValueError("No numeric feature columns found in CSV!")

    label_names = le_label.classes_.tolist()
    return X, y, X.columns.tolist(), label_names


# ─── Train on Real Dataset ────────────────────────────────────────────────────

def train_on_real_dataset(df: pd.DataFrame, dataset_type: str = "auto"):
    """
    Train ML model on real dataset (KDD99, CICIDS, or Custom CSV).
    Returns same result dict format as train_model().
    """
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(df)

    label_map = {0: "Normal", 1: "DDoS", 2: "Probe/BruteForce",
                 3: "R2L/BruteForce", 4: "Restricted/U2R"}
    label_names = None

    try:
        if dataset_type == "kdd99":
            X, y, feature_cols = load_kdd99(df.copy())
            label_map = {0: "Normal", 1: "DoS/DDoS", 2: "Probe",
                         3: "R2L", 4: "U2R"}
        elif dataset_type == "cicids":
            X, y, feature_cols = load_cicids(df.copy())
            label_map = {0: "Benign", 1: "DDoS/DoS", 2: "PortScan",
                         3: "BruteForce", 4: "Web Attack"}
        else:
            X, y, feature_cols, label_names = load_custom_dataset(df.copy())
            label_map = {i: name for i, name in enumerate(label_names)}

    except Exception as e:
        return {"error": str(e)}

    # Sample if too large (max 10000 rows for speed)
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    if y.nunique() < 2:
        return {"error": "Dataset has only one class — cannot train!"}

    # Hybrid Feature Selection
    try:
        selected_features, filter_features = hybrid_feature_selection(X, y)
    except Exception:
        selected_features = X.columns[:4].tolist()
        filter_features = X.columns[:5].tolist()

    if not selected_features:
        selected_features = feature_cols[:3]

    # Train / Test Split
    X_sel = X[selected_features]
    min_class_count = y.value_counts().min()
    use_stratify = y if (y.nunique() <= 10 and min_class_count >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.25, random_state=42,
        stratify=use_stratify
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ── FIX: derive labels only from what actually appears in y_test/y_pred ──
    # Using y.unique() caused a mismatch when minority classes were absent
    # from the test split, making target_names longer than classes seen.
    eval_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [label_map.get(i, f"Class {i}") for i in eval_labels]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=eval_labels)
    report = classification_report(
        y_test, y_pred,
        labels=eval_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # ─────────────────────────────────────────────────────────────────────────

    importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "feature_importance": importance,
        "selected_features": selected_features,
        "filter_features": filter_features,
        "label_map": label_map,
        "unique_labels": eval_labels,          # ← store eval_labels, not y.unique()
        "dataset_type": dataset_type,
        "total_samples": len(X),
        "error": None,
    }


# ─── Train on Simulated Data (original) ──────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Full ML pipeline on simulated/live data.
    Same as before — used when no real dataset is uploaded.
    """
    if df is None or df.empty or len(df) < 20:
        return None

    encoded_df, _ = encode_features(df)

    feature_cols = [
        "ip", "endpoint", "method",
        "status_code", "request_count",
        "bytes_sent", "response_time_ms"
    ]
    X = encoded_df[feature_cols]
    y = assign_labels(df)

    if y.nunique() < 2:
        return None

    selected_features, filter_features = hybrid_feature_selection(X, y)
    if not selected_features:
        selected_features = feature_cols[:3]

    X_sel = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    label_map = {0: "Normal", 1: "DDoS", 2: "Brute Force", 3: "Restricted Access"}
    accuracy = accuracy_score(y_test, y_pred)

    # ── FIX: same fix applied to simulated path ───────────────────────────────
    eval_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [label_map[i] for i in eval_labels]

    cm = confusion_matrix(y_test, y_pred, labels=eval_labels)
    report = classification_report(
        y_test, y_pred,
        labels=eval_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # ─────────────────────────────────────────────────────────────────────────

    importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "feature_importance": importance,
        "selected_features": selected_features,
        "filter_features": filter_features,
        "label_map": label_map,
        "unique_labels": eval_labels,          # ← store eval_labels, not y.unique()
        "dataset_type": "simulated",
        "total_samples": len(df),
        "error": None,
    }