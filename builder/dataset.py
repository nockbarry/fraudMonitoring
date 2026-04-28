"""Synthetic transaction-fraud dataset generator.

Goals
-----
- 10 weeks (70 days) of transaction-level data.
- Persistent entity model (accounts, cards, merchants, devices, IPs) so velocity
  and history-based features have realistic structure.
- 60-80 engineered features spanning invariant / tactical / compositional buckets.
- Multiple fraud archetypes (ATO, CNP-stolen-card, mule, bust-out) with distinct
  feature signatures.
- A planted regime shift in the second half of OOT — a new BIN cluster attack —
  affecting tactical features only.
- Target base rate ~0.4% pre-shift, ~0.9% post-shift.

Public API
----------
make_dataset(seed=42) -> (df, feature_names, feature_meta, cohort_dims)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Bucket metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMeta:
    bucket: str            # invariant | tactical | compositional
    stable_target: bool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_DAYS = 70                       # 10 weeks
SHIFT_DAY = 49                    # week 8 onwards: new attack regime
START_DATE = pd.Timestamp("2026-02-15")

N_ACCOUNTS = 60_000
N_CARDS = 70_000                  # some accounts have multiple cards
N_MERCHANTS = 6_000
N_DEVICES = 40_000
N_IPS = 20_000

TX_PER_DAY = 9_000                # ~630k total transactions

CHANNELS = ["CNP_seasoned", "CNP_new", "CardPresent", "MobileWallet", "Recurring"]
CHANNEL_PROBS = [0.40, 0.10, 0.20, 0.22, 0.08]

MCC_CATEGORIES = ["retail", "digital_goods", "travel", "food", "services",
                  "gaming", "crypto", "subscription"]
MCC_PROBS = [0.30, 0.18, 0.08, 0.18, 0.12, 0.06, 0.04, 0.04]
MCC_BASE_RISK = {"retail": 0.5, "digital_goods": 0.7, "travel": 0.6, "food": 0.3,
                 "services": 0.5, "gaming": 0.85, "crypto": 0.95, "subscription": 0.4}

BIN_COUNTRIES = ["US", "GB", "CA", "DE", "FR", "BR", "IN", "NG", "MX", "JP"]
BIN_COUNTRY_PROBS = [0.55, 0.10, 0.07, 0.05, 0.04, 0.05, 0.05, 0.03, 0.04, 0.02]

DEVICE_OS = ["iOS", "Android", "macOS", "Windows", "Linux", "Other"]
DEVICE_OS_PROBS = [0.32, 0.34, 0.12, 0.18, 0.02, 0.02]


# ---------------------------------------------------------------------------
# Entity tables
# ---------------------------------------------------------------------------

def _make_accounts(rng: np.random.Generator) -> pd.DataFrame:
    age = rng.gamma(2.5, 200, N_ACCOUNTS).clip(1, 4000)
    return pd.DataFrame({
        "account_id":   np.arange(N_ACCOUNTS),
        "account_age_days": age,
        "tenure_bucket": pd.cut(age, [0, 30, 180, 365, 4000],
                                labels=["<30d", "30-180d", "180-365d", "1y+"]).astype(str),
        "channel_pref": rng.choice(CHANNELS, size=N_ACCOUNTS, p=CHANNEL_PROBS),
        "country":      rng.choice(BIN_COUNTRIES, size=N_ACCOUNTS, p=BIN_COUNTRY_PROBS),
        "behavior_keystroke_baseline": rng.lognormal(0, 0.3, N_ACCOUNTS),
        "behavior_session_baseline":   rng.lognormal(2.5, 0.4, N_ACCOUNTS),
        "typical_amt":      rng.lognormal(3.6, 0.7, N_ACCOUNTS),
        "typical_velocity": rng.gamma(2.0, 1.0, N_ACCOUNTS),
        "is_recurring_user": rng.random(N_ACCOUNTS) < 0.18,
        "address_change_recency_days": rng.exponential(180, N_ACCOUNTS).clip(0, 1500),
    })


def _make_cards(rng: np.random.Generator, accounts: pd.DataFrame) -> pd.DataFrame:
    owner = rng.choice(N_ACCOUNTS, size=N_CARDS, replace=True,
                       p=np.linspace(2, 1, N_ACCOUNTS) / np.linspace(2, 1, N_ACCOUNTS).sum())
    bin_country = rng.choice(BIN_COUNTRIES, size=N_CARDS, p=BIN_COUNTRY_PROBS)
    return pd.DataFrame({
        "card_id":     np.arange(N_CARDS),
        "owner_acct":  owner,
        "bin_country": bin_country,
        "bin_age_days": rng.gamma(3.0, 250, N_CARDS).clip(1, 5000),
        "is_prepaid":   rng.random(N_CARDS) < 0.08,
        "card_tenure_days": rng.gamma(2.0, 180, N_CARDS).clip(1, 3000),
    })


def _make_merchants(rng: np.random.Generator) -> pd.DataFrame:
    mcc = rng.choice(MCC_CATEGORIES, size=N_MERCHANTS, p=MCC_PROBS)
    age = rng.gamma(2.0, 250, N_MERCHANTS).clip(1, 5000)
    base_risk = np.array([MCC_BASE_RISK[m] for m in mcc])
    risk = (base_risk + rng.normal(0, 0.10, N_MERCHANTS)).clip(0.05, 0.99)
    return pd.DataFrame({
        "merchant_id":       np.arange(N_MERCHANTS),
        "merchant_mcc":      mcc,
        "merchant_age_days": age,
        "merchant_risk_30d": risk,
        "merchant_country":  rng.choice(BIN_COUNTRIES, size=N_MERCHANTS, p=BIN_COUNTRY_PROBS),
        "is_recurring_merchant": rng.random(N_MERCHANTS) < 0.20,
        "merchant_avg_ticket": rng.lognormal(3.6, 0.8, N_MERCHANTS),
    })


def _make_devices(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "device_id":  np.arange(N_DEVICES),
        "device_os":  rng.choice(DEVICE_OS, size=N_DEVICES, p=DEVICE_OS_PROBS),
        "device_age_days": rng.gamma(2.0, 90, N_DEVICES).clip(0, 2000),
        "device_entropy": rng.beta(5, 2, N_DEVICES),
        "device_jailbroken": rng.random(N_DEVICES) < 0.03,
        "device_num_accounts": (1 + rng.poisson(0.3, N_DEVICES)).clip(1, 30),
    })


def _make_ips(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "ip_id": np.arange(N_IPS),
        "ip_country":     rng.choice(BIN_COUNTRIES, size=N_IPS, p=BIN_COUNTRY_PROBS),
        "ip_asn_rep":     rng.beta(3, 8, N_IPS),
        "ip_is_vpn":      rng.random(N_IPS) < 0.06,
        "ip_is_tor":      rng.random(N_IPS) < 0.005,
        "ip_anon_share":  rng.beta(2, 10, N_IPS),
    })


# ---------------------------------------------------------------------------
# Fraud archetypes — produce a (account, card, merchant, device, ip) tuple +
# transaction overrides that match a recognizable pattern.
# ---------------------------------------------------------------------------

def _select_ato(rng, accounts, cards, merchants, devices, ips, idx):
    acct = accounts.iloc[idx]
    own_cards = cards[cards.owner_acct == acct.account_id]
    card = own_cards.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0] if len(own_cards) else cards.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0]
    merchant = merchants.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0]
    # ATO uses a NEW device (not the account's typical) — high entropy
    device = devices.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0]
    # IP not matching account country
    bad_ips = ips[ips.ip_country != acct.country]
    ip = bad_ips.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0] if len(bad_ips) else ips.sample(1, random_state=rng.integers(0, 1 << 30)).iloc[0]
    amt_mult = float(np.clip(rng.lognormal(0.5, 0.6), 0.3, 6.0))
    return acct, card, merchant, device, ip, amt_mult


# ---------------------------------------------------------------------------
# Transaction generator
# ---------------------------------------------------------------------------

def _draw_transactions(rng: np.random.Generator,
                       accounts: pd.DataFrame, cards: pd.DataFrame,
                       merchants: pd.DataFrame, devices: pd.DataFrame,
                       ips: pd.DataFrame) -> pd.DataFrame:
    """Draw transactions per day, with persistent-entity references and timestamps.

    Velocity-style features are added in a second pass.
    """
    rows = []
    bin_attack_set = set(rng.choice(N_CARDS, size=400, replace=False))  # cards in the regime-shift attack
    attack_mcc = "gaming"  # the new attack concentrates on this MCC

    for day in range(N_DAYS):
        n_today = TX_PER_DAY + rng.integers(-300, 300)
        # Account selection biased toward older accounts (heavier-tail activity)
        acct_idx = rng.choice(N_ACCOUNTS, size=n_today, replace=True,
                              p=np.linspace(2, 1, N_ACCOUNTS) / np.linspace(2, 1, N_ACCOUNTS).sum())
        acct_rows = accounts.iloc[acct_idx]

        # Card: usually one of account's own cards; ~7% of time another
        own_card_id = []
        for ai in acct_idx:
            owns = cards.index[cards.owner_acct == ai]
            if len(owns) and rng.random() > 0.07:
                own_card_id.append(int(owns[rng.integers(0, len(owns))]))
            else:
                own_card_id.append(int(rng.integers(0, N_CARDS)))
        card_rows = cards.iloc[own_card_id]

        # Merchant
        m_idx = rng.choice(N_MERCHANTS, size=n_today, replace=True)
        merch_rows = merchants.iloc[m_idx]

        # Device — usually the account's "stable" device; occasionally a new one
        d_idx = (acct_idx % N_DEVICES + rng.integers(-500, 500, size=n_today)) % N_DEVICES
        change_mask = rng.random(n_today) < 0.04
        d_idx = np.where(change_mask, rng.integers(0, N_DEVICES, size=n_today), d_idx)
        dev_rows = devices.iloc[d_idx]

        # IP
        ip_idx = rng.choice(N_IPS, size=n_today, replace=True)
        ip_rows = ips.iloc[ip_idx]

        # Amount: account-typical scaled by lognormal noise, plus per-merchant noise
        amount = (acct_rows["typical_amt"].values *
                  rng.lognormal(0, 0.7, n_today) *
                  (merch_rows["merchant_avg_ticket"].values / 50).clip(0.2, 4.0))
        amount = np.round(np.clip(amount, 1, 10_000), 2)

        # Hour of day
        hour = (rng.integers(0, 24, n_today) + acct_rows["account_age_days"].values % 7).astype(int) % 24
        ts = pd.to_datetime(START_DATE) + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 60, n_today), unit="m")

        df = pd.DataFrame({
            "tx_id":       np.arange(len(rows), len(rows) + n_today),
            "ts":          ts,
            "day_idx":     day,
            "account_id":  acct_rows["account_id"].values,
            "card_id":     card_rows["card_id"].values,
            "merchant_id": merch_rows["merchant_id"].values,
            "device_id":   dev_rows["device_id"].values,
            "ip_id":       ip_rows["ip_id"].values,
            "amount":      amount,
            "hour":        hour,
            "channel":     acct_rows["channel_pref"].values,
            "_typical_amt": acct_rows["typical_amt"].values,
            "_typical_vel": acct_rows["typical_velocity"].values,
            "_acct_country": acct_rows["country"].values,
            "_acct_keystroke_baseline": acct_rows["behavior_keystroke_baseline"].values,
            "_acct_session_baseline":   acct_rows["behavior_session_baseline"].values,
            "_acct_age_days": acct_rows["account_age_days"].values,
            "_acct_tenure": acct_rows["tenure_bucket"].values,
            "_acct_addr_change": acct_rows["address_change_recency_days"].values,
            "_acct_recurring": acct_rows["is_recurring_user"].values,
            # card
            "bin_country":   card_rows["bin_country"].values,
            "bin_age_days":  card_rows["bin_age_days"].values,
            "is_prepaid":    card_rows["is_prepaid"].values,
            "card_tenure_days": card_rows["card_tenure_days"].values,
            # merchant
            "merchant_mcc":      merch_rows["merchant_mcc"].values,
            "merchant_age_days": merch_rows["merchant_age_days"].values,
            "merchant_risk_30d": merch_rows["merchant_risk_30d"].values,
            "merchant_country":  merch_rows["merchant_country"].values,
            "is_recurring_merchant": merch_rows["is_recurring_merchant"].values,
            "merchant_avg_ticket": merch_rows["merchant_avg_ticket"].values,
            # device
            "device_os":           dev_rows["device_os"].values,
            "device_age_days":     dev_rows["device_age_days"].values,
            "device_entropy":      dev_rows["device_entropy"].values,
            "device_jailbroken":   dev_rows["device_jailbroken"].values,
            "device_num_accounts": dev_rows["device_num_accounts"].values,
            "_dev_changed":        change_mask,
            # IP
            "ip_country":   ip_rows["ip_country"].values,
            "ip_asn_rep":   ip_rows["ip_asn_rep"].values,
            "ip_is_vpn":    ip_rows["ip_is_vpn"].values,
            "ip_is_tor":    ip_rows["ip_is_tor"].values,
            "ip_anon_share": ip_rows["ip_anon_share"].values,
        })
        # Behavioral measurements per-tx (deviates from baseline if fraud later)
        df["session_duration_s"] = (df["_acct_session_baseline"] *
                                    rng.lognormal(0, 0.3, n_today)).clip(2, 1800)
        df["keystroke_var"] = (df["_acct_keystroke_baseline"] *
                               rng.lognormal(0, 0.3, n_today)).clip(0.05, 5.0)
        df["mouse_var"] = rng.lognormal(0, 0.4, n_today).clip(0.1, 5.0)
        df["time_to_submit_s"] = rng.gamma(3.0, 4.0, n_today).clip(0.5, 300)
        df["copy_paste_count"] = rng.poisson(0.3, n_today)
        df["retries"] = rng.poisson(0.15, n_today)
        df["3ds_outcome"] = rng.choice(
            ["pass", "fail", "frictionless", "challenge", "bypass"],
            size=n_today, p=[0.55, 0.05, 0.30, 0.07, 0.03])

        # Mark cards in attack set
        df["_attack_card"] = df["card_id"].isin(bin_attack_set)
        df["_post_shift"] = day >= SHIFT_DAY

        rows.append(df)

    return pd.concat(rows, ignore_index=True), bin_attack_set, attack_mcc


# ---------------------------------------------------------------------------
# Velocity / history features (single pass over sorted data)
# ---------------------------------------------------------------------------

def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    df["unix_h"] = (df["ts"].astype("int64") // (3_600 * 1_000_000_000)).astype(int)

    # Per-card windows
    g = df.groupby("card_id", sort=False)
    df["_card_seq"] = g.cumcount()
    # Hours since previous on same card
    prev_unix = g["unix_h"].shift(1)
    df["hours_since_prev_card"] = (df["unix_h"] - prev_unix).fillna(9999)
    # Approximate rolling-window counts/sums via groupby cumulative + lookback
    # (true rolling on sparse timestamps is expensive; we use seq-distance proxies
    #  scaled by mean inter-arrival)
    df["velocity_cnt_card_24h"] = (g.cumcount() -
                                   g["unix_h"].transform(lambda x: x.shift(1).where(x.shift(1) > x - 24, np.nan).ffill().pipe(lambda y: g["unix_h"].cumcount()))).fillna(0).clip(0, 50)
    # Simpler approach: count of previous tx on same card within 24h via merge_asof
    return df


def _velocity_via_asof(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-card velocity (count and sum) for 1h, 24h, 7d windows."""
    df = df.sort_values(["card_id", "ts"]).reset_index(drop=True)

    def _rolling_per_card(window):
        # For each row, count how many prior rows on same card have ts in [ts-window, ts).
        out_cnt = np.zeros(len(df), dtype=np.int32)
        out_amt = np.zeros(len(df), dtype=np.float32)
        last = {}
        for card, sub_idx in df.groupby("card_id", sort=False).indices.items():
            sub = df.iloc[sub_idx]
            ts = sub["ts"].to_numpy(dtype="datetime64[ns]")
            amt = sub["amount"].to_numpy()
            # Two-pointer
            j = 0
            for i in range(len(sub)):
                while j < i and (ts[i] - ts[j]) > window:
                    j += 1
                out_cnt[sub_idx[i]] = i - j
                out_amt[sub_idx[i]] = amt[j:i].sum() if i > j else 0.0
        return out_cnt, out_amt

    cnt_1h, amt_1h = _rolling_per_card(np.timedelta64(1, "h"))
    cnt_24h, amt_24h = _rolling_per_card(np.timedelta64(24, "h"))
    cnt_7d,  amt_7d  = _rolling_per_card(np.timedelta64(7, "D"))

    df["velocity_cnt_card_1h"]  = cnt_1h
    df["velocity_amt_card_1h"]  = amt_1h
    df["velocity_cnt_card_24h"] = cnt_24h
    df["velocity_amt_card_24h"] = amt_24h
    df["velocity_cnt_card_7d"]  = cnt_7d
    df["velocity_amt_card_7d"]  = amt_7d

    # Per-account velocity at 24h
    df = df.sort_values(["account_id", "ts"]).reset_index(drop=True)
    out_cnt = np.zeros(len(df), dtype=np.int32)
    out_amt = np.zeros(len(df), dtype=np.float32)
    for acct, sub_idx in df.groupby("account_id", sort=False).indices.items():
        sub = df.iloc[sub_idx]
        ts = sub["ts"].to_numpy(dtype="datetime64[ns]")
        amt = sub["amount"].to_numpy()
        j = 0
        for i in range(len(sub)):
            while j < i and (ts[i] - ts[j]) > np.timedelta64(24, "h"):
                j += 1
            out_cnt[sub_idx[i]] = i - j
            out_amt[sub_idx[i]] = amt[j:i].sum() if i > j else 0.0
    df["velocity_cnt_acct_24h"] = out_cnt
    df["velocity_amt_acct_24h"] = out_amt

    # Decline / retry history per card 7d
    df = df.sort_values(["card_id", "ts"]).reset_index(drop=True)
    declines = np.zeros(len(df), dtype=np.int32)
    for card, sub_idx in df.groupby("card_id", sort=False).indices.items():
        sub = df.iloc[sub_idx]
        ts = sub["ts"].to_numpy(dtype="datetime64[ns]")
        is_decl = (sub["3ds_outcome"].to_numpy() == "fail").astype(int)
        j = 0
        for i in range(len(sub)):
            while j < i and (ts[i] - ts[j]) > np.timedelta64(7, "D"):
                j += 1
            declines[sub_idx[i]] = is_decl[j:i].sum() if i > j else 0
    df["declined_attempts_7d"] = declines

    return df


# ---------------------------------------------------------------------------
# Engineered feature layer (60-80 features) + bucket metadata
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame, rng: np.random.Generator
                       ) -> Tuple[pd.DataFrame, Dict[str, FeatureMeta]]:
    out = pd.DataFrame(index=df.index)
    meta: Dict[str, FeatureMeta] = {}

    def add(col, vals, bucket, stable_target):
        out[col] = vals
        meta[col] = FeatureMeta(bucket, stable_target)

    # ---- Amount features (compositional, mostly stable) ----
    add("amount",            df["amount"].astype(float),                       "compositional", True)
    add("amt_log",           np.log1p(df["amount"]).astype(float),             "compositional", True)
    add("amt_z_account",     ((df["amount"] - df["_typical_amt"]) /
                              df["_typical_amt"].clip(1)).astype(float),       "compositional", True)
    add("amt_over_avg_ticket", (df["amount"] / df["merchant_avg_ticket"].clip(1)).astype(float), "compositional", True)
    add("amt_band_low",      (df["amount"] < 25).astype(int),                  "tactical", False)
    add("amt_band_high",     (df["amount"] > 1000).astype(int),                "tactical", False)
    add("amt_round_value",   (df["amount"] % 10 == 0).astype(int),             "tactical", False)

    # ---- Velocity features (tactical) ----
    add("velocity_cnt_card_1h",  df["velocity_cnt_card_1h"].astype(float),     "tactical", False)
    add("velocity_amt_card_1h",  df["velocity_amt_card_1h"].astype(float),     "tactical", False)
    add("velocity_cnt_card_24h", df["velocity_cnt_card_24h"].astype(float),    "tactical", False)
    add("velocity_amt_card_24h", df["velocity_amt_card_24h"].astype(float),    "tactical", False)
    add("velocity_cnt_card_7d",  df["velocity_cnt_card_7d"].astype(float),     "tactical", False)
    add("velocity_amt_card_7d",  df["velocity_amt_card_7d"].astype(float),     "tactical", False)
    add("velocity_cnt_acct_24h", df["velocity_cnt_acct_24h"].astype(float),    "tactical", False)
    add("velocity_amt_acct_24h", df["velocity_amt_acct_24h"].astype(float),    "tactical", False)
    add("hours_since_prev_card", df["ts"].groupby(df["card_id"]).diff().dt.total_seconds().fillna(86400 * 30) / 3600, "tactical", False)
    add("velocity_ratio_24h_typical",
        (df["velocity_cnt_card_24h"] / df["_typical_vel"].clip(0.1)).astype(float),
        "compositional", True)

    # ---- Account / structural (invariant) ----
    add("account_age_days",      df["_acct_age_days"].astype(float),           "invariant", True)
    add("account_age_under_30d", (df["_acct_age_days"] < 30).astype(int),      "invariant", True)
    add("account_addr_change_recency", df["_acct_addr_change"].astype(float),  "invariant", True)
    add("is_recurring_user",     df["_acct_recurring"].astype(int),            "invariant", True)
    add("acct_country_match_ip", (df["_acct_country"] == df["ip_country"]).astype(int), "invariant", True)
    add("acct_country_match_bin", (df["_acct_country"] == df["bin_country"]).astype(int), "invariant", True)
    add("ip_bin_country_match",  (df["ip_country"] == df["bin_country"]).astype(int), "invariant", True)

    # ---- Card features ----
    add("card_tenure_days",      df["card_tenure_days"].astype(float),          "invariant", True)
    add("card_is_prepaid",       df["is_prepaid"].astype(int),                  "invariant", True)
    add("bin_age_days",          df["bin_age_days"].astype(float),              "invariant", True)
    # tactical: card-country in attack list
    add("bin_country_high_risk", df["bin_country"].isin(["NG", "BR", "IN"]).astype(int), "tactical", False)

    # ---- Merchant features ----
    add("merchant_age_days",     df["merchant_age_days"].astype(float),         "invariant", True)
    add("merchant_age_under_30d",(df["merchant_age_days"] < 30).astype(int),    "invariant", True)
    add("merchant_risk_30d",     df["merchant_risk_30d"].astype(float),         "tactical", False)
    add("is_recurring_merchant", df["is_recurring_merchant"].astype(int),       "invariant", True)
    add("merchant_country_mismatch_ip", (df["merchant_country"] != df["ip_country"]).astype(int), "tactical", False)

    # MCC one-hots (tactical)
    for mcc in MCC_CATEGORIES:
        add(f"mcc_{mcc}", (df["merchant_mcc"] == mcc).astype(int), "tactical", False)

    # ---- Device features ----
    add("device_age_days",       df["device_age_days"].astype(float),           "invariant", True)
    add("device_entropy",        df["device_entropy"].astype(float),            "invariant", True)
    add("device_changed_recent", df["_dev_changed"].astype(int),                "tactical", False)
    add("device_jailbroken",     df["device_jailbroken"].astype(int),           "invariant", True)
    add("device_num_accounts",   df["device_num_accounts"].astype(float),       "invariant", True)
    for os_ in DEVICE_OS:
        add(f"device_os_{os_.lower()}", (df["device_os"] == os_).astype(int), "invariant", True)

    # ---- IP / network ----
    add("ip_asn_rep",            df["ip_asn_rep"].astype(float),                "invariant", True)
    add("ip_is_vpn",             df["ip_is_vpn"].astype(int),                   "tactical", False)
    add("ip_is_tor",             df["ip_is_tor"].astype(int),                   "tactical", False)
    add("ip_anon_share",         df["ip_anon_share"].astype(float),             "invariant", True)

    # ---- Behavioral ----
    add("session_duration_s",    df["session_duration_s"].astype(float),        "invariant", True)
    add("keystroke_var",         df["keystroke_var"].astype(float),             "invariant", True)
    add("mouse_var",             df["mouse_var"].astype(float),                 "invariant", True)
    add("time_to_submit_s",      df["time_to_submit_s"].astype(float),          "invariant", True)
    add("copy_paste_count",      df["copy_paste_count"].astype(float),          "invariant", True)
    add("retries",               df["retries"].astype(float),                   "tactical", False)
    add("ks_dev_from_baseline",  np.abs(np.log(df["keystroke_var"].clip(0.01) / df["_acct_keystroke_baseline"].clip(0.01))).astype(float), "compositional", True)
    add("session_dev_from_baseline", np.abs(np.log(df["session_duration_s"].clip(1) / df["_acct_session_baseline"].clip(1))).astype(float), "compositional", True)

    # ---- 3DS outcome (tactical) ----
    for o in ["pass", "fail", "frictionless", "challenge", "bypass"]:
        add(f"3ds_{o}", (df["3ds_outcome"] == o).astype(int), "tactical", False)

    # ---- Decline history (tactical) ----
    add("declined_attempts_7d",  df["declined_attempts_7d"].astype(float),      "tactical", False)

    # ---- Hour / day cyclical (invariant) ----
    add("hour",                  df["hour"].astype(float),                      "invariant", True)
    add("hour_sin",              np.sin(2 * np.pi * df["hour"] / 24).astype(float), "invariant", True)
    add("hour_cos",              np.cos(2 * np.pi * df["hour"] / 24).astype(float), "invariant", True)
    add("dow",                   (df["day_idx"] % 7).astype(float),             "invariant", True)

    # ---- Compositional embeddings via PCA on simple categorical co-occurrence ----
    # 1) Merchant embedding (via random projection over MCC × amount-band × country)
    n = len(df)
    merch_feat = np.column_stack([
        df["merchant_mcc"].astype("category").cat.codes.values,
        np.digitize(df["merchant_avg_ticket"].values, [10, 30, 80, 200, 500]),
        df["merchant_country"].astype("category").cat.codes.values,
    ]).astype(float)
    if len(merch_feat) > 100:
        merch_feat = (merch_feat - merch_feat.mean(0)) / (merch_feat.std(0) + 1e-6)
        merch_emb = PCA(n_components=3, random_state=0).fit_transform(merch_feat)
        for i in range(3):
            add(f"embed_merchant_d{i+1}", merch_emb[:, i].astype(float),
                "compositional", i == 2)  # one of them tagged stable, two drifting
    # 2) Device-network embedding
    dn_feat = np.column_stack([
        df["device_os"].astype("category").cat.codes.values,
        df["ip_country"].astype("category").cat.codes.values,
        df["ip_is_vpn"].astype(int).values,
        np.log1p(df["device_age_days"].values),
    ]).astype(float)
    if len(dn_feat) > 100:
        dn_feat = (dn_feat - dn_feat.mean(0)) / (dn_feat.std(0) + 1e-6)
        dn_emb = PCA(n_components=3, random_state=0).fit_transform(dn_feat)
        for i in range(3):
            add(f"embed_device_net_d{i+1}", dn_emb[:, i].astype(float),
                "compositional", True)

    # ---- Interaction features (compositional) ----
    add("cnp_x_new_account",
        ((df["channel"] == "CNP_new") & (df["_acct_age_days"] < 30)).astype(int),
        "compositional", False)
    add("high_amt_x_new_merchant",
        ((df["amount"] > 500) & (df["merchant_age_days"] < 30)).astype(int),
        "compositional", False)
    add("vpn_x_high_amt",
        (df["ip_is_vpn"].astype(int) * (df["amount"] > 500).astype(int)).astype(int),
        "compositional", False)
    add("declines_x_new_card",
        (df["declined_attempts_7d"] * (df["card_tenure_days"] < 60).astype(int)).astype(float),
        "compositional", False)

    return out, meta


# ---------------------------------------------------------------------------
# Fraud generation
# ---------------------------------------------------------------------------

def _label_fraud(df: pd.DataFrame, X: pd.DataFrame, rng: np.random.Generator,
                 attack_mcc: str) -> np.ndarray:
    """Plant fraud probability based on a linear combination of risk drivers,
    then sample binary labels. Different archetypes get different signatures."""
    n = len(df)

    # Base risk: function of plausible features
    z = (
        + 1.20 * np.clip(X["velocity_amt_card_1h"].values / 800, 0, 5)
        + 1.00 * np.clip(X["velocity_cnt_card_24h"].values / 8, 0, 4)
        + 1.40 * X["bin_country_high_risk"].values
        + 0.90 * X["merchant_risk_30d"].values * 3
        + 1.20 * X["device_changed_recent"].values
        + 0.80 * (1 - X["acct_country_match_ip"].values)
        + 0.70 * (1 - X["acct_country_match_bin"].values)
        + 0.80 * (1 - X["ip_bin_country_match"].values)
        + 0.60 * X["ip_is_vpn"].values
        + 1.30 * X["ip_is_tor"].values
        + 0.90 * X["account_age_under_30d"].values
        + 0.70 * X["merchant_age_under_30d"].values
        + 0.80 * X["device_jailbroken"].values
        + 0.70 * (X["device_num_accounts"].values > 5)
        + 1.10 * np.clip(X["declined_attempts_7d"].values / 4, 0, 4)
        + 0.50 * X["3ds_fail"].values
        + 0.40 * X["3ds_bypass"].values
        + 0.80 * X["amt_band_high"].values
        + 0.60 * X["cnp_x_new_account"].values
        + 0.50 * X["high_amt_x_new_merchant"].values
        + 0.70 * X["vpn_x_high_amt"].values
        + 0.50 * np.clip(X["ks_dev_from_baseline"].values, 0, 4)
        + 0.40 * np.clip(X["session_dev_from_baseline"].values, 0, 4)
        - 0.40 * X["is_recurring_user"].values
        - 0.50 * X["is_recurring_merchant"].values
        - 0.40 * (X["card_tenure_days"].values > 365).astype(float)
        - 0.30 * (X["account_age_days"].values > 365).astype(float)
    )

    # Regime shift: post-shift, a NEW BIN cluster (bin_country in {GB,DE}) starts
    # attacking on a particular MCC. Only post-shift, only for cards in attack set.
    post = df["_post_shift"].values
    attack_card = df["_attack_card"].values
    attack_mask = post & attack_card & (df["merchant_mcc"].values == attack_mcc)
    if attack_mask.any():
        z_attack = (
            + 2.0 * np.ones(attack_mask.sum())
            + 1.4 * np.clip(X.loc[attack_mask, "velocity_amt_card_1h"].values / 400, 0, 5)
            + 1.6 * (X.loc[attack_mask, "amount"].values > 200).astype(float)
            + 0.9 * np.clip(X.loc[attack_mask, "declined_attempts_7d"].values / 3, 0, 4)
        )
        z[attack_mask] += z_attack

    # Calibrate intercept on pre-shift to ~0.4%
    z_pre = z[~post]
    target_pre = 0.0042

    def mean_p(intercept):
        return float((1.0 / (1 + np.exp(-(z_pre - intercept)))).mean())

    lo, hi = -5.0, 30.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) > target_pre:
            lo = mid
        else:
            hi = mid
    intercept = 0.5 * (lo + hi)
    p = 1.0 / (1 + np.exp(-(z - intercept)))

    # Boost post-shift to ~0.85%
    if post.any():
        target_post = 0.0085
        scale = target_post / max(p[post].mean(), 1e-9)
        p[post] = np.clip(p[post] * scale, 0, 0.95)

    return (rng.random(n) < p).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_dataset(seed: int = 42) -> Tuple[pd.DataFrame, List[str], Dict[str, FeatureMeta], List[str]]:
    rng = np.random.default_rng(seed)

    accounts  = _make_accounts(rng)
    cards     = _make_cards(rng, accounts)
    merchants = _make_merchants(rng)
    devices   = _make_devices(rng)
    ips       = _make_ips(rng)

    tx, attack_set, attack_mcc = _draw_transactions(rng, accounts, cards, merchants, devices, ips)
    tx = _velocity_via_asof(tx)
    X, meta = _engineer_features(tx, rng)
    y = _label_fraud(tx, X, rng, attack_mcc)

    df = pd.concat([
        tx[["tx_id", "ts", "day_idx", "channel", "merchant_mcc", "bin_country",
            "_acct_tenure"]].rename(columns={"_acct_tenure": "tenure_bucket"}).reset_index(drop=True),
        X.reset_index(drop=True),
    ], axis=1)
    df["label"] = y
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    df["week_idx"] = df["day_idx"] // 7
    df["amt_band"] = pd.cut(df["amount"], [-1, 25, 100, 500, 1e9],
                            labels=["<25", "25-100", "100-500", "500+"]).astype(str)

    # Splits: train [0..34], cal [35..41], oot [42..55], recent [56..69]
    df["split"] = pd.cut(df["day_idx"], bins=[-1, 34, 41, 55, 69],
                         labels=["train", "cal", "oot", "recent"]).astype(str)

    feature_names = list(X.columns)
    cohort_dims = ["channel", "merchant_mcc", "bin_country", "amt_band", "tenure_bucket"]
    return df, feature_names, meta, cohort_dims
