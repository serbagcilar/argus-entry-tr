"""
ARGUS BIST SCAN — AEI v18.3 SINYAL-ONLY TARAMA (Railway cron)
=================================================
BIST evrenini (Yahoo Finance) AEI v18.3 ile tarar.
Bu script SADECE tarama/sinyal yapar — emir GONDERMEZ (US/Alpaca'daki
main.py'den farkli, o gercek paper trading yapiyor). Amac: Claude Code'un
results_bist.json dosyasini okuyup "hangi ticker'lar ENTRY_ARMED/IN_TRADE"
sorularina aninda cevap verebilmesi (her seferinde canli hesaplamak
yerine).

ONEMLI: Railway cron, bu process'in isini bitirip KENDINI KAPATMASINI
bekler. Sonsuz donguye girmez (while True KULLANILMAZ).

Env variables:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID -- opsiyonel, yeni event bildirimi icin
  STATE_FILE_PATH   -- varsayilan "state_bist_scan.json" (dedup icin, son bildirilen event)
  RESULTS_FILE_PATH -- varsayilan "results_bist.json" (tam anlik durum snapshotu)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
import os
import json
import requests
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
warnings.filterwarnings("ignore")


import yfinance as yf

BENCHMARK = "XU100.IS"
_YF_PERIOD_MAP = {"5m": ("5m", "60d"), "15m": ("15m", "60d"), "1h": ("60m", "730d"), "1d": ("1d", "1000d")}

def _resample_4h(df):
    o = df["Open"].resample("4h").first()
    h = df["High"].resample("4h").max()
    l = df["Low"].resample("4h").min()
    c = df["Close"].resample("4h").last()
    v = df["Volume"].resample("4h").sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out.dropna()

def fetch_bars(symbol, tf, days=60):
    yf_symbol = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    if tf == "4h":
        interval, period = _YF_PERIOD_MAP["1h"]
    else:
        interval, period = _YF_PERIOD_MAP.get(tf, ("1d", "1000d"))
    try:
        raw = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    if tf == "4h":
        df = _resample_4h(df)
    return df


TELEGRAM_BOT_TOKEN = ""  # Telegram bildirimi KAPALI — bu scriptler sadece results_*.json'a yazar
TELEGRAM_CHAT_ID   = ""  # sorgular Claude Code uzerinden results_*.json okunarak cevaplanir
STATE_FILE_PATH    = os.environ.get("STATE_FILE_PATH", "state_bist_scan.json")
RESULTS_FILE_PATH  = os.environ.get("RESULTS_FILE_PATH", "results_bist.json")

FIXED_TFS = ["5m", "15m", "1h", "4h"]

BIST_TICKERS = ['THYAO', 'ASELS', 'GARAN', 'AKBNK', 'EREGL', 'SISE', 'KCHOL', 'BIMAS']
TICKERS = BIST_TICKERS


def send_telegram(message: str):
    # Devre disi - bu scriptler sadece tarama yapar, results_*.json'a yazar.
    # Sorgular Claude Code uzerinden o dosya okunarak cevaplanir.
    return


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ============================================================
# AEI v18.3 ENGINE CORE — Pine AEI_v18_2.pine ile birebir
# ============================================================
BIAS_ENTRY_MIN    = 20
BIAS_EXIT_MIN     = 50
BIAS_INVALID_BUF  = 5
SMOOTH_LEN        = 3
DI_RATIO_MAX      = 6
STOP_PCT          = 4.7
RS_PERIOD         = 20

HL_U       = 20
K_FRAC     = 0.6  # v18.3: sabit-yuzde (HL_K) yerine range-bazli bant genisligi
ELI_ALPHA1 = 0.33
ELI_ALPHA2 = 0.25

BSL_PIVOT_LEN      = 5
BSL_VOL_LOOKBACK   = 20
BSL_STRENGTH_VETO  = 2.0
BSL_VETO_LOOKBACK  = 15
CANDLE_BODY_MIN    = 0.6
CANDLE_CLOSE_MIN   = 0.7
BSL_EQ_TOL_ATR     = 0.15
BSL_SWEEP_VOL_MULT = 1.5


def compute_bias_score_series(df, bench_df, rs_period=RS_PERIOD):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    open_ = df["Open"]

    adx_data = ta.adx(high, low, close, length=14)
    adx, dip, dim = adx_data["ADX_14"], adx_data["DMP_14"], adx_data["DMN_14"]

    adx_up = np.select([adx > 60, adx > 45, adx > 35, adx > 25, adx > 20], [0, 10, 15, 12, 5], default=0)
    adx_dn = np.select([adx > 35, adx > 25, adx > 20], [-15, -10, -5], default=0)
    adx_s = np.where(dip > dim, adx_up, adx_dn)

    di_diff = dip - dim
    di_s = np.where(dip > dim, np.where(di_diff >= 10, 20, 12), -10)

    sr = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    ck = [c for c in sr.columns if "k" in c.lower()][0]
    cd = [c for c in sr.columns if "d" in c.lower()][0]
    sk, sd, skp = sr[ck], sr[cd], sr[ck].shift(1)
    strong_up = (adx > 25) & (dip > dim)
    stoch_s = np.select(
        [(sk >= 40) & (sk <= 65) & (sk > skp),
         (sk > 65) & (sk <= 80) & (sk > skp),
         (sk > 20) & (sk < 40) & ((sk - skp) > 5),
         (sk > 80) & strong_up,
         (sk > 80) & ~strong_up,
         (sk < sd) & (sk > 30),
         (sk < 20)],
        [20, 14, 9, 6, -10, -5, -7], default=0)

    def _ema_dema(s, p):
        e1 = s.ewm(span=p, adjust=False).mean()
        return 2 * e1 - e1.ewm(span=p, adjust=False).mean()
    d10, d20, d50 = _ema_dema(close, 10), _ema_dema(close, 20), _ema_dema(close, 50)
    d100 = _ema_dema(close, 100)
    d200 = _ema_dema(close, 200) if len(df) >= 200 else d100
    aligned = (d10 > d20) & (d20 > d50) & (d50 > d100) & (d100 > d200)
    a10, a20, a50, a100, a200 = close > d10, close > d20, close > d50, close > d100, close > d200
    dema_s = np.select(
        [a10 & a20 & a50 & a100 & a200 & aligned,
         a10 & a20 & a50 & a100,
         a10 & a20 & a50,
         a10 & a20],
        [20, 14, 8, 3], default=0)
    sm1 = (d10 - d20) / d20 * 100
    sm2 = (d20 - d50) / d50 * 100
    sm3 = (d50 - d100) / d100 * 100
    sp_s = np.select([(sm1 > 3) & (sm2 > 2) & (sm3 > 1), (sm1 > 2) & (sm2 > 1), sm1 > 1, sm1 < 0], [8, 5, 2, -3], default=0)
    dema_s = dema_s + np.where(aligned, sp_s, 0)

    rs_s = np.zeros(len(df))
    try:
        if bench_df is not None and len(bench_df) > rs_period:
            bc = bench_df["Close"].reindex(df.index, method="ffill")
            rs_h = (close / close.shift(rs_period) - 1) * 100
            rs_b = (bc / bc.shift(rs_period) - 1) * 100
            rsd = rs_h - rs_b
            rs_s = np.select(
                [rsd > 10, rsd > 5, rsd > 2, rsd > 0, rsd > -3, rsd > -7],
                [15, 10, 7, 3, -3, -7], default=-12)
    except Exception:
        pass

    vm = vol.rolling(20).mean()
    vr = vol / vm.replace(0, np.nan)
    bull = dip > dim
    strong_trend = adx > 20
    is_green = close > open_
    rv_raw = np.select([(vr > 2.0) & is_green, (vr > 2.0) & ~is_green, vr > 1.5, vr > 1.2, vr > 0.8],
                        [15, 0, 10, 5, 0], default=-5)
    rv_s = np.where(~strong_trend, rv_raw * 0.3, np.where(bull, rv_raw, -rv_raw))

    total = adx_s + di_s + stoch_s + dema_s + rs_s + rv_s
    return pd.Series(total, index=df.index).astype(float)


def compute_eli_hm_lm(df, hl_u=HL_U, k_frac=K_FRAC, alpha1=ELI_ALPHA1, alpha2=ELI_ALPHA2):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)

    def f_var(data):
        r = np.zeros(n)
        r[0] = data[0] if n > 0 else 0.0
        for i in range(1, n):
            b = abs(data[i] - data[i - 9]) if i >= 9 else abs(data[i] - data[0])
            window = data[max(0, i - 8):i + 1]
            c = np.sum(np.abs(np.diff(window))) if len(window) > 1 else 0.0
            d = (b / c) if c != 0 else 0.0
            e = 2.0 / 3.0
            r[i] = d * e * (data[i] - r[i - 1]) + r[i - 1]
        return r

    def f_ott_range(data, range_val, frac):
        b = range_val * frac
        a = np.where(data != 0, b / data, 0.0)
        c = data - b
        dd = data + b
        for i in range(1, n):
            c[i]  = c[i]  if (c[i]  > c[i - 1]  or data[i] < c[i - 1])  else c[i - 1]
            dd[i] = dd[i] if (dd[i] < dd[i - 1] or data[i] > dd[i - 1]) else dd[i - 1]
        e = c.copy()
        for i in range(1, n):
            if data[i] > e[i - 1]:
                e[i] = c[i]
            elif data[i] < e[i - 1]:
                e[i] = dd[i]
            else:
                e[i] = e[i - 1]
        h = np.where(data > e, e * (1 + a / 2), e * (1 - a / 2))
        return np.roll(h, 2)

    highest_u = pd.Series(high).rolling(hl_u).max().bfill().values
    lowest_u  = pd.Series(low).rolling(hl_u).min().bfill().values
    range_u   = highest_u - lowest_u

    hm = f_ott_range(f_var(highest_u), range_u, k_frac)
    lm = f_ott_range(f_var(lowest_u), range_u, k_frac)

    lead = np.zeros(n); eli = np.zeros(n)
    lead[0] = close[0]; eli[0] = close[0]
    for i in range(1, n):
        lead[i] = 2 * close[i] + (alpha1 - 2) * close[i - 1] + (1 - alpha1) * lead[i - 1]
        eli[i]  = alpha2 * lead[i] + (1 - alpha2) * eli[i - 1]

    return (pd.Series(eli, index=df.index), pd.Series(hm, index=df.index), pd.Series(lm, index=df.index))


@dataclass
class BSLLevel:
    price: float
    strength: float
    swept: bool = False
    sweep_bar: int = None
    confirmed: bool = False
    confirmed_bar: int = None


def _atr(df, length=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _pivot_high_series(high, left, right):
    n = len(high)
    out = pd.Series(np.nan, index=high.index)
    vals = high.values
    for i in range(left, n - right):
        window = vals[i - left:i + right + 1]
        if vals[i] == window.max() and (window == vals[i]).sum() == 1:
            out.iloc[i + right] = vals[i]
    return out


def build_bsl_levels_and_veto_series(df, pivot_len=BSL_PIVOT_LEN, vol_lookback=BSL_VOL_LOOKBACK,
                                      eq_tol_atr=BSL_EQ_TOL_ATR, sweep_vol_mult=BSL_SWEEP_VOL_MULT,
                                      veto_lookback=BSL_VETO_LOOKBACK,
                                      candle_body_min=CANDLE_BODY_MIN, candle_close_min=CANDLE_CLOSE_MIN,
                                      strength_veto=BSL_STRENGTH_VETO, is_intraday=True):
    n = len(df)
    high, low, close, open_, vol = (df["High"].values, df["Low"].values, df["Close"].values,
                                     df["Open"].values, df["Volume"].values)
    atr14 = _atr(df, 14).values
    avg_vol = pd.Series(vol).rolling(vol_lookback).mean().values
    ph_confirmed_at = _pivot_high_series(df["High"], pivot_len, pivot_len)

    levels = []
    nearest_strength_arr = np.full(n, np.nan)
    bsl_risk_arr = np.zeros(n, dtype=bool)

    if is_intraday:
        daily_hl = df.groupby(df.index.date).agg(High=("High", "max"), Low=("Low", "min"))
        daily_hl.index = pd.to_datetime(daily_hl.index)
        prev_hl = daily_hl.shift(1)
        date_series = pd.Series(pd.to_datetime(df.index.date), index=df.index)
        pdh_ref = date_series.map(prev_hl["High"]).values
        pdl_ref = date_series.map(prev_hl["Low"]).values
    else:
        pdh_ref = np.full(n, np.nan)
        pdl_ref = np.full(n, np.nan)

    last_pdh_pdl_sweep_bar = None
    recent_pdh_pdl_sweep_arr = np.zeros(n, dtype=bool)
    is_strong_candle_arr = np.zeros(n, dtype=bool)

    for i in range(n):
        if not np.isnan(ph_confirmed_at.iloc[i]):
            pivot_price = ph_confirmed_at.iloc[i]
            src_idx = i - pivot_len
            pivot_vol = vol[src_idx] if src_idx >= 0 else vol[0]
            pivot_avg_vol = avg_vol[src_idx] if src_idx >= 0 and not np.isnan(avg_vol[src_idx]) else np.nan
            new_strength = (pivot_vol / pivot_avg_vol) if pivot_avg_vol and pivot_avg_vol > 0 else 1.0
            eq_tol = (atr14[i] if not np.isnan(atr14[i]) else 0.0) * eq_tol_atr

            merged = False
            for lv in levels:
                if not lv.swept and abs(lv.price - pivot_price) <= eq_tol:
                    lv.strength = max(lv.strength, new_strength)
                    merged = True
                    break
            if not merged:
                levels.append(BSLLevel(price=pivot_price, strength=new_strength))

        for lv in levels:
            if lv.swept:
                continue
            if high[i] >= lv.price:
                lv.swept = True
                lv.sweep_bar = i
                vol_ok = (not np.isnan(avg_vol[i])) and avg_vol[i] > 0 and vol[i] > avg_vol[i] * sweep_vol_mult
                if close[i] < lv.price and vol_ok:
                    lv.confirmed = True
                    lv.confirmed_bar = i

        levels[:] = [lv for lv in levels if not (lv.swept and lv.sweep_bar is not None and (i - lv.sweep_bar) > 200)]

        best = None
        for lv in levels:
            if lv.confirmed and lv.confirmed_bar is not None and (i - lv.confirmed_bar) <= veto_lookback:
                if best is None or lv.strength > best:
                    best = lv.strength
        nearest_strength_arr[i] = best if best is not None else np.nan
        bsl_risk_arr[i] = (best is not None) and (best > strength_veto)

        if is_intraday and not np.isnan(pdh_ref[i]) and not np.isnan(avg_vol[i]) and avg_vol[i] > 0:
            vol_ok_pdh = vol[i] > avg_vol[i] * sweep_vol_mult
            pdh_sweep_now = high[i] >= pdh_ref[i] and close[i] < pdh_ref[i] and vol_ok_pdh
            pdl_sweep_now = (not np.isnan(pdl_ref[i])) and low[i] <= pdl_ref[i] and close[i] > pdl_ref[i] and vol_ok_pdh
            if pdh_sweep_now or pdl_sweep_now:
                last_pdh_pdl_sweep_bar = i
        recent_pdh_pdl_sweep_arr[i] = (last_pdh_pdl_sweep_bar is not None) and ((i - last_pdh_pdl_sweep_bar) <= veto_lookback)

        bar_range = high[i] - low[i]
        if bar_range > 0:
            body_ratio = abs(close[i] - open_[i]) / bar_range
            close_pos = (close[i] - low[i]) / bar_range
        else:
            body_ratio = 0.0
            close_pos = 0.0
        is_strong_candle_arr[i] = (body_ratio > candle_body_min) and (close_pos > candle_close_min)

    return {
        "nearest_bsl_strength": pd.Series(nearest_strength_arr, index=df.index),
        "bsl_risk": pd.Series(bsl_risk_arr, index=df.index),
        "recent_pdh_pdl_sweep": pd.Series(recent_pdh_pdl_sweep_arr, index=df.index),
        "is_strong_candle": pd.Series(is_strong_candle_arr, index=df.index),
    }


def run_state_machine(df, bias_score, eli, hm, lm, diP, diM, bsl,
                       bias_entry_min=BIAS_ENTRY_MIN, bias_exit_min=BIAS_EXIT_MIN,
                       bias_invalid_buf=BIAS_INVALID_BUF, smooth_len=SMOOTH_LEN,
                       di_ratio_max=DI_RATIO_MAX, stop_pct=STOP_PCT):
    bias_smooth = bias_score.rolling(smooth_len).mean()
    close = df["Close"].values
    n = len(df)

    in_position = False
    entry_armed = False
    exit_armed = False
    entry_price = None
    entry_idx = None
    events = []

    nearest_bsl = bsl["nearest_bsl_strength"]
    bsl_risk = bsl["bsl_risk"]
    recent_pdh_pdl = bsl["recent_pdh_pdl_sweep"]
    strong_candle = bsl["is_strong_candle"]

    for i in range(1, n):
        b = bias_score.iloc[i]
        bs = bias_smooth.iloc[i]
        eli_now, eli_prev = eli.iloc[i], eli.iloc[i - 1]
        hm_now, hm_prev = hm.iloc[i], hm.iloc[i - 1]
        lm_now, lm_prev = lm.iloc[i], lm.iloc[i - 1]
        di_ratio = (diP.iloc[i] / diM.iloc[i]) if diM.iloc[i] > 0 else 999.0
        di_ok = (diP.iloc[i] > diM.iloc[i]) and (di_ratio < di_ratio_max)

        if b >= bias_entry_min and not in_position:
            entry_armed = True
        if bs < (bias_entry_min - bias_invalid_buf) and entry_armed and not in_position:
            entry_armed = False

        eli_cross_up = (eli_prev <= hm_prev) and (eli_now > hm_now)
        bsl_veto_now = bool(eli_cross_up and (bool(bsl_risk.iloc[i]) or bool(recent_pdh_pdl.iloc[i])) and not bool(strong_candle.iloc[i]))
        buy_cond = entry_armed and eli_cross_up and di_ok and not in_position and not bsl_veto_now

        if bsl_veto_now and (entry_armed and eli_cross_up and di_ok and not in_position):
            reason = f"{nearest_bsl.iloc[i]:.1f}x BSL" if bool(bsl_risk.iloc[i]) else "PDH/PDL"
            events.append((i, df.index[i], "VETO", close[i], reason))

        if b >= bias_exit_min and in_position:
            exit_armed = True
        eli_cross_dn = (eli_prev >= lm_prev) and (eli_now < lm_now)
        sell_cond = exit_armed and eli_cross_dn and in_position and (entry_idx is not None and i > entry_idx)

        stop_price = entry_price * (1 - stop_pct / 100) if in_position and entry_price else None
        stop_cond = in_position and stop_price is not None and close[i] < stop_price

        if buy_cond:
            in_position = True
            entry_price = close[i]
            entry_idx = i
            entry_armed = False
            events.append((i, df.index[i], "BUY", close[i], None))
        elif stop_cond:
            events.append((i, df.index[i], "STOP", close[i], None))
            in_position = False; entry_price = None; entry_idx = None
            entry_armed = False; exit_armed = False
        elif sell_cond:
            events.append((i, df.index[i], "SELL", close[i], None))
            in_position = False; entry_price = None; entry_idx = None
            entry_armed = False; exit_armed = False

    status = "IN_TRADE" if in_position else ("ENTRY_ARMED" if entry_armed else "WAIT")
    last_bsl_note = "-"
    if not np.isnan(nearest_bsl.iloc[-1]):
        last_bsl_note = f"{nearest_bsl.iloc[-1]:.1f}x" if bool(bsl_risk.iloc[-1]) else "risk yok"
    last_event = events[-1] if events else None
    bars_ago = (n - 1 - last_event[0]) if last_event else None
    return {
        "events": events,
        "in_position": in_position,
        "status": status,
        "last_bias": round(float(bias_score.iloc[-1]), 1),
        "last_bias_smooth": round(float(bias_smooth.iloc[-1]), 1),
        "last_bsl_note": last_bsl_note,
        "last_pdh_pdl_risk": bool(recent_pdh_pdl.iloc[-1]),
        "last_strong_candle": bool(strong_candle.iloc[-1]),
        "last_event": ({"type": last_event[2], "time": str(last_event[1]), "price": round(float(last_event[3]), 4),
                         "bars_ago": bars_ago} if last_event else None),
    }
TITLE = "ARGUS BIST SCAN"

def scan_one(ticker, tf, bench_df, is_intraday):
    df = fetch_bars(ticker, tf, days=60)
    if df is None or len(df) < 110:
        return None
    bias_score = compute_bias_score_series(df, bench_df)
    eli, hm, lm = compute_eli_hm_lm(df)
    adx_data = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    diP, diM = adx_data["DMP_14"], adx_data["DMN_14"]
    bsl = build_bsl_levels_and_veto_series(df, is_intraday=is_intraday)
    result = run_state_machine(df, bias_score, eli, hm, lm, diP, diM, bsl)
    result["price"] = round(float(df["Close"].iloc[-1]), 6)
    result["bar_time"] = str(df.index[-1])
    return result


def run_scan():
    tz_tr = timezone(timedelta(hours=3))
    now = datetime.now(tz_tr)
    print(f"{'=' * 65}")
    print(f"{TITLE} — AEI v18.3 sinyal taramasi — {now.strftime('%Y-%m-%d %H:%M')} TR")
    print(f"{len(TICKERS)} ticker x {len(FIXED_TFS)} TF = {len(TICKERS) * len(FIXED_TFS)} kombinasyon")
    print(f"{'=' * 65}")

    results_snapshot = {}
    entry_armed_list = []
    in_trade_list = []

    bench_cache = {}
    for tf in FIXED_TFS:
        bench_cache[tf] = fetch_bars(BENCHMARK, tf, days=60)

    for ticker in TICKERS:
        for tf in FIXED_TFS:
            key = f"{ticker}_{tf}"
            is_intraday = tf != "1d"
            try:
                r = scan_one(ticker, tf, bench_cache.get(tf), is_intraday)
            except Exception as e:
                print(f"  ERR {key}: {e}")
                continue
            if r is None:
                continue

            results_snapshot[key] = {
                "ticker": ticker, "tf": tf, "status": r["status"],
                "bias": r["last_bias"], "bias_smooth": r["last_bias_smooth"],
                "price": r["price"], "bar_time": r["bar_time"],
                "bsl_note": r["last_bsl_note"], "pdh_pdl_risk": r["last_pdh_pdl_risk"],
                "strong_candle": r["last_strong_candle"], "last_event": r["last_event"],
            }

            if r["status"] == "ENTRY_ARMED":
                entry_armed_list.append(f"{ticker} ({tf}) bias={r['last_bias']}")
            elif r["status"] == "IN_TRADE":
                in_trade_list.append(f"{ticker} ({tf}) bias={r['last_bias']}")

            print(f"  {ticker:10s} {tf:4s} {r['status']:12s} bias={r['last_bias']:6.1f} price={r['price']}")

    save_json(RESULTS_FILE_PATH, {
        "scan_time": now.isoformat(),
        "entry_armed": entry_armed_list,
        "in_trade": in_trade_list,
        "all": results_snapshot,
    })

    print(f"\n{len(entry_armed_list)} ENTRY_ARMED | {len(in_trade_list)} IN_TRADE")
    if entry_armed_list:
        print("ENTRY_ARMED:", ", ".join(entry_armed_list))
    if in_trade_list:
        print("IN_TRADE:", ", ".join(in_trade_list))
    print(f"\nSonuc kaydedildi -> {RESULTS_FILE_PATH}")


def main():
    print(f"{TITLE} calisiyor (tek seferlik, cron tetiklemesi)")
    try:
        run_scan()
        print("\nTarama tamamlandi, cikiliyor.")
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"BEKLENMEDIK HATA: {e}\n{error_detail}")

if __name__ == "__main__":
    main()
