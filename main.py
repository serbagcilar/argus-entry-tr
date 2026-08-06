"""
AEI v18.3 -- BIST100 (Yahoo Finance) -- Railway cron, SADECE SINYAL/TELEGRAM
==============================================================================
US/Alpaca'daki main.py ile ayni AEI motoru (bias latch + ELI/HM-LM kesisim +
BSL/PDH-PDL veto), farkli veri kaynagi (yfinance, BIST100) ve FARKLI DAVRANIS:

  ** BU SCRIPT HICBIR EMIR GONDERMEZ ** - BIST'te broker/execution API'miz
  yok (Midas/TradingView execution yolu daha once terk edilmisti). Sadece
  durum hesaplayip YENI bir BUY/SELL/STOP olayi olustugunda Telegram'a
  bildirim atar. Watchlist/held/pozisyon takibi yok - sadece "en son
  bildirdigim olaydan sonra yeni bir olay oldu mu" kontrolu var (tekrar
  tekrar ayni alert'i atmamak icin).

  Yahoo Finance BIST intraday verisi duzensiz olabilir (bazi gunler/barlar
  atlanabilir) - kullanici bunu bilerek kabul etti ("risk alarak devam
  edecegiz, otomatik olmadigindan beni korur").

Env variables (Railway'de ayarlanmali):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID -- Telegram bildirimi icin
  STATE_FILE_PATH                       -- son bildirilen olay zaman damgalari
                                           (varsayilan: state_bist.json, Railway
                                           Volume onerilir, orn. /data/state_bist.json)
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
import warnings
import os
import json
import requests
import traceback
from datetime import datetime, timezone, timedelta
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
STATE_FILE_PATH     = os.environ.get("STATE_FILE_PATH", "state_bist.json")

# ============================================================
# STRATEJI PARAMETRELERI (Pine input'lariyla ayni varsayilan - main.py/US ile
# senkron. NOT: kullanicinin TradingView'de canli ince ayar yaptigi degerler
# (bias_entry_min=20, stop_pct=4.7 vb.) henuz buraya YANSITILMADI - ayri
# onay bekliyor.)
# ============================================================
BIAS_ENTRY_MIN   = 20
BIAS_EXIT_MIN    = 50
BIAS_INVALID_BUF = 5
SMOOTH_LEN       = 3
DI_RATIO_MAX     = 5
STOP_PCT         = 5.0
RS_PERIOD        = 20
HL_U             = 20
K_FRAC           = 0.6  # v18.3: sabit-yuzde (HL_K) yerine range-bazli bant genisligi -
                         # Pine'da Strategy Tester ile dogrulandi (GOOGL, k_frac=0.6)
ELI_ALPHA1       = 0.33
ELI_ALPHA2       = 0.25
BENCHMARK_DEFAULT = "XU100.IS"

# Ticker'lara ozel TF secmeye GEREK YOK - hepsi ayni sabit TF setinde
# taranir. Yahoo Finance'te DOGAL 4 saatlik bar yok (sadece 15dk/30dk/60dk/
# 1g gibi standart araliklar) - 4h burada 1h verisinden YENIDEN ORNEKLENEREK
# (resample) turetiliyor, BIST'in kabaca oturum saatlerine gore.
FIXED_TFS = ["5m", "15m", "1h", "4h"]

# BIST100 + Others (kullanicinin verdigi tam liste)
TICKERS = [
    "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "BTCIM.IS", "CCOLA.IS",
    "CIMSA.IS", "DOAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GUBRF.IS", "HEKTS.IS", "KRDMD.IS", "KUYAS.IS", "MAVI.IS", "MIATK.IS",
    "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SASA.IS", "SISE.IS", "SOKM.IS",
    "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TRALT.IS", "TRMET.IS",
    "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "AHGAZ.IS", "AKSA.IS", "AKSEN.IS",
    "ALARK.IS", "ALFAS.IS", "ANSGR.IS", "ARCLK.IS", "ARDYZ.IS", "BERA.IS",
    "BRYAT.IS", "BSOKE.IS", "CWENE.IS", "DOHOL.IS", "ECILC.IS", "EGEEN.IS",
    "ENERY.IS", "ENJSA.IS", "EUPWR.IS", "GESAN.IS", "IZENR.IS", "KARSN.IS",
    "KCHOL.IS", "KONYA.IS", "ODAS.IS", "OTKAR.IS", "QUAGR.IS", "RALYH.IS",
    "REEDR.IS", "RYGYO.IS", "SAHOL.IS", "SELEC.IS", "SMRTG.IS", "TABGD.IS",
    "TKFEN.IS", "TKNSA.IS", "TMSN.IS", "TTRAK.IS", "ZOREN.IS", "AKCNS.IS",
    "BFREN.IS", "BIOEN.IS", "BOBET.IS", "CANTE.IS", "CVKMD.IS", "DARDL.IS",
    "ECZYT.IS", "ENTRA.IS", "ESEN.IS", "EUREN.IS", "GWIND.IS", "ISDMR.IS",
    "IZMDC.IS", "KAYSE.IS", "KCAER.IS", "SARKY.IS", "SDTTR.IS", "VAKKO.IS",
    "YEOTK.IS", "BANVT.IS", "FADE.IS", "KNFRT.IS", "KRVGD.IS", "KTSKR.IS",
    "OFSYM.IS", "PNSUT.IS", "ANGEN.IS", "BAGFS.IS", "BRISA.IS", "DYOBY.IS",
    "EGGUB.IS", "EGPRO.IS", "EPLAS.IS", "IZFAS.IS", "KBORU.IS", "KMPUR.IS",
    "KOPOL.IS", "KRPLS.IS", "MEDTR.IS", "ONCSM.IS", "POLTK.IS", "TRILC.IS",
    "BMSCH.IS", "ERBOS.IS", "ERCB.IS", "PNLSN.IS", "YKSLN.IS", "ASUZU.IS",
    "EKOS.IS", "GEREL.IS", "HATSN.IS", "HKTM.IS", "JANTS.IS", "KATMR.IS",
    "KLMSN.IS", "MAKIM.IS", "MAKTK.IS", "PARSN.IS", "PRKAB.IS", "SAYAS.IS",
    "SNICA.IS", "ALKA.IS", "BAKAB.IS", "DGNMO.IS", "KARTN.IS", "KONKA.IS",
    "TEZOL.IS", "VKING.IS", "AFYON.IS", "DOGUB.IS", "KLKIM.IS", "KUTPO.IS",
    "MARBL.IS", "NUHCM.IS", "USAK.IS", "DERIM.IS", "DESA.IS", "ENSRI.IS",
    "HATEK.IS", "ISSEN.IS", "KORDS.IS", "KRTEK.IS", "SKTAS.IS", "YATAS.IS",
    "AYDEM.IS", "CATES.IS", "NTGAZ.IS", "PAMEL.IS", "TATEN.IS", "ZEDUR.IS",
    "ANELE.IS", "BRLSM.IS", "DAPGM.IS", "EDIP.IS", "ORGE.IS", "SANEL.IS",
    "TURGG.IS", "YYAPI.IS", "ARZUM.IS", "GENIL.IS", "INTEM.IS", "TGSAS.IS",
    "BEYAZ.IS", "CLEBI.IS", "RYSAS.IS", "TLMAN.IS", "AGYO.IS", "ALGYO.IS",
    "BASGZ.IS", "FZLGY.IS", "HLGYO.IS", "KRGYO.IS", "KZBGY.IS", "NUGYO.IS",
    "PSGYO.IS", "SNGYO.IS", "TRGYO.IS", "INDES.IS", "KAREL.IS", "KFEIN.IS",
    "KRONT.IS", "LINK.IS", "LOGO.IS", "MTRKS.IS", "PAPIL.IS", "PKART.IS",
    "SMART.IS", "GLCVY.IS", "KRDMA.IS",
]

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Telegram credentials eksik, bildirim atlanidi."); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        print("  Telegram OK" if r.status_code == 200 else f"  Telegram HATA: {r.text}")
    except Exception as e:
        print(f"  Telegram exception: {e}")

# ============================================================
# STATE -- sadece "en son bildirilen olay zamani" (emir/pozisyon yok)
# ============================================================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE_PATH):
        return {}
    with open(STATE_FILE_PATH, "r") as f:
        return json.load(f)

def save_state(state: dict):
    with open(STATE_FILE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# FETCH -- yfinance (BIST100). Intraday duzensiz olabilir, D1 en guvenilir.
# ============================================================
_YF_PERIOD_MAP = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "1d": "3y",
}

def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Yahoo'da dogal 4h yok - 1h barlarindan turetiliyor (yaklasik, BIST
    seans saatlerine tam hizali degil ama kabaca dogru)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    df4 = df_1h.resample("4h").agg(agg).dropna(how="all")
    return df4

def fetch_yahoo_bars(symbol: str, tf: str):
    if tf == "4h":
        df_1h = fetch_yahoo_bars(symbol, "1h")
        if df_1h is None or df_1h.empty:
            return None
        df4 = _resample_4h(df_1h)
        return df4 if not df4.empty else None

    period = _YF_PERIOD_MAP.get(tf, "60d")
    interval_map = {"1h": "60m"}
    interval = interval_map.get(tf, tf)
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    except Exception as e:
        print(f"  fetch hatasi {symbol} ({tf}): {e}")
        return None

# 1) ALPHA/BIAS SKORU -- tam seri (vektorize), Pine f_alpha_score() ile ayni agirliklar
# ============================================================
def compute_bias_score_series(df: pd.DataFrame, bench_df: pd.DataFrame, rs_period: int = RS_PERIOD) -> pd.Series:
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    open_ = df["Open"]

    adx_data = ta.adx(high, low, close, length=14)
    adx, dip, dim = adx_data["ADX_14"], adx_data["DMP_14"], adx_data["DMN_14"]

    adx_up = np.select([adx>60, adx>45, adx>35, adx>25, adx>20], [0,10,15,12,5], default=0)
    adx_dn = np.select([adx>35, adx>25, adx>20], [-15,-10,-5], default=0)
    adx_s = np.where(dip > dim, adx_up, adx_dn)

    di_diff = dip - dim
    di_s = np.where(dip > dim, np.where(di_diff >= 10, 20, 12), -10)

    sr = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    ck = [c for c in sr.columns if "k" in c.lower()][0]
    cd = [c for c in sr.columns if "d" in c.lower()][0]
    sk, sd, skp = sr[ck], sr[cd], sr[ck].shift(1)
    strong_up = (adx > 25) & (dip > dim)
    stoch_s = np.select(
        [ (sk >= 40) & (sk <= 65) & (sk > skp),
          (sk > 65)  & (sk <= 80) & (sk > skp),
          (sk > 20)  & (sk < 40)  & ((sk - skp) > 5),
          (sk > 80)  & strong_up,
          (sk > 80)  & ~strong_up,
          (sk < sd)  & (sk > 30),
          (sk < 20) ],
        [20, 14, 9, 6, -10, -5, -7], default=0)

    def _dema(s, p):
        e1 = s.ewm(span=p, adjust=False).mean()
        return 2 * e1 - e1.ewm(span=p, adjust=False).mean()
    d10, d20, d50 = _dema(close, 10), _dema(close, 20), _dema(close, 50)
    d100 = _dema(close, 100)
    d200 = _dema(close, 200) if len(df) >= 200 else d100
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
    sp_s = np.select([ (sm1>3)&(sm2>2)&(sm3>1), (sm1>2)&(sm2>1), sm1>1, sm1<0 ], [8,5,2,-3], default=0)
    dema_s = dema_s + np.where(aligned, sp_s, 0)

    rs_s = np.zeros(len(df))
    try:
        if bench_df is not None and len(bench_df) > rs_period:
            bc = bench_df["Close"].reindex(df.index, method="ffill")
            rs_h = (close / close.shift(rs_period) - 1) * 100
            rs_b = (bc / bc.shift(rs_period) - 1) * 100
            rsd = rs_h - rs_b
            rs_s = np.select(
                [rsd>10, rsd>5, rsd>2, rsd>0, rsd>-3, rsd>-7],
                [15, 10, 7, 3, -3, -7], default=-12)
    except Exception:
        pass

    vm = vol.rolling(20).mean()
    vr = vol / vm.replace(0, np.nan)
    bull = dip > dim
    strong_trend = adx > 20
    # v4.1 fix: yuksek hacim + kirmizi mum -> odul yok (dagitim riski)
    is_green = close > open_
    rv_raw = np.select([ (vr>2.0)&is_green, (vr>2.0)&~is_green, vr>1.5, vr>1.2, vr>0.8 ],
                        [15, 0, 10, 5, 0], default=-5)
    rv_s = np.where(~strong_trend, rv_raw * 0.3, np.where(bull, rv_raw, -rv_raw))

    total = adx_s + di_s + stoch_s + dema_s + rs_s + rv_s
    return pd.Series(total, index=df.index).astype(float)


# ============================================================
# 2) ELI / HM / LM -- Pine'daki f_var / f_ott ile birebir (IIR filtre, bar-bar)
# ============================================================
def compute_eli_hm_lm(df: pd.DataFrame, hl_u=HL_U, k_frac=K_FRAC, alpha1=ELI_ALPHA1, alpha2=ELI_ALPHA2):
    high, low, close = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)

    def f_var(data: np.ndarray) -> np.ndarray:
        r = np.zeros(n)
        r[0] = data[0] if n > 0 else 0.0
        for i in range(1, n):
            b = abs(data[i] - data[i-9]) if i >= 9 else abs(data[i] - data[0])
            window = data[max(0, i-8):i+1]
            c = np.sum(np.abs(np.diff(window))) if len(window) > 1 else 0.0
            d = (b / c) if c != 0 else 0.0
            e = 2.0 / 3.0
            r[i] = d * e * (data[i] - r[i-1]) + r[i-1]
        return r

    def f_ott_range(data: np.ndarray, range_val: np.ndarray, frac: float) -> np.ndarray:
        # v18.3: bant genisligi artik fiyatin sabit yuzdesi degil, son hl_u
        # barin kendi (highest-lowest) range'inin bir kesri - Pine f_ott_range
        # ile birebir ayni (bkz. AEI_v18_3.pine).
        b = range_val * frac
        a = np.where(data != 0, b / data, 0.0)
        c = data - b
        dd = data + b
        for i in range(1, n):
            c[i]  = c[i]  if (c[i]  > c[i-1]  or data[i] < c[i-1])  else c[i-1]
            dd[i] = dd[i] if (dd[i] < dd[i-1] or data[i] > dd[i-1]) else dd[i-1]
        e = c.copy()
        for i in range(1, n):
            if data[i] > e[i-1]:
                e[i] = c[i]
            elif data[i] < e[i-1]:
                e[i] = dd[i]
            else:
                e[i] = e[i-1]
        h = np.where(data > e, e * (1 + a/2), e * (1 - a/2))
        return np.roll(h, 2)  # Pine .shift(2) (gecikme)

    highest_u = pd.Series(high).rolling(hl_u).max().bfill().values
    lowest_u  = pd.Series(low).rolling(hl_u).min().bfill().values
    range_u   = highest_u - lowest_u

    hm = f_ott_range(f_var(highest_u), range_u, k_frac)
    lm = f_ott_range(f_var(lowest_u), range_u, k_frac)

    lead = np.zeros(n); eli = np.zeros(n)
    lead[0] = close[0]; eli[0] = close[0]
    for i in range(1, n):
        lead[i] = 2*close[i] + (alpha1 - 2)*close[i-1] + (1 - alpha1)*lead[i-1]
        eli[i]  = alpha2*lead[i] + (1 - alpha2)*eli[i-1]

    return (pd.Series(eli, index=df.index),
            pd.Series(hm,  index=df.index),
            pd.Series(lm,  index=df.index))


# ============================================================
# 2.5) BSL SEVIYE TESPITI + MUM KALITESI (kullanicinin 6 aylik gozlemi:
# "ilk BSL cok guclu (hacim carpani>2) ise ELI kesisimi cogu zaman
# stop-hunt/sweep, gercek kirilim degil - zarar ediyoruz". Cozum:
# BSL guclu VE mum zayifsa veto et; mum gucluyse (gercek momentum
# breakout'u) BSL gucune bakmaksizin gir.
#
# v18.3 eklentisi (EonMetrics Liquidity Toolkit'ten esinlenerek): "swept"
# tanimi sikilastirildi -> artik sadece seviyeyi GECMEK yetmiyor, fitil
# gecip KAPANISIN GERI ICERIDE olmasi (red) VE hacmin ortalamanin
# uzerinde olmasi gerekiyor ("confirmed" sweep = gercek stop-hunt).
# Seviyeyi gecip devam eden (rejection olmayan) durumlar artik veto
# hesabina girmiyor - bu gercek breakout'lari daha da az cezalandirir.
# Ayrica PDH/PDL (onceki gun yuksek/dusuk) da ayni "confirmed sweep"
# mantigiyla takip edilip veto kaynagina ekleniyor - PDH/PDL hacim
# carpanina bakilmaksizin HER ZAMAN "onemli seviye" sayilir (swing
# pivotlardan farkli olarak evrensel olarak izlenen bir seviye).
# ============================================================
from dataclasses import dataclass

@dataclass
class BSLLevel:
    price: float
    pivot_idx: int
    strength: float       # hacim carpani (pivot barin hacmi / o barin 20-bar ort. hacmi)
    swept: bool = False        # seviye gecildi (artik resting degil, takip durur)
    sweep_idx: int = None
    confirmed: bool = False    # gecis + kapanis geri icerde (red) + hacim filtresi -> gercek sweep
    confirmed_idx: int = None

SWEEP_VOL_MULT = 1.5   # EonMetrics'teki "Volume Multiplier vs. Average" ile ayni varsayilan
SWEEP_VOL_LOOKBACK = 20

def detect_pivot_highs(df: pd.DataFrame, length: int = 5) -> pd.Series:
    high = df["High"]
    n = len(df)
    pivot_high = pd.Series(np.nan, index=df.index)
    for i in range(length, n - length):
        window = high.iloc[i-length:i+length+1]
        if high.iloc[i] == window.max() and (window == high.iloc[i]).sum() == 1:
            pivot_high.iloc[i] = high.iloc[i]
    return pivot_high

def build_bsl_levels(df: pd.DataFrame, pivot_len: int = 5, eq_tol_atr_mult: float = 0.15,
                      vol_lookback: int = 20, sweep_vol_mult: float = SWEEP_VOL_MULT):
    """ARGUS CORE'daki build_levels_and_sweeps'in BSL-only sadelestirilmis hali.
    v18.3: sweep artik iki asamali - swept (gecildi) ve confirmed (gecti+
    kapanis geri icerde+hacim yuksek, yani gercek red/stop-hunt)."""
    high, close, vol = df["High"], df["Close"], df["Volume"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-prev_close).abs(), (df["Low"]-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    avg_vol = vol.rolling(vol_lookback).mean()
    ph = detect_pivot_highs(df, pivot_len)

    levels: list[BSLLevel] = []
    n = len(df)
    for i in range(n):
        eq_tol = (atr.iloc[i] or 0) * eq_tol_atr_mult
        if i >= pivot_len and not np.isnan(ph.iloc[i-pivot_len]):
            price = ph.iloc[i-pivot_len]
            pvol = vol.iloc[i-pivot_len]
            pavg = avg_vol.iloc[i-pivot_len]
            strength = (pvol / pavg) if pavg and pavg > 0 else 1.0
            merged = False
            for lv in levels:
                if not lv.swept and abs(lv.price - price) <= eq_tol:
                    lv.strength = max(lv.strength, strength)
                    merged = True
                    break
            if not merged:
                levels.append(BSLLevel(price, i - pivot_len, strength))
        for lv in levels:
            if lv.swept:
                continue
            if high.iloc[i] >= lv.price:
                lv.swept = True
                lv.sweep_idx = i
                bar_avg_vol = avg_vol.iloc[i]
                vol_ok = bar_avg_vol and bar_avg_vol > 0 and vol.iloc[i] > bar_avg_vol * sweep_vol_mult
                if close.iloc[i] < lv.price and vol_ok:
                    lv.confirmed = True
                    lv.confirmed_idx = i
    return levels

def compute_pdh_pdl_confirmed_sweep(df: pd.DataFrame, sweep_vol_mult: float = SWEEP_VOL_MULT,
                                     vol_lookback: int = SWEEP_VOL_LOOKBACK) -> pd.Series:
    """Onceki gunun High/Low'unu her bar icin hesaplar, confirmed-sweep
    (fitil gecmis + kapanis geri icerde + hacim yuksek) tespit eder.
    Sadece intraday TF'lerde anlamli - gunluk TF'de cagirilmamali (scan_ticker_alpaca
    bunu tf='1d' oldugunda atlar). Donen seri: her bar icin True/False
    (o barda PDH VEYA PDL confirmed-sweep oldu mu)."""
    dates = pd.Series(df.index.date, index=df.index)
    daily_high = df.groupby(dates)["High"].max()
    daily_low = df.groupby(dates)["Low"].min()
    prev_daily_high = daily_high.shift(1)
    prev_daily_low = daily_low.shift(1)
    pdh = dates.map(prev_daily_high)
    pdl = dates.map(prev_daily_low)

    avg_vol = df["Volume"].rolling(vol_lookback).mean()
    vol_ok = df["Volume"] > (avg_vol * sweep_vol_mult)

    pdh_confirmed = (df["High"] >= pdh) & (df["Close"] < pdh) & vol_ok
    pdl_confirmed = (df["Low"] <= pdl) & (df["Close"] > pdl) & vol_ok
    return (pdh_confirmed | pdl_confirmed).fillna(False)

def nearest_recent_sweep_strength(levels: list[BSLLevel], bar_idx: int, lookback: int = 15):
    """Son `lookback` bar icinde CONFIRMED (gercek red+hacim) supurulmus
    BSL'ler arasinda EN GUCLU olanini dondur (en son supurulen degil -
    ELI gecikmeli oldugu icin asil tehlikeli/guclu seviye birkac bar once
    kirilmis, sonra daha zayif bir seviye daha yakinda kirilmis olabilir;
    risk acisindan max onemli). Sadece "confirmed" (gercek red) seviyeler
    sayilir - duz gecip devam eden (rejection olmayan) seviyeler veto
    hesabina girmez. Yoksa None."""
    best_strength = None
    for lv in levels:
        if lv.confirmed and lv.confirmed_idx is not None and (bar_idx - lv.confirmed_idx) <= lookback:
            if best_strength is None or lv.strength > best_strength:
                best_strength = lv.strength
    return best_strength

def recent_pdh_pdl_sweep(pdh_pdl_confirmed_series: pd.Series, bar_idx: int, lookback: int = 15) -> bool:
    """Son `lookback` bar icinde (bar_idx dahil) confirmed bir PDH/PDL
    sweep'i oldu mu."""
    if pdh_pdl_confirmed_series is None:
        return False
    start = max(0, bar_idx - lookback)
    return bool(pdh_pdl_confirmed_series.iloc[start:bar_idx+1].any())

def is_strong_candle(df: pd.DataFrame, i: int, body_ratio_min: float = 0.6, close_pos_min: float = 0.7) -> bool:
    o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
    rng = h - l
    if rng <= 0:
        return False
    body_ratio = abs(c - o) / rng
    close_pos = (c - l) / rng
    return body_ratio > body_ratio_min and close_pos > close_pos_min



def run_state_machine(df, bias_score, eli, hm, lm, diP, diM, levels=None, pdh_pdl_series=None,
                       bias_entry_min=BIAS_ENTRY_MIN, bias_exit_min=BIAS_EXIT_MIN,
                       bias_invalid_buf=BIAS_INVALID_BUF, smooth_len=SMOOTH_LEN,
                       di_ratio_max=DI_RATIO_MAX, stop_pct=STOP_PCT,
                       bsl_strength_veto=2.0, bsl_veto_lookback=15):
    """levels: build_bsl_levels(df) sonucu. None ise BSL/mum filtresi devre disi
    kalir (eski davranis - sadece ELI cross + DI kosulu).
    pdh_pdl_series: compute_pdh_pdl_confirmed_sweep(df) sonucu (sadece intraday
    TF'lerde anlamli). None ise PDH/PDL vetosu devre disi kalir."""
    bias_smooth = bias_score.rolling(smooth_len).mean()
    close = df["Close"].values
    n = len(df)

    in_position = False
    entry_armed = False
    exit_armed = False
    entry_price = None
    entry_idx = None
    events = []  # (idx, timestamp, event_type, price)

    for i in range(1, n):
        b = bias_score.iloc[i]
        bs = bias_smooth.iloc[i]
        eli_now, eli_prev = eli.iloc[i], eli.iloc[i-1]
        hm_now, hm_prev = hm.iloc[i], hm.iloc[i-1]
        lm_now, lm_prev = lm.iloc[i], lm.iloc[i-1]
        di_ratio = (diP.iloc[i] / diM.iloc[i]) if diM.iloc[i] > 0 else 999.0
        di_ok = (diP.iloc[i] > diM.iloc[i]) and (di_ratio < di_ratio_max)

        # ENTRY LATCH
        if b >= bias_entry_min and not in_position:
            entry_armed = True
        # ENTRY LATCH INVALIDATION
        if bs < (bias_entry_min - bias_invalid_buf) and entry_armed and not in_position:
            entry_armed = False

        eli_cross_up = (eli_prev <= hm_prev) and (eli_now > hm_now)

        # BSL guc + PDH/PDL + mum kalitesi vetosu (kullanicinin 6 aylik gozlemi):
        # yakinda CONFIRMED (gercek red+hacim) supurulmus guclu bir BSL varsa
        # (hacim carpani > bsl_strength_veto) VEYA yakinda confirmed bir
        # PDH/PDL sweep'i olduysa, VE bu barin mumu zayifsa -> veto. Mum
        # gucluyse (gercek momentum breakout) risk kaynagina bakilmaksizin izin ver.
        bsl_veto = False
        if eli_cross_up:
            risk_source = False
            if levels is not None:
                strength = nearest_recent_sweep_strength(levels, i, lookback=bsl_veto_lookback)
                if strength is not None and strength > bsl_strength_veto:
                    risk_source = True
            if not risk_source and pdh_pdl_series is not None:
                if recent_pdh_pdl_sweep(pdh_pdl_series, i, lookback=bsl_veto_lookback):
                    risk_source = True
            if risk_source and not is_strong_candle(df, i):
                bsl_veto = True

        buy_cond = entry_armed and eli_cross_up and di_ok and not in_position and not bsl_veto

        # EXIT LATCH
        if b >= bias_exit_min and in_position:
            exit_armed = True
        eli_cross_dn = (eli_prev >= lm_prev) and (eli_now < lm_now)
        sell_cond = exit_armed and eli_cross_dn and in_position and (entry_idx is not None and i > entry_idx)

        stop_price = entry_price * (1 - stop_pct/100) if in_position and entry_price else None
        stop_cond = in_position and stop_price is not None and close[i] < stop_price

        if buy_cond:
            in_position = True
            entry_price = close[i]
            entry_idx = i
            entry_armed = False
            events.append((i, df.index[i], "BUY", close[i]))
        elif stop_cond:
            events.append((i, df.index[i], "STOP", close[i]))
            in_position = False; entry_price = None; entry_idx = None
            entry_armed = False; exit_armed = False
        elif sell_cond:
            events.append((i, df.index[i], "SELL", close[i]))
            in_position = False; entry_price = None; entry_idx = None
            entry_armed = False; exit_armed = False
        elif bsl_veto and not in_position:
            events.append((i, df.index[i], "VETO", close[i]))

    status = "IN_TRADE" if in_position else ("ENTRY_ARMED" if entry_armed else "WAIT")
    return {
        "events": events,
        "in_position": in_position,
        "entry_price": entry_price,
        "status": status,
        "last_bias": round(float(bias_score.iloc[-1]), 1),
        "last_bias_smooth": round(float(bias_smooth.iloc[-1]), 1),
    }

# ============================================================
# TEK TICKER TARAMASI (yfinance/BIST) -- sadece sinyal
# ============================================================
def scan_ticker_bist(ticker: str, tf: str, benchmark: str = BENCHMARK_DEFAULT, verbose=True):
    df = fetch_yahoo_bars(ticker, tf)
    if df is None or df.empty:
        print(f"  {ticker}: veri alinamadi ({tf})")
        return None
    bench_df = fetch_yahoo_bars(benchmark, tf)

    if len(df) < 110:
        print(f"  {ticker}: yetersiz bar sayisi ({len(df)}, {tf})")
        return None

    bias_score = compute_bias_score_series(df, bench_df)
    eli, hm, lm = compute_eli_hm_lm(df)
    adx_data = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    diP, diM = adx_data["DMP_14"], adx_data["DMN_14"]

    levels = build_bsl_levels(df)
    pdh_pdl_series = compute_pdh_pdl_confirmed_sweep(df) if tf != "1d" else None
    result = run_state_machine(df, bias_score, eli, hm, lm, diP, diM, levels=levels, pdh_pdl_series=pdh_pdl_series)
    result["last_price"] = float(df["Close"].iloc[-1])
    result["last_bar_time"] = df.index[-1]

    if verbose:
        print(f"  {ticker} ({tf}) — {len(df)} bar, son bar: {result['last_bar_time']}")
        print(f"  Bias a: {result['last_bias']} (smooth: {result['last_bias_smooth']}) | Durum: {result['status']} | Fiyat: {result['last_price']:.2f}")
    return result


# ============================================================
# SADECE BILDIRIM -- yeni olay varsa Telegram'a at, emir gonderme
# ============================================================
def check_and_notify(ticker: str, tf: str, state: dict, results_out: dict):
    result = scan_ticker_bist(ticker, tf)
    if result is None:
        return

    key = f"{ticker}_{tf}"
    results_out[key] = {
        "ticker": ticker, "tf": tf, "status": result["status"],
        "last_bias": result["last_bias"], "last_price": result["last_price"],
        "last_bar_time": str(result["last_bar_time"]),
    }

    if not result["events"]:
        return

    last_event = result["events"][-1]
    _, ts, ev_type, price = last_event
    ts_key = str(ts)

    prev_ts = state.get(key)
    if prev_ts is None:
        # ilk calisma - gecmis olaylari bildirme, sadece referans noktasi kaydet
        state[key] = ts_key
        print(f"  [{ticker} {tf}] ilk calisma, referans kaydedildi ({ev_type} @ {ts})")
        return

    if ts_key == prev_ts:
        print(f"  [{ticker} {tf}] yeni olay yok (son: {ev_type} @ {ts})")
        return

    # yeni olay -> bildir
    print(f"  [{ticker} {tf}] YENI OLAY: {ev_type} @ {price:.2f} ({ts})")
    emoji = {"BUY": "✅", "SELL": "🔴", "STOP": "⚠️", "VETO": "🟠"}.get(ev_type, "ℹ️")
    msg = (f"{emoji} <b>{ev_type} — {ticker}</b>\n"
           f"TF: {tf} | Fiyat: {price:.2f}\n"
           f"Bias a: {result['last_bias']} | Durum: {result['status']}\n"
           f"{ts}\n"
           f"AEI BIST (sinyal-only, emir yok)")
    send_telegram(msg)
    state[key] = ts_key


def run_scan():
    tz_tr = timezone(timedelta(hours=3))
    now = datetime.now(tz_tr)
    print(f"\n{'='*65}")
    print(f"AEI BIST100 (Yahoo Finance) — {now.strftime('%Y-%m-%d %H:%M')} TR")
    print(f"SADECE SINYAL/TELEGRAM - emir gonderilmiyor | TF'ler: {FIXED_TFS}")
    print(f"{'='*65}")

    state = load_state()
    results_out = {}
    for ticker in TICKERS:
        for tf in FIXED_TFS:
            print(f"\n[{ticker}] ({tf}) taraniyor...")
            try:
                check_and_notify(ticker, tf, state, results_out)
            except Exception as e:
                print(f"  HATA {ticker} {tf}: {e}")

    save_state(state)
    results_path = STATE_FILE_PATH.replace("state_bist.json", "results_bist.json")
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\nGuncel durum tablosu kaydedildi: {results_path}")


# ============================================================
# RAILWAY CRON GIRIS NOKTASI (tek seferlik, while True YOK)
# ============================================================
def main():
    print("AEI BIST100 calisiyor (tek seferlik, cron tetiklemesi)")
    try:
        run_scan()
        print("\nTarama tamamlandi, cikiliyor.")
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"BEKLENMEDIK HATA: {e}\n{error_detail}")
        send_telegram("⚠️ AEI BIST100'de geçici bir hata oluştu. Bir sonraki zamanlanmış çalışmada tekrar denenecek.")

if __name__ == "__main__":
    main()
