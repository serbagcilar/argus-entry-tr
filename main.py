import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import gc
from datetime import datetime, timezone, timedelta
import warnings, os, time
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")

TICKERS = [
    ("TAVHL", "Havacılık"),
    ("PAMEL", "Savunma"),
    ("THYAO", "Havacılık"),
    ("SISE",  "Cam / Holding"),
    ("ASELS", "Savunma"),
    ("EFOR",  "Fintech"),
]

BENCHMARK   = "XU100.IS"
BUY_ESIK    = 50
STRONG_ESIK = 70
INTERVAL    = 900  # 15 dakika

# ============================================================
# FETCH
# ============================================================
def fetch_15m(symbol, days=5):
    try:
        ticker = f"{symbol}.IS"
        df = yf.download(ticker, period=f"{days}d", interval="15m", progress=False)
        if df is None or len(df) < 30: return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except Exception as e:
        print(f"  15M fetch hatasi {symbol}: {e}")
        return None

def fetch_d1(symbol, days=300):
    try:
        ticker = f"{symbol}.IS" if symbol != "XU100" else "XU100.IS"
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if df is None or len(df) < 50: return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except Exception as e:
        print(f"  D1 fetch hatasi {symbol}: {e}")
        return None

# ============================================================
# KRİTER 1: VR x YON (max 20p)
# ============================================================
def calculate_vr_score(df, vr_len=50):
    if len(df) < vr_len + 5: return 0, None, "Unknown"
    close  = df["Close"]
    log_r1 = np.log(close / close.shift(1)).dropna()
    log_r2 = np.log(close / close.shift(2)).dropna()
    if len(log_r1) < vr_len or len(log_r2) < vr_len: return 0, None, "Unknown"
    var1 = log_r1.rolling(vr_len).var().iloc[-1]
    var2 = log_r2.rolling(vr_len).var().iloc[-1]
    if var1 <= 0 or np.isnan(var2): return 0, None, "Unknown"
    vr = round(var2 / (2 * var1), 3)

    if   vr > 1.15: regime = "Strong Trending"
    elif vr > 1.05: regime = "Mild Trending"
    elif vr > 0.95: regime = "Random"
    elif vr > 0.85: regime = "Mild Reverting"
    else:           regime = "Strong Reverting"

    e1   = close.ewm(span=10, adjust=False).mean()
    dema = 2*e1 - e1.ewm(span=10, adjust=False).mean()
    yon_yukari = dema.iloc[-1] > dema.iloc[-3] and close.iloc[-1] > dema.iloc[-1]

    VR_MATRIX = {
        "Strong Trending":  {"up": +20, "down": -20},
        "Mild Trending":    {"up": +10, "down": -10},
        "Random":           {"up":   0, "down":   0},
        "Mild Reverting":   {"up": -10, "down": -10},
        "Strong Reverting": {"up": -10, "down": -10},
    }
    puan = VR_MATRIX.get(regime, {"up":0,"down":0})["up" if yon_yukari else "down"]
    return puan, round(vr, 3), regime

# ============================================================
# KRİTER 2: SMC (max 20p)
# ============================================================
def calculate_smc_score(df_15m, zone_d1):
    if len(df_15m) < 55: return 0, {}
    puan  = 0
    close = df_15m["Close"].iloc[-1]
    low   = df_15m["Low"].iloc[-1]
    detail = {}

    ssl       = df_15m["Low"].shift(1).rolling(50).min().iloc[-1]
    ssl_sweep = (low < ssl) and (close > ssl)
    if ssl_sweep:
        puan += 15
        detail["ssl"] = "SSL Sweep"
    else:
        detail["ssl"] = "—"

    fvg_low  = df_15m["High"].iloc[-3]
    fvg_high = df_15m["Low"].iloc[-1]
    fvg_ok   = (fvg_low < fvg_high) and (fvg_low <= close <= fvg_high)
    if fvg_ok:
        puan += 10
        detail["fvg"] = "FVG"
    else:
        detail["fvg"] = "—"

    if zone_d1 in ("Discount", "Eq-Low"):
        puan += 5
        detail["zone"] = zone_d1
    elif zone_d1 == "Equilibrium":
        puan += 3
        detail["zone"] = zone_d1
    else:
        detail["zone"] = zone_d1

    return min(puan, 20), detail

# ============================================================
# KRİTER 3: AEI (max 20p)
# ============================================================
def calculate_aei_score(df, bench_df,
                        bias_entry_min=20, rs_period=20,
                        hl_u=20, hl_k=1.5,
                        eli_alpha1=0.33, eli_alpha2=0.25):
    if len(df) < 100: return 0, "NONE"
    close = df["Close"]; high = df["High"]
    low   = df["Low"];   vol  = df["Volume"]

    adx_data = ta.adx(high, low, close, length=14)
    if adx_data is None or adx_data.empty: return 0, "NONE"
    adx = adx_data["ADX_14"].iloc[-1]
    dip = adx_data["DMP_14"].iloc[-1]
    dim = adx_data["DMN_14"].iloc[-1]

    if dip > dim:
        adx_s = (0 if adx>60 else 10 if adx>45 else 15 if adx>35 else 12 if adx>25 else 5 if adx>20 else 0)
    else:
        adx_s = (-15 if adx>35 else -10 if adx>25 else -5 if adx>20 else 0)

    di_s = (20 if (dip-dim)>=10 else 12) if dip>dim else -10

    try:
        sr  = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
        ck  = [c for c in sr.columns if "k" in c.lower()][0]
        cd  = [c for c in sr.columns if "d" in c.lower()][0]
        sk  = sr[ck].iloc[-1]; sd = sr[cd].iloc[-1]; skp = sr[ck].iloc[-2]
        su  = adx>25 and dip>dim
        if   40<=sk<=65 and sk>skp:     stoch_s=20
        elif 65<sk<=80  and sk>skp:     stoch_s=14
        elif 20<sk<40   and (sk-skp)>5: stoch_s=9
        elif sk>80 and su:              stoch_s=6
        elif sk>80:                     stoch_s=-10
        elif sk<sd and sk>30:           stoch_s=-5
        elif sk<20:                     stoch_s=-7
        else:                           stoch_s=0
    except: stoch_s=0

    def _dema(s,p):
        e1=s.ewm(span=p,adjust=False).mean()
        return 2*e1-e1.ewm(span=p,adjust=False).mean()

    d10=_dema(close,10).iloc[-1]; d20=_dema(close,20).iloc[-1]
    d50=_dema(close,50).iloc[-1]; d100=_dema(close,100).iloc[-1]
    cl=close.iloc[-1]; aligned=d10>d20 and d20>d50 and d50>d100

    if   cl>d10 and cl>d20 and cl>d50 and cl>d100 and aligned: dema_s=20
    elif cl>d10 and cl>d20 and cl>d50 and cl>d100:             dema_s=14
    elif cl>d10 and cl>d20 and cl>d50:                         dema_s=8
    elif cl>d10 and cl>d20:                                    dema_s=3
    else:                                                       dema_s=0

    rs_s=0
    try:
        if bench_df is not None and len(df)>rs_period and len(bench_df)>rs_period:
            rsd=((cl/close.iloc[-rs_period-1]-1)*100)-((bench_df["Close"].iloc[-1]/bench_df["Close"].iloc[-rs_period-1]-1)*100)
            rs_s=(15 if rsd>10 else 10 if rsd>5 else 7 if rsd>2 else 3 if rsd>0 else -3 if rsd>-3 else -7 if rsd>-7 else -12)
    except: pass

    vm=vol.rolling(20).mean().iloc[-1]; vr_r=vol.iloc[-1]/vm if vm>0 else 0
    rv_raw=(15 if vr_r>2.0 else 10 if vr_r>1.5 else 5 if vr_r>1.2 else 0 if vr_r>0.8 else -5)
    rv_s=rv_raw*0.3 if adx<=20 else (rv_raw if dip>dim else -rv_raw)

    bias_score=adx_s+di_s+stoch_s+dema_s+rs_s+rv_s
    if bias_score < bias_entry_min: return 0, "NONE"

    try:
        def fv(data,length):
            sma2=data.rolling(2).mean(); b=data.diff(9).abs(); c=data.diff().abs().rolling(9).sum()
            d=(b/c.replace(0,np.nan)).fillna(0); e=2.0/(length+1)
            arr=sma2.values.copy(); d_arr=d.values; data_arr=data.values
            for i in range(1,len(arr)):
                arr[i]=d_arr[i]*e*(data_arr[i]-arr[i-1])+arr[i-1] if not np.isnan(arr[i-1]) else arr[i]
            return pd.Series(arr,index=data.index)

        def fo(data,mult):
            a=mult/100; b=data*a; c_arr=(data-b).values.copy(); dd_arr=(data+b).values.copy(); data_arr=data.values
            for i in range(1,len(data_arr)):
                c_arr[i] =c_arr[i]  if (c_arr[i] >c_arr[i-1]  or data_arr[i]<c_arr[i-1])  else c_arr[i-1]
                dd_arr[i]=dd_arr[i] if (dd_arr[i]<dd_arr[i-1] or data_arr[i]>dd_arr[i-1]) else dd_arr[i-1]
            e_arr=c_arr.copy()
            for i in range(1,len(data_arr)):
                e_arr[i]=c_arr[i] if data_arr[i]>e_arr[i-1] else (dd_arr[i] if data_arr[i]<e_arr[i-1] else e_arr[i-1])
            h=np.where(data_arr>e_arr,e_arr*(1+a/2),e_arr*(1-a/2))
            return pd.Series(h,index=data.index).shift(2)

        hm=fo(fv(high.rolling(hl_u).max(),2),hl_k)

        lead_arr=close.values.copy().astype(float); eli_arr=close.values.copy().astype(float)
        for i in range(1,len(lead_arr)):
            lead_arr[i]=2*close.values[i]+(eli_alpha1-2)*close.values[i-1]+(1-eli_alpha1)*lead_arr[i-1]
            eli_arr[i] =eli_alpha2*lead_arr[i]+(1-eli_alpha2)*eli_arr[i-1]

        eli_s=pd.Series(eli_arr,index=close.index)
        eli_cross=(eli_s.iloc[-1]>hm.iloc[-1]) and (eli_s.iloc[-2]<=hm.iloc[-2])
        di_ratio=dip/dim if dim>0 else 999
        di_ok=dip>dim and di_ratio<2.5

        if eli_cross and di_ok: return 20,"BUY"
        else:                   return 10,"ARMED"
    except: return 10,"ARMED"

# ============================================================
# KRİTER 4: JMA Filter (max 20p)
# ============================================================
def calculate_jma_score(df, tf_minutes=385, fast=3, slow=5, c=1):
    if len(df)<50: return 0,"BLUE"
    close=df["Close"]
    length=max(int((tf_minutes/15)*7),10)
    beta=0.45*(length-1)/(0.45*(length-1)+2); alpha=beta**2
    e0_v=float(close.iloc[0]); e1_v=0.0; e2_v=0.0; jma_v=float(close.iloc[0]); jma_arr=[]
    for src in close.values:
        e0_v=(1-alpha)*src+alpha*e0_v
        e1_v=(src-e0_v)*(1-beta)+beta*e1_v
        e2_v=(e0_v+1.5*e1_v-jma_v)*((1-alpha)**2)+(alpha**2)*e2_v
        jma_v=e2_v+jma_v; jma_arr.append(jma_v)
    jma_s=pd.Series(jma_arr,index=close.index)
    p=jma_s/close; p2=p/-1; z=(p2-p)+2; x=z*100
    x1=(x.rolling(fast).mean()+x.rolling(slow).mean()).rolling(c).mean()
    if x1.empty or np.isnan(x1.iloc[-1]): return 0,"BLUE"
    return (20,"WHITE") if x1.iloc[-1]>=0 else (0,"BLUE")

# ============================================================
# KRİTER 5: AEC (max 10p)
# ============================================================
def calculate_aec_score(df, rsi_len=100, sup_len=50, ott_mult=0.2):
    if len(df)<rsi_len+20: return 0,"SHORT"
    close=df["Close"]
    def fv(data,length):
        sma2=data.rolling(2).mean(); b=data.diff(9).abs(); c=data.diff().abs().rolling(9).sum()
        d=(b/c.replace(0,np.nan)).fillna(0); e=2.0/(length+1)
        arr=sma2.values.copy(); d_arr=d.values; data_arr=data.values
        for i in range(1,len(arr)):
            arr[i]=d_arr[i]*e*(data_arr[i]-arr[i-1])+arr[i-1] if not np.isnan(arr[i-1]) else arr[i]
        return pd.Series(arr,index=data.index)
    def fo(data,mult):
        a=mult/100; b=data*a; c_arr=(data-b).values.copy(); dd_arr=(data+b).values.copy(); data_arr=data.values
        for i in range(1,len(data_arr)):
            c_arr[i] =c_arr[i]  if (c_arr[i] >c_arr[i-1]  or data_arr[i]<c_arr[i-1])  else c_arr[i-1]
            dd_arr[i]=dd_arr[i] if (dd_arr[i]<dd_arr[i-1] or data_arr[i]>dd_arr[i-1]) else dd_arr[i-1]
        e_arr=c_arr.copy()
        for i in range(1,len(data_arr)):
            e_arr[i]=c_arr[i] if data_arr[i]>e_arr[i-1] else (dd_arr[i] if data_arr[i]<e_arr[i-1] else e_arr[i-1])
        h=np.where(data_arr>e_arr,e_arr*(1+a/2),e_arr*(1-a/2))
        return pd.Series(h,index=data.index).shift(2)
    try:
        rsi_v=ta.rsi(close,length=rsi_len).dropna()
        sup=fv(rsi_v,sup_len)+1000
        line=fo(fv(sup,2),ott_mult)
        return (10,"LONG") if float(sup.iloc[-1])>float(line.iloc[-1]) else (0,"SHORT")
    except: return 0,"SHORT"

# ============================================================
# KRİTER 6: DEMA100 (max 10p)
# ============================================================
def calculate_dema100_score(df):
    if len(df)<110: return 0,False
    close=df["Close"]; e1=close.ewm(span=100,adjust=False).mean()
    dema=2*e1-e1.ewm(span=100,adjust=False).mean()
    above=bool(close.iloc[-1]>dema.iloc[-1])
    return (10,True) if above else (0,False)

# ============================================================
# D1 ZONE
# ============================================================
def classify_zone_d1(df_d1, lookback=60):
    if df_d1 is None or len(df_d1)<lookback: return "No-Data"
    r=df_d1.tail(lookback); sh=r["High"].max(); sl=r["Low"].min()
    cur=df_d1["Close"].iloc[-1]; rng=sh-sl
    if rng==0: return "Flat"
    pos=(cur-sl)/rng
    if   pos<0.30: return "Discount"
    elif pos<0.45: return "Eq-Low"
    elif pos<0.55: return "Equilibrium"
    elif pos<0.70: return "Eq-High"
    elif pos<0.85: return "Premium"
    else:          return "Weak-High"

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Telegram credentials eksik."); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data={"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"HTML"}
    try:
        r=requests.post(url,data=data,timeout=10)
        print("  Telegram OK" if r.status_code==200 else f"  Telegram HATA: {r.text}")
    except Exception as e: print(f"  Telegram exception: {e}")

# ============================================================
# TEK TARAMA
# ============================================================
def run_scan():
    tz_tr = timezone(timedelta(hours=3))
    now   = datetime.now(tz_tr)
    print(f"\n{'='*65}")
    print(f"ARGUS ENTRY TR — {now.strftime('%Y-%m-%d %H:%M')} TR")
    print(f"{'='*65}")

    bench_d1 = fetch_d1("XU100")

    for ticker, sector in TICKERS:
        print(f"[{ticker}] taranıyor...")
        try:
            df_15m = fetch_15m(ticker, days=5)
            df_d1  = fetch_d1(ticker, days=300)

            if df_15m is None or len(df_15m) < 60:
                print(f"  15M veri yetersiz"); continue

            zone_d1             = classify_zone_d1(df_d1)
            s1, vr_val, regime  = calculate_vr_score(df_15m)
            s2, smc_detail      = calculate_smc_score(df_15m, zone_d1)
            s3, aei_status      = calculate_aei_score(df_15m, bench_d1)
            s4, jma_color       = calculate_jma_score(df_15m)
            s5, aec_status      = calculate_aec_score(df_15m)
            s6, dema_above      = calculate_dema100_score(df_15m)

            total = s1+s2+s3+s4+s5+s6
            price = round(float(df_15m["Close"].iloc[-1]), 2)

            if   total >= STRONG_ESIK: signal = "STRONG BUY"
            elif total >= BUY_ESIK:    signal = "BUY"
            else:                      signal = "BEKLE"

            print(f"  Skor:{total}/100 | {signal}")

            if total >= BUY_ESIK:
                emoji = "🔥 STRONG BUY" if total >= STRONG_ESIK else "✅ BUY"
                msg = (f"<b>{emoji} — {ticker}</b>\n"
                       f"Fiyat: <b>{price} TL</b>\n"
                       f"Skor: <b>{total}/100</b>\n"
                       f"Sektor: {sector} | D1 Zone: {zone_d1}\n\n"
                       f"VR: {s1:+d}p ({regime})\n"
                       f"SMC: {s2:+d}p ({smc_detail.get('ssl','—')} / {smc_detail.get('fvg','—')})\n"
                       f"AEI: {s3:+d}p ({aei_status})\n"
                       f"JMA: {s4:+d}p ({jma_color})\n"
                       f"AEC: {s5:+d}p ({aec_status})\n"
                       f"DEMA100: {s6:+d}p\n\n"
                       f"{now.strftime('%H:%M')} TR | ARGUS ENTRY TR")
                send_telegram(msg)

            # RAM temizle
            del df_15m, df_d1
        except Exception as e:
            print(f"  HATA {ticker}: {e}")

    del bench_d1
    gc.collect()

# ============================================================
# ANA DÖNGÜ
# ============================================================
print("ARGUS ENTRY TR başlatıldı — 7/24 aktif")
send_telegram("🤖 <b>ARGUS ENTRY TR başlatıldı</b>\n7/24 aktif | 15dk tarama")

while True:
    try:
        run_scan()
    except Exception as e:
        print(f"DÖNGÜ HATA: {e}")
        send_telegram(f"⚠️ ARGUS ENTRY TR hata: {e}")
    print(f"⏳ {INTERVAL//60} dakika bekleniyor...")
    time.sleep(INTERVAL)
