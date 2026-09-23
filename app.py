import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_poisson_deviance
import sqlite3
from datetime import datetime
import yfinance as yf
import requests
from math import erf
from scipy.optimize import minimize

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI VE CSS STİLİ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Finansal Veri Bilimi & Aktüeryal Lab",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
.block-container { color: #0b1f33 !important; }
.block-container p, .block-container span, .block-container label, .block-container div, .block-container li { color: #0b1f33 !important; }
.stApp { background-color: #f8f9fa; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
header[data-testid="stHeader"] { background-color: #ffffff !important; }
header[data-testid="stHeader"] * { color: #000000 !important; fill: #000000 !important; }
[data-testid="collapsedControl"] svg, [data-testid="collapsedControl"] path, [data-testid="stSidebarCollapsedControl"] svg, button[kind="header"] svg { color: #000000 !important; fill: #000000 !important; }
[data-testid="stSidebar"] { background-color: #0b1f33; color: #ffffff; }
[data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] span { color: #ffffff !important; }
h1, h2, h3, h4, h5, h6 { color: #0b1f33 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #0055a5 !important; border-color: #0055a5 !important; }
.stSlider [data-baseweb="slider"] div > div > div > div { background-color: #0055a5 !important; }
div.stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #0055a5; }
.stButton>button { background-color: #0055a5; color: white; border-radius: 6px; border: none; padding: 0.5rem 1rem; font-weight: 600; }
.stButton>button:hover { background-color: #003d73; color: white; }
.model-badge { background-color: #e6f4ea; color: #1e6b34; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }
.demo-badge { background-color: #fff4e5; color: #8a5a00; padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; display: inline-block; margin-bottom: 10px; }

.stMultiSelect [data-baseweb="tag"] {
    background-color: #0055a5 !important;
    border-radius: 20px !important;
    padding: 3px 6px 3px 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(0,85,165,0.25);
    transition: background-color 0.15s ease;
}
.stMultiSelect [data-baseweb="tag"]:hover { background-color: #003d73 !important; }
.stMultiSelect [data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"] * { color: #ffffff !important; }
.stMultiSelect [data-baseweb="tag"] svg { fill: #ffffff !important; opacity: 0.8; }
.stMultiSelect [data-baseweb="tag"] svg:hover { opacity: 1; }
.stMultiSelect > div > div,
.stSelectbox > div > div {
    border-radius: 8px !important;
    border-color: #d5dce3 !important;
}
.stMultiSelect > div > div:focus-within,
.stSelectbox > div > div:focus-within {
    border-color: #0055a5 !important;
    box-shadow: 0 0 0 1px #0055a5 !important;
}
li[role="option"]:hover { background-color: #e8f1fb !important; }
li[role="option"][aria-selected="true"] { background-color: #d7e6f7 !important; }
ul[data-testid="stMultiSelectPopover"] li:hover,
div[data-baseweb="popover"] li:hover { background-color: #e8f1fb !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; margin-bottom: 20px;">
        <a href="https://www.linkedin.com/in/sultan-kuş/" target="_blank" style="color: #0077b5; font-size: 32px; text-decoration: none;" title="LinkedIn">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/SultanKus" target="_blank" style="color: #ffffff; font-size: 32px; text-decoration: none;" title="GitHub">
            <i class="fab fa-github"></i>
        </a>
        <a href="mailto:kussultannn34@gmail.com" style="color: #ea4335; font-size: 32px; text-decoration: none;" title="Email Gönder">
            <i class="fas fa-envelope"></i>
        </a>
    </div>
    <hr style="border-top: 1px solid #ffffff; opacity: 0.2;">
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# VERİTABANI (SQLite) - Üyelik ve Ziyaretçi Logları Entegre Edildi
# ---------------------------------------------------------
def veritabani_olustur():
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS simulasyonlar (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            kullanici TEXT, 
            tarih TEXT, 
            modul_adi TEXT, 
            girdi_detayi TEXT, 
            sonuc_deger TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS kullanicilar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kullanici_adi TEXT UNIQUE,
            sifre TEXT,
            kayit_tarihi TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ziyaretci_loglari (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zaman TEXT,
            islem_tipi TEXT,
            detay TEXT
        )
    ''')
    conn.commit()
    conn.close()

veritabani_olustur()

def ziyaret_logla(islem_tipi, detay):
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    c = conn.cursor()
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO ziyaretci_loglari (zaman, islem_tipi, detay) VALUES (?, ?, ?)", (zaman, islem_tipi, detay))
    conn.commit()
    conn.close()

def kayit_ekle(modul_adi, girdi_detayi, sonuc_deger):
    kullanici = st.session_state.get("aktif_kullanici", "Misafir")
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    c = conn.cursor()
    tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO simulasyonlar (kullanici, tarih, modul_adi, girdi_detayi, sonuc_deger) VALUES (?, ?, ?, ?, ?)",
              (kullanici, tarih, modul_adi, girdi_detayi, sonuc_deger))
    conn.commit()
    conn.close()

def kullanici_gecmisi_getir():
    kullanici = st.session_state.get("aktif_kullanici", "Misafir")
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    df = pd.read_sql("SELECT tarih, modul_adi, girdi_detayi, sonuc_deger FROM simulasyonlar WHERE kullanici = ? ORDER BY id DESC", conn, params=(kullanici,))
    conn.close()
    return df

def tum_loglari_getir():
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    df_sim = pd.read_sql("SELECT * FROM simulasyonlar ORDER BY id DESC", conn)
    df_log = pd.read_sql("SELECT * FROM ziyaretci_loglari ORDER BY id DESC", conn)
    df_user = pd.read_sql("SELECT * FROM kullanicilar ORDER BY id DESC", conn)
    conn.close()
    return df_sim, df_log, df_user

# --- SIDEBAR KULLANICI GİRİŞ & ÜYELİK PANELİ ---
if "aktif_kullanici" not in st.session_state:
    st.session_state["aktif_kullanici"] = None

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Kullanıcı Paneli")

if st.session_state["aktif_kullanici"] is None:
    islem = st.sidebar.radio("İşlem Seçin", ["Giriş Yap", "Üye Ol"], horizontal=True, key="auth_radio")
    k_adi = st.sidebar.text_input("Kullanıcı Adı", key="sidebar_k_adi")
    sifre = st.sidebar.text_input("Şifre", type="password", key="sidebar_sifre")

    if islem == "Üye Ol":
        if st.sidebar.button("Kayıt Ol"):
            if k_adi and sifre:
                try:
                    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
                    c = conn.cursor()
                    c.execute("INSERT INTO kullanicilar (kullanici_adi, sifre, kayit_tarihi) VALUES (?, ?, ?)",
                              (k_adi, sifre, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                    ziyaret_logla("Yeni Üye", f"Kullanıcı kayıt oldu: {k_adi}")
                    st.sidebar.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
                except Exception:
                    st.sidebar.error("Bu kullanıcı adı zaten alınmış.")
            else:
                st.sidebar.warning("Tüm alanları doldurun.")
    else:
        if st.sidebar.button("Giriş Yap"):
            conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
            df_u = pd.read_sql("SELECT * FROM kullanicilar WHERE kullanici_adi = ? AND sifre = ?", conn, params=(k_adi, sifre))
            conn.close()
            if not df_u.empty:
                st.session_state["aktif_kullanici"] = k_adi
                ziyaret_logla("Giriş", f"Kullanıcı giriş yaptı: {k_adi}")
                st.sidebar.success(f"Hoş geldin, {k_adi}!")
                st.rerun()
            else:
                st.sidebar.error("Hatalı kullanıcı adı veya şifre.")
else:
    st.sidebar.success(f"Oturum Açık: **{st.session_state['aktif_kullanici']}**")
    if st.sidebar.button("Çıkış Yap"):
        ziyaret_logla("Çıkış", f"Kullanıcı çıkış yaptı: {st.session_state['aktif_kullanici']}")
        st.session_state["aktif_kullanici"] = None
        st.rerun()

def model_rozeti(auc, f1, kaynak):
    st.markdown(
        f'<div class="model-badge">✅ Eğitilmiş model — AUC: {auc:.3f} · F1: {f1:.3f} · Kaynak: {kaynak}</div>',
        unsafe_allow_html=True
    )

def demo_rozeti(metin="Bu sayfa kavramsal bir formül gösterimidir; canlı veri veya eğitilmiş model kullanmaz."):
    st.markdown(f'<div class="demo-badge">🧪 {metin}</div>', unsafe_allow_html=True)

def egitim_notu(icerik, baslik="📚 Bu modül nasıl çalışıyor? (Teori + Yöntem)"):
    """Her modülün altında/üstünde açılır bir eğitim/açıklama bloğu gösterir."""
    with st.expander(baslik):
        st.markdown(icerik)

# ---------------------------------------------------------
# GERÇEK VERİ: KASKO (freMTPL2freq)
# ---------------------------------------------------------
@st.cache_data
def varsayilan_kasko_verisi_getir():
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    return dataset.frame[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

@st.cache_resource
def kasko_glm_egit():
    df = varsayilan_kasko_verisi_getir().sample(50000, random_state=42)
    X = df[['DrivAge', 'VehAge', 'VehPower']]
    y = df['ClaimNb']
    exposure = df['Exposure'].clip(lower=0.01)

    X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
        X, y, exposure, test_size=0.25, random_state=42
    )
    model = PoissonRegressor(alpha=1e-4, max_iter=500)
    model.fit(X_train, y_train / exp_train, sample_weight=exp_train)

    y_pred_test = model.predict(X_test) * exp_test
    deviance = mean_poisson_deviance(y_test.clip(lower=1e-6), y_pred_test.clip(lower=1e-6))
    return model, deviance, len(X_train), len(X_test)

# ---------------------------------------------------------
# SENTETİK EĞİTİM VERİSİ ÜRETİCİLERİ
# ---------------------------------------------------------
def _sentetik_veri_kredi(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    gelir = rng.normal(35000, 12000, n).clip(8000, None)
    borc = rng.normal(10000, 8000, n).clip(0, None)
    yas = rng.integers(20, 70, n)
    sure_ay = rng.integers(6, 61, n)
    oran = borc / gelir
    logit = 3.0 * oran + 0.02 * sure_ay - 0.015 * (yas - 40) - 1.6 + rng.normal(0, 0.6, n)
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({'gelir': gelir, 'borc': borc, 'yas': yas, 'sure_ay': sure_ay, 'hedef': y})

def _sentetik_veri_churn(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    kredi_skoru = rng.integers(350, 851, n)
    musterilik_yil = rng.integers(1, 21, n)
    sikayet_sayisi = rng.poisson(1.2, n)
    urun_sayisi = rng.integers(1, 6, n)
    logit = (-0.015 * (kredi_skoru - 600)) + (-0.25 * musterilik_yil) + (0.55 * sikayet_sayisi) + (-0.35 * urun_sayisi) + rng.normal(0, 0.8, n)
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({'kredi_skoru': kredi_skoru, 'musterilik_yil': musterilik_yil,
                          'sikayet_sayisi': sikayet_sayisi, 'urun_sayisi': urun_sayisi, 'hedef': y})

def _sentetik_veri_fraud(n=4000, seed=11):
    rng = np.random.default_rng(seed)
    hasar_saati = rng.integers(0, 24, n)
    police_yasi_gun = rng.integers(1, 1000, n)
    hasar_tutari = rng.gamma(2, 3000, n)
    onceki_hasar_sayisi = rng.poisson(0.5, n)
    gece_faktoru = np.where((hasar_saati < 5) | (hasar_saati > 22), 1, 0)
    logit = 1.4 * gece_faktoru - 0.004 * police_yasi_gun + 0.00015 * hasar_tutari + 0.6 * onceki_hasar_sayisi - 2.0 + rng.normal(0, 0.7, n)
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return pd.DataFrame({'hasar_saati': hasar_saati, 'police_yasi_gun': police_yasi_gun,
                          'hasar_tutari': hasar_tutari, 'onceki_hasar_sayisi': onceki_hasar_sayisi, 'hedef': y})

@st.cache_resource
def kredi_risk_modelini_egit():
    try:
        veri = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
        df = veri.frame.copy()
        y = (df['class'] == 'bad').astype(int)
        X = pd.get_dummies(df.drop(columns=['class']), drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=3000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
        for kolon in ['age', 'duration', 'credit_amount', 'existing_credits']:
            if kolon not in X.columns:
                raise KeyError(f"Beklenen sütun bulunamadı: {kolon}")
        return {"tip": "gercek", "model": model, "kolonlar": X.columns, "varsayilan": X.median(),
                "auc": auc, "f1": f1, "kaynak": "OpenML German Credit (credit-g)"}
    except Exception:
        df = _sentetik_veri_kredi()
        X, y = df.drop(columns=['hedef']), df['hedef']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
        return {"tip": "sentetik", "model": model, "auc": auc, "f1": f1,
                "kaynak": "Sentetik veri (gerçek veri setine erişilemedi)"}

@st.cache_resource
def churn_modelini_egit():
    df = _sentetik_veri_churn()
    X, y = df.drop(columns=['hedef']), df['hedef']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
    return model, auc, f1

@st.cache_resource
def fraud_modelini_egit():
    df = _sentetik_veri_fraud()
    X, y = df.drop(columns=['hedef']), df['hedef']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, (y_prob > 0.5).astype(int))
    return model, auc, f1

# ---------------------------------------------------------
# GERÇEK PORTFÖY OPTİMİZASYONU
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def markowitz_veri_getir(hisseler, periyot="2y"):
    fiyatlar = pd.DataFrame()
    for h in hisseler:
        veri = yf.Ticker(h).history(period=periyot)['Close']
        if not veri.empty:
            fiyatlar[h] = veri
    fiyatlar = fiyatlar.dropna()
    getiriler = fiyatlar.pct_change().dropna()
    ort_getiri = getiriler.mean() * 252
    kovaryans = getiriler.cov() * 252
    return ort_getiri, kovaryans

def _portfoy_varyansi(agirliklar, kovaryans):
    return agirliklar @ kovaryans.values @ agirliklar

def min_varyans_agirliklari(kovaryans, hedef_getiri, ort_getiri):
    n = len(ort_getiri)
    kisitlar = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, ort_getiri) - hedef_getiri},
    ]
    sinirlar = tuple((0.0, 1.0) for _ in range(n))
    sonuc = minimize(_portfoy_varyansi, x0=np.repeat(1 / n, n), args=(kovaryans,),
                      method='SLSQP', bounds=sinirlar, constraints=kisitlar)
    return sonuc.x if sonuc.success else None

def maksimum_sharpe_agirliklari(kovaryans, ort_getiri, risksiz_oran=0.30):
    n = len(ort_getiri)
    def negatif_sharpe(w):
        getiri = np.dot(w, ort_getiri)
        risk = np.sqrt(w @ kovaryans.values @ w)
        return -(getiri - risksiz_oran) / risk if risk > 0 else 0
    kisitlar = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    sinirlar = tuple((0.0, 1.0) for _ in range(n))
    sonuc = minimize(negatif_sharpe, x0=np.repeat(1 / n, n), method='SLSQP', bounds=sinirlar, constraints=kisitlar)
    return sonuc.x if sonuc.success else None

# ---------------------------------------------------------
# CANLI PİYASA VERİSİ
# ---------------------------------------------------------
def tr_sayi(x, ondalik=2):
    try:
        s = f"{float(x):,.{ondalik}f}"
    except (TypeError, ValueError):
        return "—"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")

def _degisim_yuzde(seri, gun):
    seri = seri.dropna()
    if len(seri) < 2:
        return None
    konum = max(0, len(seri) - 1 - gun)
    onceki = seri.iloc[konum]
    if onceki == 0:
        return None
    return (seri.iloc[-1] / onceki - 1) * 100

def _ybb_degisim(seri):
    seri = seri.dropna()
    if seri.empty:
        return None
    bu_yil = seri[seri.index.year == seri.index[-1].year]
    if len(bu_yil) < 2 or bu_yil.iloc[0] == 0:
        return None
    return (bu_yil.iloc[-1] / bu_yil.iloc[0] - 1) * 100

def _rsi(seri, periyot=14):
    delta = seri.diff()
    kazanc = delta.clip(lower=0).ewm(alpha=1 / periyot, adjust=False).mean()
    kayip = (-delta.clip(upper=0)).ewm(alpha=1 / periyot, adjust=False).mean()
    rs = kazanc / kayip.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

ONS_GRAM = 31.1034768

PANO_ENSTRUMANLARI = [
    {"kod": "XU100.IS",   "ad": "BIST 100",       "birim": "",   "ondalik": 0},
    {"kod": "TRY=X",      "ad": "Dolar / TL",     "birim": "₺",  "ondalik": 4},
    {"kod": "EURTRY=X",   "ad": "Euro / TL",      "birim": "₺",  "ondalik": 4},
    {"kod": "GRAM_ALTIN", "ad": "Gram Altın",     "birim": "₺",  "ondalik": 2},
    {"kod": "GC=F",       "ad": "Ons Altın",      "birim": "$",  "ondalik": 2},
    {"kod": "BZ=F",       "ad": "Brent Petrol",   "birim": "$",  "ondalik": 2},
    {"kod": "SI=F",       "ad": "Ons Gümüş",      "birim": "$",  "ondalik": 2},
    {"kod": "BTC-USD",    "ad": "Bitcoin",        "birim": "$",  "ondalik": 0},
]

PANO_SEMBOLLERI = [e["kod"] for e in PANO_ENSTRUMANLARI if e["kod"] != "GRAM_ALTIN"]
PERIYOT_SECENEKLERI = {"1 Hafta": 5, "1 Ay": 22, "3 Ay": 66, "6 Ay": 132, "1 Yıl": 252}

@st.cache_data(ttl=900, show_spinner=False)
def pano_verisi_getir():
    ham = yf.download(PANO_SEMBOLLERI, period="1y", progress=False, auto_adjust=True, threads=True)
    if ham is None or ham.empty:
        return pd.DataFrame()
    if isinstance(ham.columns, pd.MultiIndex):
        kapanis = ham["Close"].copy()
    else:
        kapanis = ham[["Close"]].copy()
        kapanis.columns = PANO_SEMBOLLERI[:1]
    try:
        if getattr(kapanis.index, "tz", None) is not None:
            kapanis.index = kapanis.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    kapanis.index = pd.to_datetime(kapanis.index).normalize()
    kapanis = kapanis.ffill().dropna(how="all")
    if {"GC=F", "TRY=X"}.issubset(kapanis.columns):
        kapanis["GRAM_ALTIN"] = kapanis["GC=F"] / ONS_GRAM * kapanis["TRY=X"]
    return kapanis

@st.cache_data(ttl=3600, show_spinner=False)
def canli_piyasa_verisi_getir(sembol, periyot="1y"):
    return yf.Ticker(sembol).history(period=periyot)

@st.cache_data(ttl=1800, show_spinner=False)
def hisse_ara(sorgu, max_sonuc=8):
    try:
        sonuc = yf.Search(sorgu, max_results=max_sonuc)
        quotes = getattr(sonuc, "quotes", [])
        return [(q.get("symbol"), q.get("shortname") or q.get("longname") or q.get("symbol"))
                for q in quotes if q.get("symbol")]
    except Exception:
        return []

BIST_POPULER = [
    ("THYAO.IS", "Türk Hava Yolları"), ("AKBNK.IS", "Akbank"), ("KCHOL.IS", "Koç Holding"),
    ("SASA.IS", "Sasa Polyester"), ("FROTO.IS", "Ford Otosan"), ("BIMAS.IS", "BİM"),
    ("EREGL.IS", "Ereğli Demir Çelik"), ("TUPRS.IS", "Tüpraş"), ("ASELS.IS", "Aselsan"),
    ("GARAN.IS", "Garanti BBVA"), ("SISE.IS", "Şişecam"), ("PGSUS.IS", "Pegasus"),
    ("YKBNK.IS", "Yapı Kredi"), ("TCELL.IS", "Turkcell"), ("ISCTR.IS", "İş Bankası C"),
    ("HALKB.IS", "Halkbank"), ("VAKBN.IS", "VakıfBank"), ("HEKTS.IS", "Hektaş"),
    ("KOZAL.IS", "Koza Altın"), ("KOZAA.IS", "Koza Madencilik"), ("ARCLK.IS", "Arçelik"),
    ("TOASO.IS", "Tofaş"), ("TAVHL.IS", "TAV Havalimanları"), ("MGROS.IS", "Migros"),
    ("ENJSA.IS", "Enerjisa"), ("PETKM.IS", "Petkim"), ("ALARK.IS", "Alarko Holding"),
    ("TTKOM.IS", "Türk Telekom"), ("CCOLA.IS", "Coca-Cola İçecek"), ("ULKER.IS", "Ülker"),
    ("DOAS.IS", "Doğuş Otomotiv"), ("VESTL.IS", "Vestel"), ("GUBRF.IS", "Gübre Fabrikaları"),
    ("SAHOL.IS", "Sabancı Holding"), ("ANSGR.IS", "Anadolu Sigorta"), ("TURSG.IS", "Türkiye Sigorta"),
    ("AGESA.IS", "Agesa Hayat ve Emeklilik"), ("EKGYO.IS", "Emlak Konut GYO"), ("TKFEN.IS", "Tekfen Holding"),
    ("ISGYO.IS", "İş GYO"), ("KRDMD.IS", "Kardemir (D)"), ("SOKM.IS", "Şok Marketler"),
    ("AEFES.IS", "Anadolu Efes"), ("MPARK.IS", "MLP Sağlık"), ("LOGO.IS", "Logo Yazılım"),
    ("KARSN.IS", "Karsan Otomotiv"), ("OTKAR.IS", "Otokar"), ("CIMSA.IS", "Çimsa"),
    ("OYAKC.IS", "OYAK Çimento"), ("KLKIM.IS", "Kalekim"), ("KONTR.IS", "Kontrolmatik"),
]

BIST_SEKTORU = {
    "THYAO.IS": "Havacılık & Ulaştırma", "PGSUS.IS": "Havacılık & Ulaştırma", "TAVHL.IS": "Havacılık & Ulaştırma",
    "AKBNK.IS": "Bankacılık", "GARAN.IS": "Bankacılık", "YKBNK.IS": "Bankacılık",
    "ISCTR.IS": "Bankacılık", "HALKB.IS": "Bankacılık", "VAKBN.IS": "Bankacılık",
    "KCHOL.IS": "Holding & Sanayi", "SASA.IS": "Holding & Sanayi", "EREGL.IS": "Holding & Sanayi",
    "SISE.IS": "Holding & Sanayi", "ALARK.IS": "Holding & Sanayi", "HEKTS.IS": "Holding & Sanayi",
    "GUBRF.IS": "Holding & Sanayi", "SAHOL.IS": "Holding & Sanayi", "TKFEN.IS": "Holding & Sanayi",
    "KRDMD.IS": "Holding & Sanayi", "CIMSA.IS": "Holding & Sanayi", "OYAKC.IS": "Holding & Sanayi",
    "KLKIM.IS": "Holding & Sanayi", "KONTR.IS": "Holding & Sanayi",
    "FROTO.IS": "Otomotiv", "TOASO.IS": "Otomotiv", "DOAS.IS": "Otomotiv",
    "KARSN.IS": "Otomotiv", "OTKAR.IS": "Otomotiv",
    "BIMAS.IS": "Perakende & Gıda", "MGROS.IS": "Perakende & Gıda", "CCOLA.IS": "Perakende & Gıda",
    "ULKER.IS": "Perakende & Gıda", "SOKM.IS": "Perakende & Gıda", "AEFES.IS": "Perakende & Gıda",
    "ASELS.IS": "Teknoloji & Telekom", "TCELL.IS": "Teknoloji & Telekom", "TTKOM.IS": "Teknoloji & Telekom",
    "LOGO.IS": "Teknoloji & Telekom",
    "TUPRS.IS": "Enerji & Madencilik", "ENJSA.IS": "Enerjisa",
    "KOZAL.IS": "Enerji & Madencilik", "KOZAA.IS": "Enerji & Madencilik", "PETKM.IS": "Petkim",
    "ARCLK.IS": "Dayanıklı Tüketim", "VESTL.IS": "Dayanıklı Tüketim",
    "ANSGR.IS": "Sigorta", "TURSG.IS": "Sigorta", "AGESA.IS": "Sigorta",
    "EKGYO.IS": "İnşaat & GYO", "ISGYO.IS": "İnşaat & GYO",
    "MPARK.IS": "Sağlık",
}

SEKTOR_SIRASI = [
    "Bankacılık", "Holding & Sanayi", "Havacılık & Ulaştırma", "Otomotiv",
    "Perakende & Gıda", "Teknoloji & Telekom", "Enerji & Madencilik",
    "Dayanıklı Tüketim", "Sigorta", "İnşaat & GYO", "Sağlık", "Diğer",
]

def hisse_secim_paneli(key_prefix, varsayilan_semboller):
    ad_sozlugu = dict(BIST_POPULER)
    sektorler = {}
    for sembol, ad in BIST_POPULER:
        sektor = BIST_SEKTORU.get(sembol, "Diğer")
        sektorler.setdefault(sektor, []).append((sembol, ad))

    ust1, ust2 = st.columns([4, 1])
    with ust1:
        arama = st.text_input("🔍 Hisse veya sektör ara", value="", key=f"{key_prefix}_ara",
                              placeholder="örn. banka, ASELS, tüpraş, otomotiv")
    with ust2:
        st.markdown('<div style="height:1.65rem;"></div>', unsafe_allow_html=True)
        if st.button("Temizle", key=f"{key_prefix}_temizle", width="stretch"):
            for sembol, _ in BIST_POPULER:
                st.session_state[f"{key_prefix}_chk_{sembol}"] = False
            st.rerun()

    arama_l = arama.strip().lower()
    sira = [s for s in SEKTOR_SIRASI if s in sektorler] + [s for s in sektorler if s not in SEKTOR_SIRASI]
    for sektor in sira:
        hisseler_bu_sektor = sektorler[sektor]
        if arama_l and arama_l not in sektor.lower():
            hisseler_bu_sektor = [(s, a) for s, a in hisseler_bu_sektor
                                  if arama_l in a.lower() or arama_l in s.lower()]
        if arama_l and not hisseler_bu_sektor:
            continue
        with st.expander(f"{sektor}  ·  {len(hisseler_bu_sektor)} hisse", expanded=bool(arama_l)):
            kolonlar = st.columns(3)
            for i, (sembol, ad) in enumerate(hisseler_bu_sektor):
                with kolonlar[i % 3]:
                    st.checkbox(ad, value=(sembol in varsayilan_semboller), key=f"{key_prefix}_chk_{sembol}")

    secili = [sembol for sembol, _ in BIST_POPULER
              if st.session_state.get(f"{key_prefix}_chk_{sembol}", sembol in varsayilan_semboller)]
    return secili

def hisse_secici(key_prefix, varsayilan="THYAO.IS"):
    mod = st.radio(
        "Hisse Seçim Yöntemi",
        ["Popüler Listeden Seç", "Şirket Adıyla Ara (canlı)", "Manuel Sembol Gir"],
        horizontal=True, key=f"{key_prefix}_mod"
    )
    if mod == "Popüler Listeden Seç":
        etiketler = [f"{ad} ({sembol})" for sembol, ad in BIST_POPULER]
        secim = st.selectbox("Hisse", etiketler, key=f"{key_prefix}_sel")
        return BIST_POPULER[etiketler.index(secim)][0]
    elif mod == "Şirket Adıyla Ara (canlı)":
        sorgu = st.text_input("Şirket adı veya sembol yazın (örn. 'Turkcell', 'Apple', 'Tesla')", key=f"{key_prefix}_q")
        if not sorgu:
            return None
        sonuclar = hisse_ara(sorgu)
        if not sonuclar:
            return None
        etiketler = [f"{ad} ({sembol})" for sembol, ad in sonuclar]
        secim = st.selectbox("Eşleşen Sonuçlar", etiketler, key=f"{key_prefix}_aramasel")
        return sonuclar[etiketler.index(secim)][0]
    else:
        return st.text_input("Sembol (örn. THYAO.IS, AAPL)", value=varsayilan, key=f"{key_prefix}_manuel")

INV_CSS = """
<style>
.inv-ticker { white-space: nowrap; overflow: hidden; background:#0b1220; padding:8px 0; border-radius:6px; margin-bottom:18px; }
.inv-ticker-track { display:inline-block; padding-left:100%; animation: inv-scroll 34s linear infinite; }
.inv-ticker:hover .inv-ticker-track { animation-play-state: paused; }
@keyframes inv-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }
.inv-ticker-item { display:inline-block; padding:0 22px; color:#a9b4c0 !important; font-size:0.78rem; font-weight:600; border-right:1px solid rgba(255,255,255,0.08); font-variant-numeric: tabular-nums; }
.inv-ticker-item b { color:#ffffff !important; font-weight:700; margin:0 6px; }
.inv-up { color:#16c784 !important; font-weight:700; }
.inv-down { color:#f0475c !important; font-weight:700; }
.inv2-wrap { border:1px solid #e4e8ec; border-radius:10px; overflow-x:auto; overflow-y:hidden; background:#ffffff; margin-bottom:6px; box-shadow:0 1px 2px rgba(11,31,51,0.04); }
table.inv2-table { width:100%; border-collapse:collapse; font-size:0.865rem; min-width:900px; }
table.inv2-table thead th { background:#f7f9fb; color:#7a8794 !important; text-transform:uppercase; font-size:0.66rem; letter-spacing:0.5px; font-weight:800; padding:11px 14px; border-bottom:1px solid #e4e8ec; white-space:nowrap; }
table.inv2-table thead th:not(:first-child) { text-align:right; }
table.inv2-table thead th:first-child { text-align:left; }
table.inv2-table td { padding:10px 14px; border-bottom:1px solid #eef1f4; vertical-align:middle; white-space:nowrap; color:#0b1f33 !important; }
table.inv2-table td:not(:first-child) { text-align:right; font-variant-numeric: tabular-nums; }
table.inv2-table tbody tr:hover { background:#f6f9fc; }
.inv2-sector-hdr { display:flex; align-items:center; gap:10px; margin:22px 0 8px 0; }
.inv2-sector-hdr .inv2-sector-bar { width:4px; height:16px; border-radius:2px; background:#0055a5; }
.inv2-sector-hdr .inv2-sector-name { font-size:0.95rem; font-weight:800; color:#0b1f33 !important; }
.inv2-sector-hdr .inv2-sector-count { font-size:0.76rem; color:#8a97a3 !important; font-weight:600; }
</style>
"""

_INV2_RENKLER = ["#0055a5", "#7a4fd6", "#c9820a", "#0f9d58", "#c2364d", "#1b8fa0", "#8a5a00", "#4b5f7a", "#a0479a", "#2f7d3a"]
_SEKTOR_RENKLERI = {
    "Bankacılık": "#0055a5", "Holding & Sanayi": "#4b5f7a", "Havacılık & Ulaştırma": "#1b8fa0",
    "Otomotiv": "#c9820a", "Perakende & Gıda": "#0f9d58", "Teknoloji & Telekom": "#7a4fd6",
    "Enerji & Madencilik": "#8a5a00", "Dayanıklı Tüketim": "#a0479a", "Sigorta": "#c2364d",
    "İnşaat & GYO": "#2f7d3a", "Sağlık": "#0aa3a3", "Diğer": "#5a6b7b",
}

def inv_ticker_goster(satirlar):
    parcalar = []
    for ad, deger, chg, birim, ondalik in satirlar:
        pos = chg >= 0
        ok = "▲" if pos else "▼"
        renk = "inv-up" if pos else "inv-down"
        parcalar.append(f'<span class="inv-ticker-item">{ad} <b>{birim}{tr_sayi(deger, ondalik)}</b> <span class="{renk}">{ok} %{tr_sayi(abs(chg), 2)}</span></span>')
    icerik = "".join(parcalar)
    st.markdown(f'<div class="inv-ticker"><div class="inv-ticker-track">{icerik}{icerik}</div></div>', unsafe_allow_html=True)

def _inv2_spark_svg(seri, pos, w=104, h=34):
    deger = seri.dropna().values
    if len(deger) < 2:
        return '<span style="color:#c3cad1;">—</span>'
    if len(deger) > 60:
        idx = np.linspace(0, len(deger) - 1, 60).astype(int)
        deger = deger[idx]
    mn, mx = float(deger.min()), float(deger.max())
    rng = (mx - mn) or 1.0
    n = len(deger)
    pts = [f"{(i / (n - 1)) * w:.1f},{h - ((v - mn) / rng) * (h - 4) - 2:.1f}" for i, v in enumerate(deger)]
    cizgi = " ".join(pts)
    renk = "#0f9d58" if pos else "#e0323e"
    alan = f"0,{h} " + cizgi + f" {w},{h}"
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="display:block;"><polyline points="{alan}" fill="{renk}20" stroke="none"/><polyline points="{cizgi}" fill="none" stroke="{renk}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'

def _inv2_hacim_metni(v):
    if v is None or pd.isna(v) or v <= 0:
        return "—"
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B".replace(".", ",")
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M".replace(".", ",")
    if v >= 1_000:
        return f"{v / 1_000:.1f}K".replace(".", ",")
    return tr_sayi(v, 0)

def inv2_watchlist_goster(satirlar, kolon_araligi_baslik="Dönem Aralığı"):
    satirlar_html = []
    for i, r in enumerate(satirlar):
        pos = r["degisim_yuzde"] >= 0
        klas_chg = "inv2-chg-pos" if pos else "inv2-chg-neg"
        klas_pct = "inv2-pct-pos" if pos else "inv2-pct-neg"
        ok = "▲" if pos else "▼"
        renk_dot = r.get("renk") or _INV2_RENKLER[i % len(_INV2_RENKLER)]
        aralik = r["maks_"] - r["min_"] if r["maks_"] > r["min_"] else 1.0
        konum = max(0.0, min(100.0, (r["son"] - r["min_"]) / aralik * 100))
        satirlar_html.append(f"""
        <tr>
            <td>
                <div class="inv2-name-wrap">
                    <span class="inv2-dot" style="background:{renk_dot};"></span>
                    <div class="inv2-name-txt"><b>{r['ad']}</b><span>{r.get('sektor', r['sembol'])}</span></div>
                </div>
            </td>
            <td>{tr_sayi(r['son'], 2)}</td>
            <td class="{klas_chg}">{'+' if pos else ''}{tr_sayi(r['degisim'], 2)}</td>
            <td><span class="inv2-pct {klas_pct}">{ok} %{tr_sayi(abs(r['degisim_yuzde']), 2)}</span></td>
            <td>
                <div class="inv2-range">
                    <div class="inv2-range-track"><div class="inv2-range-dot" style="left:{konum:.1f}%;"></div></div>
                    <div class="inv2-range-labels"><span>{tr_sayi(r['min_'], 2)}</span><span>{tr_sayi(r['maks_'], 2)}</span></div>
                </div>
            </td>
            <td>{_inv2_hacim_metni(r.get('hacim'))}</td>
            <td>{'—' if pd.isna(r.get('volatilite', np.nan)) else f"%{tr_sayi(r['volatilite'], 1)}"}</td>
            <td class="inv2-spark-cell">{_inv2_spark_svg(r['seri'], pos)}</td>
        </tr>""")
    basliklar = ["Hisse", "Son", "Değişim", "Değişim %", kolon_araligi_baslik, "Hacim", "Yıllık Vol.", "Grafik"]
    th = "".join(f"<th>{b}</th>" for b in basliklar)
    st.markdown(f'<div class="inv2-wrap"><table class="inv2-table"><thead><tr>{th}</tr></thead><tbody>{"".join(satirlar_html)}</tbody></table></div>', unsafe_allow_html=True)

def inv2_watchlist_sektorel_goster(satirlar, kolon_araligi_baslik="Dönem Aralığı"):
    gruplar = {}
    for r in satirlar:
        gruplar.setdefault(r.get("sektor", "Diğer"), []).append(r)
    sira = [s for s in SEKTOR_SIRASI if s in gruplar] + [s for s in gruplar if s not in SEKTOR_SIRASI]
    for sektor in sira:
        grup = sorted(gruplar[sektor], key=lambda r: r["degisim_yuzde"], reverse=True)
        renk = _SEKTOR_RENKLERI.get(sektor, "#0055a5")
        st.markdown(f'<div class="inv2-sector-hdr"><span class="inv2-sector-bar" style="background:{renk};"></span><span class="inv2-sector-name">{sektor}</span><span class="inv2-sector-count">· {len(grup)} hisse</span></div>', unsafe_allow_html=True)
        for r in grup:
            r["renk"] = renk
        inv2_watchlist_goster(grup, kolon_araligi_baslik=kolon_araligi_baslik)

def piyasa_karti(sutun, enstruman, seri, gun_sayisi):
    seri = seri.dropna()
    with sutun:
        if len(seri) < 2:
            st.markdown(f'<div class="piyasa-karti"><div class="pk-ad">{enstruman["ad"]}</div><div class="pk-deger">—</div><div class="pk-alt">Veri alınamadı</div></div>', unsafe_allow_html=True)
            return
        dilim = seri.iloc[-(gun_sayisi + 1):] if len(seri) > gun_sayisi else seri
        son = dilim.iloc[-1]
        degisim = (son / dilim.iloc[0] - 1) * 100 if dilim.iloc[0] else 0.0
        artis = degisim >= 0
        renk = "#1e6b34" if artis else "#b3261e"
        arka = "rgba(30,107,52,0.10)" if artis else "rgba(179,38,30,0.10)"
        ok = "▲" if artis else "▼"
        st.markdown(f"""
        <div class="piyasa-karti" style="border-left: 4px solid {renk};">
            <div class="pk-ad">{enstruman['ad']}</div>
            <div class="pk-deger">{enstruman['birim']}{tr_sayi(son, enstruman['ondalik'])}</div>
            <div class="pk-rozet" style="color:{renk}; background:{arka};">{ok} %{tr_sayi(abs(degisim), 2)}</div>
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=dilim.index, y=dilim.values, mode="lines",
            line=dict(color=renk, width=2), fill="tozeroy", fillcolor=arka,
            hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f}<extra></extra>"
        ))
        fig.update_layout(height=70, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")

# ---------------------------------------------------------
# GENEL BAKIŞ & SAYFALAR
# ---------------------------------------------------------
def ml_rehberi_sayfasi():
    st.header("Yöntem Notları")
    st.markdown("Model seçimi ve arkasındaki istatistiki gerekçeler.")

def ana_sayfa():
    st.title("Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("---")
    st.markdown("""
    ### 🏛️ Platform Vizyonu
    Bu platform üç segmentten oluşur:
    1. **✅ Doğrulanmış ML Modelleri** — gerçek train/test ayrımı, AUC/F1 metrikleriyle raporlanan eğitilmiş modeller.
    2. **📐 Aktüeryal Yöntemler** — sektörde birebir kullanılan matematiksel formüller (IBNR, Black-Scholes, Solvency II vb.).
    3. **🧪 Kavramsal Vitrin** — fikir/konsept gösterimi amaçlı, henüz gerçek veriyle doğrulanmamış modüller.
    """)

def finansal_bilgi_sayfasi():
    st.header("Canlı Makroekonomi & Piyasalar")

def veri_analizi_sayfasi():
    st.markdown(INV_CSS, unsafe_allow_html=True)
    st.header("Canlı Hisse Korelasyonu")

def kasko_fiyatlama_sayfasi():
    st.header("Kasko Saf Prim Fiyatlama")

def kredi_risk_sayfasi():
    st.header("Kredi Risk Skorlama")

def churn_sayfasi():
    st.header("Churn Tahmini")

def fraud_sayfasi():
    st.header("Fraud Uyarı Sistemi")

def ibnr_sayfasi():
    st.header("IBNR Rezerv")

def hayat_sigortasi_sayfasi():
    st.header("Hayat Sigortası")

def hasar_frekans_sayfasi():
    st.header("Hasar Frekans")

def monte_carlo_sayfasi():
    st.header("Monte Carlo")

def solvency_sayfasi():
    st.header("Solvency II")

def black_scholes_sayfasi():
    st.header("Black-Scholes")

def kredi_var_sayfasi():
    st.header("Kredi VaR")

def reasurans_sayfasi():
    st.header("Reasürans")

def stres_testi_sayfasi():
    st.header("Stres Testi")

def katilim_fon_sayfasi():
    st.header("Katılım Bankacılığı")

def alm_nakit_sayfasi():
    st.header("ALM Nakit")

def alm_durasyon_sayfasi():
    st.header("ALM Durasyon")

def markowitz_sayfasi():
    st.header("Markowitz Optimizasyonu")

def varlik_dagilimi_sayfasi():
    st.header("Varlık Dağılımı")

def benchmark_sayfasi():
    st.header("Benchmark")

def telematik_sayfasi():
    st.header("Telematik")

def clv_sayfasi():
    st.header("CLV")

# --- VERİTABANI & GEÇMİŞ SAYFASI (Kişisel Geçmiş + Şifreli Ziyaretçi Takip Paneli) ---
def veritabani_sayfasi():
    st.header("📂 Veritabanı & Geçmiş Paneli")
    aktif = st.session_state.get("aktif_kullanici", None)
    
    if aktif is None:
        st.info("🔒 Kendi geçmiş simülasyonlarınızı görmek için lütfen sol menüden giriş yapın.")
    else:
        st.subheader(f"✨ {aktif} - Kişisel Simülasyon Geçmişiniz")
        df_kisi = kullanici_gecmisi_getir()
        if not df_kisi.empty:
            st.dataframe(df_kisi, width='stretch')
        else:
            st.markdown("Henüz kayıtlı bir simülasyonunuz yok. Modüllerde işlem yaptıkça burada listelenecektir.")
            
    st.markdown("---")
    st.subheader("🕵️ Sistem Sahibi / Ziyaretçi Takip Paneli")
    yonetici_sifresi = st.secrets.get("YONETICI_SIFRE", "sultan123")
    girilen_sifre = st.text_input("Yönetici Şifresi (Sadece sizin erişiminiz için)", type="password", key="admin_sifre_giris")
    
    if girilen_sifre == yonetici_sifresi:
        st.success("🔓 Yönetici yetkisi doğrulandı. Ziyaretçi ve sistem logları yükleniyor...")
        df_sim, df_log, df_user = tum_loglari_getir()
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Üye Sayısı", len(df_user))
        col2.metric("Toplam Simülasyon", len(df_sim))
        col3.metric("Toplam Ziyaret/İşlem Logu", len(df_log))
        
        st.markdown("#### 👥 Kayıtlı Üyeler")
        st.dataframe(df_user, width='stretch')
        st.markdown("#### 📈 Ziyaret ve Etkileşim Logları")
        st.dataframe(df_log, width='stretch')
        st.markdown("#### 🌐 Tüm Kullanıcıların Simülasyon Geçmişi")
        st.dataframe(df_sim, width='stretch')
    elif girilen_sifre != "":
        st.error("Hatalı yönetici şifresi!")

# --- HAKKINDA & İLETİŞİM SAYFASI (Foto Yükleme, Kadın Avatar, Gerçek İkonlar, CV Yükleme) ---
def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    col1, col2 = st.columns([1, 3])
    with col1:
        yuklenen_foto = st.file_uploader("Profil Fotoğrafı Yükle", type=["png", "jpg", "jpeg"], key="profil_foto_up")
        if yuklenen_foto is not None:
            st.image(yuklenen_foto, width=180, caption="Sultan Kuş")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2922/2922561.png", width=180, caption="Örnek Profil (Kadın)")
            
    with col2:
        st.markdown("""
        Merhaba! Ben **Sultan Kuş**.
        Matematik altyapımla finans, sigorta ve risk analitiği alanlarına yönelik veri bilimi çözümleri geliştiriyorum.
        Hedefim; finans, sigorta ve **katılım bankacılığı** alanlarında, matematiksel titizliği veri bilimiyle
        birleştiren bir rol.
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 15px; line-height: 2.2;">
            <i class="fas fa-envelope" style="color: #ea4335; font-size: 18px; width: 25px;"></i> <b>Email:</b> <a href="mailto:kussultannn34@gmail.com" style="text-decoration: none; color: #0055a5;">kussultannn34@gmail.com</a><br>
            <i class="fab fa-linkedin" style="color: #0077b5; font-size: 18px; width: 25px;"></i> <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/sultan-kuş/" target="_blank" style="text-decoration: none; color: #0055a5;">linkedin.com/in/sultan-kuş</a><br>
            <i class="fab fa-github" style="color: #24292e; font-size: 18px; width: 25px;"></i> <b>GitHub:</b> <a href="https://github.com/SultanKus" target="_blank" style="text-decoration: none; color: #0055a5;">github.com/SultanKus</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Yetkinlik Haritası")
    st.markdown("""
| Alan | Yetkinlikler |
|---|---|
| **Matematik & İstatistik** | Olasılık Teorisi, Stokastik Süreçler, Doğrusal Cebir, Optimizasyon (Karesel Programlama), İstatistiksel Modelleme |
| **Aktüerya & Risk** | IBNR (Chain Ladder), Solvency II, VaR, Poisson/GLM Frekans Modelleme, Aktüerlik Sınavına Hazırlık |
| **Finansal Mühendislik** | Black-Scholes, Markowitz Portföy Optimizasyonu, Reasürans, Katılım Bankacılığı Ürünleri (Murabaha, Sukuk) |
| **Makine Öğrenmesi** | Lojistik Regresyon, Poisson Regresyonu, Model Doğrulama (AUC, F1), Scikit-learn |
| **Yazılım & Araçlar** | Python, SQL, Streamlit, Plotly, Git/GitHub, Excel (İleri Düzey), SAP |
""")
    st.markdown("---")
    st.subheader("📄 Özgeçmiş (CV)")
    
    yuklenen_cv = st.file_uploader("Gerçek Özgeçmişini Yükle (PDF formatında)", type=["pdf"], key="cv_upload")
    
    if yuklenen_cv is not None:
        st.success("✅ CV'niz başarıyla yüklendi!")
        st.download_button(
            label="📥 Yüklenen Özgeçmişimi İndir (PDF)",
            data=yuklenen_cv,
            file_name="Sultan_Kus_CV.pdf",
            mime="application/pdf"
        )
    else:
        try:
            with open("Sultan_Kus_CV.pdf", "rb") as pdf_file:
                st.download_button(label="📥 Özgeçmişimi İndir (PDF)", data=pdf_file, file_name="Sultan_Kus_CV.pdf", mime="application/pdf")
        except FileNotFoundError:
            st.info("💡 Kendi gerçek PDF CV'nizi yukarıdaki alandan yükleyebilirsiniz. Yüklediğinizde indirme butonu aktif olacaktır.")

# ---------------------------------------------------------
# NAVİGASYON
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış & Canlı Piyasa": [
        st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠"),
        st.Page(ml_rehberi_sayfasi, title="ML & Aktüerya Rehberi", icon="🎓"),
        st.Page(finansal_bilgi_sayfasi, title="Makroekonomi & Piyasalar", icon="🌍"),
        st.Page(veri_analizi_sayfasi, title="Canlı Hisse Korelasyon (EDA)", icon="📈"),
    ],
    "✅ Doğrulanmış ML Modelleri": [
        st.Page(kasko_fiyatlama_sayfasi, title="Kasko Saf Prim (Poisson GLM)", icon="🚗"),
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🏦"),
        st.Page(churn_sayfasi, title="Churn Tahmini", icon="🚪"),
        st.Page(fraud_sayfasi, title="Fraud Uyarı Sistemi", icon="🕵️"),
    ],
    "📊 Kantitatif Finans (Canlı Optimizasyon)": [
        st.Page(markowitz_sayfasi, title="Markowitz Portföy Optimizasyonu", icon="🥧"),
    ],
    "🕌 Katılım Bankacılığı": [
        st.Page(katilim_fon_sayfasi, title="Murabaha & Sukuk Araçları", icon="🕌"),
    ],
    "📐 Aktüeryal Yöntemler": [
        st.Page(ibnr_sayfasi, title="IBNR Muallak Hasar", icon="📐"),
        st.Page(hayat_sigortasi_sayfasi, title="Hayat Sigortası Fiyatlama", icon="👨‍🦳"),
        st.Page(hasar_frekans_sayfasi, title="Hasar Frekans & Risk", icon="📉"),
        st.Page(monte_carlo_sayfasi, title="Monte Carlo Simülatörü", icon="🎲"),
        st.Page(solvency_sayfasi, title="Solvency II", icon="🏛️"),
        st.Page(black_scholes_sayfasi, title="Black-Scholes", icon="📈"),
        st.Page(kredi_var_sayfasi, title="Kredi Portföyü VaR", icon="📉"),
        st.Page(reasurans_sayfasi, title="Dinamik Reasürans", icon="🌐"),
    ],
    "🧪 Kavramsal Vitrin (Demo)": [
        st.Page(stres_testi_sayfasi, title="Aktüeryal Stres Testi", icon="⚡"),
        st.Page(alm_nakit_sayfasi, title="ALM Nakit Eşitleme", icon="🔄"),
        st.Page(alm_durasyon_sayfasi, title="ALM Durasyon", icon="⚖️"),
        st.Page(varlik_dagilimi_sayfasi, title="Varlık Dağılımı", icon="📊"),
        st.Page(benchmark_sayfasi, title="Piyasa Kıyaslama", icon="📈"),
        st.Page(telematik_sayfasi, title="Telematik Risk Skorlama", icon="🚗"),
        st.Page(clv_sayfasi, title="Müşteri Yaşam Değeri", icon="💎"),
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻"),
    ]
})

pg.run()
