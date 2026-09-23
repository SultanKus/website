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
# VERİTABANI (SQLite)
# ---------------------------------------------------------
def veritabani_olustur():
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS simulasyonlar (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tarih TEXT, modul_adi TEXT, girdi_detayi TEXT, sonuc_deger TEXT)
    ''')
    conn.commit()
    conn.close()

veritabani_olustur()

def kayit_ekle(modul_adi, girdi_detayi, sonuc_deger):
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    c = conn.cursor()
    tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO simulasyonlar (tarih, modul_adi, girdi_detayi, sonuc_deger) VALUES (?, ?, ?, ?)",
              (tarih, modul_adi, girdi_detayi, sonuc_deger))
    conn.commit()
    conn.close()

def gecmisi_getir():
    conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
    df = pd.read_sql("SELECT * FROM simulasyonlar ORDER BY id DESC", conn)
    conn.close()
    return df

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
    """
    Hasar frekansı için Poisson GLM (endüstri standardı yöntem).
    Exposure, sample_weight olarak offset görevi görür: model,
    E[ClaimNb / Exposure] = exp(X . beta) ilişkisini öğrenir.
    """
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
# SENTETİK (AMA MANTIKSAL İLİŞKİLİ) EĞİTİM VERİSİ ÜRETİCİLERİ
# Not: Bu üreticiler rastgele etiket atamaz; hedef değişken,
# bilinen risk faktörlerinin lojistik bir fonksiyonu olarak kurulur.
# Gerçek şirket verisi yerine geçmez — amaç, doğru ML metodolojisini
# (train/test split + AUC/F1 doğrulaması) dürüstçe göstermektir.
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
    """Önce gerçek OpenML German Credit veri setini dener; olmazsa sentetik veriye düşer (graceful degradation)."""
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
# GERÇEK PORTFÖY OPTİMİZASYONU (Markowitz / Karesel Programlama)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def markowitz_veri_getir(hisseler, periyot="2y"):
    """Canlı fiyat verisinden yıllıklandırılmış beklenen getiri ve kovaryans matrisini hesaplar."""
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
    """Belirli bir hedef getiriyi sağlayan minimum varyanslı portföyü SLSQP (Sequential Least Squares
    Quadratic Programming) ile çözer — KKT koşullarını sayısal olarak sağlayan gerçek bir optimizasyondur."""
    n = len(ort_getiri)
    kisitlar = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, ort_getiri) - hedef_getiri},
    ]
    sinirlar = tuple((0.0, 1.0) for _ in range(n))  # açığa satış yok
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
# CANLI PİYASA VERİSİ (yfinance) — Makroekonomi & Piyasalar sayfası için ortak yardımcılar
# ---------------------------------------------------------
def tr_sayi(x, ondalik=2):
    """1234567.89 -> '1.234.567,89' (Türkçe sayı biçimi)."""
    try:
        s = f"{float(x):,.{ondalik}f}"
    except (TypeError, ValueError):
        return "—"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


def _degisim_yuzde(seri, gun):
    """`gun` kadar geri gidip yüzde değişim hesaplar; yetersiz veri varsa None."""
    seri = seri.dropna()
    if len(seri) < 2:
        return None
    konum = max(0, len(seri) - 1 - gun)
    onceki = seri.iloc[konum]
    if onceki == 0:
        return None
    return (seri.iloc[-1] / onceki - 1) * 100


def _ybb_degisim(seri):
    """Yılbaşından bugüne değişim."""
    seri = seri.dropna()
    if seri.empty:
        return None
    bu_yil = seri[seri.index.year == seri.index[-1].year]
    if len(bu_yil) < 2 or bu_yil.iloc[0] == 0:
        return None
    return (bu_yil.iloc[-1] / bu_yil.iloc[0] - 1) * 100


def _rsi(seri, periyot=14):
    """Wilder yöntemiyle Relative Strength Index."""
    delta = seri.diff()
    kazanc = delta.clip(lower=0).ewm(alpha=1 / periyot, adjust=False).mean()
    kayip = (-delta.clip(upper=0)).ewm(alpha=1 / periyot, adjust=False).mean()
    rs = kazanc / kayip.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


ONS_GRAM = 31.1034768  # 1 troy ons = 31.1034768 gram

# Panoda ve tabloda gösterilecek enstrümanlar
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
    """
    Tüm pano enstrümanlarının 1 yıllık kapanış serilerini TEK bir yfinance
    çağrısıyla indirir (8 ayrı istek yerine 1 istek — sayfa çok daha hızlı açılır).
    Gram altın, ons altın ve USD/TRY serilerinden türetilir:
        Gram Altın (TL) = Ons Altın ($) / 31.1035 × USD/TRY
    """
    ham = yf.download(PANO_SEMBOLLERI, period="1y", progress=False,
                      auto_adjust=True, threads=True)
    if ham is None or ham.empty:
        return pd.DataFrame()

    if isinstance(ham.columns, pd.MultiIndex):
        kapanis = ham["Close"].copy()
    else:  # tek sembol dönerse
        kapanis = ham[["Close"]].copy()
        kapanis.columns = PANO_SEMBOLLERI[:1]

    # Farklı borsalar farklı saat dilimi döndürebiliyor; indeksi normalize ediyoruz.
    try:
        if getattr(kapanis.index, "tz", None) is not None:
            kapanis.index = kapanis.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    kapanis.index = pd.to_datetime(kapanis.index).normalize()

    # BIST tatilde, kripto 7/24 → boşlukları son bilinen fiyatla dolduruyoruz.
    kapanis = kapanis.ffill().dropna(how="all")

    if {"GC=F", "TRY=X"}.issubset(kapanis.columns):
        kapanis["GRAM_ALTIN"] = kapanis["GC=F"] / ONS_GRAM * kapanis["TRY=X"]

    return kapanis


@st.cache_data(ttl=3600, show_spinner=False)
def canli_piyasa_verisi_getir(sembol, periyot="1y"):
    return yf.Ticker(sembol).history(period=periyot)


@st.cache_data(ttl=1800, show_spinner=False)
def hisse_ara(sorgu, max_sonuc=8):
    """Yahoo Finance canlı arama servisiyle şirket adı/sembol eşleştirir (gerçek zamanlı otomatik tamamlama)."""
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
    "TUPRS.IS": "Enerji & Madencilik", "ENJSA.IS": "Enerji & Madencilik",
    "KOZAL.IS": "Enerji & Madencilik", "KOZAA.IS": "Enerji & Madencilik", "PETKM.IS": "Enerji & Madencilik",
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
    """Sektöre göre gruplanmış, aranabilir, açılır kutucuklu (checkbox) hisse seçim paneli.
    Investing.com/TradingView'daki 'watchlist oluştur' deneyimine benzer;
    kırmızı/mavi renk sorununa yol açan multiselect etiketleri yerine geçer."""
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

    # Görünürde olmasa bile (arama filtreliyken) tüm işaretli hisseleri topla
    secili = [sembol for sembol, _ in BIST_POPULER
              if st.session_state.get(f"{key_prefix}_chk_{sembol}", sembol in varsayilan_semboller)]

    if secili:
        pilller = "".join(
            f'<span style="display:inline-block; background:#0055a5; color:#fff !important; '
            f'padding:4px 13px; border-radius:14px; font-size:0.82rem; font-weight:600; '
            f'margin:3px 5px 3px 0;">{ad_sozlugu.get(s, s)}</span>'
            for s in secili
        )
        st.markdown(
            f'<div style="margin-top:6px;"><span style="font-size:0.82rem; color:#5a6b7b !important; '
            f'font-weight:600; margin-right:6px;">Seçili ({len(secili)}):</span>{pilller}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Hiç hisse seçilmedi — yukarıdaki sektörlerden en az iki hisse işaretleyin.")

    return secili


def hisse_secici(key_prefix, varsayilan="THYAO.IS"):
    """Kullanıcının popüler listeden, canlı aramadan veya manuel girişten hisse seçmesini sağlar."""
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
            st.caption("Aramak için bir şirket adı veya sembol girin.")
            return None
        sonuclar = hisse_ara(sorgu)
        if not sonuclar:
            st.warning("Sonuç bulunamadı veya Yahoo Finance arama servisine şu an ulaşılamıyor.")
            return None
        etiketler = [f"{ad} ({sembol})" for sembol, ad in sonuclar]
        secim = st.selectbox("Eşleşen Sonuçlar", etiketler, key=f"{key_prefix}_aramasel")
        return sonuclar[etiketler.index(secim)][0]
    else:
        return st.text_input("Sembol (örn. THYAO.IS, AAPL)", value=varsayilan, key=f"{key_prefix}_manuel")


# ---------------------------------------------------------
# INVESTING.COM TARZI İZLEME LİSTESİ (watchlist) BİLEŞENLERİ
# ---------------------------------------------------------
INV_CSS = """
<style>
/* --- üstteki ince akan şerit --- */
.inv-ticker { white-space: nowrap; overflow: hidden; background:#0b1220;
              padding:8px 0; border-radius:6px; margin-bottom:18px; }
.inv-ticker-track { display:inline-block; padding-left:100%;
                     animation: inv-scroll 34s linear infinite; }
.inv-ticker:hover .inv-ticker-track { animation-play-state: paused; }
@keyframes inv-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }
.inv-ticker-item { display:inline-block; padding:0 22px; color:#a9b4c0 !important;
                    font-size:0.78rem; font-weight:600; border-right:1px solid rgba(255,255,255,0.08);
                    font-variant-numeric: tabular-nums; }
.inv-ticker-item b { color:#ffffff !important; font-weight:700; margin:0 6px; }
.inv-up { color:#16c784 !important; font-weight:700; }
.inv-down { color:#f0475c !important; font-weight:700; }

/* --- büyük izleme listesi tablosu --- */
.inv2-wrap { border:1px solid #e4e8ec; border-radius:10px; overflow-x:auto; overflow-y:hidden;
             background:#ffffff; margin-bottom:6px; box-shadow:0 1px 2px rgba(11,31,51,0.04); }
table.inv2-table { width:100%; border-collapse:collapse; font-size:0.865rem; min-width:900px; }
table.inv2-table thead th { background:#f7f9fb; color:#7a8794 !important; text-transform:uppercase;
                             font-size:0.66rem; letter-spacing:0.5px; font-weight:800;
                             padding:11px 14px; border-bottom:1px solid #e4e8ec; white-space:nowrap; }
table.inv2-table thead th:not(:first-child) { text-align:right; }
table.inv2-table thead th:first-child { text-align:left; }
table.inv2-table td { padding:10px 14px; border-bottom:1px solid #eef1f4; vertical-align:middle;
                       white-space:nowrap; color:#0b1f33 !important; }
table.inv2-table td:not(:first-child) { text-align:right; font-variant-numeric: tabular-nums; }
table.inv2-table tbody tr:hover { background:#f6f9fc; }
table.inv2-table tbody tr:last-child td { border-bottom:none; }
.inv2-name-wrap { display:flex; align-items:center; gap:10px; }
.inv2-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.inv2-name-txt { display:flex; flex-direction:column; line-height:1.25; }
.inv2-name-txt b { font-size:0.885rem; color:#0b1f33 !important; font-weight:700; }
.inv2-name-txt span { font-size:0.70rem; color:#98a3af !important; font-weight:600; }
.inv2-chg-pos { color:#0f9d58 !important; font-weight:700; }
.inv2-chg-neg { color:#e0323e !important; font-weight:700; }
.inv2-pct { display:inline-block; padding:3px 10px; border-radius:5px; font-weight:700;
            min-width:62px; font-size:0.82rem; }
.inv2-pct-pos { background:rgba(15,157,88,0.10); color:#0f9d58 !important; }
.inv2-pct-neg { background:rgba(224,50,62,0.10); color:#e0323e !important; }
.inv2-range { width:104px; margin-left:auto; }
.inv2-range-track { position:relative; width:100%; height:4px; background:#e7ebee; border-radius:2px; }
.inv2-range-dot { position:absolute; top:50%; width:8px; height:8px; border-radius:50%;
                   background:#0055a5; border:2px solid #fff; box-shadow:0 0 0 1px #cfd8e0;
                   transform:translate(-50%, -50%); }
.inv2-range-labels { display:flex; justify-content:space-between; font-size:0.66rem;
                      color:#a4aeb8 !important; margin-top:3px; font-weight:600; }
.inv2-spark-cell { line-height:0; }
.inv2-hdr-row { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:10px; }
.inv2-hdr-row .inv2-count { font-size:0.8rem; color:#7a8794 !important; font-weight:600; }
</style>
"""

_INV2_RENKLER = ["#0055a5", "#7a4fd6", "#c9820a", "#0f9d58", "#c2364d",
                  "#1b8fa0", "#8a5a00", "#4b5f7a", "#a0479a", "#2f7d3a"]


def inv_ticker_goster(satirlar):
    """satirlar: [(ad, deger, yuzde_degisim, birim, ondalik), ...] — üstteki ince akan şerit."""
    parcalar = []
    for ad, deger, chg, birim, ondalik in satirlar:
        pos = chg >= 0
        ok = "▲" if pos else "▼"
        renk = "inv-up" if pos else "inv-down"
        parcalar.append(
            f'<span class="inv-ticker-item">{ad} <b>{birim}{tr_sayi(deger, ondalik)}</b> '
            f'<span class="{renk}">{ok} %{tr_sayi(abs(chg), 2)}</span></span>'
        )
    icerik = "".join(parcalar)
    st.markdown(f'<div class="inv-ticker"><div class="inv-ticker-track">{icerik}{icerik}</div></div>',
                unsafe_allow_html=True)


def _inv2_spark_svg(seri, pos, w=104, h=34):
    """Küçük, hafif bir alan (sparkline) grafiği — SVG olarak, plotly'siz. Investing.com'daki
    mini grafik sütununun karşılığı; 60'tan fazla noktayı seyrekleştirir ki HTML şişmesin."""
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
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
            f'<polyline points="{alan}" fill="{renk}20" stroke="none"/>'
            f'<polyline points="{cizgi}" fill="none" stroke="{renk}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


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
    """satirlar: her biri dict — {ad, sembol, sektor, son, degisim, degisim_yuzde,
    min_, maks_, hacim, seri (pd.Series), renk}. Investing.com ana sayfasındaki
    büyük izleme listesi tablosunun karşılığı."""
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
    st.markdown(
        f'<div class="inv2-wrap"><table class="inv2-table"><thead><tr>{th}</tr></thead>'
        f'<tbody>{"".join(satirlar_html)}</tbody></table></div>',
        unsafe_allow_html=True
    )

def piyasa_karti(sutun, enstruman, seri, gun_sayisi):
    """Değer + yüzde rozeti + mini sparkline gösteren tek bir piyasa kartı (Makroekonomi sayfası için)."""
    seri = seri.dropna()
    with sutun:
        if len(seri) < 2:
            st.markdown(
                f'<div class="piyasa-karti"><div class="pk-ad">{enstruman["ad"]}</div>'
                f'<div class="pk-deger">—</div>'
                f'<div class="pk-alt">Veri alınamadı</div></div>',
                unsafe_allow_html=True
            )
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
            <div class="pk-rozet" style="color:{renk}; background:{arka};">
                {ok} %{tr_sayi(abs(degisim), 2)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Scatter(
            x=dilim.index, y=dilim.values, mode="lines",
            line=dict(color=renk, width=2),
            fill="tozeroy", fillcolor=arka,
            hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f}<extra></extra>"
        ))
        fig.update_layout(
            height=70, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False, range=[dilim.min() * 0.995, dilim.max() * 1.005]),
            showlegend=False
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")

# ---------------------------------------------------------
# GENEL BAKIŞ & CANLI PİYASA SAYFALARI
# ---------------------------------------------------------
def ml_rehberi_sayfasi():
    st.header("Yöntem Notları")
    st.markdown("""
Bu sayfa bir ders anlatımı değil; sitedeki modülleri kurarken hangi yöntemi neden seçtiğimin
notları. Mülakatta "burada neden lojistik regresyon kullandın?" diye sorulduğunda vereceğim
cevaplar da burada.
""")

    st.subheader("Önce soru, sonra model")
    st.markdown("""
Modeli veri değil, sorunun kendisi belirliyor. "Bu sürücü yılda kaç hasar yapar?" dediğimde
cevap bir sayı; "bu müşteri krediyi öder mi?" dediğimde cevap iki kategoriden biri. Birincisi
regresyon, ikincisi sınıflandırma. Sitede Kasko modülü birinci gruba, Kredi Risk / Churn /
Fraud modülleri ikinci gruba giriyor.

Ayrım basit görünüyor ama pratikte karıştırılıyor: "temerrüt olasılığı" sürekli bir sayı
üretir, yine de problem sınıflandırmadır — çünkü gerçek hayatta gözlemlediğin etiket 0 veya 1.
""")

    st.subheader("Neden düz regresyon değil de GLM?")
    st.markdown("""
Klasik doğrusal regresyon (OLS) iki şey varsayar: hata terimi normal dağılır ve tahmin
istediği değeri alabilir. Sigorta verisinde ikisi de tutmuyor.

Kasko modülündeki `ClaimNb` sütununu düşünün: poliçelerin büyük çoğunluğu 0 hasar, bir kısmı 1,
çok azı 2 ve üzeri. Dağılım sıfıra yığılmış ve sağa çarpık. Bu veriye OLS uydurursanız model
bazı segmentler için **negatif hasar sayısı** tahmin eder. "Bu sürücü yılda -0.04 hasar yapar"
cümlesinin bir karşılığı yok.

GLM'in çözümü, tahmini doğru aralığa sıkıştıran bir link fonksiyonu kullanmak:
""")
    st.latex(r"g\big(E[Y \mid X]\big) = X\beta")
    st.markdown("""
Poisson GLM'de `g` logaritma olduğu için tahmin `exp(Xβ)` şeklinde çıkar ve hiçbir zaman
negatif olamaz. Lojistik regresyonda link logit'tir, çıktı 0-1 arasına hapsolur. Gamma GLM ise
pozitif ve çarpık büyüklükler için kullanılır — hasar tutarı gibi.

Aktüeryada standart kurulum, hasar sayısı için Poisson, hasar tutarı için Gamma modeli kurup
ikisini çarpmaktır. Kasko modülünde frekans kısmı gerçek veriyle eğitilmiş durumda; tutar
tarafı için elimdeki veri setinde sütun olmadığından o kısım varsayım olarak giriliyor ve bunu
sayfada açıkça yazdım.
""")

    st.subheader("Eğitim ve test verisini neden ayırıyorum")
    st.markdown("""
Bir model, gördüğü veriyi ezberleyip henüz görmediği veride çökebilir. Bunu anlamanın tek yolu,
modele hiç göstermediğiniz bir parça veriyi kenara ayırıp performansı orada ölçmek.

Sitedeki her eğitilmiş modelde veri %75 eğitim / %25 test olarak bölünüyor ve rozette gördüğünüz
metrik **test** kümesinden geliyor, eğitim kümesinden değil. Sınıflandırma modellerinde ayrıca
`stratify` kullanıyorum; aksi halde azınlık sınıfı test kümesine dengesiz dağılabiliyor ve skor
gürültülü çıkıyor.
""")

    st.subheader("Doğruluk (accuracy) neden yanıltıcı")
    st.markdown("""
Fraud verisinde vakaların diyelim %2'si gerçek suistimal. Hiçbir şey öğrenmeyen, her başvuruya
"temiz" diyen bir model %98 doğruluk alır. Rakam muhteşem görünür, model tamamen işe yaramazdır.

Bu yüzden dengesiz veride iki metriğe bakıyorum:

**AUC**, modelin rastgele seçilmiş bir riskli ve bir risksiz kaydı doğru sıralama olasılığıdır.
0.50 yazı tura demek, 1.00 kusursuz ayrım. Eşik değerinden (0.5 vb.) bağımsız çalıştığı için
modelin sıralama gücünü ölçer.

**F1**, kaçırdığınız gerçek vakalar (recall) ile boşuna alarm verdiğiniz temiz vakalar
(precision) arasındaki dengeyi tek sayıya indirir. İş tarafında bu dengeyi seçmek teknik değil
ticari bir karardır: bir fraud incelemesinin maliyeti, kaçan bir dolandırıcılığın maliyetinden
ucuzsa recall'u yükseltmek mantıklıdır.

Sayım verisinde (Kasko) bunların ikisi de anlamsız; orada **Poisson deviance** raporluyorum,
düşük olması iyi.
""")

    st.subheader("Bazı modüllerde sentetik veri var, sebebi şu")
    st.markdown("""
Fraud, churn ve müşteri davranışı verisi şirket içi ve gizli. Halka açık olanlar ya çok eski,
ya başka bir ülkenin pazarına ait, ya da hedef değişkeni bu modüllerin anlattığı şeyle
örtüşmüyor. İki seçeneğim vardı: modülü hiç yapmamak, ya da veriyi kendim üretip bunu açıkça
söylemek. İkincisini seçtim.

Ürettiğim veri rastgele etiket atamıyor. Hedef değişken, bilinen risk faktörlerinin lojistik
bir fonksiyonu olarak kuruluyor — örneğin churn'de müşterilik süresi uzadıkça terk olasılığı
düşüyor, şikayet sayısı arttıkça yükseliyor. Yani model gerçek bir sinyali öğreniyor, ama o
sinyali ben koydum. Bu şu demek: metrikler metodolojinin doğru kurulduğunu gösterir, modelin
gerçek dünyada bu performansı vereceğini **göstermez**.

Kredi Risk modülü bunun istisnası: önce gerçek bir açık veri setine (OpenML German Credit)
bağlanmayı deniyor, erişim olmazsa aynı şeffaflıkla sentetiğe düşüyor ve rozet hangisinin
kullanıldığını yazıyor.
""")

    st.subheader("Hangi sayfa ne kadar 'gerçek'")
    st.markdown("""
Sitede üç tür içerik var ve her sayfanın üstündeki rozet hangisi olduğunu söylüyor:

Gerçek veriyle eğitilmiş ve test kümesinde doğrulanmış modeller (Kasko GLM, Kredi Risk),
sektörde birebir kullanılan deterministik aktüeryal formüller (IBNR, Black-Scholes,
Solvency II — bunlar model değil, hesap), ve metodoloji göstermek için kurulmuş kavramsal
modüller (Telematik, CLV, stres testi).

Bu ayrımı yapmasam sayfa sayısı daha etkileyici görünürdü. Ama bir kredi risk modelini bir CLV
formülüyle aynı vitrine koymak, ikisini de değersizleştiriyor.
""")

def ana_sayfa():
    st.title("Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("---")
    st.markdown("""
    ### 🏛️ Platform Vizyonu
    Bu platform üç segmentten oluşur:
    1. **✅ Doğrulanmış ML Modelleri** — gerçek train/test ayrımı, AUC/F1 metrikleriyle raporlanan eğitilmiş modeller.
    2. **📐 Aktüeryal Yöntemler** — sektörde birebir kullanılan matematiksel formüller (IBNR, Black-Scholes, Solvency II vb.).
    3. **🧪 Kavramsal Vitrin** — fikir/konsept gösterimi amaçlı, henüz gerçek veriyle doğrulanmamış modüller.

    Her sayfanın üstünde hangi segmentte olduğunu gösteren bir rozet bulunur.
    """)

def finansal_bilgi_sayfasi():
    st.markdown("""
    <style>
    .piyasa-karti { background:#ffffff; border-radius:10px; padding:14px 16px 6px 16px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.06); margin-bottom:-10px; }
    .pk-ad     { font-size:0.80rem; text-transform:uppercase; letter-spacing:0.6px;
                 color:#5a6b7b !important; font-weight:600; }
    .pk-deger  { font-size:1.65rem; font-weight:700; color:#0b1f33 !important; line-height:1.4; }
    .pk-rozet  { display:inline-block; padding:2px 8px; border-radius:6px;
                 font-size:0.82rem; font-weight:700; }
    .pk-alt    { font-size:0.85rem; color:#8a97a3 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.header("🌍 Canlı Makroekonomi & Küresel Piyasalar")
    st.markdown(
        '<div class="model-badge">✅ Tüm rakamlar Yahoo Finance üzerinden canlı çekilir '
        '(15 dk önbellekli). Gram altın, ons altın ve USD/TRY kurundan türetilir.</div>',
        unsafe_allow_html=True
    )

    egitim_notu("""
**Gram altın neden ayrı bir sembol değil?** Çünkü piyasada gram altının bağımsız bir fiyatı yoktur —
türetilmiş bir büyüklüktür. Dünya piyasasında altın **ons** cinsinden ve **dolarla** işlem görür
(`XAU/USD`, vadeli kontratı `GC=F`). Türkiye'de gördüğünüz gram fiyatı şu dönüşümün sonucudur:

`Gram Altın (TL) = Ons Altın ($) ÷ 31,1035 × USD/TRY`

Bunun pratik sonucu şu: **gram altın iki ayrı riske aynı anda maruzdur** — ons altın fiyatı ve kur.
Ons altın sabit kalsa bile dolar yükselirse gram altın yükselir. Bu yüzden "altın aldım, riskten korundum"
cümlesi Türkiye'de tam doğru değildir; aslında kısmen döviz pozisyonu taşınmaktadır. Aşağıdaki tabloda
ons altın ile gram altının farklı dönemlerdeki getirilerini yan yana karşılaştırarak bu ayrışmayı
somut olarak görebilirsiniz.

**Yüzde değişimler neden dönem seçimine bağlı?** Kartlardaki oran, seçtiğiniz periyodun **ilk gününe**
göre hesaplanır. Finans sitelerinde varsayılan olarak günlük değişim gösterilir; burada periyodu
serbest bıraktım çünkü tek günlük hareket çoğu zaman gürültüdür, trendi 1 ay/3 ay ölçeğinde görmek
daha anlamlıdır.
""", baslik="📚 Gram altın nasıl hesaplanıyor? (Yöntem notu)")

    periyot_etiket = st.radio(
        "Değişim Periyodu", list(PERIYOT_SECENEKLERI.keys()),
        index=1, horizontal=True, key="pano_periyot"
    )
    gun_sayisi = PERIYOT_SECENEKLERI[periyot_etiket]

    with st.spinner("Canlı piyasa verileri çekiliyor..."):
        pano = pano_verisi_getir()

    if pano.empty:
        st.warning("Yahoo Finance verileri şu an çekilemiyor. İnternet bağlantınızı kontrol edin.")
        return

    son_tarih = pano.index[-1].strftime("%d.%m.%Y")
    st.caption(f"Son veri tarihi: {son_tarih} · Kaynak: Yahoo Finance")

    # --- Kart paneli (2 satır × 4 sütun) ---
    mevcut = [e for e in PANO_ENSTRUMANLARI if e["kod"] in pano.columns]
    for satir_baslangic in range(0, len(mevcut), 4):
        sutunlar = st.columns(4)
        for sutun, enstruman in zip(sutunlar, mevcut[satir_baslangic:satir_baslangic + 4]):
            piyasa_karti(sutun, enstruman, pano[enstruman["kod"]], gun_sayisi)

    st.markdown("---")

    # --- Çok dönemli getiri tablosu ---
    st.subheader("📋 Dönemsel Getiri Karşılaştırması")
    st.caption("Her enstrümanın farklı zaman ölçeklerindeki yüzde değişimi — hangi varlığın hangi dönemde öne çıktığını gösterir.")

    satirlar = []
    for e in mevcut:
        seri = pano[e["kod"]].dropna()
        if seri.empty:
            continue
        satirlar.append({
            "Enstrüman": e["ad"],
            "Son Fiyat": f"{e['birim']}{tr_sayi(seri.iloc[-1], e['ondalik'])}",
            "1 Gün %": _degisim_yuzde(seri, 1),
            "1 Hafta %": _degisim_yuzde(seri, 5),
            "1 Ay %": _degisim_yuzde(seri, 22),
            "3 Ay %": _degisim_yuzde(seri, 66),
            "YBB %": _ybb_degisim(seri),
            "1 Yıl %": _degisim_yuzde(seri, 252),
        })

    tablo = pd.DataFrame(satirlar)
    yuzde_kolonlari = [k for k in tablo.columns if k.endswith("%")]

    def _renk(v):
        if pd.isna(v):
            return "color:#8a97a3;"
        return "color:#1e6b34; font-weight:600;" if v >= 0 else "color:#b3261e; font-weight:600;"

    stil = tablo.style.format({k: lambda v: "—" if pd.isna(v) else f"%{v:+.2f}" for k in yuzde_kolonlari})
    stil = (stil.map(_renk, subset=yuzde_kolonlari) if hasattr(stil, "map")
            else stil.applymap(_renk, subset=yuzde_kolonlari))
    st.dataframe(stil, width="stretch", hide_index=True)

    # --- Normalize edilmiş karşılaştırma grafiği ---
    st.subheader("📈 Bazlanmış Performans Karşılaştırması")
    egitim_notu("""
Farklı ölçekteki serileri (BIST 100 ≈ 10.000 puan, dolar ≈ 40 TL) aynı grafikte ham haliyle çizmek
anlamsızdır — büyük olan diğerlerini ezer. Bu yüzden her seri, seçilen dönemin başlangıcında
**100'e eşitlenir** (`seri / ilk_değer × 100`). Böylece grafikte okunan şey fiyat değil, **göreli
performanstır**: 100 çizgisinin üzerindeki her nokta, dönem başına göre kazanç demektir.

Yüksek enflasyon ortamında bu grafiğin asıl faydası şu: TL bazlı bir yatırımın "kazanç" sayılabilmesi
için dolar/euro ve gram altın eğrilerinin **üzerinde** kalması gerekir. Nominal getiri değil, göreli
getiri kritiktir (bkz. Piyasa Kıyaslama sayfası — reel getiri hesabı).
""", baslik="📚 Neden 100'e bazlanıyor?")

    varsayilan = [e["ad"] for e in mevcut if e["kod"] in ("XU100.IS", "TRY=X", "GRAM_ALTIN")]
    ad_kod = {e["ad"]: e["kod"] for e in mevcut}
    secilen_adlar = st.multiselect("Karşılaştırılacak Enstrümanlar", list(ad_kod.keys()),
                                   default=varsayilan, key="pano_karsilastir")

    if secilen_adlar:
        dilim = pano.iloc[-(gun_sayisi + 1):] if len(pano) > gun_sayisi else pano
        normalize = pd.DataFrame(index=dilim.index)
        for ad in secilen_adlar:
            seri = dilim[ad_kod[ad]].dropna()
            if not seri.empty and seri.iloc[0] != 0:
                normalize[ad] = seri / seri.iloc[0] * 100
        if not normalize.empty:
            fig = px.line(normalize, title=f"Göreli Performans — {periyot_etiket} (başlangıç = 100)")
            fig.add_hline(y=100, line_dash="dash", line_color="#8a97a3")
            fig.update_layout(height=420, yaxis_title="Endeks (baz 100)", xaxis_title="",
                              legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ---------------------------------------------------------
    # TCMB EVDS
    # ---------------------------------------------------------
    st.subheader("🏛️ TCMB EVDS Veri Analizi")
    EVDS_SERILERI = {
        "USD/TRY (Alış, Günlük)": "TP.DK.USD.A",
        "EUR/TRY (Alış, Günlük)": "TP.DK.EUR.A",
        "TCMB Politika Faizi (1 Hafta Repo)": "TP.APIFON4",
        "TÜFE (2003=100)": "TP.FG.J0",
    }
    try:
        TCMB_API_KEY = st.secrets.get("EVDS_API_KEY", "")
    except Exception:
        TCMB_API_KEY = ""

    if not TCMB_API_KEY:
        demo_rozeti("EVDS API anahtarı tanımlı değil — aşağıdaki grafik simüle edilmiş örnek veridir. "
                    "Gerçek veri için evds2.tcmb.gov.tr üzerinden ücretsiz anahtar alıp "
                    ".streamlit/secrets.toml dosyasına EVDS_API_KEY olarak ekleyin.")
        df_trend = pd.DataFrame({
            'Yıl': [2020, 2021, 2022, 2023, 2024, 2025],
            'TCMB Politika Faizi': [17, 14, 9, 42.5, 50, 42.5],
            'Ortalama Hasar Maliyeti Endeksi': [118, 145, 285, 465, 540, 610]
        })
        fig_tcmb = px.line(df_trend, x='Yıl', y=['TCMB Politika Faizi', 'Ortalama Hasar Maliyeti Endeksi'],
                           title="Makro Göstergeler vs Sigorta Hasar Maliyeti (simüle)", markers=True)
        fig_tcmb.update_layout(height=380, legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig_tcmb, width="stretch")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            seri_adi = st.selectbox("EVDS Serisi", list(EVDS_SERILERI.keys()), key="evds_seri")
        with c2:
            yil_sayisi = st.slider("Kaç yıllık?", 1, 10, 3, key="evds_yil")

        if st.button("EVDS Verisini Çek"):
            bitis = pd.Timestamp.today()
            baslangic = bitis - pd.DateOffset(years=yil_sayisi)
            kod = EVDS_SERILERI[seri_adi]
            url = (f"https://evds2.tcmb.gov.tr/service/evds/series={kod}"
                   f"&startDate={baslangic.strftime('%d-%m-%Y')}&endDate={bitis.strftime('%d-%m-%Y')}"
                   f"&type=json")
            try:
                with st.spinner("TCMB EVDS sisteminden canlı veri çekiliyor..."):
                    cevap = requests.get(url, headers={"key": TCMB_API_KEY}, timeout=20)
                if cevap.status_code != 200:
                    st.error(f"EVDS yanıt vermedi (HTTP {cevap.status_code}). API anahtarını kontrol edin.")
                else:
                    kayitlar = cevap.json().get("items", [])
                    df_evds = pd.DataFrame(kayitlar)
                    deger_kolonu = kod.replace(".", "_")
                    if deger_kolonu not in df_evds.columns:
                        st.error("Beklenen seri sütunu yanıtta bulunamadı.")
                    else:
                        df_evds["Tarih"] = pd.to_datetime(df_evds["Tarih"], dayfirst=True, errors="coerce")
                        df_evds[seri_adi] = pd.to_numeric(df_evds[deger_kolonu], errors="coerce")
                        df_evds = df_evds[["Tarih", seri_adi]].dropna()
                        st.success(f"{len(df_evds)} gözlem çekildi.")
                        fig_e = px.line(df_evds, x="Tarih", y=seri_adi, title=f"TCMB EVDS — {seri_adi}")
                        fig_e.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig_e, width="stretch")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Son Değer", tr_sayi(df_evds[seri_adi].iloc[-1], 4))
                        m2.metric("Dönem Ortalaması", tr_sayi(df_evds[seri_adi].mean(), 4))
                        m3.metric("Dönem Değişimi",
                                  f"%{(df_evds[seri_adi].iloc[-1] / df_evds[seri_adi].iloc[0] - 1) * 100:+.1f}")
            except Exception as hata:
                st.error(f"EVDS bağlantı hatası: {hata}")

    st.markdown("---")

    # ---------------------------------------------------------
    # TEKNİK ANALİZ
    # ---------------------------------------------------------
    st.subheader("📈 Gerçek Zamanlı Hisse Teknik Analizi")
    egitim_notu("""
Bu paneldeki göstergeler fiyatın kendisinden türetilir; yani yeni bilgi eklemezler, mevcut fiyat
hareketini farklı bir açıdan özetlerler.

**SMA20 / SMA50 (hareketli ortalamalar):** Son 20 ve 50 günün ortalama kapanışı. Kısa ortalamanın
uzun ortalamayı yukarı kesmesi piyasada "golden cross", aşağı kesmesi "death cross" olarak anılır.
Bunların tahmin gücü akademik olarak tartışmalıdır — trend takibi için kullanılırlar, sinyal olarak değil.

**Bollinger Bantları:** Orta çizgi SMA20, bantlar ±2 standart sapma. Yani bant genişliği doğrudan
**volatilitenin** görsel karşılığıdır: bantlar daralıyorsa piyasa sakinleşmiş, açılıyorsa oynaklık
artmıştır. Fiyatın bandın dışına taşması "pahalı/ucuz" demek değildir; sadece son 20 günün normalinden
istatistiksel olarak uzaklaşıldığını gösterir.

**RSI (14):** Son 14 günün kazanç/kayıp oranını 0-100 arasına sıkıştırır. Geleneksel yorum 70 üstü
"aşırı alım", 30 altı "aşırı satım"dır. Güçlü trendlerde RSI haftalarca 70'in üzerinde kalabilir —
tek başına karar aracı değildir.

**Yıllık volatilite:** Günlük getirilerin standart sapmasının `√252` ile ölçeklenmesidir. Bu sayı,
sitedeki Black-Scholes sayfasındaki `σ` girdisinin ve Markowitz optimizasyonundaki risk ölçüsünün
tam olarak kendisidir — yani buradaki teknik panel ile kantitatif modüller aynı büyüklüğü kullanır.
""", baslik="📚 Bu göstergeler ne anlatıyor?")

    secilen_hisse = hisse_secici("teknik")
    c1, c2 = st.columns([1, 1])
    with c1:
        secilen_periyot = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                       index=3, key="teknik_periyot")
    with c2:
        gostergeler = st.multiselect("Göstergeler", ["SMA20", "SMA50", "Bollinger", "Hacim", "RSI"],
                                     default=["SMA20", "SMA50", "Hacim", "RSI"], key="teknik_gosterge")

    if secilen_hisse and st.button("Teknik Analizi Getir"):
        with st.spinner("Canlı piyasa verileri çekiliyor..."):
            df = canli_piyasa_verisi_getir(secilen_hisse, secilen_periyot)

        if df.empty:
            st.error("Sembol bulunamadı (BIST hisselerinin sonuna .IS eklemeyi unutmayın, örn: KCHOL.IS).")
        else:
            df = df.copy()
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            std20 = df["Close"].rolling(20).std()
            df["BB_UST"] = df["SMA20"] + 2 * std20
            df["BB_ALT"] = df["SMA20"] - 2 * std20
            df["RSI"] = _rsi(df["Close"])
            gunluk_getiri = df["Close"].pct_change().dropna()
            yillik_vol = gunluk_getiri.std() * np.sqrt(252) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Son Kapanış", tr_sayi(df['Close'].iloc[-1], 2),
                      f"{(df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100:+.2f}%")
            m2.metric("Dönem Getirisi", f"%{(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.1f}")
            m3.metric("Dönem Yüksek / Düşük",
                      f"{tr_sayi(df['High'].max(), 2)} / {tr_sayi(df['Low'].min(), 2)}")
            m4.metric("Yıllık Volatilite", f"%{yillik_vol:.1f}",
                      help="Günlük getirilerin standart sapması × √252. Black-Scholes'taki σ ile aynı büyüklük.")

            rsi_var = "RSI" in gostergeler
            hacim_var = "Hacim" in gostergeler
            satir_sayisi = 1 + int(hacim_var) + int(rsi_var)
            yukseklikler = [0.62] + ([0.18] if hacim_var else []) + ([0.20] if rsi_var else [])
            yukseklikler = [y / sum(yukseklikler) for y in yukseklikler]

            fig = make_subplots(rows=satir_sayisi, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=yukseklikler)

            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                         low=df["Low"], close=df["Close"], name="Fiyat",
                                         increasing_line_color="#1e6b34",
                                         decreasing_line_color="#b3261e"), row=1, col=1)

            if "Bollinger" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_UST"], line=dict(color="rgba(11,31,51,0.25)", width=1),
                                         name="Bollinger Üst"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_ALT"], line=dict(color="rgba(11,31,51,0.25)", width=1),
                                         fill="tonexty", fillcolor="rgba(11,31,51,0.06)",
                                         name="Bollinger Alt"), row=1, col=1)
            if "SMA20" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(color="#0055a5", width=1.5),
                                         name="SMA20"), row=1, col=1)
            if "SMA50" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="#e08b00", width=1.5),
                                         name="SMA50"), row=1, col=1)

            siradaki = 2
            if hacim_var:
                renkler = np.where(df["Close"] >= df["Open"], "rgba(30,107,52,0.55)", "rgba(179,38,30,0.55)")
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=renkler, name="Hacim"),
                              row=siradaki, col=1)
                fig.update_yaxes(title_text="Hacim", row=siradaki, col=1)
                siradaki += 1
            if rsi_var:
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#6a3fa0", width=1.5),
                                         name="RSI(14)"), row=siradaki, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#b3261e", row=siradaki, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#1e6b34", row=siradaki, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=siradaki, col=1)

            fig.update_layout(title=f"{secilen_hisse.upper()} — Canlı Teknik Analiz ({secilen_periyot})",
                              height=300 + 180 * (satir_sayisi - 1),
                              xaxis_rangeslider_visible=False, hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            fig.update_yaxes(title_text="Fiyat", row=1, col=1)
            st.plotly_chart(fig, width="stretch")

            if rsi_var and not np.isnan(df["RSI"].iloc[-1]):
                son_rsi = df["RSI"].iloc[-1]
                if son_rsi > 70:
                    st.caption(f"RSI {son_rsi:.0f} — geleneksel yorumla 'aşırı alım' bölgesinde. "
                               "Güçlü trendlerde bu seviyenin uzun süre korunabileceğini unutmayın.")
                elif son_rsi < 30:
                    st.caption(f"RSI {son_rsi:.0f} — 'aşırı satım' bölgesinde. Tek başına alım sinyali değildir.")
                else:
                    st.caption(f"RSI {son_rsi:.0f} — nötr bölgede (30-70).")

            kayit_ekle("Canlı Teknik Analiz", f"{secilen_hisse} / {secilen_periyot} incelendi", "Başarılı")

@st.cache_data(ttl=3600, show_spinner=False)
def _coklu_hisse_kapanis_getir(hisseler, periyot):
    """Birden çok sembolün kapanış fiyatı VE hacmini TEK yfinance çağrısıyla indirir.
    (kapanis_df, hacim_df) döner — watchlist tablosundaki Hacim sütunu için."""
    hisseler = list(hisseler)
    ham = yf.download(hisseler, period=periyot, progress=False, auto_adjust=True, threads=True)
    if ham is None or ham.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(ham.columns, pd.MultiIndex):
        kapanis = ham["Close"].copy()
        hacim = ham["Volume"].copy() if "Volume" in ham.columns.get_level_values(0) else pd.DataFrame()
    else:
        kapanis = ham[["Close"]].copy()
        kapanis.columns = hisseler[:1]
        hacim = ham[["Volume"]].copy() if "Volume" in ham.columns else pd.DataFrame()
        if not hacim.empty:
            hacim.columns = hisseler[:1]
    try:
        if getattr(kapanis.index, "tz", None) is not None:
            kapanis.index = kapanis.index.tz_localize(None)
            if not hacim.empty:
                hacim.index = hacim.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return kapanis.dropna(how="all"), hacim


def veri_analizi_sayfasi():
    st.markdown(INV_CSS, unsafe_allow_html=True)

    st.header("📈 Canlı Hisse Korelasyon & Performans Analizi")
    st.markdown(
        '<div class="model-badge">✅ Yahoo Finance üzerinden gerçek fiyat verisiyle hesaplanır — '
        'tüm semboller tek istekte indirilir.</div>',
        unsafe_allow_html=True
    )

    egitim_notu("""
Burada ölçtüğüm şey, iki hissenin fiyatının aynı gün aynı yönde mi hareket ettiği. Korelasyon
katsayısı +1'e yakınsa ikisi neredeyse birlikte hareket ediyor demektir, 0'a yakınsa aralarında
doğrusal bir ilişki yok, -1'e yakınsa biri çıkarken diğeri düşüyor demektir.

Neden önemli? Bir portföyün riskini asıl belirleyen şey, içindeki hisselerin tek tek riski
değil, birbirleriyle ne kadar birlikte hareket ettikleri. Beşi de bankacılık hissesi olan bir
portföy, faiz kararı geldiğinde hepsi aynı anda düşer — çeşitlendirme sadece kağıt üzerinde
kalır. Markowitz sayfasındaki optimizasyonun girdilerinden biri tam olarak bu matris; etkin
sınırın şeklini kovaryans (dolayısıyla korelasyon) belirliyor.

Hesapladığım şey fiyatın kendisi değil, günlük getiri (`pct_change`) korelasyonu. Fiyat
serilerini doğrudan karşılaştırsaydım, ikisi de sadece genel piyasa trendiyle birlikte
yükseldiği için sahte bir yüksek korelasyon çıkardı. Getiriye geçmek bu ortak trendi büyük
ölçüde temizler.

Sınırlaması: bu doğrusal (Pearson) korelasyon. Piyasa çöküşü gibi kriz anlarında hisseler
arasındaki bağımlılık genelde sakin dönemlerden daha güçlüdür — buna korelasyon kırılması
denir ve bu basit matris bunu yakalamaz. Kurumsal risk yönetiminde bu yüzden stres senaryoları
ayrıca test edilir (bkz. Solvency II ve Stres Testi sayfaları).
""")

    ust_c1, ust_c2 = st.columns([3, 1])
    with ust_c2:
        periyot = st.selectbox("Periyot", ["6mo", "1y", "2y", "5y"], index=1, key="korr_periyot")

    st.markdown("**Karşılaştırılacak Hisseler** — sektöre göre gruplanmış listeden seçin")
    varsayilan_semboller = [s for s, _ in BIST_POPULER[:5]]
    hisse_listesi = hisse_secim_paneli("korr", varsayilan_semboller)
    ekstra = st.text_input("İsteğe bağlı ek semboller (virgülle ayırın, örn. AAPL, TSLA)",
                           value="", key="korr_ekstra")
    if ekstra.strip():
        hisse_listesi += [h.strip() for h in ekstra.split(',') if h.strip()]
    hisse_listesi = list(dict.fromkeys(hisse_listesi))  # aynı sembol iki kez eklenmişse tekilleştir

    if len(hisse_listesi) < 2:
        st.info("En az 2 hisse seçin.")
        return

    if not st.button("Analizi Getir", type="primary"):
        return

    with st.spinner("Hisse verileri indiriliyor..."):
        df_fiyat, df_hacim = _coklu_hisse_kapanis_getir(tuple(hisse_listesi), periyot)

    if df_fiyat.empty:
        st.error("Hiçbir sembol için veri bulunamadı.")
        return

    basarisiz = [h for h in hisse_listesi if h not in df_fiyat.columns or df_fiyat[h].dropna().empty]
    if basarisiz:
        st.warning(f"Şu semboller için veri bulunamadı, analizden çıkarıldı: {', '.join(basarisiz)}")
    df_fiyat = df_fiyat.drop(columns=[h for h in basarisiz if h in df_fiyat.columns])

    if df_fiyat.shape[1] < 2:
        st.error("Korelasyon hesaplamak için en az 2 hissenin verisi gerekiyor.")
        return

    ad_sozlugu = dict(BIST_POPULER)
    df_getiri_tum = df_fiyat.pct_change().dropna()
    hisseler = list(df_fiyat.columns)

    # --- Üstte ince akan şerit ---
    ticker_satirlari = []
    for sembol in hisseler:
        seri = df_fiyat[sembol].dropna()
        if len(seri) < 2 or seri.iloc[0] == 0:
            continue
        chg = (seri.iloc[-1] / seri.iloc[0] - 1) * 100
        ticker_satirlari.append((ad_sozlugu.get(sembol, sembol), seri.iloc[-1], chg, "", 2))
    if ticker_satirlari:
        inv_ticker_goster(ticker_satirlari)

    # --- BÜYÜK İZLEME LİSTESİ TABLOSU (Investing.com tarzı) ---
    st.markdown(
        f'<div class="inv2-hdr-row"><h3 style="margin:0;">🧾 İzleme Listesi</h3>'
        f'<span class="inv2-count">{len(hisseler)} hisse · {periyot} · Yahoo Finance</span></div>',
        unsafe_allow_html=True
    )
    satirlar = []
    for sembol in hisseler:
        seri = df_fiyat[sembol].dropna()
        if len(seri) < 2:
            continue
        son, ilk = seri.iloc[-1], seri.iloc[0]
        degisim_yuzde = (son / ilk - 1) * 100 if ilk else 0.0
        getiri_serisi = df_getiri_tum[sembol].dropna() if sembol in df_getiri_tum.columns else pd.Series(dtype=float)
        yillik_vol = getiri_serisi.std() * np.sqrt(252) * 100 if not getiri_serisi.empty else float("nan")
        son_hacim = (df_hacim[sembol].dropna().iloc[-1]
                     if (not df_hacim.empty and sembol in df_hacim.columns and not df_hacim[sembol].dropna().empty)
                     else None)
        satirlar.append({
            "ad": ad_sozlugu.get(sembol, sembol), "sembol": sembol,
            "sektor": BIST_SEKTORU.get(sembol, sembol),
            "son": son, "degisim": son - ilk, "degisim_yuzde": degisim_yuzde,
            "min_": seri.min(), "maks_": seri.max(), "hacim": son_hacim,
            "volatilite": yillik_vol, "seri": seri,
        })
    satirlar.sort(key=lambda r: r["degisim_yuzde"], reverse=True)
    inv2_watchlist_goster(satirlar, kolon_araligi_baslik=f"{periyot} Aralığı")
    st.caption("Dönem Aralığı çubuğu, son fiyatın seçilen dönemin en düşük–en yüksek bandı içindeki "
               "konumunu gösterir; nokta sağa yakınsa fiyat dönem tepesine, sola yakınsa dibine yakındır.")

    st.markdown("---")

    # --- Bazlanmış performans grafiği ---
    st.subheader("📊 Bazlanmış Performans Karşılaştırması")
    st.caption(f"Her hisse, seçilen {periyot} döneminin başında 100'e eşitlenmiştir — göreli performansı gösterir.")
    normalize = pd.DataFrame(index=df_fiyat.index)
    for sembol in hisseler:
        seri = df_fiyat[sembol].dropna()
        if not seri.empty and seri.iloc[0] != 0:
            normalize[ad_sozlugu.get(sembol, sembol)] = seri / seri.iloc[0] * 100
    if not normalize.empty:
        fig_norm = px.line(normalize, title=f"Göreli Performans ({periyot}, başlangıç = 100)")
        fig_norm.add_hline(y=100, line_dash="dash", line_color="#8a97a3")
        fig_norm.update_layout(height=420, yaxis_title="Endeks (baz 100)", xaxis_title="",
                               legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig_norm, width="stretch")

    st.markdown("---")

    # --- Korelasyon ısı haritası ---
    st.subheader("🔥 Günlük Getiri Korelasyon Matrisi")
    df_getiri = df_fiyat.pct_change().dropna()
    corr_matrix = df_getiri.corr()
    corr_goster = corr_matrix.rename(index=ad_sozlugu, columns=ad_sozlugu)

    fig = px.imshow(corr_goster, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, title=f"Günlük Getiri Korelasyonu ({periyot})")
    fig.update_layout(height=max(380, 60 * len(hisseler)), coloraxis_colorbar=dict(title="Korelasyon"))
    fig.update_traces(hovertemplate="%{y} – %{x}<br>Korelasyon: %{z:.2f}<extra></extra>")
    st.plotly_chart(fig, width="stretch")

    corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
    if not corr_pairs.empty:
        en_yuksek = corr_pairs.idxmax()
        en_dusuk = corr_pairs.idxmin()
        ortalama = corr_pairs.mean()

        st.markdown("#### Okuma")
        m1, m2, m3 = st.columns(3)
        m1.metric("En Güçlü Birliktelik",
                  f"{ad_sozlugu.get(en_yuksek[0], en_yuksek[0])} – {ad_sozlugu.get(en_yuksek[1], en_yuksek[1])}",
                  f"{corr_pairs[en_yuksek]:.2f}")
        m2.metric("En Bağımsız Çift",
                  f"{ad_sozlugu.get(en_dusuk[0], en_dusuk[0])} – {ad_sozlugu.get(en_dusuk[1], en_dusuk[1])}",
                  f"{corr_pairs[en_dusuk]:.2f}")
        m3.metric("Ortalama Korelasyon", f"{ortalama:.2f}")

        if ortalama > 0.6:
            st.caption(
                f"Seçilen grup genel olarak yüksek korelasyonlu (ortalama {ortalama:.2f}) — "
                "hepsi büyük ölçüde aynı piyasa hareketine tepki veriyor, çeşitlendirme etkisi sınırlı kalır."
            )
        elif ortalama < 0.2:
            st.caption(
                f"Seçilen grup düşük korelasyonlu (ortalama {ortalama:.2f}) — "
                "bir arada tutulduklarında portföy riski, tek tek hisselerin riskinin toplamından belirgin şekilde düşük çıkar."
            )
        else:
            st.caption(f"Seçilen grubun ortalama korelasyonu {ortalama:.2f} — orta düzeyde bir çeşitlendirme etkisi var.")

    st.markdown("---")

    # --- Volatilite karşılaştırması ---
    st.subheader("⚡ Yıllıklandırılmış Volatilite Karşılaştırması")
    egitim_notu("""
Volatilite, günlük getirilerin standart sapmasının `√252` ile ölçeklenmesidir — bir hissenin
fiyatının ne kadar "oynak" olduğunun standart ölçüsü. Yüksek volatilite tek başına "kötü" demek
değildir; daha çok risk ve daha çok potansiyel getiri birlikte gelir. Bu grafikteki sıralama,
Markowitz sayfasındaki optimizasyonun neden bazı hisselere düşük ağırlık verdiğini sezgisel
olarak açıklar — aynı beklenen getiri için daha oynak bir hisse, portföy varyansına orantısız
katkı yapar.
""", baslik="📚 Volatilite neyi ölçer?")
    volatilite = (df_getiri.std() * np.sqrt(252) * 100).sort_values(ascending=False)
    volatilite.index = [ad_sozlugu.get(s, s) for s in volatilite.index]
    fig_vol = px.bar(volatilite, orientation="h", title="Yıllık Volatilite (%)",
                     color=volatilite.values, color_continuous_scale="Blues")
    fig_vol.update_layout(height=max(300, 40 * len(hisseler)), showlegend=False,
                          xaxis_title="Yıllık Volatilite (%)", yaxis_title="",
                          coloraxis_showscale=False)
    st.plotly_chart(fig_vol, width="stretch")

    kayit_ekle("Hisse Korelasyonu", f"{df_fiyat.shape[1]} hisse, {periyot}", "Matris hesaplandı")

# ---------------------------------------------------------
# ✅ DOĞRULANMIŞ ML MODELLERİ
# ---------------------------------------------------------
def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim Fiyatlama Motoru")
    egitim_notu("""
Burada tahmin ettiğim şey bir sayı: bir sürücünün yıl içinde kaç hasar yapacağı. Dolayısıyla
problem regresyon. Ama düz doğrusal regresyon bu veriye uymuyor.

Sebebi veri setine bakınca görünüyor. `ClaimNb` sütununda poliçelerin ezici çoğunluğu 0, bir
kısmı 1, çok azı 2 ve üzeri. Dağılım sıfıra yığılmış, sağa çarpık ve tam sayılardan oluşuyor.
OLS regresyonu ise hatanın normal dağıldığını ve tahminin her değeri alabileceğini varsayar —
o modeli bu veriye uydurduğumda bazı segmentler için negatif hasar sayısı tahmin ediyor.
"Bu sürücü yılda -0.03 hasar yapar" cümlesinin karşılığı yok.

Poisson regresyonu tam da bu tip sayım verisi için var. Öğrendiği ilişki şu:

`E[Hasar Sayısı] = Exposure × exp(β₀ + β₁·Yaş + β₂·AraçYaşı + β₃·MotorGücü)`

Dışarıdaki `exp` (log-link) tahminin hiçbir koşulda negatife düşmemesini garantiliyor.

Exposure kısmı önemli: poliçeler farklı sürelerde risk altında. Üç ay sigortalı biriyle on iki
ay sigortalı birinin ham hasar sayısını karşılaştırmak yanıltıcı olur. Bu yüzden Exposure'ı
`sample_weight` olarak veriyorum; model artık hasar adedini değil, birim zaman başına hasar
oranını öğreniyor.

Eksik kalan taraf şu: gerçek aktüeryal fiyatlama iki modelden oluşur — hasar *sayısını* tahmin
eden Poisson GLM (bu sayfa) ve hasar *tutarını* tahmin eden Gamma GLM. Saf prim ikisinin
çarpımıdır. Kullandığım veri setinde tutar sütunu olmadığı için şiddet tarafını slider ile
varsayım olarak giriyorum. Üretimde bunun freMTPL2sev gibi bir veriyle ayrıca eğitilmesi gerekir;
sayfada uydurulmuş bir tutar modeli varmış gibi göstermek istemedim.

Metrik olarak AUC göremezsiniz, çünkü ortada sınıflandırma yok. Sayım verisinde karşılığı
Poisson deviance: modelin tahmin ettiği dağılımla gerçek dağılım arasındaki sapma. Düşük olması iyi.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        with st.spinner("Poisson GLM eğitiliyor..."):
            model, deviance, n_train, n_test = kasko_glm_egit()
        model_rozeti(auc=float('nan'), f1=float('nan'), kaynak="freMTPL2freq (gerçek açık kaynak kasko verisi)") if False else None
        st.markdown(
            f'<div class="model-badge">✅ Poisson GLM — Ortalama Poisson Deviance (test): {deviance:.4f} · '
            f'Eğitim: {n_train:,} / Test: {n_test:,} satır · Kaynak: freMTPL2freq (gerçek veri)</div>',
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            driv_age = st.slider("Sürücü Yaşı (DrivAge)", 18, 90, 28)
            veh_power = st.slider("Araç Motor Gücü (VehPower)", 1, 15, 7)
        with c2:
            veh_age = st.slider("Araç Yaşı (VehAge)", 0, 20, 3)
            ort_siddet = st.number_input("Ortalama Hasar Şiddeti Varsayımı (TL)", 5000, 100000, 15000, step=1000,
                                          help="Bu veri setinde hasar tutarı yok, sadece hasar sayısı var. "
                                               "Bu yüzden şiddet varsayımsal girilir; gerçek uygulamada ayrı bir "
                                               "Gamma GLM ile (freMTPL2sev gibi) tahmin edilir.")

        girdi_df = pd.DataFrame([[driv_age, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower'])
        yillik_frekans = model.predict(girdi_df)[0]
        saf_prim = yillik_frekans * ort_siddet
        m1, m2 = st.columns(2)
        m1.metric("Tahmini Yıllık Hasar Frekansı", f"{yillik_frekans:.4f}")
        m2.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
        kayit_ekle("Kasko Poisson GLM", f"Yaş:{driv_age}, AraçYaşı:{veh_age}, Güç:{veh_power}", f"{saf_prim:,.2f} TL")

        yas_listesi = list(range(18, 81))
        sim_frekans = model.predict(pd.DataFrame({'DrivAge': yas_listesi,
                                                    'VehAge': [veh_age] * len(yas_listesi),
                                                    'VehPower': [veh_power] * len(yas_listesi)}))
        fig = go.Figure(go.Scatter(x=yas_listesi, y=sim_frekans * ort_siddet, line=dict(color='#0055a5')))
        fig.update_layout(title="Yaşa Göre Tahmini Saf Prim (Poisson GLM)", xaxis_title="Sürücü Yaşı", yaxis_title="Saf Prim (TL)")
        st.plotly_chart(fig, width='stretch')
    with t2:
        st.latex(r"E[\text{ClaimNb}_i \mid X_i] = \text{Exposure}_i \cdot \exp(X_i \beta)")
        st.latex(r"\text{Saf Prim} = \text{Hasar Frekansı} \times \text{Hasar Şiddeti}")
        st.markdown("Model, `sklearn.linear_model.PoissonRegressor` ile eğitilmiştir; `Exposure` değişkeni "
                     "`sample_weight` olarak kullanılarak GLM'in offset yapısı taklit edilmiştir.")
    with t3:
        st.markdown("Poisson GLM, hasar sayımı verisinin doğasına (negatif olamayan, sağa çarpık) uygun "
                     "tek yöntemdir; OLS regresyonun aksine sigorta/aktüerya sektöründe fiilen kullanılır.")

def kredi_risk_sayfasi():
    st.header("Kredi Risk Skorlama (Lojistik Regresyon)")
    egitim_notu("""
Bu sayfada cevap aradığım soru "ne kadar" değil, "hangisi": başvuru sahibi krediyi öder mi,
ödemez mi. İki kategori, dolayısıyla sınıflandırma.

Model çıktısı yine de bir sayı — temerrüt olasılığı. Lojistik regresyonun yaptığı iş tam olarak
bu: skoru sigmoid fonksiyonuyla 0-1 aralığına sıkıştırmak. Düz doğrusal regresyon kullansaydım
model 1.4 veya -0.2 gibi olasılık olarak okunamayan değerler üretirdi.

Daha güçlü algoritmalar dururken neden lojistik regresyon? Çünkü bankacılıkta model yorumlanabilir
olmak zorunda. Her katsayı, o değişkenin riski hangi yönde ve ne kadar ittiğini söylüyor. Bu
sadece akademik bir zarafet değil; regülasyon müşteriye başvurusunun neden reddedildiğinin
açıklanmasını istiyor ve "gradient boosting öyle dedi" kabul edilebilir bir cevap değil.

`class_weight='balanced'` ayarı burada kritik. Gerçek kredi portföyünde temerrüde düşen müşteri
azınlıkta. Bu ayar olmadan model çoğunluğu ezberleyip herkese "iyi müşteri" demeyi öğrenebilir —
yüksek doğruluk, sıfır fayda. Ağırlıklandırma, azınlık sınıfındaki hataları modele daha pahalıya
mal ediyor.

Raporladığım iki metrik: AUC, modelin rastgele seçilmiş bir iyi ve bir kötü müşteriyi doğru
sıralama olasılığı (0.50 yazı tura, 1.00 kusursuz). F1 ise kaçırılan kötü müşterilerle boşuna
reddedilen iyi müşteriler arasındaki dengeyi ölçüyor. Dengesiz veride ham doğruluğa bakmak
anlamsız olduğu için ikisini birlikte veriyorum, ikisi de test kümesinden.
""")
    sonuc = kredi_risk_modelini_egit()
    model_rozeti(sonuc['auc'], sonuc['f1'], sonuc['kaynak'])

    if sonuc['tip'] == 'gercek':
        c1, c2 = st.columns(2)
        with c1:
            yas = st.slider("Yaş", 18, 75, 35)
            sure_ay = st.slider("Kredi Vadesi (Ay)", 6, 72, 24)
        with c2:
            tutar = st.number_input("Kredi Tutarı (Yerel Para Birimi)", 500, 20000, 3000, step=100)
            mevcut_kredi = st.slider("Mevcut Kredi Sayısı", 1, 4, 1)
        girdi = sonuc['varsayilan'].copy()
        for kolon, deger in [('age', yas), ('duration', sure_ay), ('credit_amount', tutar), ('existing_credits', mevcut_kredi)]:
            if kolon in girdi.index:
                girdi[kolon] = deger
        X_girdi = pd.DataFrame([girdi])[sonuc['kolonlar']]
        risk_skoru = sonuc['model'].predict_proba(X_girdi)[0, 1] * 100
    else:
        gelir = st.number_input("Aylık Gelir (TL)", 8000, 200000, 35000, step=1000)
        borc = st.number_input("Mevcut Kredi Borcu (TL)", 0, 500000, 10000, step=1000)
        yas = st.slider("Yaş", 18, 75, 35)
        sure_ay = st.slider("Kredi Vadesi (Ay)", 6, 60, 24)
        X_girdi = pd.DataFrame([[gelir, borc, yas, sure_ay]], columns=['gelir', 'borc', 'yas', 'sure_ay'])
        risk_skoru = sonuc['model'].predict_proba(X_girdi)[0, 1] * 100

    st.metric("Temerrüt (Default) Olasılığı", f"%{risk_skoru:.1f}")
    if risk_skoru > 50:
        st.error("⚠️ Yüksek risk — manuel inceleme önerilir.")
    if st.button("Riski Kaydet"):
        kayit_ekle("Kredi Risk Skoru (LogReg)", "girdi kaydedildi", f"%{risk_skoru:.1f}")
        st.success("Veritabanına kaydedildi.")

def churn_sayfasi():
    st.header("Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    egitim_notu("""
Yöntem olarak Kredi Risk sayfasındakiyle aynı yerdeyiz — lojistik regresyon, iki sınıf. Değişen
şey girdiler.

Kredi modelinde ağırlıklı olarak finansal durum verisi vardı. Churn'de işe yarayan sinyaller
davranışsal: müşterilik süresi, şikayet sayısı, sahip olunan ürün adedi. Sektörde genel kabul,
demografik bilginin churn tahmininde zayıf kaldığı yönünde — kimin gideceğini yaşı değil, son
haftalardaki davranış değişimi haber veriyor. Kullanım sıklığındaki düşüş, arka arkaya açılan
destek kaydı, tek ürüne inme gibi.

Ürün sayısının etkisi burada özellikle görünür durumda. Birden fazla ürünü olan müşterinin
ayrılması daha maliyetli ve daha zahmetli; bu yüzden ürün sayısı arttıkça terk olasılığı
düşüyor. Bankacılıkta "çapraz satış müşteriyi bağlar" sezgisinin sayısal karşılığı bu.

Churn de dengesiz bir problem — çoğu müşteri kalır. O yüzden accuracy yerine yine AUC ve F1
raporluyorum.
""")
    model, auc, f1 = churn_modelini_egit()
    model_rozeti(auc, f1, "Sentetik veri (metodoloji gösterimi — bkz. not aşağıda)")
    demo_rozeti("Gerçek şirket verisi yerine, bilinen churn risk faktörlerinin (kısa müşterilik süresi, "
                "düşük kredi skoru, yüksek şikayet sayısı) lojistik ilişkisiyle üretilmiş sentetik veri kullanılmıştır. "
                "Üretimde şirketin gerçek CRM/işlem verisiyle yeniden eğitilmelidir.")
    kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
    aktif_yil = st.slider("Müşterilik Süresi (Yıl)", 1, 20, 3)
    sikayet_sayisi = st.slider("Son 1 Yıldaki Şikayet Sayısı", 0, 10, 1)
    urun_sayisi = st.slider("Sahip Olduğu Ürün Sayısı", 1, 5, 2)
    X_girdi = pd.DataFrame([[kredi_skoru, aktif_yil, sikayet_sayisi, urun_sayisi]],
                            columns=['kredi_skoru', 'musterilik_yil', 'sikayet_sayisi', 'urun_sayisi'])
    churn_prob = model.predict_proba(X_girdi)[0, 1] * 100
    st.metric("Terk (Churn) Olasılığı", f"%{churn_prob:.1f}")
    if st.button("Analizi Kaydet"):
        kayit_ekle("Churn Skoru (LogReg)", f"Skor:{kredi_skoru}, Yıl:{aktif_yil}", f"%{churn_prob:.1f}")
        st.success("Veritabanına kaydedildi.")

def fraud_sayfasi():
    st.header("ML Hasar Suistimali (Fraud) Uyarı Sistemi")
    egitim_notu("""
Yine sınıflandırma, ama fraud'da dengesizlik başka bir boyutta. Gerçek portföylerde suistimalli
hasar oranı çoğu zaman %1-2'yi bulmaz. Bu şu tuhaf sonucu doğurur: hiçbir şey öğrenmeyen, her
dosyaya "temiz" diyen bir model %98 doğruluk alır. Metrik seçimi burada modelin kendisinden
daha kritik hale geliyor.

Sayfadaki model lojistik regresyon ve metodolojiyi doğru kuruyor: train/test ayrımı var, metrik
test kümesinden geliyor, azınlık sınıfı ağırlıklandırılmış. Ama üretim kalitesinde bir fraud
sistemi için tek başına yeterli olmadığını söylemem lazım. Sahada tipik olarak SMOTE benzeri
örnekleme teknikleri, anomali tespiti için isolation forest veya doğrusal olmayan etkileşimleri
yakalayan gradient boosting (XGBoost, LightGBM) tercih ediliyor.

Bir de modelin kendisinden bağımsız bir sorun var: fraud verisinde etiket güvenilmezdir.
"Fraud değil" diye işaretlenen dosyaların bir kısmı aslında yakalanamamış fraud'dur. Yani model
gerçek suistimali değil, şirketin geçmişte *tespit edebildiği* suistimali öğrenir. Bu yüzden
fraud modelleri genelde tek başına karar vermez; insan incelemesine düşecek dosyaları önceliklendirir.
""")
    model, auc, f1 = fraud_modelini_egit()
    model_rozeti(auc, f1, "Sentetik veri (metodoloji gösterimi — bkz. not aşağıda)")
    demo_rozeti("Gerçek fraud verisi genellikle gizlidir; burada gece saatleri, yeni poliçe, yüksek hasar tutarı "
                "ve tekrarlayan hasar geçmişi gibi bilinen risk sinyalleriyle üretilmiş sentetik veri kullanılmıştır. "
                "Üretimde şirketin etiketlenmiş gerçek hasar geçmişiyle yeniden eğitilmelidir.")
    hasar_saati = st.slider("Hasar Saati", 0, 23, 2)
    police_yasi = st.slider("Poliçe Yaşı (Gün)", 1, 999, 10)
    hasar_tutari = st.number_input("Bildirilen Hasar Tutarı (TL)", 500, 200000, 15000, step=500)
    onceki_hasar = st.slider("Son 2 Yıldaki Hasar Sayısı", 0, 5, 0)
    X_girdi = pd.DataFrame([[hasar_saati, police_yasi, hasar_tutari, onceki_hasar]],
                            columns=['hasar_saati', 'police_yasi_gun', 'hasar_tutari', 'onceki_hasar_sayisi'])
    skor = model.predict_proba(X_girdi)[0, 1] * 100
    st.metric("Fraud Olasılık Skoru", f"%{skor:.1f}")
    if skor > 50:
        st.error("⚠️ İnceleme Gerekli!")
    if st.button("Sisteme Kaydet"):
        kayit_ekle("Fraud Modeli (LogReg)", f"Saat:{hasar_saati}, PoliçeYaşı:{police_yasi}", f"%{skor:.1f} Risk")
        st.success("Loglandı.")

# ---------------------------------------------------------
# 📐 AKTÜERYAL YÖNTEMLER (gerçek formüller, deterministik)
# ---------------------------------------------------------
def ibnr_sayfasi():
    st.header("IBNR (Chain Ladder) Muallak Hasar Rezervi Aracı")
    demo_rozeti("Bu modül deterministik bir aktüeryal yöntemdir (Chain Ladder); ML modeli değildir.")
    egitim_notu("""
**Bu bir makine öğrenmesi modeli değil, klasik bir aktüeryal projeksiyon yöntemidir.** Chain Ladder, geçmiş
kaza yıllarının hasar gelişim örüntüsünün (bir yıldan diğerine kümülatif hasarın nasıl büyüdüğü) gelecekte de
benzer şekilde devam edeceği varsayımına dayanır. Her gelişim yılı için bir "link ratio" (`f_j`) hesaplanır ve
henüz tamamlanmamış (eksik) hücreler bu oranlarla projekte edilir — sonuç, o kaza yılının "nihai" hasar
tutarıdır. IBNR (Incurred But Not Reported), bu nihai tahmin ile şimdiye kadar ödenen tutar arasındaki farktır.

**Sınırlaması:** Yöntem, geçmiş gelişim örüntüsünün stabil kaldığını varsayar; enflasyon şoku veya poliçe
koşullarındaki ani değişimler bu varsayımı bozar ve daha gelişmiş yöntemler (Bornhuetter-Ferguson, GLM tabanlı
rezervleme) gerektirir.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.info("Hasar gelişim üçgeni verinizi yükleyerek (CSV/Excel) IBNR rezerv hesaplamasını başlatın.")
        yuklenen_dosya = st.file_uploader("📂 Hasar Gelişim Üçgeni Yükle", type=["csv", "xlsx"], key="ibnr_up")
        if yuklenen_dosya is not None:
            df = pd.read_csv(yuklenen_dosya, index_col=0) if yuklenen_dosya.name.endswith('.csv') else pd.read_excel(yuklenen_dosya, index_col=0)
        else:
            df = pd.DataFrame({
                'Gelisim_1': [5000, 5500, 6000, 6500, 7200],
                'Gelisim_2': [7500, 8000, 8800, 9500, np.nan],
                'Gelisim_3': [8500, 9200, 10000, np.nan, np.nan],
                'Gelisim_4': [9000, 9800, np.nan, np.nan, np.nan],
                'Gelisim_5': [9200, np.nan, np.nan, np.nan, np.nan]
            }, index=['2019', '2020', '2021', '2022', '2023'])
        st.write("**Mevcut Hasar Üçgeni (Kümülatif)**")
        st.dataframe(df)
        if st.button("IBNR Rezervini Hesapla"):
            n = len(df)
            f_factors = []
            for j in range(n - 1):
                sum_y_j1, sum_y_j = df.iloc[:n - 1 - j, j + 1].sum(), df.iloc[:n - 1 - j, j].sum()
                f_factors.append(sum_y_j1 / sum_y_j if sum_y_j != 0 else 1)
            df_proj = df.copy()
            for i in range(1, n):
                for j in range(n - i, n):
                    df_proj.iloc[i, j] = df_proj.iloc[i, j - 1] * f_factors[j - 1]
            ibnr = df_proj.iloc[:, -1].sum() - np.nansum(np.diag(df.values[::-1]))
            st.metric("Hesaplanan Toplam IBNR Rezervi", f"{ibnr:,.2f} TL")
            kayit_ekle("IBNR Rezervi", "Chain Ladder Projeksiyonu", f"{ibnr:,.2f} TL")
            fig = go.Figure()
            for index, row in df_proj.iterrows():
                fig.add_trace(go.Scatter(x=df_proj.columns, y=row, mode='lines+markers', name=str(index)))
            fig.update_layout(title="Kaza Yıllarına Göre Hasar Gelişim Projeksiyonu", xaxis_title="Gelişim Yılı", yaxis_title="Kümülatif Hasar (TL)")
            st.plotly_chart(fig, width='stretch')
    with t2:
        st.latex(r"f_j = \frac{\sum_{i=1}^{n-j} C_{i, j+1}}{\sum_{i=1}^{n-j} C_{i, j}}")
    with t3:
        st.markdown("Bilançodaki en büyük yükümlülük kalemini doğru tahmin ederek Solvency rasyolarının SEDDK regülasyonlarına uyumunu sağlar.")

def hayat_sigortasi_sayfasi():
    st.header("Hayat Sigortası ve Aktüeryal Anüite Fiyatlama Motoru")
    demo_rozeti("Basitleştirilmiş sabit mortalite varsayımı kullanır; gerçek uygulamada CSO/TRH gibi resmi mortalite tabloları kullanılır.")
    egitim_notu("""
**Bu da bir makine öğrenmesi modeli değil, aktüeryal bir bugünkü değer (present value) hesabıdır.**
Fikir şudur: müşteri öldüğünde şirketin ödeyeceği teminatın **bugünkü karşılığı** ne kadar olmalı ki, alınan
tek prim uzun vadede beklenen ödemeyi karşılasın?

**Formüldeki üç bileşen:**
- **`q_x` (ölüm olasılığı):** Belirli yaştaki bir kişinin o yıl içinde ölme olasılığı — gerçek uygulamada
  yaşa, cinsiyete ve sigara/sağlık durumuna göre değişen resmi **mortalite tablolarından** (Türkiye'de TRH 2010,
  uluslararası CSO tabloları) okunur. Bu demoda basitleştirilmiş sabit bir değer kullanılıyor.
- **`v^t` (iskonto faktörü):** Paranın zaman değeri — gelecekteki 1 TL'nin bugünkü karşılığı, teknik faiz
  oranıyla iskonto edilir (`v = 1/(1+i)`).
- **`_t p_x` (yaşama olasılığı):** Kişinin t yıl daha hayatta kalma olasılığı.

**Neden önemli?** Bu üçünün çarpımı toplanarak (**aktüeryal bugünkü değer**), şirketin karşılayacağı beklenen
yükümlülüğün bugünkü tutarı bulunur — hayat sigortası ve emeklilik fiyatlamasının temelidir.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        col1, col2 = st.columns(2)
        with col1:
            yas = st.slider("Müşteri Yaşı", 20, 80, 35)
            cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        with col2:
            teknik_faiz = st.slider("Teknik Faiz Oranı (%)", 1.0, 15.0, 3.5)
            teminat = st.number_input("Ölüm Teminatı (TL)", 100000, 5000000, 500000)
        if st.button("Aktüeryal Fiyatlamayı Çalıştır"):
            q_x = 0.0015 if cinsiyet == "Erkek" else 0.0011
            iskonto = 1 / (1 + teknik_faiz / 100)
            nsp = teminat * q_x * iskonto * (80 - yas) * 0.4
            st.metric("Hayat Sigortası Net Tek Prim", f"{nsp:,.2f} TL")
            kayit_ekle("Hayat Sigortası", f"Yaş:{yas}, Cinsiyet:{cinsiyet}", f"NSP: {nsp:,.2f} TL")
    with t2:
        st.latex(r"A_x = \sum_{t=0}^{\infty} v^{t+1} \cdot _{t}p_x \cdot q_{x+t}")
    with t3:
        st.markdown("Mortalite risklerinin matematiksel kesinlikle fiyatlanması, hayat/BES portföyünde kârlılığı korur.")

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Portföy Dağılımı")
    demo_rozeti("Gerçek freMTPL2freq verisi üzerinde tanımlayıcı (descriptive) istatistiktir; tahmin modeli için Kasko GLM sayfasına bakın.")
    egitim_notu("""
**Bu sayfa bir model değil, betimsel istatistiktir (descriptive statistics)** — yani veriyi olduğu gibi
özetler, gelecek tahmini yapmaz (o iş Kasko GLM sayfasında).

**Neden ortalama değil, `Toplam Hasar / Toplam Exposure` oranı kullanılıyor?** Her poliçe farklı sürelerde
(exposure) risk altındadır — 3 ay sigortalı biriyle 12 ay sigortalı birinin hasar sayısını doğrudan karşılaştırmak
yanıltıcıdır. Aktüeryada frekans her zaman **"birim zamana normalize edilmiş"** olarak hesaplanır; bu da
segmentler arası adil karşılaştırmayı sağlar (örn. genç sürücü grubu gerçekten mi daha riskli, yoksa
sadece daha kısa süredir mi poliçesi var?).
""")
    df_hesap = varsayilan_kasko_verisi_getir()
    toplam_hasar, toplam_exposure = df_hesap['ClaimNb'].sum(), df_hesap['Exposure'].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Hasar Adedi", f"{toplam_hasar:,.0f}")
    c2.metric("Toplam Poliçe Yılı", f"{toplam_exposure:,.2f}")
    c3.metric("Genel Portföy Frekansı", f"%{(toplam_hasar / toplam_exposure) * 100:.2f}")
    yas_gruplari = df_hesap.groupby('DrivAge').agg({'ClaimNb': 'sum', 'Exposure': 'sum'}).reset_index()
    yas_gruplari = yas_gruplari[yas_gruplari['Exposure'] > 0]
    yas_gruplari['Frekans'] = yas_gruplari['ClaimNb'] / yas_gruplari['Exposure']
    fig = px.line(yas_gruplari, x='DrivAge', y='Frekans', title="Yaş Bazlı Gerçek Hasar Frekansı", markers=True)
    st.plotly_chart(fig, width='stretch')

def monte_carlo_sayfasi():
    st.header("Monte Carlo ile Toplu Hasar Simülatörü")
    demo_rozeti("Parametrik varsayımlarla (Poisson frekans / Lognormal şiddet) çalışan gerçek bir simülasyon yöntemidir.")
    egitim_notu("""
**Neden simülasyon, neden kapalı formül değil?** Bir yıldaki toplam hasar, rastgele sayıda hasarın
(`N ~ Poisson`) her birinin rastgele bir tutarının (`X ~ Lognormal`) toplamıdır. Bu "bileşik dağılımın"
(compound distribution) kapalı-form bir formülü genelde yoktur; bu yüzden binlerce senaryo rastgele üretilip
(Monte Carlo) sonuçların dağılımına bakılır. Bu yaklaşım, sigorta şirketlerinin sermaye yeterliliği ve
reasürans ihtiyacı hesaplarında (iç model / Solvency II) fiilen kullanılır.

**%99 VaR ne anlama gelir?** "Vakaların %99'unda toplam hasar bu tutarı geçmez" demektir; kalan %1'lik kuyruk,
şirketin kendi öz kaynağıyla veya reasürans ile karşılaması gereken aşırı senaryodur.
""")
    frekans = st.slider("Beklenen Hasar Sayısı (Poisson)", 100, 5000, 1000)
    siddet_mu = st.slider("Ortalama Hasar Şiddeti (Lognormal, log-ölçek)", 5.0, 15.0, 9.0)
    if st.button("Simülasyonu Başlat"):
        rng = np.random.default_rng(42)
        sim_sonuclar = [np.sum(rng.lognormal(mean=siddet_mu, sigma=1.2, size=rng.poisson(frekans))) for _ in range(1000)]
        var_99 = np.percentile(sim_sonuclar, 99)
        st.plotly_chart(px.histogram(sim_sonuclar, nbins=50, title="1 Yıllık Toplam Hasar Dağılımı"), width="stretch")
        st.metric("%99 VaR (İflas Riski Sınırı)", f"{var_99:,.0f} TL")
        kayit_ekle("Monte Carlo", f"Frekans:{frekans}, Mu:{siddet_mu}", f"VaR: {var_99:,.0f} TL")
    st.latex(r"S = \sum_{i=1}^{N} X_i \quad (N \sim Poisson,\ X \sim Lognormal)")

def solvency_sayfasi():
    st.header("Solvency II Sermaye Yeterliliği (Basitleştirilmiş Standart Formül)")
    demo_rozeti("Korelasyon katsayısı (0.25) EIOPA standart formülünün basitleştirilmiş bir yaklaşımıdır.")
    egitim_notu("""
**Bu bölüm bir istatistiksel model değil, Avrupa Birliği'nin Solvency II sigorta regülasyonundan gelen bir
sermaye yeterliliği formülüdür.** Fikir: bir sigorta şirketi piyasa riski, kredi riski ve hayat-dışı sigorta
riski gibi farklı risklere maruzdur; bunları basitçe toplarsak (`Mkt + Kredi + HayatDışı`) gerçekte olmayan bir
"hepsi aynı anda gerçekleşir" varsayımı yapmış oluruz.

**Neden karekök içinde toplama (kareler toplamı) kullanılıyor?** Bu, istatistikteki **çeşitlendirme etkisini**
(diversification benefit) modellemenin standart yoludur — riskler birbiriyle tam ilişkili değilse (örn.
piyasa çöktüğünde hayat-dışı hasarlar otomatik artmaz), toplam risk, tek tek risklerin aritmetik toplamından
daha küçük çıkar. `Corr_{i,j}` matrisi, iki riskin ne kadar birlikte hareket ettiğini gösterir; bu demoda
sabit 0.25 kullanılıyor, EIOPA'nın gerçek standart formülünde her risk çifti için ayrı bir katsayı vardır.

**BSCR (Basic Solvency Capital Requirement):** Şirketin, 1 yıl içinde %99.5 güvenle iflas etmemesi için elinde
bulundurması gereken minimum öz kaynak tutarıdır.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        mkt_risk = st.number_input("Piyasa Riski", value=15000000)
        def_risk = st.number_input("Kredi Riski", value=5000000)
        nl_risk = st.number_input("Hayat Dışı Risk", value=20000000)
        bscr = np.sqrt(mkt_risk**2 + def_risk**2 + nl_risk**2 + 2 * 0.25 * (mkt_risk * def_risk + mkt_risk * nl_risk + def_risk * nl_risk))
        st.metric("Gerekli Temel Özkaynak (BSCR)", f"{bscr:,.0f} TL")
        kayit_ekle("Solvency II", f"Mkt:{mkt_risk}, Kredi:{def_risk}, HayatDışı:{nl_risk}", f"BSCR: {bscr:,.0f} TL")
    with t2:
        st.latex(r"BSCR = \sqrt{ \sum_i \sum_j Corr_{i,j} \cdot SCR_i \cdot SCR_j }")
    with t3:
        st.markdown("Şirketi lisans iptallerinden kurtarır, rasyonel risk yönetimi kültürü inşa eder.")

def black_scholes_sayfasi():
    st.header("Black-Scholes Opsiyon Fiyatlama")
    egitim_notu("""
**Bu, kapalı-form (closed-form) analitik bir çözümdür** — Monte Carlo gibi simülasyona gerek kalmadan, tek bir
formülle "adil" opsiyon fiyatı hesaplanır. 1973'te Black, Scholes ve Merton tarafından geliştirildi ve hâlâ
türev ürün fiyatlamasının temel taşıdır.

**Sezgi:** Bir opsiyonun (call/put) değeri, dayanak varlığın (`S`) gelecekteki fiyatının, kullanım fiyatını
(`K`) aşma **olasılığına** ve aştığında ne kadar aşacağına bağlıdır. Model, hisse fiyatının **geometrik
Brown hareketi** izlediğini (yani logaritmik getirilerin normal dağıldığını) varsayar; `N(d1)` ve `N(d2)`
terimleri standart normal dağılımın kümülatif fonksiyonudur ve bu olasılıkları temsil eder.

**Gerçekçi sınırlamalar (mülakatta sorulabilir):** Model sabit volatilite (`σ`) ve sabit faiz oranı varsayar;
gerçek piyasalarda volatilite zamana ve strike'a göre değişir ("volatility smile/skew"), bu yüzden kurumsal
masalarda Black-Scholes bir başlangıç noktasıdır, tek başına yeterli değildir.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot (S)", value=100.0)
            K = st.number_input("Strike (K)", value=100.0)
            T = st.slider("Vade (Yıl)", 0.05, 5.0, 1.0)
        with c2:
            r = st.slider("Faiz (%)", 1, 50, 15) / 100
            sigma = st.slider("Volatilite (%)", 5, 100, 25) / 100
            opt_tipi = st.selectbox("Opsiyon Tipi", ["Call", "Put"])
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        norm_cdf = lambda x: (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        fiyat = (S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)) if opt_tipi == "Call" \
            else (K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1))
        st.metric("Teorik Opsiyon Primi", f"{fiyat:,.2f}")
        kayit_ekle("Black-Scholes", f"S:{S}, K:{K}, T:{T}, {opt_tipi}", f"{fiyat:,.2f}")
    with t2:
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")
        st.latex(r"C = S_t N(d_1) - K e^{-rT} N(d_2)")
    with t3:
        st.markdown("Kurumsal hazine departmanları için standart bir Risk Hedging (korunma) aracıdır.")

def kredi_var_sayfasi():
    st.header("Kredi Portföyü VaR Hesaplayıcı (Parametrik VaR)")
    demo_rozeti("Parametrik (varyans-kovaryans) VaR yöntemidir; kuyruk riskini (tail risk) tam yakalamaz.")
    egitim_notu("""
**VaR (Value at Risk), "belirli bir güven düzeyinde, belirli bir sürede en fazla ne kadar kaybedebilirim?"**
sorusuna cevap verir. Bu sayfa **parametrik (analitik) VaR** kullanır — yani kaybın normal dağıldığını varsayıp
tek bir formülle hesaplar; alternatifi, Monte Carlo sayfasında gördüğün gibi binlerce senaryo simüle etmektir
(**tarihsel VaR** veya **Monte Carlo VaR**).

**Formüldeki `z_α` nereden geliyor?** Standart normal dağılımın belirli bir güven düzeyine karşılık gelen
kuantilidir (`%95` için `1.65`, `%99` için `2.33`) — "dağılımın bu noktasından sonrasına düşme olasılığı
%5/%1'dir" demektir. `√T` terimi ise riskin zamanla karekök kuralına göre büyüdüğü varsayımıdır (günlük
volatiliteden 10 günlük volatiliteye geçiş).

**En büyük eleştirisi:** Parametrik VaR, gerçek piyasa kayıplarının normal dağılımdan çok daha "kalın kuyruklu"
(fat-tailed) olduğunu göz ardı eder — 2008 krizi gibi aşırı olayları hafife alır. Bu yüzden kurumlar VaR'ı tek
başına değil, stres testi ve Monte Carlo ile birlikte kullanır.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        portfoy = st.number_input("Kredi Portföyü (TL)", value=50000000)
        guven = st.selectbox("Güven Aralığı", ["%95", "%99"])
        z_skor = 1.65 if "%95" in guven else 2.33
        volatilite = st.slider("Yıllık Portföy Volatilitesi (%)", 1, 40, 12) / 100
        var_degeri = portfoy * z_skor * volatilite / np.sqrt(252) * np.sqrt(10)
        st.metric("10 Günlük Portföy VaR", f"{var_degeri:,.0f} TL")
        kayit_ekle("Kredi VaR", f"Portföy:{portfoy}, Güven:{guven}", f"VaR: {var_degeri:,.0f} TL")
    with t2:
        st.latex(r"VaR = V_p \cdot z_{\alpha} \cdot \sigma_p \cdot \sqrt{T}")
    with t3:
        st.markdown("Yönetim kurulunun risk iştahını matematiksel olarak sınırlandırır.")

def reasurans_sayfasi():
    st.header("Dinamik Reasürans Optimizasyonu (Excess of Loss)")
    egitim_notu("""
**Reasürans, sigorta şirketinin kendi sigortasıdır** — büyük bir felaket (deprem, sel vb.) olduğunda tüm
hasarı tek başına karşılamak yerine, riskin bir kısmını bir başka şirkete (reasüröre) devreder.

**Excess of Loss (XoL) mantığı:** Şirket, "saklama payı" (retention) adı verilen bir eşiği kendisi üstlenir;
bu eşiğin üzerindeki her TL, reasüröre devredilir. Formül basit bir eşik fonksiyonudur
(`max(0, Brüt Hasar − Saklama Payı)`), ama gerçek hayatta bu eşik ve devredilen oran, aktüeryal olarak
şirketin sermaye yapısına, risk iştahına ve reasürans maliyetine göre optimize edilir — "dinamik" kelimesi
buradan gelir: saklama payı sabit değil, piyasa koşullarına göre yeniden hesaplanabilir bir değişkendir.

**Neden önemli?** Saklama payı çok düşük seçilirse şirket gereğinden fazla prim reasüröre öder (kârdan
kaybeder); çok yüksek seçilirse büyük bir felaket şirketi iflasa sürükleyebilir. Bu modül, bu dengeyi görsel
olarak keşfetmeyi sağlıyor.
""")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Matematiksel Model", "💼 İş Değeri"])
    with t1:
        brut_hasar = st.slider("Afet Hasarı (Milyon TL)", 10, 500, 150)
        retention = st.slider("Saklama Payı (Milyon TL)", 1, 100, 25)
        devir = max(0, brut_hasar - retention)
        st.metric("Reasüröre Devredilen Hasar", f"{devir} Milyon TL")
        kayit_ekle("Reasürans", f"Brüt:{brut_hasar}, Saklama:{retention}", f"{devir} Milyon TL")
    with t2:
        st.latex(r"\text{Reasürör Payı} = \max(0, \text{Brüt Hasar} - \text{Saklama Payı})")
    with t3:
        st.markdown("Katastrofik riskler karşısında şirketin iflas etmesini engeller (Excess of Loss anlaşması).")

# ---------------------------------------------------------
# 🧪 KAVRAMSAL VİTRİN (formül gösterimi, henüz gerçek veriyle doğrulanmamış)
# ---------------------------------------------------------
def stres_testi_sayfasi():
    st.header("Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    demo_rozeti()
    egitim_notu("""
**Stres testi, bir modelin çıktısı değil, "ne olursa ne olur?" sorusuna verilen sistematik bir cevaptır**
(senaryo analizi). Solvency II ve BDDK regülasyonları, şirketlerden düzenli olarak "aşırı ama makul" (severe
but plausible) senaryolar altında sermaye yeterliliğini test etmesini ister — örn. "enflasyon %20 artarsa,
faiz 5 puan düşerse ne olur?".

**Neden önemli?** Ortalama/beklenen senaryo altında sağlıklı görünen bir şirket, kuyruk (tail) senaryolarında
iflas edebilir. Bu sayfadaki formül basitleştirilmiş bir duyarlılık fonksiyonudur; gerçek stres testleri
genelde tarihsel kriz senaryolarının (2008, 2018 kur şoku vb.) tekrar oynatılmasına (historical scenario
replay) veya çok değişkenli Monte Carlo simülasyonlarına dayanır.
""")
    enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
    faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
    simule_kar = 10000000 * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
    st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")
    st.latex(r"\Delta \text{Kâr} = f(\Delta \text{Faiz}, \Delta \text{Enflasyon})")

def murabaha_hesaplayici():
    st.subheader("🕌 Murabaha (Maliyet+Kâr Satışı) Hesaplayıcı")
    st.markdown(
        '<div class="model-badge">✅ Gerçek katılım bankacılığı ürün formülü — faizsiz finansmanın temel yapısıdır</div>',
        unsafe_allow_html=True
    )
    egitim_notu("""
**Murabaha nedir?** Faizli kredide banka size doğrudan para (faizle) verir; Murabaha'da ise banka **malın
kendisini** satıcıdan peşin alır, size (genelde vadeli ve şeffaf bir kâr marjıyla) satar. Fark kritik:
para değil, mal alınıp satılıyor — bu yüzden "faiz" değil "kâr" olarak adlandırılır ve katılım bankacılığının
en yaygın finansman yöntemidir (araç, konut, ticari mal finansmanı).

**Formül:** `Satış Bedeli = Mal Bedeli × (1 + Kâr Oranı × Vade/12)`, taksitler bu toplamın vadeye eşit
bölünmesiyle bulunur (bazı ürünlerde azalan bakiye yöntemi de kullanılır, burada sabit taksit varsayılıyor).
""")
    c1, c2 = st.columns(2)
    with c1:
        mal_bedeli = st.number_input("Mal/Varlık Bedeli (TL)", 10000, 5000000, 500000, step=10000)
        kar_orani = st.slider("Yıllık Kâr Oranı (%)", 1.0, 60.0, 35.0)
    with c2:
        vade_ay = st.slider("Vade (Ay)", 3, 120, 24)
    toplam_kar = mal_bedeli * (kar_orani / 100) * (vade_ay / 12)
    satis_bedeli = mal_bedeli + toplam_kar
    aylik_taksit = satis_bedeli / vade_ay
    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam Kâr Tutarı", f"{toplam_kar:,.0f} TL")
    m2.metric("Toplam Satış Bedeli", f"{satis_bedeli:,.0f} TL")
    m3.metric("Aylık Taksit", f"{aylik_taksit:,.0f} TL")
    if st.button("Murabaha Hesabını Kaydet"):
        kayit_ekle("Murabaha Hesaplama", f"Mal:{mal_bedeli}, Kâr%:{kar_orani}, Vade:{vade_ay}ay", f"Taksit: {aylik_taksit:,.0f} TL")
        st.success("Kaydedildi.")

def sukuk_degerleme():
    st.subheader("🕌 İcara Sukuk Değerleme (Kira Sertifikası)")
    egitim_notu("""
**Sukuk nedir?** Tahvilin faizsiz karşılığıdır. Klasik tahvilde yatırımcı borç verir ve faiz alır; İcara
Sukuk'ta yatırımcı bir varlığın (bina, uçak vb.) **ortak sahibi** olur ve o varlığın kira gelirinden düzenli
"kira payı" alır — vade sonunda varlık ihraççıya geri satılır (nominal değer ödenir).

**Değerleme mantığı, tahvil fiyatlamasıyla matematiksel olarak aynıdır** (bugünkü değer/present value):
periyodik kira ödemeleri ve vade sonu nominal değer, beklenen kâr oranıyla iskonto edilip toplanır.
""")
    st.latex(r"P = \sum_{t=1}^{n} \frac{\text{Kira Ödemesi}_t}{(1+r)^t} + \frac{\text{Nominal Değer}}{(1+r)^n}")
    c1, c2 = st.columns(2)
    with c1:
        nominal = st.number_input("Nominal Değer (TL)", 1000, 1000000, 100000, step=1000, key="sukuk_nominal")
        kira_orani = st.slider("Yıllık Kira Getiri Oranı (%)", 1.0, 60.0, 32.0, key="sukuk_kira")
    with c2:
        vade_yil = st.slider("Vade (Yıl)", 1, 10, 3, key="sukuk_vade")
        iskonto_orani = st.slider("Beklenen Piyasa Kâr Oranı (%)", 1.0, 60.0, 34.0, key="sukuk_iskonto",
                                   help="Piyasadaki benzer risk profilli araçların beklenen getirisi; sukuk'un fiyatını belirler.")
    yillik_kira = nominal * (kira_orani / 100)
    pv = sum(yillik_kira / (1 + iskonto_orani / 100) ** t for t in range(1, vade_yil + 1))
    pv += nominal / (1 + iskonto_orani / 100) ** vade_yil
    m1, m2 = st.columns(2)
    m1.metric("Yıllık Kira Ödemesi", f"{yillik_kira:,.0f} TL")
    m2.metric("Sukuk'un Bugünkü Değeri", f"{pv:,.0f} TL")
    if pv > nominal:
        st.info("💡 Bugünkü değer nominalin üzerinde — kira oranı, piyasa beklentisinden yüksek (sukuk primli işlem görür).")
    elif pv < nominal:
        st.info("💡 Bugünkü değer nominalin altında — kira oranı, piyasa beklentisinden düşük (sukuk iskontolu işlem görür).")

def katilim_fon_sayfasi():
    st.header("Katılım Bankacılığı Araçları")
    t1, t2, t3 = st.tabs(["🕌 Murabaha Hesaplayıcı", "🕌 Sukuk Değerleme", "📊 Fon Performans Karşılaştırma (Demo)"])
    with t1:
        murabaha_hesaplayici()
    with t2:
        sukuk_degerleme()
    with t3:
        demo_rozeti("Fon getirileri rastgele üretilmiştir; gerçek fon verisi değildir.")
        tarihler = pd.date_range(start='2025-01-01', periods=60, freq='W')
        rng = np.random.default_rng(42)
        df_fonlar = pd.DataFrame({
            'Tarih': tarihler,
            'Hisse Katılım': 100 * (1 + rng.normal(0.003, 0.02, 60)).cumprod(),
            'Altın Katılım': 100 * (1 + rng.normal(0.0025, 0.012, 60)).cumprod(),
            'Sukuk Fonu': 100 * (1 + rng.normal(0.0015, 0.004, 60)).cumprod()
        })
        secilenler = st.multiselect("Fonları Seçin", ['Hisse Katılım', 'Altın Katılım', 'Sukuk Fonu'], default=['Hisse Katılım'])
        if secilenler:
            st.plotly_chart(px.line(df_fonlar, x='Tarih', y=secilenler, title="Performans Kıyaslaması (Baz: 100 TL, simüle)"), width='stretch')

def alm_nakit_sayfasi():
    st.header("ALM Nakit Akışı Eşitleme")
    demo_rozeti()
    egitim_notu("""
**ALM (Asset-Liability Management / Varlık-Yükümlülük Yönetimi), bir şirketin varlıklarından gelecek nakit
akışlarının, yükümlülüklerinden çıkacak nakit akışlarını her dönemde karşılayıp karşılamadığını kontrol eder.**
Bu, kâr/zarar tablosundan farklı bir bakış açısıdır — şirket kâğıt üzerinde kârlı görünse bile, belirli bir
yılda elindeki nakit, o yıl ödemesi gereken tazminatı karşılamıyorsa **likidite krizi** yaşar.

**Sigorta şirketleri için özel önemi:** Hayat sigortası ve emeklilik gibi uzun vadeli yükümlülüklerde, varlık
portföyünün (tahvil, hisse vb.) getiri zamanlaması, yükümlülük ödeme zamanlamasıyla eşleşmelidir. Bu sayfadaki
kısıt (`Varlık Nakit Akışı ≥ Yükümlülük Nakit Akışı`), her dönem için ayrı ayrı sağlanmalıdır.
""")
    yil_1_yuk = st.number_input("1. Yıl Tazminat Yükü (TL)", 1000000, 50000000, 15000000)
    faiz_orani = st.slider("Piyasa Getirisi (%)", 5, 50, 25)
    varlik_tahvil = st.number_input("Tahvil Portföyü (TL)", 10000000, 100000000, 60000000)
    yillar = ['1. Yıl', '2. Yıl', '3. Yıl']
    yukumlulukler = [yil_1_yuk, yil_1_yuk * 1.2, yil_1_yuk * 1.4]
    varlik_getirileri = [varlik_tahvil * (faiz_orani / 100)] * 3
    alm_df = pd.DataFrame({'Yıl': yillar, 'Yükümlülük': yukumlulukler, 'Varlık Getirisi': varlik_getirileri})
    st.plotly_chart(px.bar(alm_df, x='Yıl', y=['Yükümlülük', 'Varlık Getirisi'], barmode='group'), width='stretch')
    st.latex(r"CF_{\text{Varlık}, t} \ge CF_{\text{Yükümlülük}, t}")

def alm_durasyon_sayfasi():
    st.header("ALM Durasyon Eşleştirme Simülatörü")
    demo_rozeti()
    egitim_notu("""
**Durasyon (Macaulay Duration), bir nakit akışı setinin faiz oranı değişimlerine ne kadar duyarlı olduğunu**
tek bir sayıyla özetler — kabaca, "ağırlıklı ortalama vade" olarak düşünülebilir. Uzun durasyonlu bir varlık/
yükümlülük, faiz değiştiğinde değeri daha çok değişir.

**Neden önemli?** Bir sigorta şirketinin varlıklarının durasyonu ile yükümlülüklerinin durasyonu birbirinden
çok farklıysa, faiz oranları değiştiğinde ikisinin değeri **farklı hızda** değişir ve aradaki fark (surplus/
açık) büyür — buna **durasyon uyumsuzluğu (duration mismatch)** denir. "Durasyon eşleştirme" (immunization)
stratejisi, bu ikisini birbirine yakın tutarak bilançoyu faiz şoklarına karşı korumayı hedefler; bu sayfa,
farklı katsayılarla (4.5 vs 6.2) bu duyarlılık farkını gösteriyor.
""")
    f_orani = st.slider("Piyasa Faiz Oranı Şoku (%)", -5.0, 5.0, 0.0)
    v_deger = 100000000 * (1 - 4.5 * (f_orani / 100))
    y_deger = 90000000 * (1 - 6.2 * (f_orani / 100))
    st.plotly_chart(px.bar(pd.DataFrame({'Tür': ['Varlık', 'Yükümlülük'], 'Tutar': [v_deger, y_deger]}), x='Tür', y='Tutar', color='Tür'), width="stretch")
    st.latex(r"D_{Mac} = \frac{\sum_{t=1}^{T} \frac{t \cdot CF_t}{(1+y)^t}}{\sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}}")

def markowitz_sayfasi():
    st.header("Markowitz Etkin Sınır — Canlı Veriyle Gerçek Portföy Optimizasyonu")
    st.markdown(
        '<div class="model-badge">✅ Karesel programlama (SLSQP) ile çözülen gerçek optimizasyon — canlı Yahoo Finance verisi kullanır</div>',
        unsafe_allow_html=True
    )
    egitim_notu("""
**Bu artık simülasyon değil, gerçek bir kısıtlı optimizasyon problemidir.** Modern Portföy Teorisi'nin (Markowitz,
1952) temel sorusu: "Belirli bir hedef getiriyi sağlayan, en düşük riskli (varyanslı) hisse ağırlık kombinasyonu
nedir?"

**Matematiksel yapı:** Portföy varyansı, ağırlıklar (`w`) ve kovaryans matrisi (`Σ`) cinsinden `w^T Σ w`
şeklinde **karesel (quadratic)** bir fonksiyondur. Bunu, `Σw=1` (ağırlıklar toplamı 1) ve `w·μ=hedef getiri`
kısıtları altında minimize ediyoruz. Bu tam olarak bir **Karesel Programlama (Quadratic Programming)**
problemidir; `scipy.optimize.minimize` içindeki **SLSQP** algoritması, KKT (Karush-Kuhn-Tucker) koşullarını
sayısal olarak çözerek optimal ağırlıkları buluyor.

**Efficient Frontier (Etkin Sınır):** Farklı hedef getiriler için bu optimizasyonu tekrarlayıp risk-getiri
noktalarını çizdiğimizde ortaya çıkan eğridir — eğrinin altında kalan hiçbir portföy, aynı riskte daha yüksek
getiri sağlayamaz.

**Sharpe Oranı:** `(Portföy Getirisi − Risksiz Oran) / Portföy Riski` — birim risk başına elde edilen fazla
getiriyi ölçer; Maksimum Sharpe portföyü, etkin sınır üzerindeki "en verimli" noktadır.
""")

    etiketler = [f"{ad} ({sembol})" for sembol, ad in BIST_POPULER]
    secilen_etiketler = st.multiselect("Optimize Edilecek Hisseler (en az 3 seçin)", etiketler, default=etiketler[:5])
    hisse_listesi = [BIST_POPULER[etiketler.index(e)][0] for e in secilen_etiketler]
    risksiz_oran = st.slider("Risksiz Faiz Oranı Varsayımı (%)", 5, 60, 30,
                              help="Sharpe oranı hesaplaması için kullanılır; Türkiye'de genelde TCMB politika faizi baz alınır.") / 100

    if len(hisse_listesi) >= 3 and st.button("Etkin Sınırı Hesapla"):
        with st.spinner("Canlı veri çekiliyor ve optimizasyon çözülüyor..."):
            ort_getiri, kovaryans = markowitz_veri_getir(hisse_listesi)
            if len(ort_getiri) < 3:
                st.error("Yeterli veri çekilemedi, farklı hisseler deneyin.")
            else:
                hedef_araligi = np.linspace(ort_getiri.min(), ort_getiri.max() * 0.98, 30)
                sonuclar = []
                for hg in hedef_araligi:
                    w = min_varyans_agirliklari(kovaryans, hg, ort_getiri.values)
                    if w is not None:
                        risk = np.sqrt(w @ kovaryans.values @ w)
                        sonuclar.append((risk, hg))
                if sonuclar:
                    riskler, getiriler_egri = zip(*sonuclar)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(riskler), y=list(getiriler_egri), mode='lines+markers',
                                              name='Etkin Sınır', line=dict(color='#0055a5')))

                    w_sharpe = maksimum_sharpe_agirliklari(kovaryans, ort_getiri.values, risksiz_oran)
                    if w_sharpe is not None:
                        r_sh = np.dot(w_sharpe, ort_getiri.values)
                        risk_sh = np.sqrt(w_sharpe @ kovaryans.values @ w_sharpe)
                        fig.add_trace(go.Scatter(x=[risk_sh], y=[r_sh], mode='markers',
                                                  marker=dict(color='red', size=14, symbol='star'),
                                                  name='Maksimum Sharpe Portföyü'))
                    fig.update_layout(title="Etkin Sınır (Yıllıklandırılmış Risk vs Getiri)",
                                       xaxis_title="Risk (Std. Sapma)", yaxis_title="Beklenen Getiri")
                    st.plotly_chart(fig, width='stretch')

                    if w_sharpe is not None:
                        st.subheader("⭐ Maksimum Sharpe Portföyü Ağırlıkları")
                        agirlik_df = pd.DataFrame({'Hisse': hisse_listesi, 'Ağırlık (%)': (w_sharpe * 100).round(2)})
                        agirlik_df = agirlik_df[agirlik_df['Ağırlık (%)'] > 0.1].sort_values('Ağırlık (%)', ascending=False)
                        st.dataframe(agirlik_df, width='stretch', hide_index=True)
                        c1, c2 = st.columns(2)
                        c1.metric("Beklenen Yıllık Getiri", f"%{r_sh * 100:.1f}")
                        c2.metric("Beklenen Yıllık Risk", f"%{risk_sh * 100:.1f}")
                        kayit_ekle("Markowitz Optimizasyonu", f"{len(hisse_listesi)} hisse", f"Sharpe getiri: %{r_sh*100:.1f}")
                else:
                    st.warning("Optimizasyon bu hisse kombinasyonu için çözüm bulamadı.")
    st.latex(r"\min_w\ w^T \Sigma w \quad \text{s.t.} \quad \sum w_i = 1,\ \ w^T \mu = \text{hedef getiri},\ \ w_i \ge 0")

def varlik_dagilimi_sayfasi():
    st.header("Varlık Dağılım Simülatörü")
    demo_rozeti("Elle girilen ağırlıkları görselleştirir; Markowitz sayfasındaki gibi optimize etmez.")
    egitim_notu("""
**Bu sayfa bir optimizasyon değil, "ne görürsün" görselleştirmesidir** — Kantitatif Finans segmentindeki
Markowitz sayfası, ağırlıkları matematiksel olarak optimize ederken, burada kullanıcı ağırlıkları elle girip
sonucu görür.

**Neden varlık dağılımı önemli?** Akademik çalışmalar (Brinson vd.), bir portföyün uzun vadeli getiri
değişkenliğinin büyük kısmının, hangi hisseyi seçtiğinizden çok, **hangi varlık sınıflarına ne oranda
yatırım yaptığınızdan** (asset allocation) kaynaklandığını gösterir. Bu yüzden kurumsal portföy yönetiminde
"hangi hisse" sorusundan önce "hisse/tahvil/altın dengesi ne olmalı" sorusu sorulur.
""")
    w_hisse = st.slider("Hisse (%)", 0, 100, 50)
    w_tahvil = st.slider("Tahvil (%)", 0, 100, 30)
    w_altin = st.slider("Altın (%)", 0, 100, 20)
    if w_hisse + w_tahvil + w_altin == 100:
        st.plotly_chart(px.pie(names=['Hisse', 'Tahvil', 'Altın'], values=[w_hisse, w_tahvil, w_altin], hole=0.4), width='stretch')
    else:
        st.warning("⚠️ Toplam %100 olmalıdır!")

def benchmark_sayfasi():
    st.header("Piyasa Kıyaslama (Benchmark)")
    demo_rozeti()
    egitim_notu("""
**Nominal getiri ile reel (enflasyondan arındırılmış) getiriyi karıştırmak, finansta en sık yapılan hatalardan
biridir.** "%35 kazandım" cümlesi, enflasyon %40 ise aslında bir kayıptır. Formüldeki `Reel Getiri` hesabı
tam olarak bunu düzeltir — nominal getiriyi enflasyon oranına bölerek "gerçek satın alma gücü" cinsinden
getiriyi bulur (basit çıkarma — `Nominal − Enflasyon` — yüksek enflasyon dönemlerinde yanıltıcı olduğu için
tercih edilmez).

**Benchmark'ın (kıyaslama endeksinin) rolü:** Bir portföy yöneticisinin "başarılı" olup olmadığı, mutlak
getiriyle değil, ilgili piyasa endeksine (örn. BIST 100) veya enflasyona göre **relatif performansla**
değerlendirilir — bu, fon yönetimi endüstrisinde standart bir KPI'dır.
""")
    portfoy_getiri = st.slider("Yıllık Getiri (%)", 0, 100, 35)
    enflasyon = st.slider("Enflasyon (%)", 0, 80, 25)
    st.plotly_chart(px.bar(pd.DataFrame({'Endeks': ['Portföy', 'BIST 100', 'Enflasyon'], 'Getiri (%)': [portfoy_getiri, 28.5, enflasyon]}),
                            x='Endeks', y='Getiri (%)', color='Endeks'), width='stretch')
    st.latex(r"R_{reel} = \frac{1 + R_{nominal}}{1 + R_{enflasyon}} - 1")

def telematik_sayfasi():
    st.header("Telematik Tabanlı Risk Skorlama")
    demo_rozeti("Elle belirlenmiş ağırlıklarla kurulmuş bir skor formülüdür; eğitilmiş bir ML modeli değildir.")
    egitim_notu("""
**Telematik (Usage-Based Insurance / UBI), sigortayı "kim olduğun"dan (yaş, cinsiyet, meslek) "nasıl
davrandığın"a kaydıran bir yaklaşımdır** — araca takılan bir cihaz veya mobil uygulama, ani fren, gece
sürüşü, hız gibi gerçek sürüş verisini toplar.

**Neden önemli?** Geleneksel fiyatlama (bu sitede Kasko GLM sayfasındaki gibi) demografik/araç özelliklerine
dayanır ve **korelasyona** dayalıdır ("genç sürücüler istatistiksel olarak daha riskli"); telematik ise
**doğrudan davranışı** ölçer, bu yüzden daha adil ve daha az riskli sürücüyü doğru fiyatlandırma potansiyeli
sunar. Bu sayfadaki formül basit bir ağırlıklı skorlama; üretim sistemlerinde genelde bu ham sinyaller,
Kasko GLM'deki gibi bir GLM'e ek değişken olarak beslenir (telematik skoru → prim çarpanı).
""")
    ani_fren = st.slider("Ani Fren (adet/ay)", 0, 50, 12)
    gece_suruş = st.slider("Gece Sürüşü (%)", 0, 100, 45)
    skor = max(0, 100 - (ani_fren * 1.5) - (gece_suruş * 0.5))
    st.metric("Güvenli Sürüş Skoru", f"{skor}")
    st.latex(r"\text{Sürüş Skoru} = 100 - \left(\sum_{i=1}^{n} w_i \cdot X_i\right)")

def clv_sayfasi():
    st.header("Müşteri Yaşam Boyu Değeri (CLV)")
    demo_rozeti()
    egitim_notu("""
**CLV (Customer Lifetime Value), bir müşterinin şirketle olan ilişkisi boyunca yaratacağı toplam kârın bugünkü
tahminidir** — pazarlama ve müşteri ilişkileri kararlarının temel finansal ölçütlerinden biridir.

**Neden önemli?** Bir müşteriyi kazanmanın maliyeti (CAC — Customer Acquisition Cost) ile CLV karşılaştırılır:
CLV, CAC'den anlamlı ölçüde yüksekse o kanal/segment kârlıdır. Bu formül basitleştirilmiş; gerçek CLV
modellerinde genelde müşteri elde tutma olasılığı (survival/churn olasılığı — bu sitedeki Churn modülüyle
doğrudan bağlantılı) ve zaman değeri (iskonto) de hesaba katılır, yani CLV ve Churn modelleri üretimde
genelde birlikte çalışır.
""")
    police_tutari = st.number_input("Poliçe Tutarı", value=4500.0)
    islem_sayisi = st.slider("Yıllık İşlem Sayısı", 1, 12, 2)
    omur = st.slider("Beklenen Müşteri Ömrü (Yıl)", 1, 20, 5)
    marj = st.slider("Kâr Marjı (%)", 5, 50, 20) / 100
    clv_deger = police_tutari * islem_sayisi * omur * marj
    st.metric("Ortalama CLV", f"{clv_deger:,.2f} TL")
    st.latex(r"CLV = (\text{Ort. Harcama} \times \text{Frekans} \times \text{Ömür}) \times \text{Marj}")

# ---------------------------------------------------------
# SİSTEM & İLETİŞİM
# ---------------------------------------------------------
def veritabani_sayfasi():
    st.header("SQLite Veritabanı Geçmişi")
    st.info("Not: Streamlit Cloud gibi ephemeral (geçici) barındırmalarda bu veritabanı her yeniden dağıtımda sıfırlanır.")
    st.dataframe(gecmisi_getir(), width='stretch')

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
    with col2:
        st.markdown("""
        Merhaba! Ben **Sultan Kuş**.
        Matematik altyapımla finans, sigorta ve risk analitiği alanlarına yönelik veri bilimi çözümleri geliştiriyorum.
        Hedefim; finans, sigorta ve **katılım bankacılığı** alanlarında, matematiksel titizliği veri bilimiyle
        birleştiren bir rol.

        * **📧 Email:** [kussultannn34@gmail.com](mailto:kussultannn34@gmail.com)
        * **💼 LinkedIn:** [linkedin.com/in/sultan-kuş](https://www.linkedin.com/in/sultan-kuş/)
        * **💻 GitHub:** [github.com/SultanKus](https://github.com/SultanKus)
        """)
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
    st.caption("Bu tablo, proje boyunca fiilen uygulanan yöntemlere dayanır — her satır, sitedeki ilgili modülle doğrulanabilir.")
    st.markdown("---")
    st.subheader("📄 Özgeçmiş (CV)")
    try:
        with open("Sultan_Kus_CV.pdf", "rb") as pdf_file:
            st.download_button(label="Özgeçmişimi İndir (PDF)", data=pdf_file, file_name="Sultan_Kus_CV.pdf", mime="application/pdf")
    except FileNotFoundError:
        st.warning("⚠️ 'Sultan_Kus_CV.pdf' dosyası proje klasöründe bulunamadı.")

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
