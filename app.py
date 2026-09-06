import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_poisson_deviance
import sqlite3
from datetime import datetime
import yfinance as yf
import requests
from math import erf

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
# CANLI PİYASA VERİSİ (yfinance)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def canli_piyasa_verisi_getir(sembol, periyot="1y"):
    return yf.Ticker(sembol).history(period=periyot)

@st.cache_data(ttl=1800)
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
]

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
# GENEL BAKIŞ & CANLI PİYASA SAYFALARI
# ---------------------------------------------------------
def ml_rehberi_sayfasi():
    st.header("🎓 Makine Öğrenmesi & Aktüeryal Yöntemler Rehberi")
    st.markdown("Bu sayfa, sitedeki modüllerde kullanılan yöntemlerin teorik arka planını özetler. "
                 "Her modülün kendi sayfasında da 📚 açılır bölümde o modüle özgü detay bulunur.")

    st.subheader("1️⃣ Regresyon mu, Sınıflandırma mı?")
    st.markdown("""
| Soru tipi | Çıktı | Bu sitedeki örnek |
|---|---|---|
| "Ne kadar?" (regresyon) | Sayısal değer | Kasko saf prim (Poisson GLM) |
| "Hangisi?" (sınıflandırma) | Kategori / olasılık | Kredi risk, churn, fraud (Lojistik Regresyon) |
""")

    st.subheader("2️⃣ GLM (Genelleştirilmiş Doğrusal Model) Ailesi")
    st.markdown("""
Sıradan doğrusal regresyon (OLS), verinin normal dağıldığını varsayar. Gerçek sigorta/finans verisi nadiren
bu şekildedir — bu yüzden GLM ailesi kullanılır; her biri farklı bir "link fonksiyonu" ile çıktıyı doğru
aralığa (pozitif, 0-1 vb.) sıkıştırır.
""")
    st.latex(r"g(E[Y|X]) = X\beta \quad \text{(g: link fonksiyonu)}")
    st.markdown("""
- **Poisson GLM** (log-link): sayım verisi için (hasar adedi) — bu sitede Kasko modülünde kullanılıyor.
- **Gamma GLM** (log-link): pozitif, sürekli ve çarpık veri için (hasar tutarı/şiddeti) — bu sitede varsayımsal olarak ele alınıyor, üretimde ayrı eğitilmesi gerekir.
- **Logistic (Binomial) GLM** (logit-link): ikili sınıflandırma için (temerrüt, churn, fraud) — bu sitede Kredi Risk/Churn/Fraud modüllerinde kullanılıyor.
""")

    st.subheader("3️⃣ Neden Eğitim/Test Ayrımı (Train/Test Split)?")
    st.markdown("""
Bir model, gördüğü veriyi ezberleyebilir (**overfitting**) — eğitim verisinde mükemmel ama hiç görmediği yeni
veride kötü performans gösterir. Bunu tespit etmek için veri ikiye bölünür: model sadece eğitim (train)
kümesiyle öğrenir, performansı ise hiç görmediği test kümesinde ölçülür. Bu sitedeki her "✅ Doğrulanmış ML
Modeli" sayfasında bu ayrım yapılır ve metrik test kümesinden raporlanır.
""")

    st.subheader("4️⃣ Metrik Sözlüğü")
    st.markdown("""
| Metrik | Ne ölçer | Ne zaman kullanılır |
|---|---|---|
| **AUC** | Modelin pozitif/negatif sınıfı rastgele bir çiftte doğru sıralama olasılığı (0.5=rastgele, 1.0=mükemmel) | İkili sınıflandırma |
| **F1** | Precision ve Recall'un harmonik ortalaması | Dengesiz (imbalanced) sınıflandırma verisi |
| **Poisson Deviance** | Tahmin ile gerçek sayım verisi arasındaki sapma (düşük iyi) | Sayım verisi regresyonu (frekans modelleri) |
| **VaR (Value at Risk)** | Belirli bir güven düzeyinde aşılmayacak maksimum kayıp | Risk sermayesi / reasürans kararları |
""")

    st.subheader("5️⃣ Sentetik Veri Neden Bazı Modüllerde Kullanılıyor?")
    st.markdown("""
Gerçek fraud/churn/kredi verisi genelde şirket içi ve gizlidir; halka açık olanların çoğu ya çok büyük ya da
alan (domain) uyuşmuyor. Bu sitedeki Churn ve Fraud modülleri, **bilinen risk faktörlerinin mantıksal bir
ilişkisiyle** (rastgele değil) üretilmiş sentetik veriyle eğitilmiştir — amaç, doğru ML metodolojisini dürüstçe
göstermektir. Kredi Risk modülü ise mümkün olduğunda gerçek bir açık veri setine (OpenML German Credit)
bağlanır; erişilemezse aynı şeffaflıkla sentetik veriye döner. Her modülün kendi sayfasında hangi veriyle
çalıştığı rozetle (✅ gerçek / 🧪 demo) açıkça belirtilir.
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
    st.header("🌍 Canlı Makroekonomi & Küresel Piyasalar")
    st.caption("✅ Bu sayfa Yahoo Finance API üzerinden gerçek zamanlı piyasa verisi çeker.")
    try:
        df_bist = canli_piyasa_verisi_getir("XU100.IS", "5d")
        df_usd = canli_piyasa_verisi_getir("TRY=X", "5d")
        df_gold = canli_piyasa_verisi_getir("GC=F", "5d")
        c1, c2, c3 = st.columns(3)
        bist_son, bist_onceki = df_bist['Close'].iloc[-1], df_bist['Close'].iloc[-2]
        c1.metric("BIST 100 (Canlı)", f"{bist_son:,.2f}", f"{((bist_son - bist_onceki) / bist_onceki) * 100:.2f}%")
        usd_son, usd_onceki = df_usd['Close'].iloc[-1], df_usd['Close'].iloc[-2]
        c2.metric("USD/TRY (Canlı)", f"{usd_son:,.2f} ₺", f"{((usd_son - usd_onceki) / usd_onceki) * 100:.2f}%", delta_color="inverse")
        gold_son, gold_onceki = df_gold['Close'].iloc[-1], df_gold['Close'].iloc[-2]
        c3.metric("Altın Ons (Canlı)", f"${gold_son:,.2f}", f"{((gold_son - gold_onceki) / gold_onceki) * 100:.2f}%")
    except Exception:
        st.warning("Yahoo Finance verileri şu an çekilemiyor. İnternet bağlantınızı kontrol edin.")

    st.markdown("---")
    st.subheader("🏛️ TCMB Veri Analizi (EVDS API)")
    TCMB_API_KEY = "BURAYA_TCMB_API_ANAHTARINI_YAZIN"
    if TCMB_API_KEY == "BURAYA_TCMB_API_ANAHTARINI_YAZIN":
        demo_rozeti("EVDS API anahtarı tanımlı değil — aşağıdaki grafik simüle edilmiş örnek veridir.")
        df_trend = pd.DataFrame({
            'Yıl': [2020, 2021, 2022, 2023, 2024],
            'TCMB Politika Faizi': [17, 14, 9, 42.5, 50],
            'Ortalama Hasar Maliyeti Endeksi': [118, 145, 285, 465, 540]
        })
        fig_tcmb = px.line(df_trend, x='Yıl', y=['TCMB Politika Faizi', 'Ortalama Hasar Maliyeti Endeksi'],
                            title="Makro Göstergeler vs Sigorta Hasar Maliyeti", markers=True)
        st.plotly_chart(fig_tcmb, width="stretch")
    else:
        try:
            with st.spinner("TCMB EVDS sisteminden canlı veri çekiliyor..."):
                url = f"https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A&startDate=01-01-2023&endDate=01-01-2024&type=json&key={TCMB_API_KEY}"
                response = requests.get(url)
                if response.status_code == 200:
                    st.success("TCMB Verisi Başarıyla Çekildi!")
                    st.write(response.json().get('items', [])[:5])
                else:
                    st.error("API Anahtarı hatalı veya TCMB servisi yanıt vermiyor.")
        except Exception:
            st.error("EVDS bağlantı hatası oluştu.")

    st.markdown("---")
    st.subheader("📈 Gerçek Zamanlı Hisse Teknik Analizi")
    secilen_hisse = hisse_secici("teknik")
    secilen_periyot = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="teknik_periyot")
    if secilen_hisse and st.button("Teknik Analizi Getir"):
        with st.spinner('Canlı piyasa verileri çekiliyor...'):
            df_hisse = canli_piyasa_verisi_getir(secilen_hisse, secilen_periyot)
            if not df_hisse.empty:
                df_hisse['SMA20'] = df_hisse['Close'].rolling(window=20).mean()
                df_hisse['SMA50'] = df_hisse['Close'].rolling(window=50).mean()
                fig = go.Figure(data=[go.Candlestick(x=df_hisse.index, open=df_hisse['Open'], high=df_hisse['High'],
                                                      low=df_hisse['Low'], close=df_hisse['Close'], name='Fiyat')])
                fig.add_trace(go.Scatter(x=df_hisse.index, y=df_hisse['SMA20'], line=dict(color='blue', width=1.5), name='SMA20'))
                fig.add_trace(go.Scatter(x=df_hisse.index, y=df_hisse['SMA50'], line=dict(color='orange', width=1.5), name='SMA50'))
                fig.update_layout(title=f"{secilen_hisse.upper()} Canlı Teknik Analiz", yaxis_title="Fiyat",
                                   xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig, width="stretch")
                kayit_ekle("Canlı Teknik Analiz", f"{secilen_hisse} incelendi", "Başarılı")
            else:
                st.error("Sembol bulunamadı (BIST hisselerinin sonuna .IS eklemeyi unutmayın, örn: KCHOL.IS).")

def veri_analizi_sayfasi():
    st.header("📈 Canlı Hisse Korelasyon Lab (EDA)")
    st.caption("✅ Gerçek zamanlı Yahoo Finance verisiyle hesaplanır.")
    etiketler = [f"{ad} ({sembol})" for sembol, ad in BIST_POPULER]
    secilen_etiketler = st.multiselect(
        "Korelasyon Hesaplanacak Hisseler (listeden seçin)",
        etiketler, default=etiketler[:5]
    )
    hisse_listesi = [BIST_POPULER[etiketler.index(e)][0] for e in secilen_etiketler]

    ekstra = st.text_input("İsteğe bağlı ek semboller (virgülle ayırın, örn. AAPL, TSLA)", value="")
    if ekstra.strip():
        hisse_listesi += [h.strip() for h in ekstra.split(',') if h.strip()]

    if hisse_listesi and st.button("Gerçek Zamanlı Korelasyon Matrisini Çiz"):
        with st.spinner('Hisse verileri indiriliyor...'):
            df_korelasyon = pd.DataFrame()
            for hisse in hisse_listesi:
                veri = canli_piyasa_verisi_getir(hisse, "1y")
                if not veri.empty:
                    df_korelasyon[hisse] = veri['Close']
            if not df_korelasyon.empty:
                corr_matrix = df_korelasyon.pct_change().corr()
                fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                                 title="1 Yıllık Getiri Korelasyon Isı Haritası")
                st.plotly_chart(fig, width="stretch")
                kayit_ekle("Canlı Korelasyon", f"{len(hisse_listesi)} Hisse Analiz Edildi", "Isı Haritası Çizildi")

# ---------------------------------------------------------
# ✅ DOĞRULANMIŞ ML MODELLERİ
# ---------------------------------------------------------
def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim Fiyatlama Motoru")
    egitim_notu("""
**Bu bir regresyon problemidir** — çıktımız (hasar frekansı) sayısal bir değer, kategori değil.

**Neden sıradan doğrusal regresyon (OLS) değil de Poisson GLM?**
Hasar sayısı (`ClaimNb`) hiç negatif olamaz, tam sayıdır ve genelde 0'a yığılmış, sağa çarpık bir dağılıma sahiptir.
OLS regresyonu ise verinin normal dağıldığını ve tahminlerin negatif de olabileceğini varsayar — bu yüzden
sigorta/aktüerya sektöründe sayım verisi (count data) için **Genelleştirilmiş Doğrusal Modeller (GLM)** ailesinden
**Poisson regresyonu** kullanılır. Model şunu öğrenir:

`E[Hasar Sayısı] = Exposure × exp(β₀ + β₁·Yaş + β₂·AraçYaşı + β₃·MotorGücü)`

Burada `exp(...)` fonksiyonu (log-link), tahminin her zaman pozitif çıkmasını garanti eder.

**Frekans-Şiddet ayrımı:** Gerçek aktüeryal fiyatlama, iki ayrı model kurar: hasar **sayısını** tahmin eden bir
Poisson GLM (bu sayfa) ve hasar **tutarını** tahmin eden bir Gamma GLM. Saf prim, ikisinin çarpımıdır. Bu veri
setinde tutar bilgisi olmadığı için şiddet kısmı varsayımsal bir slider ile giriliyor — gerçek üretimde ayrı bir
Gamma GLM ile öğrenilmesi gerekir.

**Değerlendirme metriği — Poisson Deviance:** Modelin gerçek ve tahmin edilen dağılımlar arasındaki farkı ne
kadar iyi açıkladığını ölçer (R² karesinin sayım verisi için karşılığı gibi düşünülebilir); düşük olması iyidir.
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
**Bu bir sınıflandırma (classification) problemidir** — çıktı "temerrüt (1) / temerrüt değil (0)" gibi iki
kategoriden biri, sayısal bir miktar değil.

**Neden Lojistik Regresyon?** Çıktının 0 ile 1 arasında bir **olasılık** olarak yorumlanabilmesi lazım
(doğrusal regresyon 1.3 ya da -0.4 gibi anlamsız değerler üretebilir). Lojistik regresyon, sigmoid fonksiyonu
ile skoru olasılığa sıkıştırır: `P(Temerrüt) = 1 / (1 + e^-(β·X))`. Bankacılık/kredi sektöründe hâlâ en çok
tercih edilen yöntemlerden biridir çünkü **yorumlanabilir**: her katsayı, o değişkenin riski ne yönde
etkilediğini gösterir — bu, regülasyon ve müşteriye "neden reddedildiniz" açıklaması için önemlidir.

**`class_weight='balanced'` neden var?** Gerçek kredi verilerinde "kötü" müşteri sayısı azınlıktadır
(dengesiz/imbalanced veri). Bu ayar olmadan model, çoğunluk sınıfını ezberleyip azınlığı gözden kaçırabilir.

**Metrikler:**
- **AUC (0.5–1.0):** Modelin iyi müşteriyle kötü müşteriyi rastgele bir çiftte doğru sıralama olasılığı. 0.5 = yazı tura, 1.0 = mükemmel ayrım.
- **F1:** Precision (yanlış alarm oranı düşük mü) ile Recall (kötü müşteriyi kaçırmıyor mu) arasındaki dengeyi tek sayıya indirir; dengesiz veri setlerinde ham doğruluk (accuracy) yanıltıcı olduğu için tercih edilir.
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
**Yine bir sınıflandırma problemi** ("terk edecek / etmeyecek"), Kredi Risk sayfasındaki gibi Lojistik Regresyon
kullanılıyor. Farkı, girdi değişkenleri: kredi skoru yerine burada müşterilik süresi, şikayet sayısı ve ürün
sayısı gibi **davranışsal (behavioral)** sinyaller kullanılıyor — churn modellemesinde genelde demografik
verilerden çok "son 30-90 gün içindeki davranış değişimi" en güçlü sinyali verir.

**Neden AUC/F1 hâlâ önemli?** Churn de tipik olarak dengesizdir (çoğu müşteri kalır, azınlık terk eder), bu
yüzden accuracy yerine yine AUC/F1 raporlanıyor.
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
**Yine sınıflandırma**, ama fraud modellerinde tipik zorluk çok daha uçtur: gerçek hayatta fraud vakaları
genelde toplam hasarların %1-2'sini bile bulmaz (aşırı dengesiz veri). Bu yüzden üretimde tek başına Lojistik
Regresyon yerine genelde **SMOTE gibi örnekleme teknikleri**, **isolation forest** (anomali tespiti) veya
**gradient boosting** (XGBoost/LightGBM) tercih edilir. Bu sayfa metodolojiyi (train/test + AUC/F1) doğru
gösterir ama üretim kalitesinde bir fraud sistemi için tek başına yeterli değildir.
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
    enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
    faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
    simule_kar = 10000000 * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
    st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")
    st.latex(r"\Delta \text{Kâr} = f(\Delta \text{Faiz}, \Delta \text{Enflasyon})")

def katilim_fon_sayfasi():
    st.header("Katılım Emeklilik & Faizsiz Yatırım Fonları Takip Aracı")
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
    f_orani = st.slider("Piyasa Faiz Oranı Şoku (%)", -5.0, 5.0, 0.0)
    v_deger = 100000000 * (1 - 4.5 * (f_orani / 100))
    y_deger = 90000000 * (1 - 6.2 * (f_orani / 100))
    st.plotly_chart(px.bar(pd.DataFrame({'Tür': ['Varlık', 'Yükümlülük'], 'Tutar': [v_deger, y_deger]}), x='Tür', y='Tutar', color='Tür'), width="stretch")
    st.latex(r"D_{Mac} = \frac{\sum_{t=1}^{T} \frac{t \cdot CF_t}{(1+y)^t}}{\sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}}")

def markowitz_sayfasi():
    st.header("Markowitz Etkin Sınır (Efficient Frontier)")
    demo_rozeti("Rastgele üretilmiş getiri/risk noktalarıdır; gerçek portföy optimizasyonu değildir.")
    if st.button("Rastgele Portföy Simüle Et"):
        rng = np.random.default_rng(42)
        getiri, risk = rng.normal(0.20, 0.10, 1000), rng.normal(0.15, 0.05, 1000)
        st.plotly_chart(px.scatter(x=risk, y=getiri, color=getiri / risk), width="stretch")
    st.latex(r"\sigma_p^2 = \sum_{i} \sum_{j} w_i w_j Cov(R_i, R_j)")

def varlik_dagilimi_sayfasi():
    st.header("Varlık Dağılım Simülatörü")
    demo_rozeti()
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
    portfoy_getiri = st.slider("Yıllık Getiri (%)", 0, 100, 35)
    enflasyon = st.slider("Enflasyon (%)", 0, 80, 25)
    st.plotly_chart(px.bar(pd.DataFrame({'Endeks': ['Portföy', 'BIST 100', 'Enflasyon'], 'Getiri (%)': [portfoy_getiri, 28.5, enflasyon]}),
                            x='Endeks', y='Getiri (%)', color='Endeks'), width='stretch')
    st.latex(r"R_{reel} = \frac{1 + R_{nominal}}{1 + R_{enflasyon}} - 1")

def telematik_sayfasi():
    st.header("Telematik Tabanlı Risk Skorlama")
    demo_rozeti("Elle belirlenmiş ağırlıklarla kurulmuş bir skor formülüdür; eğitilmiş bir ML modeli değildir.")
    ani_fren = st.slider("Ani Fren (adet/ay)", 0, 50, 12)
    gece_suruş = st.slider("Gece Sürüşü (%)", 0, 100, 45)
    skor = max(0, 100 - (ani_fren * 1.5) - (gece_suruş * 0.5))
    st.metric("Güvenli Sürüş Skoru", f"{skor}")
    st.latex(r"\text{Sürüş Skoru} = 100 - \left(\sum_{i=1}^{n} w_i \cdot X_i\right)")

def clv_sayfasi():
    st.header("Müşteri Yaşam Boyu Değeri (CLV)")
    demo_rozeti()
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
        Matematik altyapımla veri bilimi, finansal risk analitiği ve karar destek sistemleri geliştiriyorum.

        * **📧 Email:** [kussultannn34@gmail.com](mailto:kussultannn34@gmail.com)
        * **💼 LinkedIn:** [linkedin.com/in/sultan-kuş](https://www.linkedin.com/in/sultan-kuş/)
        * **💻 GitHub:** [github.com/SultanKus](https://github.com/SultanKus)
        """)
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
        st.Page(katilim_fon_sayfasi, title="Katılım Fon Takibi", icon="🪙"),
        st.Page(alm_nakit_sayfasi, title="ALM Nakit Eşitleme", icon="🔄"),
        st.Page(alm_durasyon_sayfasi, title="ALM Durasyon", icon="⚖️"),
        st.Page(markowitz_sayfasi, title="Markowitz Optimizasyonu", icon="🥧"),
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
