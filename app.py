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
# VERİTABANI (SQLite) - Üyelik ve Loglar Eklendi
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

# --- SIDEBAR KULLANICI GİRİŞ & YÖNETİMİ ---
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
    st.markdown(f'<div class="model-badge">✅ Eğitilmiş model — AUC: {auc:.3f} · F1: {f1:.3f} · Kaynak: {kaynak}</div>', unsafe_allow_html=True)

def demo_rozeti(metin="Bu sayfa kavramsal bir formül gösterimidir; canlı veri veya eğitilmiş model kullanmaz."):
    st.markdown(f'<div class="demo-badge">🧪 {metin}</div>', unsafe_allow_html=True)

def egitim_notu(icerik, baslik="📚 Bu modül nasıl çalışıyor? (Teori + Yöntem)"):
    with st.expander(baslik):
        st.markdown(icerik)

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

def tr_sayi(x, ondalik=2):
    try:
        s = f"{float(x):,.{ondalik}f}"
    except (TypeError, ValueError):
        return "—"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")

# Sayfalar
def ml_rehberi_sayfasi():
    st.header("Yöntem Notları")
    st.markdown("Model seçimi ve arkasındaki istatistiki gerekçeler.")

def ana_sayfa():
    st.title("Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("---")
    st.markdown("### 🏛️ Platform Vizyonu")

def finansal_bilgi_sayfasi():
    st.header("Canlı Makroekonomi & Piyasalar")

def veri_analizi_sayfasi():
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

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    col1, col2 = st.columns([1, 3])
    with col1:
        yuklenen_foto = st.file_uploader("Profil Fotoğrafı Yükle", type=["png", "jpg", "jpeg"], key="profil_foto_up")
        if yuklenen_foto is not None:
            st.image(yuklenen_foto, width=180, caption="Sultan Kuş")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2922/2922561.png", width=180, caption="Örnek Profil")
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
        st.download_button(label="📥 Yüklenen Özgeçmişimi İndir (PDF)", data=yuklenen_cv, file_name="Sultan_Kus_CV.pdf", mime="application/pdf")
    else:
        try:
            with open("Sultan_Kus_CV.pdf", "rb") as pdf_file:
                st.download_button(label="📥 Özgeçmişimi İndir (PDF)", data=pdf_file, file_name="Sultan_Kus_CV.pdf", mime="application/pdf")
        except FileNotFoundError:
            st.info("💡 Kendi gerçek PDF CV'nizi yukarıdaki alandan yükleyebilirsiniz.")

# ---------------------------------------------------------
# NAVİGASYON
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış & Canlı Piyasa": [
        st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠"),
        st.Page(ml_rehberi_sayfasi, title="ML & Aktüeryal Rehberi", icon="🎓"),
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
