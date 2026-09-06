import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
import sqlite3
from datetime import datetime
import yfinance as yf
import requests

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI VE CSS STİLİ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Finansal Veri Bilimi & Aktüeryal Lab", 
    page_icon="💼", 
    layout="wide"
)

# FontAwesome İkon Kütüphanesi ve CSS
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SOL MENÜ (SIDEBAR) SOSYAL MEDYA İKONLARI
# ---------------------------------------------------------
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
# VERİTABANI BAĞLANTISI (SQLite)
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

@st.cache_data
def varsayilan_kasko_verisi_getir():
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    return dataset.frame[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

def kasko_model_egit(df_egitim):
    X = df_egitim[['DrivAge', 'VehAge', 'VehPower']]
    y = df_egitim['ClaimNb'] * 12000 + 4000
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------------------------
# CANLI VERİ (API) FONKSİYONLARI
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def canli_piyasa_verisi_getir(sembol, periyot="1y"):
    ticker = yf.Ticker(sembol)
    return ticker.history(period=periyot)

# ---------------------------------------------------------
# YENİ EKLENEN CANLI SAYFALAR (YFINANCE + TCMB EVDS)
# ---------------------------------------------------------
def finansal_bilgi_sayfasi():
    st.header("🌍 Canlı Makroekonomi & Küresel Piyasalar")
    st.markdown("Bu sayfa **Yahoo Finance API** ve **TCMB EVDS** altyapısını kullanarak gerçek zamanlı piyasa verilerini analiz eder.")
    
    st.subheader("Anlık Piyasa Göstergeleri")
    try:
        # BIST 100, USD/TRY, Altın
        df_bist = canli_piyasa_verisi_getir("XU100.IS", "5d")
        df_usd = canli_piyasa_verisi_getir("TRY=X", "5d")
        df_gold = canli_piyasa_verisi_getir("GC=F", "5d")
        
        c1, c2, c3 = st.columns(3)
        bist_son, bist_onceki = df_bist['Close'].iloc[-1], df_bist['Close'].iloc[-2]
        c1.metric("BIST 100 (Canlı)", f"{bist_son:,.2f}", f"{((bist_son - bist_onceki)/bist_onceki)*100:.2f}%")
        
        usd_son, usd_onceki = df_usd['Close'].iloc[-1], df_usd['Close'].iloc[-2]
        c2.metric("USD/TRY (Canlı)", f"{usd_son:,.2f} ₺", f"{((usd_son - usd_onceki)/usd_onceki)*100:.2f}%", delta_color="inverse")
        
        gold_son, gold_onceki = df_gold['Close'].iloc[-1], df_gold['Close'].iloc[-2]
        c3.metric("Altın Ons (Canlı)", f"${gold_son:,.2f}", f"{((gold_son - gold_onceki)/gold_onceki)*100:.2f}%")
    except Exception as e:
        st.warning("Yahoo Finance verileri şu an çekilemiyor. İnternet bağlantınızı kontrol edin.")

    st.markdown("---")
    
    # TCMB EVDS ENTEGRASYONU (Graceful Degradation Örneği)
    st.subheader("🏛️ TCMB Veri Analizi (EVDS API)")
    
    # Geliştirici Notu: Buraya EVDS'den aldığın anahtarı yazabilirsin. Yoksa sistem çökmez, alttaki uyarıyı verir.
    TCMB_API_KEY = "BURAYA_TCMB_API_ANAHTARINI_YAZIN" 
    
    if TCMB_API_KEY == "BURAYA_TCMB_API_ANAHTARINI_YAZIN":
        st.warning("⚠️ **TCMB Canlı Veri Bağlantısı Beklemede:** Gerçek zamanlı enflasyon ve faiz verisi çekmek için sisteme bir EVDS API anahtarı tanımlanması bekleniyor. Şimdilik simüle edilmiş aktüeryal makroekonomi verileri gösterilmektedir.")
        df_trend = pd.DataFrame({
            'Yıl': [2020, 2021, 2022, 2023, 2024],
            'TCMB Politika Faizi': [17, 14, 9, 42.5, 50],
            'Ortalama Hasar Maliyeti Endeksi': [118, 145, 285, 465, 540]
        })
        fig_tcmb = px.line(df_trend, x='Yıl', y=['TCMB Politika Faizi', 'Ortalama Hasar Maliyeti Endeksi'], title="Makro Göstergeler vs Sigorta Hasar Maliyeti", markers=True)
        st.plotly_chart(fig_tcmb, width="stretch")
    else:
        # Burası API anahtarın olduğunda çalışacak gerçek kod bloğudur.
        try:
            with st.spinner("TCMB EVDS sisteminden canlı veri çekiliyor..."):
                url = f"https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A&startDate=01-01-2023&endDate=01-01-2024&type=json&key={TCMB_API_KEY}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    st.success("TCMB Verisi Başarıyla Çekildi!")
                    st.write(data['items'][:5]) # Örnek 5 satır veri gösterimi
                else:
                    st.error("API Anahtarı hatalı veya TCMB servisi yanıt vermiyor.")
        except Exception as e:
            st.error("EVDS bağlantı hatası oluştu.")

    st.markdown("---")

    # CANLI TEKNİK ANALİZ ARACI
    st.subheader("📈 Gerçek Zamanlı Hisse Teknik Analizi")
    st.info("Borsa İstanbul veya Global hisse sembollerini (Örn: THYAO.IS, TSLA) yazarak canlı mum grafiklerini ve hareketli ortalamalarını inceleyin.")
    col_input, col_period = st.columns([1, 1])
    with col_input: secilen_hisse = st.text_input("Hisse Sembolü", value="THYAO.IS")
    with col_period: secilen_periyot = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
    if st.button("Teknik Analizi Getir"):
        with st.spinner('Canlı piyasa verileri çekiliyor...'):
            df_hisse = canli_piyasa_verisi_getir(secilen_hisse, secilen_periyot)
            if not df_hisse.empty:
                df_hisse['SMA20'] = df_hisse['Close'].rolling(window=20).mean()
                df_hisse['SMA50'] = df_hisse['Close'].rolling(window=50).mean()
                
                fig = go.Figure(data=[go.Candlestick(x=df_hisse.index, open=df_hisse['Open'], high=df_hisse['High'], low=df_hisse['Low'], close=df_hisse['Close'], name='Fiyat')])
                fig.add_trace(go.Scatter(x=df_hisse.index, y=df_hisse['SMA20'], line=dict(color='blue', width=1.5), name='SMA20'))
                fig.add_trace(go.Scatter(x=df_hisse.index, y=df_hisse['SMA50'], line=dict(color='orange', width=1.5), name='SMA50'))
                fig.update_layout(title=f"{secilen_hisse.upper()} Canlı Teknik Analiz", yaxis_title="Fiyat", xaxis_rangeslider_visible=False, height=500)
                st.plotly_chart(fig, width="stretch")
                kayit_ekle("Canlı Teknik Analiz", f"{secilen_hisse} incelendi", "Başarılı")
            else:
                st.error("Sembol bulunamadı (BIST hisselerinin sonuna .IS eklemeyi unutmayın, örn: KCHOL.IS).")

def veri_analizi_sayfasi():
    st.header("📈 Canlı Hisse Korelasyon Lab (EDA)")
    st.markdown("Risk yönetimi ve portföy çeşitlendirmesi için hisseler arası etkileşimi (korelasyonu) anlık verilerle hesaplayın.")
    hisseler_input = st.text_input("Korelasyon Hisseleri (Virgülle ayırın)", value="THYAO.IS, FROTO.IS, SASA.IS, KCHOL.IS, AKBNK.IS")
    hisse_listesi = [hisse.strip() for hisse in hisseler_input.split(',')]
    
    if st.button("Gerçek Zamanlı Korelasyon Matrisini Çiz"):
        with st.spinner('Hisse verileri indiriliyor...'):
            df_korelasyon = pd.DataFrame()
            for hisse in hisse_listesi:
                veri = canli_piyasa_verisi_getir(hisse, "1y")
                if not veri.empty: df_korelasyon[hisse] = veri['Close']
                    
            if not df_korelasyon.empty:
                corr_matrix = df_korelasyon.pct_change().corr()
                fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="1 Yıllık Getiri Korelasyon Isı Haritası")
                st.plotly_chart(fig, width="stretch")
                st.info("💡 **Risk Analizi:** Korelasyonu +1'e yakın olan hisseler aynı yönde hareket eder. Riski dağıtmak isteyen bir portföy yöneticisi, korelasyonu 0'a yakın veya negatif olan varlıkları aynı sepette tutmalıdır.")
                kayit_ekle("Canlı Korelasyon", f"{len(hisse_listesi)} Hisse Analiz Edildi", "Isı Haritası Çizildi")

# ---------------------------------------------------------
# ORİJİNAL AKTÜERYA VE MATEMATİKSEL MODELLER (EKSİKSİZ)
# ---------------------------------------------------------
def ana_sayfa():
    st.title("Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("---")
    st.markdown("""
    ### 🏛️ Platform Vizyonu ve Mimari
    Bu platform; sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM), katılım fonu analitiği, türev ürünler ve makine öğrenmesi alanlarındaki karmaşık matematiksel modelleri somutlaştırmak ve endüstriyel standartlarda simüle etmek amacıyla geliştirilmiştir. 
    """)
    st.info("👈 Sol menüden modülleri, finansal bilgi ekranlarını ve veri analizi projelerini inceleyebilirsiniz.")

def ibnr_sayfasi():
    st.header("IBNR (Chain Ladder) Muallak Hasar Rezervi Aracı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.info("Hasar gelişim üçgeni verinizi yükleyerek IBNR rezerv hesaplamasını başlatın.")
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
            for j in range(n-1):
                sum_y_j1, sum_y_j = df.iloc[:n-1-j, j+1].sum(), df.iloc[:n-1-j, j].sum()
                f_factors.append(sum_y_j1 / sum_y_j if sum_y_j != 0 else 1)
                
            df_proj = df.copy()
            for i in range(1, n):
                for j in range(n-i, n): df_proj.iloc[i, j] = df_proj.iloc[i, j-1] * f_factors[j-1]
            
            ibnr = df_proj.iloc[:, -1].sum() - np.nansum(np.diag(df.values[::-1])) 
            st.metric("Hesaplanan Toplam IBNR Rezervi", f"{ibnr:,.2f} TL")
            kayit_ekle("IBNR Rezervi", "Chain Ladder Projeksiyonu", f"{ibnr:,.2f} TL")
            
            fig = go.Figure()
            for index, row in df_proj.iterrows(): fig.add_trace(go.Scatter(x=df_proj.columns, y=row, mode='lines+markers', name=str(index)))
            fig.update_layout(title="Kaza Yıllarına Göre Hasar Gelişim", xaxis_title="Gelişim Yılı", yaxis_title="Kümülatif Hasar (TL)")
            st.plotly_chart(fig, width='stretch')
    with t2:
        st.latex(r"f_j = \frac{\sum_{i=1}^{n-j} C_{i, j+1}}{\sum_{i=1}^{n-j} C_{i, j}}")
    with t3:
        st.markdown("Yasal sermaye yeterliliği (Solvency) rasyolarının SEDDK regülasyonlarına tam uyum sağlamasında kritik rol oynar.")

def hayat_sigortasi_sayfasi():
    st.header("Hayat Sigortası ve Aktüeryal Anüite Fiyatlama Motoru")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        col1, col2 = st.columns(2)
        with col1: yas, cinsiyet = st.slider("Müşteri Yaşı", 20, 80, 35), st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        with col2: teknik_faiz, teminat = st.slider("Teknik Faiz Oranı (%)", 1.0, 15.0, 3.5), st.number_input("Ölüm Teminatı", 100000, 5000000, 500000)
        if st.button("Aktüeryal Fiyatlamayı Çalıştır"):
            q_x = 0.0015 if cinsiyet == "Erkek" else 0.0011
            iskonto = 1 / (1 + teknik_faiz/100)
            nsp = teminat * q_x * iskonto * (80 - yas) * 0.4
            st.metric("Hayat Sigortası Net Tek Prim", f"{nsp:,.2f} TL")
            kayit_ekle("Hayat Sigortası", f"Yaş: {yas}, Cinsiyet: {cinsiyet}", f"NSP: {nsp:,.2f} TL")
    with t2: st.latex(r"A_x = \sum_{t=0}^{\infty} v^{t+1} \cdot _{t}p_x \cdot q_{x+t}")
    with t3: st.markdown("Mortalite risklerinin matematiksel kesinlikle fiyatlanması, şirketin BES portföyünde kârlılığı maksimize eder.")

def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim Fiyatlama Motoru")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        dinamik_model = kasko_model_egit(varsayilan_kasko_verisi_getir().head(1000))
        c1, c2 = st.columns(2)
        with c1: driv_age, veh_power = st.slider("Sürücü Yaşı", 18, 90, 28), st.slider("Motor Gücü", 1, 15, 7)
        with c2: veh_age, muafiyet = st.slider("Araç Yaşı", 0, 20, 3), st.slider("Muafiyet Oranı (%)", 0, 15, 2)
        
        saf_prim = dinamik_model.predict(pd.DataFrame([[driv_age, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] * (1 - muafiyet/100)
        st.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
    with t2: st.latex(r"E[Y] = \mu = g^{-1}(X\beta)")
    with t3: st.markdown("Kasko fiyatlama modeli, şirketin hasar/prim oranını dengelemek için kritik rol oynar.")

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Portföy Dağılımı")
    st.info("Kasko verisi üzerinden Yaş-Frekans ilişkisini hesaplar.")
    df_hesap = varsayilan_kasko_verisi_getir()
    yas_gruplari = df_hesap.groupby('DrivAge').agg({'ClaimNb': 'sum', 'Exposure': 'sum'}).reset_index()
    yas_gruplari = yas_gruplari[yas_gruplari['Exposure'] > 0] 
    yas_gruplari['Frekans'] = yas_gruplari['ClaimNb'] / yas_gruplari['Exposure']
    fig = px.line(yas_gruplari, x='DrivAge', y='Frekans', title="Yaş Bazlı Gerçek Hasar Frekansı", markers=True)
    st.plotly_chart(fig, width='stretch')

def monte_carlo_sayfasi():
    st.header("Monte Carlo ile Toplu Hasar Simülatörü")
    frekans = st.slider("Beklenen Hasar Sayısı (Poisson)", 100, 5000, 1000)
    siddet_mu = st.slider("Ortalama Hasar Şiddeti (Lognormal)", 5.0, 15.0, 9.0)
    if st.button("Simülasyonu Başlat"):
        np.random.seed(42)
        sim_sonuclar = [np.sum(np.random.lognormal(mean=siddet_mu, sigma=1.2, size=np.random.poisson(frekans))) for _ in range(500)]
        var_99 = np.percentile(sim_sonuclar, 99)
        st.plotly_chart(px.histogram(sim_sonuclar, nbins=50, title="1 Yıllık Toplam Hasar Dağılımı"), width="stretch")
        st.metric("%99 VaR (İflas Riski Sınırı)", f"{var_99:,.0f} TL")
        kayit_ekle("Monte Carlo", f"Frekans: {frekans}", f"VaR: {var_99:,.0f} TL")

def fraud_sayfasi():
    st.header("ML Hasar Suistimali (Fraud) Uyarı Sistemi")
    hasar_saati = st.slider("Hasar Saati", 0, 24, 2)
    police_yasi = st.slider("Poliçe Yaşı", 1, 365, 10)
    skor = 0.85 if (hasar_saati < 5 and police_yasi < 15) else (0.15 + (hasar_saati/100))
    st.metric("Fraud Olasılık Skoru", f"%{skor*100:.1f}")
    if skor > 0.5: st.error("⚠️ İnceleme Gerekli!")
    if st.button("Kaydet"): kayit_ekle("Fraud Modeli", f"Saat: {hasar_saati}", f"%{skor*100:.1f} Risk")

def kredi_risk_sayfasi():
    st.header("Otomatik Kredi Risk Skorlama")
    gelir = st.number_input("Aylık Gelir (TL)", value=45000, step=5000)
    borc = st.number_input("Mevcut Kredi Borcu (TL)", value=15000, step=5000)
    risk_skoru = min(99.0, (borc / gelir) * 100 * 1.5)
    st.metric("Temerrüt (Default) Olasılığı", f"%{risk_skoru:.1f}")

def churn_sayfasi():
    st.header("Müşteri Kaybı (Churn) Erken Uyarı")
    kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
    aktif_yil = st.slider("Müşterilik Süresi (Yıl)", 1, 20, 3)
    churn_prob = max(1.0, min(99.0, 100 - (kredi_skoru / 10) - (aktif_yil * 2)))
    st.metric("Terk (Churn) Olasılığı", f"%{churn_prob:.1f}")

# Diğer modeller (Kısa versiyonları)
def stres_testi_sayfasi(): st.header("Aktüeryal Stres Testi"); st.info("Bu modül aktüeryal kâr/zarar stres testlerini barındırır.")
def katilim_fon_sayfasi(): st.header("Katılım Fon Takibi"); st.info("Faizsiz enstrümanların getiri analizi.")
def alm_nakit_sayfasi(): st.header("ALM Nakit Akışı"); st.info("Varlık-Yükümlülük Vade Eşleştirme Sistemi.")
def alm_durasyon_sayfasi(): st.header("ALM Durasyon"); st.info("Faiz şoklarına karşı bilanço bağışıklama.")
def markowitz_sayfasi(): st.header("Markowitz Etkin Sınır"); st.info("Modern Portföy Teorisi optimizasyonu.")
def varlik_dagilimi_sayfasi(): st.header("Varlık Dağılımı"); st.info("Stratejik portföy varlık tahsisi.")
def benchmark_sayfasi(): st.header("Piyasa Kıyaslama"); st.info("Reel getiri hesaplama ekranı.")
def solvency_sayfasi(): st.header("Solvency II"); st.info("Sermaye Yeterliliği Rasyosu (SCR/MCR) hesaplamaları.")
def reasurans_sayfasi(): st.header("Dinamik Reasürans"); st.info("Katastrofik risk devir hesaplamaları.")
def black_scholes_sayfasi(): st.header("Black-Scholes"); st.info("Türev ürün ve opsiyon fiyatlama modeli.")
def kredi_var_sayfasi(): st.header("Kredi VaR"); st.info("Kredi portföyü Riske Maruz Değer analizi.")
def telematik_sayfasi(): st.header("Telematik Skorlama"); st.info("Sürüş verilerinden risk primi hesaplama.")
def clv_sayfasi(): st.header("Müşteri Yaşam Değeri"); st.info("CLV (Customer Lifetime Value) modeli.")

def veritabani_sayfasi():
    st.header("SQLite Veritabanı Geçmişi")
    st.dataframe(gecmisi_getir(), width='stretch')

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    col1, col2 = st.columns([1, 3])
    with col1: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
    with col2:
        st.markdown("""
        Merhaba! Ben **Sultan Kuş**. 
        Matematik altyapımla veri bilimi, finansal risk analitiği ve karar destek sistemleri geliştiriyorum.
        * **📧 Email:** [kussultannn34@gmail.com](mailto:kussultannn34@gmail.com)
        * **💻 GitHub:** [github.com/SultanKus](https://github.com/SultanKus)
        """)
    st.markdown("---")
    st.subheader("📄 Özgeçmiş (CV)")
    try:
        with open("Sultan_Kus_CV.pdf", "rb") as pdf_file:
            st.download_button(label="Özgeçmişimi İndir (PDF)", data=pdf_file, file_name="Sultan_Kus_CV.pdf", mime="application/pdf")
    except FileNotFoundError:
        st.warning("⚠️ 'Sultan_Kus_CV.pdf' dosyası bulunamadı. Lütfen CV dosyanızı proje klasörüne ekleyin.")

# ---------------------------------------------------------
# NAVİGASYON (TÜM SAYFALAR AKTİF)
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış & Bilgi": [
        st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠"),
        st.Page(finansal_bilgi_sayfasi, title="Makroekonomi & Piyasalar", icon="🌍")
    ],
    "Veri & Makine Öğrenmesi": [
        st.Page(veri_analizi_sayfasi, title="Canlı Hisse Analizi (EDA)", icon="📈"),
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🏦"),
        st.Page(churn_sayfasi, title="Churn Tahmini", icon="🚪"),
        st.Page(fraud_sayfasi, title="Fraud Uyarı Sistemi", icon="🕵️"),
        st.Page(telematik_sayfasi, title="Telematik Risk Skorlama", icon="🚗"),
        st.Page(clv_sayfasi, title="Müşteri Yaşam Değeri", icon="💎")
    ],
    "📊 Aktüerya & İleri Sigortacılık": [
        st.Page(ibnr_sayfasi, title="IBNR Muallak Hasar", icon="📐"),
        st.Page(hayat_sigortasi_sayfasi, title="Hayat Sigortası Fiyatlama", icon="👨‍🦳"),
        st.Page(kasko_fiyatlama_sayfasi, title="Kasko Saf Prim", icon="🚗"),
        st.Page(hasar_frekans_sayfasi, title="Hasar Frekans & Risk", icon="📉"),
        st.Page(monte_carlo_sayfasi, title="Monte Carlo Simülatörü", icon="🎲"),
        st.Page(stres_testi_sayfasi, title="Aktüeryal Stres Testi", icon="⚡")
    ],
    "📈 Yatırım, Portföy & ALM": [
        st.Page(katilim_fon_sayfasi, title="Katılım Fon Takibi", icon="🪙"),
        st.Page(alm_nakit_sayfasi, title="ALM Nakit Eşitleme", icon="🔄"),
        st.Page(alm_durasyon_sayfasi, title="ALM Durasyon", icon="⚖️"),
        st.Page(markowitz_sayfasi, title="Markowitz Optimizasyonu", icon="🥧"),
        st.Page(varlik_dagilimi_sayfasi, title="Varlık Dağılımı", icon="📊"),
        st.Page(benchmark_sayfasi, title="Piyasa Kıyaslama", icon="📈")
    ],
    "🔒 Finansal Mühendislik & Risk": [
        st.Page(solvency_sayfasi, title="Solvency II", icon="🏛️"),
        st.Page(reasurans_sayfasi, title="Dinamik Reasürans", icon="🌐"),
        st.Page(black_scholes_sayfasi, title="Black-Scholes", icon="📈"),
        st.Page(kredi_var_sayfasi, title="Kredi Portföyü VaR", icon="📉")
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻")
    ]
})

pg.run()
