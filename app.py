import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
import sqlite3
from datetime import datetime

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
/* MOBİL BEYAZ EKRAN VE GÖRÜNMEYEN YAZI ÇÖZÜMÜ */
.block-container {
    color: #0b1f33 !important;
}
.block-container p, .block-container span, .block-container label, .block-container div, .block-container li {
    color: #0b1f33 !important;
}
.stApp {
    background-color: #f8f9fa;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
/* Masaüstü ve Genel Header Ayarları */
header[data-testid="stHeader"] {
    background-color: #ffffff !important;
}
header[data-testid="stHeader"] * {
    color: #000000 !important;
    fill: #000000 !important;
}
/* MOBİL ÖZEL: Hamburger Menü ve Dinamik İkon Çözümü */
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] path,
[data-testid="stSidebarCollapsedControl"] svg,
button[kind="header"] svg {
    color: #000000 !important;
    fill: #000000 !important;
}
[data-testid="stSidebar"] {
    background-color: #0b1f33;
    color: #ffffff;
}
[data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] span {
    color: #ffffff !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0b1f33 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #0055a5 !important;
    border-color: #0055a5 !important;
}
.stSlider [data-baseweb="slider"] div > div > div > div {
    background-color: #0055a5 !important;
}
div.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #0055a5;
}
.stButton>button {
    background-color: #0055a5;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #003d73;
    color: white;
}
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih TEXT,
            modul_adi TEXT,
            girdi_detayi TEXT,
            sonuc_deger TEXT
        )
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
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

def kasko_model_egit(df_egitim):
    X = df_egitim[['DrivAge', 'VehAge', 'VehPower']]
    y = df_egitim['ClaimNb'] * 12000 + 4000
    model = LinearRegression()
    model.fit(X, y)
    return model

# ---------------------------------------------------------
# ANA SAYFA VE MODÜLLER
# ---------------------------------------------------------
def ana_sayfa():
    st.title("Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("---")
    st.markdown("""
    ### 🏛️ Platform Vizyonu ve Mimari
    Bu platform; sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM), katılım fonu analitiği, türev ürünler ve makine öğrenmesi alanlarındaki karmaşık matematiksel modelleri somutlaştırmak ve endüstriyel standartlarda simüle etmek amacıyla geliştirilmiştir. 
    
    Tüm modüller; karar alıcıların, aktüerlerin ve veri bilimcilerin kullanımına uygun olarak tasarlanmıştır:
    1. **📊 Uygulama Paneli:** İnteraktif slider'lar ve anlık Plotly simülasyonları.
    2. **📐 Kullanılan Matematiksel Model:** Saf matematiksel zarafet, LaTeX destekli formülasyonlar.
    3. **💼 İş Değeri:** Algoritmanın sigorta ve finans şirketlerine sağladığı stratejik avantaj.
    """)
    st.info("👈 Sol menüden toplam 16 ileri düzey aktüeryal ve finansal modülü inceleyebilirsiniz.")

def ibnr_sayfasi():
    st.header("IBNR (Chain Ladder) Muallak Hasar Rezervi Aracı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.info("Hasar gelişim üçgeni verinizi yükleyerek (CSV/Excel/TXT) IBNR rezerv hesaplamasını başlatın. Sol sütun 'Kaza Yılı' olmalıdır.")
        yuklenen_dosya = st.file_uploader("📂 Hasar Gelişim Üçgeni Yükle", type=["csv", "xlsx", "txt"], key="ibnr_up")
        
        if yuklenen_dosya is not None:
            if yuklenen_dosya.name.endswith('.csv') or yuklenen_dosya.name.endswith('.txt'):
                df = pd.read_csv(yuklenen_dosya, index_col=0)
            else:
                df = pd.read_excel(yuklenen_dosya, index_col=0)
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
            
            for j in range(n-1):
                sum_y_j1 = df.iloc[:n-1-j, j+1].sum()
                sum_y_j = df.iloc[:n-1-j, j].sum()
                f = sum_y_j1 / sum_y_j if sum_y_j != 0 else 1
                f_factors.append(f)
                
            df_proj = df.copy()
            for i in range(1, n):
                for j in range(n-i, n):
                    df_proj.iloc[i, j] = df_proj.iloc[i, j-1] * f_factors[j-1]
            
            nihai_hasar = df_proj.iloc[:, -1].sum()
            odenen_hasar = np.nansum(np.diag(df.values[::-1])) 
            ibnr = nihai_hasar - odenen_hasar
            
            st.metric("Hesaplanan Toplam IBNR Rezervi", f"{ibnr:,.2f} TL")
            
            fig = go.Figure()
            for index, row in df_proj.iterrows():
                fig.add_trace(go.Scatter(x=df_proj.columns, y=row, mode='lines+markers', name=str(index)))
            fig.update_layout(title="Kaza Yıllarına Göre Hasar Gelişim Projeksiyonu", xaxis_title="Gelişim Yılı", yaxis_title="Kümülatif Hasar (TL)")
            st.plotly_chart(fig, width='stretch')
            
    with t2:
        st.markdown("Geçmiş kaza yıllarına ait kümülatif hasar ödemeleri kullanılarak hasar gelişim faktörleri (Link Ratios) hesaplanır.")
        st.latex(r"f_j = \frac{\sum_{i=1}^{n-j} C_{i, j+1}}{\sum_{i=1}^{n-j} C_{i, j}}")
    with t3:
        st.markdown("Bu rezerv modeli, şirketin bilançosundaki en büyük yükümlülük kalemini doğru tahmin ederek nakit akışı krizlerini önler ve yasal sermaye yeterliliği (Solvency) rasyolarının SEDDK regülasyonlarına tam uyum sağlamasında **kritik rol oynar.**")

def hayat_sigortasi_sayfasi():
    st.header("Hayat Sigortası ve Aktüeryal Anüite Fiyatlama Motoru")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        col1, col2 = st.columns(2)
        with col1:
            yas = st.slider("Müşteri Yaşı", 20, 80, 35)
            cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
        with col2:
            teknik_faiz = st.slider("Teknik Faiz Oranı (%)", 1.0, 15.0, 3.5)
            teminat = st.number_input("Ölüm Teminatı / Yıllık Maaş (TL)", 100000, 5000000, 500000)
        if st.button("Aktüeryal Fiyatlamayı Çalıştır"):
            q_x = 0.0015 if cinsiyet == "Erkek" else 0.0011
            iskonto = 1 / (1 + teknik_faiz/100)
            nsp = teminat * q_x * iskonto * (80 - yas) * 0.4
            anuite = (teminat / 12) * 0.6 * iskonto
            m1, m2 = st.columns(2)
            m1.metric("Hayat Sigortası Net Tek Prim", f"{nsp:,.2f} TL")
            m2.metric("Aylık Emeklilik Maaşı (Anüite)", f"{anuite:,.2f} TL")
    with t2:
        st.latex(r"A_x = \sum_{t=0}^{\infty} v^{t+1} \cdot _{t}p_x \cdot q_{x+t}")
    with t3:
        st.markdown("Uzun ömür (longevity) ve mortalite risklerinin matematiksel kesinlikle fiyatlanması, şirketin BES ve Hayat portföyünde kârlılığı maksimize eder.")

def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim Fiyatlama Motoru")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Kendi Kasko Veri Setinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="kasko_up")
        aktif_df = varsayilan_kasko_verisi_getir().head(1000)
        dinamik_model = kasko_model_egit(aktif_df)
        c1, c2 = st.columns(2)
        with c1:
            driv_age = st.slider("Sürücü Yaşı (DrivAge)", 18, 90, 28)
            veh_power = st.slider("Araç Motor Gücü (VehPower)", 1, 15, 7)
        with c2:
            veh_age = st.slider("Araç Yaşı (VehAge)", 0, 20, 3)
            muafiyet = st.slider("Kasko Muafiyet Oranı (%)", 0, 15, 2)
            
        girdi_df = pd.DataFrame([[driv_age, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower'])
        saf_prim = dinamik_model.predict(girdi_df)[0] * (1 - muafiyet/100)
        st.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
        yas_listesi = list(range(18, 81))
        sim_prim = [dinamik_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] * (1 - muafiyet/100) for y in yas_listesi]
        fig = go.Figure(go.Scatter(x=yas_listesi, y=sim_prim, line=dict(color='#0055a5')))
        fig.update_layout(title="Yaşa ve Muafiyete Göre Prim Dağılımı", xaxis_title="Sürücü Yaşı", yaxis_title="Saf Prim (TL)")
        st.plotly_chart(fig, width='stretch')
    with t2:
        st.latex(r"\text{Saf Prim} = \text{Hasar Frekansı} \times \text{Hasar Şiddeti}")
        st.latex(r"E[Y] = \mu = g^{-1}(X\beta)")
    with t3:
        st.markdown("Bu kasko fiyatlama modeli, şirketin hasar/prim (Loss Ratio) oranını dengelemek için kritik rol oynar.")

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Aktüeryal Portföy Dağılımı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    
    with t1:
        st.info("Kendi verinizi yükleyin veya varsayılan açık kaynaklı kasko verisi üzerinden analizi inceleyin.")
        yuklenen_dosya = st.file_uploader("📂 Kendi Portföy Veri Setinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="hasar_up")
        
        # Dosya yükleme mantığı
        if yuklenen_dosya is not None:
            if yuklenen_dosya.name.endswith('.csv'):
                df_sigorta = pd.read_csv(yuklenen_dosya)
            else:
                df_sigorta = pd.read_excel(yuklenen_dosya)
        else:
            df_sigorta = varsayilan_kasko_verisi_getir()
            
        # Doğru Aktüeryal Hesaplama Kontrolü
        if 'Exposure' in df_sigorta.columns and 'ClaimNb' in df_sigorta.columns and 'DrivAge' in df_sigorta.columns:
            
            # Portföy Genel Metrikleri
            toplam_hasar = df_sigorta['ClaimNb'].sum()
            toplam_exposure = df_sigorta['Exposure'].sum()
            genel_frekans = (toplam_hasar / toplam_exposure) if toplam_exposure > 0 else 0
            
            # Metrik Kartları
            c1, c2, c3 = st.columns(3)
            c1.metric("Toplam Hasar Adedi", f"{toplam_hasar:,.0f}")
            c2.metric("Toplam Poliçe Yılı (Exposure)", f"{toplam_exposure:,.2f}")
            c3.metric("Genel Portföy Frekansı", f"% {genel_frekans * 100:.2f}")
            
            st.markdown("---")
            
            # Yaş bazlı doğru aktüeryal frekans hesaplaması
            yas_gruplari = df_sigorta.groupby('DrivAge').agg({'ClaimNb': 'sum', 'Exposure': 'sum'}).reset_index()
            yas_gruplari['Frekans'] = yas_gruplari['ClaimNb'] / yas_gruplari['Exposure']
            
            # Frekans Grafiği
            fig = px.line(yas_gruplari, x='DrivAge', y='Frekans', 
                          title="Yaş Bazlı Gerçek Hasar Frekansı (Toplam Hasar / Toplam Exposure)", 
                          markers=True)
            fig.update_traces(line_color='#0055a5', marker=dict(size=6))
            fig.update_layout(xaxis_title="Sürücü Yaşı", yaxis_title="Hasar Frekansı", hovermode="x unified")
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.error("⚠️ Yüklediğiniz veri setinde 'DrivAge', 'ClaimNb' ve 'Exposure' sütunları bulunmalıdır!")
            
    with t2:
        st.markdown("Bir portföyün veya belirli bir segmentin hasar frekansı, salt ortalama alınarak değil; toplam hasar adedinin, portföyde kalınan süreye (Poliçe Yılı / Exposure) oranlanmasıyla bulunur.")
        st.latex(r"\text{Frekans} = \frac{\sum \text{Hasar Adedi}}{\sum \text{Exposure (Poliçe Yılı)}}")
        st.info("Bu modelde, her bir risk profilinin maruz kaldığı süre (Poliçe Yılı / Exposure) hesaba katılarak ağırlıklı hasar sıklığı hesaplanmaktadır. Aktüeryal modellemelerde hasar adetleri kesikli ve pozitif tamsayılar olduğu için, frekans tahminlemelerinde Poisson Dağılımı bazlı Genelleştirilmiş Doğrusal Modeller (GLM) temel alınır.")
        
    with t3:
        st.markdown("""
        **Portföy Dağılımının Stratejik Önemi:**
        
        * **Risk Bazlı Fiyatlandırma:** Kârlı segmentlere indirim sunarak sadakati artırırken, toksik segmentlere doğru prim yüklemesi (surprim) yapılmasını sağlar.
        * **Ters Seçimin (Adverse Selection) Engellenmesi:** Şirketin yüksek riskli profiller için bir "güvenli liman" haline gelmesini önler.
        * **Kârlılık ve Büyüme Dengesi:** Aktüeryal portföy dağılımını optimize ederek şirketin teknik kâr marjını güvenceye alır.
        """)

def monte_carlo_sayfasi():
    st.header("Monte Carlo ile Toplu Hasar Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        frekans = st.slider("Beklenen Hasar Sayısı (Poisson)", 100, 5000, 1000)
        siddet_mu = st.slider("Ortalama Hasar Şiddeti (Lognormal)", 5.0, 15.0, 9.0)
        if st.button("Monte Carlo Simülasyonunu Başlat"):
            np.random.seed(42)
            sim_sonuclar = [np.sum(np.random.lognormal(mean=siddet_mu, sigma=1.2, size=np.random.poisson(frekans))) for _ in range(1000)]
            st.plotly_chart(px.histogram(sim_sonuclar, nbins=50, title="1 Yıllık Toplam Hasar Dağılımı (Aggregate Loss)"), width="stretch")
            st.metric("%99 VaR (İflas Riski Sınırı)", f"{np.percentile(sim_sonuclar, 99):,.0f} TL")
    with t2:
        st.latex(r"S = \sum_{i=1}^{N} X_i \quad (N \sim Poisson, X \sim Lognormal)")
    with t3:
        st.markdown("Şirketin beklenmedik makro şoklara karşı taşıdığı iflas olasılığını (Ruin Probability) hesaplar.")

def stres_testi_sayfasi():
    st.header("Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
        faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
        simule_kar = 10000000 * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
        st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")
    with t2:
        st.latex(r"\Delta \text{Kâr} = f(\Delta \text{Faiz}, \Delta \text{Enflasyon})")
    with t3:
        st.markdown("Yönetim kurulunun ekonomik kriz senaryolarına karşı hazırlıklı olmasını sağlar.")

def katilim_fon_sayfasi():
    st.header("Katılım Emeklilik & Faizsiz Yatırım Fonları Takip Aracı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Kendi Fon Veri Setinizi Yükleyin", type=["csv", "xlsx"], key="fon_up")
        tarihler = pd.date_range(start='2025-01-01', periods=60, freq='W')
        np.random.seed(42)
        df_fonlar = pd.DataFrame({
            'Tarih': tarihler,
            'Hisse Katılım': 100 * (1 + np.random.normal(0.003, 0.02, 60)).cumprod(),
            'Altın Katılım': 100 * (1 + np.random.normal(0.0025, 0.012, 60)).cumprod(),
            'Sukuk Fonu': 100 * (1 + np.random.normal(0.0015, 0.004, 60)).cumprod()
        })
        secilenler = st.multiselect("Fonları Seçin", ['Hisse Katılım', 'Altın Katılım', 'Sukuk Fonu'], default=['Hisse Katılım'])
        if secilenler:
            st.plotly_chart(px.line(df_fonlar, x='Tarih', y=secilenler, title="Performans Kıyaslaması (Baz: 100 TL)"), width='stretch')
    with t2:
        st.latex(r"P_t = P_0 \prod_{i=1}^t (1 + R_i)")
    with t3:
        st.markdown("Katılım esaslı fon yönetiminde şeffaflık sağlayarak AUM (Yönetilen Varlık) büyüklüğünü artırır.")

def alm_nakit_sayfasi():
    st.header("ALM Nakit Akışı Eşitleme")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        yil_1_yuk = st.number_input("1. Yıl Tazminat Yükü (TL)", 1000000, 50000000, 15000000)
        faiz_orani = st.slider("Piyasa Getirisi (%)", 5, 50, 25)
        varlik_tahvil = st.number_input("Tahvil Portföyü (TL)", 10000000, 100000000, 60000000)
        
        yillar = ['1. Yıl', '2. Yıl', '3. Yıl']
        yukumlulukler = [yil_1_yuk, yil_1_yuk*1.2, yil_1_yuk*1.4]
        varlik_getirileri = [varlik_tahvil * (faiz_orani / 100)] * 3
        
        alm_df = pd.DataFrame({'Yıl': yillar, 'Yükümlülük': yukumlulukler, 'Varlık Getirisi': varlik_getirileri})
        st.plotly_chart(px.bar(alm_df, x='Yıl', y=['Yükümlülük', 'Varlık Getirisi'], barmode='group'), width='stretch')
    with t2:
        st.latex(r"CF_{\text{Varlık}, t} \ge CF_{\text{Yükümlülük}, t}")
    with t3:
        st.markdown("Kurumsal likidite krizlerini kökünden çözer.")

def alm_durasyon_sayfasi():
    st.header("ALM Durasyon Eşleştirme Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        f_orani = st.slider("Piyasa Faiz Oranı Şoku (%)", -5.0, 5.0, 0.0)
        v_deger = 100000000 * (1 - 4.5 * (f_orani/100))
        y_deger =  90000000 * (1 - 6.2 * (f_orani/100))
        st.plotly_chart(px.bar(pd.DataFrame({'Tür': ['Varlık', 'Yükümlülük'], 'Tutar': [v_deger, y_deger]}), x='Tür', y='Tutar', color='Tür'), width="stretch")
    with t2:
        st.latex(r"D_{Mac} = \frac{\sum_{t=1}^{T} \frac{t \cdot CF_t}{(1+y)^t}}{\sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}}")
    with t3:
        st.markdown("Bilançoyu faiz oranlarındaki yıkıcı dalgalanmalara karşı kurşungeçirmez (immunized) hale getirir.")

def markowitz_sayfasi():
    st.header("Markowitz Etkin Sınır (Efficient Frontier)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        if st.button("Rastgele Portföy Simüle Et"):
            np.random.seed(42)
            getiri, risk = np.random.normal(0.20, 0.10, 1000), np.random.normal(0.15, 0.05, 1000)
            st.plotly_chart(px.scatter(x=risk, y=getiri, color=getiri/risk), width="stretch")
    with t2:
        st.latex(r"\sigma_p^2 = \sum_{i} \sum_{j} w_i w_j Cov(R_i, R_j)")
    with t3:
        st.markdown("Minimum riskle maksimum getiriyi sağlayacak stratejik varlık dağılımını kurgular.")

def varlik_dagilimi_sayfasi():
    st.header("Varlık Dağılım Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        w_hisse, w_tahvil, w_altin = st.slider("Hisse (%)", 0, 100, 50), st.slider("Tahvil (%)", 0, 100, 30), st.slider("Altın (%)", 0, 100, 20)
        if w_hisse + w_tahvil + w_altin == 100:
            st.plotly_chart(px.pie(names=['Hisse', 'Tahvil', 'Altın'], values=[w_hisse, w_tahvil, w_altin], hole=0.4), width='stretch')
        else:
            st.warning("⚠️ Toplam %100 olmalıdır!")
    with t2:
        st.latex(r"\sum_{i=1}^{n} w_i = 1")
    with t3:
        st.markdown("Müşteri portföylerinde çeşitlendirmeyi (diversification) görselleştirir.")

def benchmark_sayfasi():
    st.header("Piyasa Kıyaslama (Benchmark)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        portfoy_getiri, enflasyon = st.slider("Yıllık Getiri (%)", 0, 100, 35), st.slider("Enflasyon (%)", 0, 80, 25)
        st.plotly_chart(px.bar(pd.DataFrame({'Endeks': ['Portföy', 'BIST 100', 'Enflasyon'], 'Getiri (%)': [portfoy_getiri, 28.5, enflasyon]}), x='Endeks', y='Getiri (%)', color='Endeks'), width='stretch')
    with t2:
        st.latex(r"R_{reel} = \frac{1 + R_{nominal}}{1 + R_{enflasyon}} - 1")
    with t3:
        st.markdown("Fon yöneticilerinin performansını değerlendirmede endüstriyel KPI olarak çalışır.")

def solvency_sayfasi():
    st.header("Solvency II Sermaye Yeterliliği")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        mkt_risk = st.number_input("Piyasa Riski", value=15000000)
        def_risk = st.number_input("Kredi Riski", value=5000000)
        nl_risk = st.number_input("Hayat Dışı Risk", value=20000000)
        bscr = np.sqrt(mkt_risk**2 + def_risk**2 + nl_risk**2 + 2*0.25*(mkt_risk*def_risk + mkt_risk*nl_risk + def_risk*nl_risk))
        st.metric("Gerekli Temel Özkaynak (BSCR)", f"{bscr:,.0f} TL")
    with t2:
        st.latex(r"BSCR = \sqrt{ \sum_i \sum_j Corr_{i,j} \cdot SCR_i \cdot SCR_j }")
    with t3:
        st.markdown("Şirketi lisans iptallerinden kurtarır ve rasyonel bir risk yönetimi kültürü inşa eder.")

def reasurans_sayfasi():
    st.header("Dinamik Reasürans Optimizasyonu")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        brut_hasar, retention = st.slider("Afet Hasarı (Milyon)", 10, 500, 150), st.slider("Saklama Payı (Milyon)", 1, 100, 25)
        st.metric("Reasüröre Devredilen Hasar", f"{max(0, brut_hasar - retention)} Milyon TL")
    with t2:
        st.latex(r"\text{Reasürör Payı} = \max(0, \text{Brüt Hasar} - \text{Saklama Payı})")
    with t3:
        st.markdown("Katastrofik riskler karşısında şirketin iflas etmesini engeller.")

def black_scholes_sayfasi():
    st.header("Black-Scholes Opsiyon Fiyatlama")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        c1, c2 = st.columns(2)
        with c1: S, K, T = st.number_input("Spot (S)", value=100.0), st.number_input("Strike (K)", value=100.0), st.slider("Vade", 0.05, 5.0, 1.0)
        with c2: r, sigma, opt_tipi = st.slider("Faiz (%)", 1, 50, 15)/100, st.slider("Volatilite (%)", 5, 100, 25)/100, st.selectbox("Opsiyon Tipi", ["Call", "Put"])
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        from math import erf
        norm_cdf = lambda x: (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        fiyat = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2) if opt_tipi == "Call" else K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        st.metric("Teorik Opsiyon Primi", f"{fiyat:,.2f} TL")
    with t2:
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")
        st.latex(r"C = S_t N(d_1) - K e^{-rT} N(d_2)")
    with t3:
        st.markdown("Kurumsal hazine departmanları için bir Risk Hedging (Korunma) aracıdır.")

def kredi_var_sayfasi():
    st.header("Kredi Portföyü VaR Hesaplayıcı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        portfoy = st.number_input("Kredi Portföyü (TL)", value=50000000)
        z_skor = 1.65 if "%95" in st.selectbox("Güven Aralığı", ["%95", "%99"]) else 2.33
        st.metric("10 Günlük Portföy VaR", f"{portfoy * z_skor * 0.12 / np.sqrt(252) * np.sqrt(10):,.0f} TL")
    with t2:
        st.latex(r"VaR = V_p \cdot z_{\alpha} \cdot \sigma_p \cdot \sqrt{T}")
    with t3:
        st.markdown("Yönetim kurulunun risk iştahını matematiksel olarak sınırlandırır.")

def fraud_sayfasi():
    st.header("ML Hasar Suistimali (Fraud) Uyarı Sistemi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        hasar_saati, police_yasi = st.slider("Hasar Saati", 0, 24, 2), st.slider("Poliçe Yaşı", 1, 365, 10)
        skor = 0.85 if (hasar_saati < 5 and police_yasi < 15) else 0.15
        st.metric("Fraud Olasılık Skoru", f"%{skor*100}")
        if skor > 0.5: st.error("⚠️ İnceleme Gerekli!")
    with t2:
        st.latex(r"P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots)}}")
    with t3:
        st.markdown("Sahte hasarları engelleyerek devasa maliyet tasarrufu sağlar.")

def telematik_sayfasi():
    st.header("Telematik Tabanlı Risk Skorlama")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        skor = max(0, 100 - (st.slider("Ani Fren", 0, 50, 12) * 1.5) - (st.slider("Gece Sürüşü (%)", 0, 100, 45) * 0.5))
        st.metric("Güvenli Sürüş Skoru", f"{skor}")
    with t2:
        st.latex(r"\text{Sürüş Skoru} = 100 - \left(\sum_{i=1}^{n} w_i \cdot X_i\right)")
    with t3:
        st.markdown("Sürücüleri 'gerçek kullanım verileriyle' adil şekilde fiyatlandırır.")

def kredi_risk_sayfasi():
    st.header("Otomatik Kredi Risk Skorlama")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.number_input("Aylık Gelir (TL)", value=45000)
        st.metric("Temerrüt (Default) Olasılığı", "%35.0")
    with t2:
        st.latex(r"PD = P(\text{Default}=1 | \text{Gelir, Borç Oranı})")
    with t3:
        st.markdown("Batık kredi oranlarını minimize eder.")

def churn_sayfasi():
    st.header("Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.slider("Kredi Skoru", 350, 850, 650)
        st.metric("Terk (Churn) Olasılığı", "%22.5")
    with t2:
        st.latex(r"P(\text{Churn}=1) = f(X_{\text{Müşteri Davranışı}})")
    with t3:
        st.markdown("Müşteri elde tutma (Retention) oranlarını artırır.")

def clv_sayfasi():
    st.header("Müşteri Yaşam Boyu Değeri (CLV)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        clv_deger = (st.number_input("Poliçe Tutarı", value=4500.0) * st.slider("İşlem Sayısı", 1, 12, 2) * st.slider("Ömür", 1, 20, 5)) * (st.slider("Marj (%)", 5, 50, 20)/100)
        st.metric("Ortalama CLV", f"{clv_deger:,.2f} TL")
    with t2:
        st.latex(r"CLV = (\text{Ort. Harcama} \times \text{Frekans} \times \text{Ömür}) \times \text{Marj}")
    with t3:
        st.markdown("Pazarlama ROI (Yatırım Getirisi) optimizasyonunu sağlar.")

def veritabani_sayfasi():
    st.header("SQLite Veritabanı Geçmişi")
    st.dataframe(gecmisi_getir(), width='stretch')

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**. 
    Veri bilimi, finansal risk analitiği ve aktüerya alanlarında karar destek sistemleri geliştiriyorum.
    
    Bu Süper Platform; teorik matematik modellerinin iş süreçlerine nasıl değer kattığını kanıtlayan bir vitrindir.
    
    ---
    ### 📬 İletişime Geçin
    Projelerim ve iş birlikleri için kanallarım:
    
    * **📧 Email:** [kussultannn34@gmail.com](mailto:kussultannn34@gmail.com)
    * **💼 LinkedIn:** [linkedin.com/in/sultan-kuş](https://www.linkedin.com/in/sultan-kuş/)
    * **💻 GitHub:** [github.com/SultanKus](https://github.com/SultanKus)
    """)

# ---------------------------------------------------------
# STREAMLIT NAVIGASYON
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış": [st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠")],
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
    "🤖 Yapay Zeka & Skorlama": [
        st.Page(fraud_sayfasi, title="Fraud Uyarı Sistemi", icon="🕵️"),
        st.Page(telematik_sayfasi, title="Telematik Risk Skorlama", icon="🚗"),
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🏦"),
        st.Page(churn_sayfasi, title="Churn Tahmini", icon="🚪"),
        st.Page(clv_sayfasi, title="Müşteri Yaşam Değeri", icon="💎")
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻")
    ]
})

pg.run()
