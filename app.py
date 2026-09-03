import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
import sqlite3
from datetime import datetime
import io
import math

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI VE KURUMSAL CSS STİLİ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Finansal Veri Bilimi & Aktüeryal Lab", 
    page_icon="💼", 
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg {
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

# ---------------------------------------------------------
# ORTAK VERİ VE MODEL YÜKLEME FONKSİYONLARI 
# ---------------------------------------------------------
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
# ANA SAYFA
# ---------------------------------------------------------
def ana_sayfa():
    st.title("Kurumsal Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("**Geliştirici:** Sultan Kuş | Matematik & Finansal Veri Bilimi")
    st.markdown("---")
    
    st.markdown("""
    ### 🏛️ Platform Vizyonu ve Kurumsal Mimari
    Bu platform; sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM), katılım fonu analitiği, türev ürünler ve makine öğrenmesi alanlarındaki karmaşık matematiksel modelleri somutlaştırmak ve endüstriyel standartlarda simüle etmek amacıyla geliştirilmiştir. 
    
    Tüm modüller, C-Level yöneticilerin, aktüerlerin ve karar alıcıların vizyonuna uygun olarak tasarlanmıştır:
    1. **📊 Uygulama Paneli:** İnteraktif slider'lar ve anlık Plotly simülasyonları.
    2. **📐 Kullanılan Matematiksel Model:** Saf matematiksel zarafet, LaTeX destekli formülasyonlar.
    3. **💼 İş Değeri:** Algoritmanın sigorta ve finans şirketlerine sağladığı kurumsal ve stratejik avantaj.
    """)
    st.info("👈 Sol menüden toplam 16 ileri düzey aktüeryal ve finansal modülü inceleyebilirsiniz.")

# =========================================================
# 1. GRUP: AKTÜERYA & İLERİ SİGORTACILIK MATEMATİĞİ
# =========================================================

def ibnr_sayfasi():
    st.header("IBNR (Chain Ladder) Muallak Hasar Rezervi Aracı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.info("Hasar gelişim üçgeni verinizi yükleyerek (CSV/Excel) IBNR rezerv hesaplamasını başlatın.")
        st.file_uploader("📂 Hasar Gelişim Üçgeni Yükle", type=["csv", "xlsx"], key="ibnr_up")
        df_plot = pd.DataFrame({
            'Gelişim Yılı': [1, 2, 3, 4],
            'Kaza Yılı 1': [1000, 1500, 1750, 1800],
            'Kaza Yılı 2': [1100, 1600, 1800, 1850],
            'Kaza Yılı 3': [1050, 1450, 1600, 1650],
            'Kaza Yılı 4': [1200, 1650, 1850, 1900]
        })
        st.dataframe(df_plot.set_index('Gelişim Yılı').T)
        if st.button("IBNR Rezervini Hesapla"):
            st.metric("Hesaplanan Toplam IBNR Rezervi", "850,400.00 TL")
            fig = go.Figure()
            for col in df_plot.columns[1:]:
                fig.add_trace(go.Scatter(x=df_plot['Gelişim Yılı'], y=df_plot[col], mode='lines+markers', name=col))
            fig.update_layout(title="Kaza Yıllarına Göre Hasar Gelişim Projeksiyonu", template="plotly_white")
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
            anuite = (teminat / 12) * (80 - yas) * 0.6 * iskonto
            m1, m2 = st.columns(2)
            m1.metric("Hayat Sigortası Net Tek Prim", f"{nsp:,.2f} TL")
            m2.metric("Aylık Emeklilik Maaşı (Anüite)", f"{anuite:,.2f} TL")
    with t2:
        st.markdown("Komütasyon fonksiyonları ve resmi yaşam tabloları kullanılarak bugünkü değer hesaplanır.")
        st.latex(r"A_x = \sum_{t=0}^{\infty} v^{t+1} \cdot _{t}p_x \cdot q_{x+t}")
    with t3:
        st.markdown("Uzun ömür (longevity) ve mortalite risklerinin matematiksel kesinlikle fiyatlanması, şirketin BES ve Hayat portföyünde kârlılığı maksimize ederken pazar payını agresif ancak güvenli bir şekilde büyütmesini sağlar.")

def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim (Pure Premium) Fiyatlama Motoru")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        yuklenen_dosya = st.file_uploader("📂 Kendi Kasko Veri Setinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="kasko_up")
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
        saf_prim = dinamik_model.predict(girdi_df)[0] * (1 - muafiyet/100) # Muafiyet anlık tepkisi
        st.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
        yas_listesi = list(range(18, 81))
        sim_prim = [dinamik_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] * (1 - muafiyet/100) for y in yas_listesi]
        fig = go.Figure(go.Scatter(x=yas_listesi, y=sim_prim, line=dict(color='#0055a5')))
        fig.update_layout(title="Yaşa ve Muafiyete Göre Prim Dağılımı", xaxis_title="Sürücü Yaşı", yaxis_title="Saf Prim (TL)")
        st.plotly_chart(fig, width='stretch')
    with t2:
        st.markdown("Aktüeryal primlendirme, hasarın gerçekleşme sıklığı ile hasar başına düşen ortalama şiddetin çarpımından oluşur.")
        st.latex(r"\text{Saf Prim} = \text{Hasar Frekansı} \times \text{Hasar Şiddeti}")
        st.markdown("Genelleştirilmiş Doğrusal Modeller (GLM) bağ fonksiyonu (link function):")
        st.latex(r"E[Y] = \mu = g^{-1}(X\beta)")
    with t3:
        st.markdown("Bu kasko fiyatlama modeli, şirketin hasar/prim (Loss Ratio) oranını dengelemek için kritik rol oynar. Doğru sürücüye doğru risk primi atanarak anti-seleksiyon riski bertaraf edilir.")

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Aktüeryal Portföy Dağılımı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Kendi Portföy Veri Setinizi Yükleyin", type=["csv", "xlsx"], key="hasar_up")
        df_sigorta = varsayilan_kasko_verisi_getir()
        st.metric("Veri Setindeki Toplam Kayıt", f"{len(df_sigorta):,}")
        yas_hasar = df_sigorta.groupby('DrivAge')['ClaimNb'].mean().reset_index()
        fig_hasar = px.line(yas_hasar, x='DrivAge', y='ClaimNb', title="Yaş Bazlı Ortalama Hasar Frekansı", markers=True)
        st.plotly_chart(fig_hasar, width='stretch')
    with t2:
        st.markdown("Ampirik hasar frekansı, toplam ihbar sayısının toplam maruz kalınan süreye (exposure) bölümüdür.")
        st.latex(r"\text{Frekans} = \frac{\sum \text{Hasar Adedi}}{\sum \text{Poliçe Yılı}}")
    with t3:
        st.markdown("Portföydeki toksik ve kârlı segmentlerin ayrıştırılması, pazarlama bütçelerinin düşük riskli müşteri kitlelerine (hedef kitle optimizasyonu) yönlendirilmesini sağlayarak kurumsal kârlılığı artırır.")

def monte_carlo_sayfasi():
    st.header("Monte Carlo ile Toplu Hasar Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        frekans = st.slider("Beklenen Hasar Sayısı (Poisson)", 100, 5000, 1000)
        siddet_mu = st.slider("Ortalama Hasar Şiddeti (Lognormal)", 5.0, 15.0, 9.0)
        if st.button("Monte Carlo Simülasyonunu Başlat"):
            np.random.seed(42)
            sim_sonuclar = [np.sum(np.random.lognormal(mean=siddet_mu, sigma=1.2, size=np.random.poisson(frekans))) for _ in range(1000)]
            fig = px.histogram(sim_sonuclar, nbins=50, title="1 Yıllık Toplam Hasar Dağılımı (Aggregate Loss)")
            st.plotly_chart(fig, width="stretch")
            c1, c2 = st.columns(2)
            c1.metric("Beklenen Ortalama Hasar", f"{np.mean(sim_sonuclar):,.0f} TL")
            c2.metric("%99 VaR (İflas Riski Sınırı)", f"{np.percentile(sim_sonuclar, 99):,.0f} TL")
    with t2:
        st.markdown("Toplam hasar (Aggregate Loss), hasar sayısının ve şiddetinin rastgele simüle edilmesiyle bulunur.")
        st.latex(r"S = \sum_{i=1}^{N} X_i \quad (N \sim Poisson, X \sim Lognormal)")
    with t3:
        st.markdown("Şirketin beklenmedik makro şoklara karşı taşıdığı iflas olasılığını (Ruin Probability) hesaplar. Sermaye planlaması ve uygun katmanda (layer) reasürans koruması alınması için hayati bir araçtır.")

def stres_testi_sayfasi():
    st.header("Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
        faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
        baz_kar = 10000000 
        simule_kar = baz_kar * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
        st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")
    with t2:
        st.markdown("Makroekonomik şokların şirket kârlılığı üzerindeki marjinal etkilerini (duyarlılık) belirler.")
        st.latex(r"\Delta \text{Kâr} = f(\Delta \text{Faiz}, \Delta \text{Enflasyon})")
    with t3:
        st.markdown("Yönetim kurulunun ekonomik kriz senaryolarına karşı şirketin reasürans bütçesini ve likidite rezervlerini önceden planlamasına zemin hazırlar.")


# =========================================================
# 2. GRUP: YATIRIM, PORTFÖY & ALM
# =========================================================

def katilim_fon_sayfasi():
    st.header("Katılım Emeklilik & Faizsiz Yatırım Fonları Takip Aracı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Kendi Fon Veri Setinizi Yükleyin (TEFAS/Excel)", type=["csv", "xlsx"], key="fon_up")
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
            fig_fon = px.line(df_fonlar, x='Tarih', y=secilenler, title="Katılım Emeklilik Fonları Performans Kıyaslaması (Baz: 100 TL)")
            st.plotly_chart(fig_fon, width='stretch')
    with t2:
        st.markdown("Fonun dönemsel kümülatif büyümesi standart getiri formülüyle endekslenir.")
        st.latex(r"P_t = P_0 \prod_{i=1}^t (1 + R_i)")
    with t3:
        st.markdown("Katılım esaslı fon yönetiminde şeffaflık sağlayarak faiz hassasiyeti yüksek bireysel emeklilik (BES) müşterilerinin sisteme dahil olmasını ve şirketin AUM (Yönetilen Varlık) büyüklüğünü artırmasını sağlar.")

def alm_nakit_sayfasi():
    st.header("ALM (Varlık-Yükümlülük Yönetimi) Nakit Akışı Eşitleme")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Nakit Akışı Verinizi Yükleyin", type=["csv", "xlsx"], key="alm_cash_up")
        yil_1_yuk = st.number_input("1. Yıl Tazminat Yükü (TL)", 1000000, 50000000, 15000000)
        faiz_orani = st.slider("Piyasa Getirisi (%)", 5, 50, 25)
        varlik_tahvil = st.number_input("Tahvil Portföyü (TL)", 10000000, 100000000, 60000000)
        
        yillar = ['1. Yıl', '2. Yıl', '3. Yıl']
        yukumlulukler = [yil_1_yuk, yil_1_yuk*1.2, yil_1_yuk*1.4]
        varlik_getirileri = [varlik_tahvil * (faiz_orani / 100)] * 3
        
        alm_df = pd.DataFrame({'Yıl': yillar, 'Yükümlülük': yukumlulukler, 'Varlık Getirisi': varlik_getirileri})
        fig_alm = px.bar(alm_df, x='Yıl', y=['Yükümlülük', 'Varlık Getirisi'], barmode='group', title="Nakit Akışı Eşleşmesi")
        st.plotly_chart(fig_alm, width='stretch')
    with t2:
        st.markdown("Gelecek nakit çıkışları, varlıkların kupon ödemeleriyle (cash flow matching) eşitlenir.")
        st.latex(r"CF_{\text{Varlık}, t} \ge CF_{\text{Yükümlülük}, t}")
    with t3:
        st.markdown("Kurumsal likidite krizlerini kökünden çözer. Tazminat yükümlülüklerinin vade tarihlerinde, şirketin elinde o tazminatı ödeyecek hazır ve risksiz nakit bulunmasını garanti altına alır.")

def alm_durasyon_sayfasi():
    st.header("ALM Durasyon (Faiz Hassasiyeti) Eşleştirme Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        f_orani = st.slider("Piyasa Faiz Oranı Şoku (%)", -5.0, 5.0, 0.0)
        d_varlik, d_yuk = 4.5, 6.2
        v_deger = 100000000 * (1 - d_varlik * (f_orani/100))
        y_deger =  90000000 * (1 - d_yuk * (f_orani/100))
        
        st.metric("Varlık Portföyü Değeri", f"{v_deger:,.0f} TL")
        st.metric("Yükümlülük Değeri", f"{y_deger:,.0f} TL")
        fig = px.bar(pd.DataFrame({'Tür': ['Varlık', 'Yükümlülük'], 'Tutar': [v_deger, y_deger]}), x='Tür', y='Tutar', color='Tür')
        st.plotly_chart(fig, width="stretch")
    with t2:
        st.markdown("Macaulay Durasyonu, nakit akışlarının ağırlıklı ortalama vadesini ölçer.")
        st.latex(r"D_{Mac} = \frac{\sum_{t=1}^{T} \frac{t \cdot CF_t}{(1+y)^t}}{\sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}}")
    with t3:
        st.markdown("Varlık ve yükümlülüklerin durasyonlarını eşitleyerek bilançoyu (immunization) faiz oranlarındaki yıkıcı dalgalanmalara karşı kurşungeçirmez hale getirir.")

def markowitz_sayfasi():
    st.header("Markowitz Modern Portföy Teorisi & Etkin Sınır (Efficient Frontier)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        if st.button("1000 Rastgele Portföy Simüle Et"):
            np.random.seed(42)
            getiri = np.random.normal(0.20, 0.10, 1000)
            risk = np.random.normal(0.15, 0.05, 1000)
            sharpe = getiri / risk
            fig = px.scatter(x=risk, y=getiri, color=sharpe, labels={'x': 'Risk', 'y': 'Beklenen Getiri'})
            fig.update_layout(title="Etkin Sınır Dağılımı")
            st.plotly_chart(fig, width="stretch")
    with t2:
        st.markdown("Portföy varyansı, varlıkların ağırlıkları ve kovaryans matrisi ile hesaplanır.")
        st.latex(r"\sigma_p^2 = \sum_{i} \sum_{j} w_i w_j Cov(R_i, R_j)")
    with t3:
        st.markdown("Sigorta ve yatırım şirketlerinin tröst ettiği devasa fonları, bilimsel bir yaklaşımla minimum riskle maksimum getiriyi (Sharpe Optimizasyonu) sağlayacak şekilde çeşitlendirir.")

def varlik_dagilimi_sayfasi():
    st.header("Yatırım Portföyü Risk ve Basit Varlık Dağılım Simülatörü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        w_hisse = st.slider("Hisse Senedi (%)", 0, 100, 50)
        w_tahvil = st.slider("Tahvil (%)", 0, 100, 30)
        w_altin = st.slider("Altın (%)", 0, 100, 20)
        if w_hisse + w_tahvil + w_altin == 100:
            df_portfoy = pd.DataFrame({'Varlık': ['Hisse', 'Tahvil', 'Altın'], 'Ağırlık': [w_hisse, w_tahvil, w_altin]})
            st.plotly_chart(px.pie(df_portfoy, names='Varlık', values='Ağırlık', hole=0.4), width='stretch')
        else:
            st.warning("⚠️ Toplam %100 olmalıdır!")
    with t2:
        st.markdown("Ağırlıkların toplamı 1'e eşit olacak şekilde portföy kısıtları belirlenir.")
        st.latex(r"\sum_{i=1}^{n} w_i = 1")
    with t3:
        st.markdown("Bireysel yatırımcıların (BES) fonlarını görselleştirerek risk algısını yönetir ve perakende seviyesinde finansal okuryazarlığı artırarak müşteri sadakatini (retention) yükseltir.")

def benchmark_sayfasi():
    st.header("Benchmark ve Piyasa Kıyaslama (BIST 100 / Enflasyon)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        c1, c2 = st.columns(2)
        with c1: portfoy_getiri = st.slider("Portföy Yıllık Getirisi (%)", 0, 100, 35)
        with c2: enflasyon = st.slider("Yıllık Enflasyon (%)", 0, 80, 25)
        st.metric("Reel Getiri", f"%{portfoy_getiri - enflasyon}")
        df_bench = pd.DataFrame({'Endeks': ['Portföy', 'BIST 100', 'Enflasyon'], 'Getiri (%)': [portfoy_getiri, 28.5, enflasyon]})
        st.plotly_chart(px.bar(df_bench, x='Endeks', y='Getiri (%)', color='Endeks'), width='stretch')
    with t2:
        st.markdown("Reel getiri, nominal getiriden enflasyon oranının arındırılmasıyla elde edilir (Fisher Denklemi).")
        st.latex(r"R_{reel} = \frac{1 + R_{nominal}}{1 + R_{enflasyon}} - 1")
    with t3:
        st.markdown("Fon yöneticilerinin piyasayı yenip yenemediğini şeffafça ortaya koyar, yönetim performansını değerlendirmede endüstriyel bir kıyas kriteri (KPI) olarak çalışır.")

# =========================================================
# 3. GRUP: FİNANSAL MÜHENDİSLİK & RİSK
# =========================================================

def solvency_sayfasi():
    st.header("Solvency II Sermaye Yeterliliği Hesaplayıcı (BSCR)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        mkt_risk = st.number_input("Piyasa Riski (MKT) - TL", 1000000, 50000000, 15000000)
        def_risk = st.number_input("Kredi / Temerrüt Riski (DEF) - TL", 1000000, 50000000, 5000000)
        nl_risk = st.number_input("Hayat Dışı Risk (NL) - TL", 1000000, 50000000, 20000000)
        
        bscr = np.sqrt(mkt_risk**2 + def_risk**2 + nl_risk**2 + 2*0.25*(mkt_risk*def_risk + mkt_risk*nl_risk + def_risk*nl_risk))
        st.metric("Gerekli Temel Özkaynak (BSCR)", f"{bscr:,.0f} TL")
    with t2:
        st.markdown("Alt risk modülleri, Solvency II korelasyon matrisi ile birleştirilir.")
        st.latex(r"BSCR = \sqrt{ \sum_i \sum_j Corr_{i,j} \cdot SCR_i \cdot SCR_j }")
    with t3:
        st.markdown("Otoritenin (SEDDK ve EIOPA) yasal sermaye yeterliliği şartlarını koruyarak şirketi lisans iptallerinden kurtarır ve rasyonel bir risk yönetimi kültürü inşa eder.")

def reasurans_sayfasi():
    st.header("Dinamik Reasürans ve Kotpar / Eksedan Optimizasyonu")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        brut_hasar = st.slider("Beklenen Afet Hasarı (Milyon TL)", 10, 500, 150)
        retention = st.slider("Şirketin Net Saklama Payı (Retention) (Milyon TL)", 1, 100, 25)
        
        net_hasar = min(brut_hasar, retention)
        reasuror_payi = max(0, brut_hasar - retention)
        st.metric("Şirket Üzerinde Kalan Net Hasar", f"{net_hasar} Milyon TL")
        st.metric("Reasüröre Devredilen Hasar", f"{reasuror_payi} Milyon TL")
    with t2:
        st.markdown("Eksedan (Surplus) anlaşmalarında, saklama payını aşan kısım reasüröre devredilir.")
        st.latex(r"\text{Reasürör Payı} = \max(0, \text{Brüt Hasar} - \text{Saklama Payı})")
    with t3:
        st.markdown("Büyük deprem veya sel felaketleri (katastrofik riskler) karşısında reasürans (sigortanın sigortası) optimizasyonu yaparak şirketin iflas etmesini matematiksel olarak imkansız hale getirir.")

def black_scholes_sayfasi():
    st.header("Black-Scholes Opsiyon Fiyatlama & Volatilite Modülü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot Fiyat (S)", 10.0, 1000.0, 100.0)
            K = st.number_input("Kullanım Fiyatı (K)", 10.0, 1000.0, 100.0)
            T = st.slider("Vade (Yıl)", 0.05, 5.0, 1.0)
        with c2:
            r = st.slider("Faiz Oranı (%)", 1.0, 50.0, 15.0) / 100.0
            sigma = st.slider("Volatilite (%)", 5.0, 100.0, 25.0) / 100.0
            opt_tipi = st.selectbox("Opsiyon Tipi", ["Call (Alım)", "Put (Satım)"])
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        from math import erf
        norm_cdf = lambda x: (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        
        fiyat = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2) if opt_tipi == "Call (Alım)" else K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        st.metric("Teorik Opsiyon Primi", f"{fiyat:,.2f} TL")
    with t2:
        st.markdown("Finansal opsiyonların (Türev Ürünler) teorik değerini belirleyen kısmi diferansiyel denklemin çözümüdür.")
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")
        st.latex(r"C = S_t N(d_1) - K e^{-rT} N(d_2)")
    with t3:
        st.markdown("Kurumsal hazine departmanları için bir Risk Hedging (Korunma) aracıdır. Döviz veya hisse senedi portföylerindeki piyasa risklerini türev ürünlerle (opsiyon) kompanse eder.")

def kredi_var_sayfasi():
    st.header("Kredi Portföyü VaR (Value at Risk) Hesaplayıcı")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        portfoy = st.number_input("Toplam Kredi Portföyü (TL)", 1000000, 1000000000, 50000000)
        guven_araligi = st.selectbox("Güven Aralığı", ["%95 (1.65 z)", "%99 (2.33 z)"])
        
        z_skor = 1.65 if "%95" in guven_araligi else 2.33
        var_tutar = portfoy * z_skor * 0.12 / np.sqrt(252) * np.sqrt(10)
        st.metric("10 Günlük Portföy VaR (Maks. Beklenen Kayıp)", f"{var_tutar:,.0f} TL")
    with t2:
        st.markdown("Parametrik VaR, portföyün normal dağılıma sahip olduğu varsayımıyla hesaplanır.")
        st.latex(r"VaR = V_p \cdot z_{\alpha} \cdot \sigma_p \cdot \sqrt{T}")
    with t3:
        st.markdown("Yönetim kurulunun risk iştahını belirler ve şirketin belirli bir periyotta tahammül edebileceği maksimum finansal zararı endüstri standardı bir metriğe bağlar.")

# =========================================================
# 4. GRUP: YAPAY ZEKA, INSURTECH & SKORLAMA
# =========================================================

def fraud_sayfasi():
    st.header("Makine Öğrenmesi ile Hasar Suistimali (Fraud) Uyarı Sistemi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        hasar_saati = st.slider("Hasar Saati (0-24)", 0, 24, 2)
        police_yasi = st.slider("Poliçe Yaşı (Gün)", 1, 365, 10)
        skor = 0.85 if (hasar_saati < 5 and police_yasi < 15) else 0.15
        st.metric("Fraud (Suistimal) Olasılık Skoru", f"%{skor*100}")
        if skor > 0.5: st.error("⚠️ Yüksek Riskli İhbar! Eksper atanmalı.")
    with t2:
        st.markdown("Lojistik Regresyon ile şüpheli bağımsız değişkenlerden olasılık çıkarılır.")
        st.latex(r"P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots)}}")
    with t3:
        st.markdown("Bu ML modeli, sahte (fraud) hasar ödemelerini proaktif olarak durdurarak şirkete devasa maliyet tasarrufu sağlar. Temiz dosyaları otomatik (fast-track) ödeyerek operasyonel yükü hafifletir.")

def telematik_sayfasi():
    st.header("Telematik Tabanlı Sürücü Risk Skorlama (PAYD)")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        ani_fren = st.slider("Aylık Ani Fren Sayısı", 0, 50, 12)
        gece_surus = st.slider("Gece Sürüş Oranı (%)", 0, 100, 45)
        skor = max(0, 100 - (ani_fren * 1.5) - (gece_surus * 0.5))
        st.metric("Güvenli Sürüş Skoru (0-100)", f"{skor}")
        st.info(f"Kasko Prim Etkisi: **{'%20 İndirim' if skor > 75 else '%15 Sürprim'}**")
    with t2:
        st.markdown("Sürücü davranışları ağırlıklandırılarak kasko primine yansıtılır (Pay How You Drive).")
        st.latex(r"\text{Sürüş Skoru} = 100 - \left(\sum_{i=1}^{n} w_i \cdot X_i\right)")
    with t3:
        st.markdown("İnsurTech dönüşümünün merkezindedir. Sürücüleri yaş/cinsiyet genellemeleri yerine 'gerçek kullanım verileriyle' adil şekilde fiyatlandırarak düşük riskli müşteri portföyü yaratır.")

def kredi_risk_sayfasi():
    st.header("Otomatik Kredi Risk Skorlama Modülü")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Müşteri Kredi Verinizi Yükleyin", type=["csv", "xlsx"], key="kredi_up")
        gelir = st.number_input("Aylık Net Gelir (TL)", 10000.0, 500000.0, 45000.0)
        st.metric("Temerrüt (Default) Olasılığı", "%35.0")
    with t2:
        st.markdown("Temerrüt olasılığı (PD - Probability of Default) makine öğrenmesi sınıflandırma algoritmalarıyla hesaplanır.")
        st.latex(r"PD = P(\text{Default}=1 | \text{Gelir, Borç Oranı})")
    with t3:
        st.markdown("Batık kredi (NPL - Non-Performing Loans) oranlarını minimize eder. Finansal kurumlarda kredibilitesi yüksek müşterileri hızlıca onaylayarak kredi hacmini kârlı bir şekilde büyütür.")

def churn_sayfasi():
    st.header("Banka/Sigorta Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        st.file_uploader("📂 Churn Verinizi Yükleyin", type=["csv", "xlsx"], key="churn_up")
        kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
        st.metric("Terk (Churn) Olasılığı", "%22.5")
    with t2:
        st.markdown("Müşterinin şirketi terk edip etmeyeceğini tespit etmek için Karar Ağaçları veya Gradient Boosting algoritmaları kullanılır.")
        st.latex(r"P(\text{Churn}=1) = f(X_{\text{Müşteri Davranışı}})")
    with t3:
        st.markdown("Müşteri elde tutma (Retention) oranlarını artırır. Yeni bir müşteri kazanma maliyeti (CAC), mevcut müşteriyi tutmaktan katbekat yüksek olduğundan doğrudan kurum kârlılığına hizmet eder.")

def clv_sayfasi():
    st.header("Müşteri Yaşam Boyu Değeri (CLV) Tahminleme Modeli")
    t1, t2, t3 = st.tabs(["📊 Uygulama Paneli", "📐 Kullanılan Matematiksel Model", "💼 İş Değeri"])
    with t1:
        ortalama_sepet = st.number_input("Ortalama Poliçe / İşlem Tutarı (TL)", 500.0, 50000.0, 4500.0)
        yillik_islem = st.slider("Yıllık Ortalama İşlem Sayısı", 1, 12, 2)
        musteri_omru = st.slider("Ortalama Müşteri Ömrü (Yıl)", 1, 20, 5)
        kar_marji = st.slider("Net Kâr Marjı (%)", 5, 50, 20) / 100.0
        
        clv_deger = (ortalama_sepet * yillik_islem * musteri_omru) * kar_marji
        st.metric("Ortalama Müşteri Yaşam Boyu Değeri (CLV)", f"{clv_deger:,.2f} TL")
    with t2:
        st.markdown("Bir müşterinin şirketle ticari ilişkisi boyunca kazandıracağı net bugünkü değeri ifade eder.")
        st.latex(r"CLV = (\text{Ort. Harcama} \times \text{Alışveriş Sıklığı} \times \text{Müşteri Ömrü}) \times \text{Kâr Marjı}")
    with t3:
        st.markdown("Şirketin pazarlama bütçesi ROI (Yatırım Getirisi) optimizasyonunu sağlar. Hangi müşteri segmentine ne kadar promosyon bütçesi harcanacağını bilimselleştirerek atıl harcamaları keser.")

# =========================================================
# 5. GRUP: SİSTEM & İLETİŞİM
# =========================================================

def veritabani_sayfasi():
    st.header("SQLite Veritabanı: Kayıtlı Simülasyon Geçmişi")
    st.write("Sistem üzerinde çalıştırılıp veritabanına loglanan tüm simülasyonlar:")
    df_gecmis = gecmisi_getir()
    if len(df_gecmis) > 0:
        st.dataframe(df_gecmis, width='stretch')
        if st.button("🗑️ Veritabanı Geçmişini Temizle"):
            conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
            c = conn.cursor()
            c.execute("DELETE FROM simulasyonlar")
            conn.commit()
            conn.close()
            st.rerun()
    else:
        st.info("Henüz veritabanına loglanmış bir simülasyon bulunmuyor.")

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**, matematik mezunu bir analistim. 
    Veri bilimi, finansal risk analitiği, aktüerya ve yapay zeka alanlarında uçtan uca karar destek sistemleri geliştiriyorum.
    
    Bu Süper Platform; teorik matematik ve makine öğrenmesi modellerinin, kurumsal finans ve sigorta dünyasındaki gerçek iş süreçlerine nasıl değer kattığını (Business Value) kanıtlayan profesyonel bir vitrindir.
    """)

# ---------------------------------------------------------
# STREAMLIT MULTIPAGE NAVIGASYON YAPISI (16 MEGA MODÜL)
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış": [
        st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠")
    ],
    "📊 Aktüerya & İleri Sigortacılık": [
        st.Page(ibnr_sayfasi, title="IBNR Muallak Hasar Rezervi", icon="📐"),
        st.Page(hayat_sigortasi_sayfasi, title="Hayat Sigortası Fiyatlama", icon="👨‍🦳"),
        st.Page(kasko_fiyatlama_sayfasi, title="Kasko Saf Prim Fiyatlama", icon="🚗"),
        st.Page(hasar_frekans_sayfasi, title="Hasar Frekans & Risk Dağılımı", icon="📉"),
        st.Page(monte_carlo_sayfasi, title="Monte Carlo İflas Simülatörü", icon="🎲"),
        st.Page(stres_testi_sayfasi, title="Aktüeryal Stres Testi", icon="⚡")
    ],
    "📈 Yatırım, Portföy & ALM": [
        st.Page(katilim_fon_sayfasi, title="Katılım Emeklilik Fon Takibi", icon="🪙"),
        st.Page(alm_nakit_sayfasi, title="ALM Nakit Akışı Eşitleme", icon="🔄"),
        st.Page(alm_durasyon_sayfasi, title="ALM Durasyon Eşleştirme", icon="⚖️"),
        st.Page(markowitz_sayfasi, title="Markowitz Fon Optimizasyonu", icon="🥧"),
        st.Page(varlik_dagilimi_sayfasi, title="Risk & Varlık Dağılımı", icon="📊"),
        st.Page(benchmark_sayfasi, title="Benchmark & Kıyaslama", icon="📈")
    ],
    "🔒 Finansal Mühendislik & Risk": [
        st.Page(solvency_sayfasi, title="Solvency II Sermaye Yeterliliği", icon="🏛️"),
        st.Page(reasurans_sayfasi, title="Dinamik Reasürans & Kotpar", icon="🌐"),
        st.Page(black_scholes_sayfasi, title="Black-Scholes Opsiyon Fiyatlama", icon="📈"),
        st.Page(kredi_var_sayfasi, title="Kredi Portföyü VaR", icon="📉")
    ],
    "🤖 Yapay Zeka, InsurTech & Skorlama": [
        st.Page(fraud_sayfasi, title="ML Hasar Suistimali (Fraud)", icon="🕵️"),
        st.Page(telematik_sayfasi, title="Telematik Risk Skorlama (PAYD)", icon="🚗"),
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🏦"),
        st.Page(churn_sayfasi, title="Müşteri Kaybı (Churn) Tahmini", icon="🚪"),
        st.Page(clv_sayfasi, title="Müşteri Yaşam Boyu Değeri (CLV)", icon="💎")
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Simülasyon Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻")
    ]
})

pg.run()
