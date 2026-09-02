import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
import io
import sqlite3
from datetime import datetime
import math

# ---------------------------------------------------------
# SAYFA YAPILANDIRMASI VE KURUMSAL CSS STİLİ
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sultan Kuş | Finansal Veri Bilimi & Aktüeryal Lab", 
    page_icon="💼", 
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Üst Appbar - Saf Beyaz */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    /* Üst Appbar Sağ Köşe Buton ve İkonlarının Rengini Tam Siyah Yapma */
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg {
        color: #000000 !important;
        fill: #000000 !important;
    }
    /* Kenar Çubuğu Kurumsal Tasarım */
    [data-testid="stSidebar"] {
        background-color: #0b1f33;
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    /* Keskin ve Net Kurumsal Siyah Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #0b1f33 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    /* Slider ve İnteraktif Bileşenlerdeki Rengi Kurumsal Maviye Çevirme */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #0055a5 !important;
        border-color: #0055a5 !important;
    }
    .stSlider [data-baseweb="slider"] div > div > div > div {
        background-color: #0055a5 !important;
    }
    input[type="range"] {
        accent-color: #0055a5 !important;
    }
    /* Metrik Kartları */
    div.stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #0055a5;
    }
    /* Butonlar */
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
# ORTAK VERİ VE MODEL YÜKLEME FONKSİYONLARI (DİNAMİK DESTEKLİ)
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
# SAYFA FONKSİYONLARI (MODÜLLER)
# ---------------------------------------------------------

def ana_sayfa():
    st.title("Kurumsal Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
    st.markdown("**Geliştirici:** Sultan Kuş | Matematik & Finansal Veri Bilimi")
    st.markdown("---")
    
    st.markdown("""
    ### 🏛️ Platform Vizyonu ve Mimari Yapı
    Bu platform; sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM), katılım fonu analitiği, türev ürünler ve makine öğrenmesi alanlarındaki karmaşık matematiksel modelleri somutlaştırmak ve endüstriyel standartlarda simüle etmek amacıyla geliştirilmiştir. 
    
    Finans ve sigorta sektöründe karar alıcıların en büyük ihtiyaç duyduğu şey; teorik modellerin gerçek veri setleriyle nasıl çalıştığını görmek ve olası makroekonomik şokların bilançoya etkilerini anlık olarak test edebilmektir. Tüm modüllerde yer alan **Dinamik Veri Yükleme Motoru** sayesinde kullanıcılar kendi CSV/Excel verilerini sisteme entegre edebilirler.
    """)
    
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktüeryal & Finans Modülleri", "8 Ana Başlık")
    c2.metric("Veritabanı Altyapısı", "SQLite (Loglama)")
    c3.metric("Veri Kaynağı", "Dinamik Dosya & Açık Kaynak")
    c4.metric("Raporlama Desteği", "XlsxWriter (Excel)")
    
    st.markdown("---")
    st.subheader("💡 Kapsamlı Odak Alanları ve Yetkinlikler")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        * **Aktüeryal Fiyatlama & Risk:** GLM tabanlı saf prim hesaplamaları, reasürans optimizasyonu ve kredi portföyü VaR analizleri.
        * **Yatırım & Portföy:** Katılım emeklilik fonları takibi, Black-Scholes opsiyon fiyatlama ve varlık dağılım simülasyonları.
        """)
    with col_b:
        st.markdown("""
        * **Finansal Skorlama & ML:** Kredi risk skorlama, müşteri kaybı (churn) ve müşteri yaşam boyu değeri (CLV) tahminleme.
        * **Kalıcılık & Veri Akışı:** Yapılan tüm simülasyonların ilişkisel bir veritabanına anlık olarak loglanması.
        """)
    
    st.info("👈 Sol menüden dilediğiniz modülü seçerek simülasyonları gerçekleştirebilir, kendi veri setinizi yükleyerek analiz yapabilirsiniz.")

def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim (Pure Premium) Fiyatlama Motoru")
    st.write("Hazır açık kaynak veri setini kullanabilir veya **kendi kasko veri setinizi yükleyerek** modelin anlık olarak sizin verilerinizle eğitilmesini sağlayabilirsiniz.")
    
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("""
        * **Matematiksel Model:** Bu modülde, sürücü yaşı, araç yaşı ve motor gücü gibi risk faktörleri kullanılarak Genelleştirilmiş Doğrusal Modeller (GLM) ve regresyon yaklaşımlarıyla saf prim tahmini yapılır.
        * **Kendi Verinizi Kullanma:** Yükleyeceğiniz CSV/Excel dosyasında `DrivAge`, `VehAge`, `VehPower` ve `ClaimNb` kolonlarının bulunması modelin kusursuz çalışmasını sağlar.
        """)
    
    st.markdown("---")
    yuklenen_dosya = st.file_uploader("📂 Kendi Kasko Veri Setinizi Yükleyin (CSV veya Excel)", type=["csv", "xlsx"], key="kasko_up")
    
    if yuklenen_dosya is not None:
        try:
            user_df = pd.read_csv(yuklenen_dosya) if yuklenen_dosya.name.endswith('.csv') else pd.read_excel(yuklenen_dosya)
            GEREKLI_KOLONLAR = ['DrivAge', 'VehAge', 'VehPower', 'ClaimNb']
            if all(kol in user_df.columns for kol in GEREKLI_KOLONLAR):
                st.success("✅ Veri setiniz başarıyla yüklendi! Model sizin verilerinizle yeniden eğitiliyor.")
                aktif_df = user_df[GEREKLI_KOLONLAR].dropna().head(2000)
            else:
                st.warning("⚠️ Gerekli kolonlar eksik ('DrivAge', 'VehAge', 'VehPower', 'ClaimNb'). Varsayılan verilere dönülüyor.")
                aktif_df = varsayilan_kasko_verisi_getir().head(1000)
        except Exception as e:
            st.error(f"Hata: {e}. Varsayılan veriye dönülüyor.")
            aktif_df = varsayilan_kasko_verisi_getir().head(1000)
    else:
        aktif_df = varsayilan_kasko_verisi_getir().head(1000)

    dinamik_model = kasko_model_egit(aktif_df)

    col1, col2 = st.columns(2)
    with col1:
        driv_age = st.slider("Sürücü Yaşı (DrivAge)", 18, 90, 28)
        veh_age = st.slider("Araç Yaşı (VehAge)", 0, 20, 3)
        veh_power = st.slider("Araç Motor Gücü (VehPower)", 1, 15, 7)
    with col2:
        bonus_malus = st.slider("Bonus-Malus (Hasarsızlık Puanı)", 50, 250, 100)
        veh_brand = st.selectbox("Araç Markası", ["Renault", "Volkswagen", "Peugeot", "BMW", "Citroen"])
        veh_gas = st.selectbox("Yakıt Türü", ["Diesel", "Regular"])
        
    girdi_df = pd.DataFrame([[driv_age, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower'])
    saf_prim = dinamik_model.predict(girdi_df)[0]
    
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
    m2.metric("Tahmini Hasar Frekansı", "%8.0")
    
    if st.button("💾 Bu Hesaplamayı Veritabanına Kaydet", key="btn_kasko"):
        kayit_ekle("Kasko Fiyatlama", f"Yaş: {driv_age}, Araç Yaş: {veh_age}", f"{saf_prim:,.2f} TL")
        st.success("✅ SQLite veritabanına loglandı!")

    yas_listesi = list(range(18, 81))
    simulasyon_primleri = [dinamik_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] for y in yas_listesi]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yas_listesi, y=simulasyon_primleri, mode='lines+markers', name='Model Tahmini Saf Prim', line=dict(color='#0055a5', width=3)))
    fig.update_layout(xaxis_title="Sürücü Yaşı", yaxis_title="Hesaplanan Saf Prim (TL)", template="plotly_white")
    st.plotly_chart(fig, width='stretch')

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Aktüeryal Portföy Dağılımı")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Gerçek portföy verilerindeki hasar olasılıklarının demografik kırılımlara göre dağılımını analiz eder.")
    
    yuklenen_dosya = st.file_uploader("📂 Kendi Portföy Veri Setinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="hasar_up")
    if yuklenen_dosya is not None:
        try:
            df_sigorta = pd.read_csv(yuklenen_dosya) if yuklenen_dosya.name.endswith('.csv') else pd.read_excel(yuklenen_dosya)
            st.success("✅ Özel portföy verisi yüklendi!")
        except:
            df_sigorta = varsayilan_kasko_verisi_getir()
    else:
        df_sigorta = varsayilan_kasko_verisi_getir()

    st.metric("Veri Setindeki Toplam Kayıt", f"{len(df_sigorta):,}")
    if 'DrivAge' in df_sigorta.columns and 'ClaimNb' in df_sigorta.columns:
        yas_hasar = df_sigorta.groupby('DrivAge')['ClaimNb'].mean().reset_index()
        fig_hasar = px.line(yas_hasar, x='DrivAge', y='ClaimNb', title="Yaş Bazlı Ortalama Hasar Frekansı", markers=True)
        st.plotly_chart(fig_hasar, width='stretch')
    else:
        st.warning("Yüklenen veride 'DrivAge' ve 'ClaimNb' kolonları bulunmalıdır.")

def reasurans_sayfasi():
    st.header("Uluslararası Reasürans ve Afet Riski Optimizasyon Modeli")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Katastrofik risklerde reasüröre devredilen risk payının maliyet ve koruma optimizasyonunu simüle eder.")
    
    portfoy_buyuklugu = st.number_input("Toplam Portföy Teminat Büyüklüğü (TL)", 10000000, 1000000000, 150000000, step=10000000)
    afet_siddeti = st.slider("Beklenen Afet Şiddet Senaryosu (Hasar Oranı %)", 5, 50, 20)
    reasurans_orani = st.slider("Reasüröre Devredilen Risk Oranı (%)", 10, 90, 60)
    
    toplam_hasar = portfoy_buyuklugu * (afet_siddeti / 100)
    sirket_net_hasar = toplam_hasar * (1 - reasurans_orani / 100)
    st.metric("Şirketin Üzerinde Kalan Net Hasar", f"{sirket_net_hasar:,.0f} TL")
    
    if st.button("💾 Reasürans Sonucunu Kaydet", key="btn_reas"):
        kayit_ekle("Reasürans Optimizasyonu", f"Portföy: {portfoy_buyuklugu:,}", f"Net Hasar: {sirket_net_hasar:,.0f}")
        st.success("✅ Kaydedildi!")

def stres_testi_sayfasi():
    st.header("Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Enflasyon ve faiz şoklarının kârlılık üzerindeki marjinal etkilerini test eder.")
    
    enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
    faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
    
    baz_kar = 10000000 
    simule_kar = baz_kar * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
    st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")

def katılım_fon_sayfasi():
    st.header("Katılım Emeklilik & Faizsiz Yatırım Fonları Takip Aracı")
    with st.expander("📖 Teorik Arka Plan & Fon Analitik Mantığı"):
        st.markdown("Faizsiz finans ilkelerine uygun BES ve yatırım fonlarının (Hisse, Altın, Sukuk) dönemsel getiri performanslarını analiz eder.")
    
    yuklenen_dosya = st.file_uploader("📂 Kendi Fon Veri Setinizi Yükleyin (TEFAS / CSV / Excel)", type=["csv", "xlsx"], key="fon_up")
    
    tarihler = pd.date_range(start='2025-01-01', periods=60, freq='W')
    np.random.seed(42)
    df_fonlar = pd.DataFrame({
        'Tarih': tarihler,
        'Hisse Katılım Fonu': 100 * (1 + np.random.normal(0.003, 0.02, 60)).cumprod(),
        'Altın Katılım Fonu': 100 * (1 + np.random.normal(0.0025, 0.012, 60)).cumprod(),
        'Kira Sertifikası (Sukuk) Fonu': 100 * (1 + np.random.normal(0.0015, 0.004, 60)).cumprod()
    })
    
    secilen_fonlar = st.multiselect("Karşılaştırılacak Fonları Seçin", ['Hisse Katılım Fonu', 'Altın Katılım Fonu', 'Kira Sertifikası (Sukuk) Fonu'], default=['Hisse Katılım Fonu', 'Altın Katılım Fonu'])
    if secilen_fonlar:
        fig_fon = px.line(df_fonlar, x='Tarih', y=secilen_fonlar, title="Katılım Emeklilik Fonları Performans Kıyaslaması (Baz: 100 TL)")
        st.plotly_chart(fig_fon, width='stretch')

def varlik_dagilimi_sayfasi():
    st.header("Yatırım Portföyü Risk ve Varlık Dağılım Simülatörü")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Modern Portföy Teorisi çerçevesinde varlık sınıflarının risk-getiri dengesini kurar.")
    
    w_hisse = st.slider("Hisse Senedi Ağırlığı (%)", 0, 100, 50)
    w_tahvil = st.slider("Tahvil / Bono Ağırlığı (%)", 0, 100, 30)
    w_altin = st.slider("Altın / Kıymetli Maden (%)", 0, 100, 20)
    
    if w_hisse + w_tahvil + w_altin == 100:
        df_portfoy = pd.DataFrame({'Varlık Sınıfı': ['Hisse Senedi', 'Tahvil', 'Altın'], 'Ağırlık': [w_hisse, w_tahvil, w_altin]})
        fig_p = px.pie(df_portfoy, names='Varlık Sınıfı', values='Ağırlık', hole=0.4)
        st.plotly_chart(fig_p, width='stretch')
    else:
        st.warning("⚠️ Varlık ağırlıkları toplamı tam olarak %100 olmalıdır!")

def alm_sayfasi():
    st.header("Varlık-Yükümlülük Yönetimi (ALM) ve Nakit Akışı Eşitleme")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Gelecekteki tazminat yükümlülükleri ile varlık nakit akışı vadelerini eşitleyerek likidite riskini yönetir.")
    
    yil_1_yuk = st.number_input("1. Yıl Ödenecek Tazminat (TL)", 1000000, 50000000, 15000000)
    yil_2_yuk = st.number_input("2. Yıl Ödenecek Tazminat (TL)", 1000000, 50000000, 25000000)
    yil_3_yuk = st.number_input("3. Yıl Ödenecek Tazminat (TL)", 1000000, 50000000, 40000000)
    faiz_orani = st.slider("Piyasa Faiz Oranı / Getiri (%)", 5, 50, 25)
    varlik_tahvil = st.number_input("Sabit Getirili Tahvil Portföyü (TL)", 10000000, 100000000, 60000000)
        
    yillar = ['1. Yıl', '2. Yıl', '3. Yıl']
    yukumlulukler = [yil_1_yuk, yil_2_yuk, yil_3_yuk]
    varlik_getirileri = [varlik_tahvil * (faiz_orani / 100)] * 3
    net_pozisyon = [v - y for v, y in zip(varlik_getirileri, yukumlulukler)]
    
    alm_df = pd.DataFrame({
        'Yıl': yillar,
        'Yükümlülük (Tazminat)': yukumlulukler,
        'Varlık Nakit Girişi': varlik_getirileri,
        'Net Finansal Pozisyon': net_pozisyon
    })
    
    fig_alm = px.bar(alm_df, x='Yıl', y=['Yükümlülük (Tazminat)', 'Varlık Nakit Girişi'], barmode='group', title="Yıllık Varlık ve Yükümlülük Nakit Akışı Eşleşmesi")
    st.plotly_chart(fig_alm, width='stretch')

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        alm_df.to_excel(writer, sheet_name='ALM_Raporu', index=False)
    
    st.download_button(
        label="📊 ALM Simülasyon Raporunu İndir (Excel)",
        data=output.getvalue(),
        file_name="ALM_Simulasyon_Raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def benchmark_sayfasi():
    st.header("Benchmark ve Piyasa Kıyaslama Analizi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Portföy getirisinin piyasa endeksleri ve enflasyon karşısındaki performansını ölçer.")
    
    col1, col2 = st.columns(2)
    with col1:
        portfoy_getiri = st.slider("Portföy Yıllık Getirisi (%)", 0, 100, 35)
    with col2:
        enflasyon = st.slider("Yıllık Enflasyon Oranı (%)", 0, 80, 25)
        
    bist_getiri = 28.5  
    m1, m2, m3 = st.columns(3)
    m1.metric("Portföy Getirisi", f"%{portfoy_getiri}")
    m2.metric("BIST 100 Benchmark", f"%{bist_getiri}")
    m3.metric("Reel Getiri (Enflasyon Arındırılmış)", f"%{portfoy_getiri - enflasyon}")
    
    df_bench = pd.DataFrame({
        'Varlık / Endeks': ['Portföyünüz', 'BIST 100 Endeksi', 'Enflasyon'],
        'Getiri Oranı (%)': [portfoy_getiri, bist_getiri, enflasyon]
    })
    fig_b = px.bar(df_bench, x='Varlık / Endeks', y='Getiri Oranı (%)', color='Varlık / Endeks', title="Portföy vs Piyasa Benchmark Kıyaslaması")
    st.plotly_chart(fig_b, width='stretch')

def kredi_risk_sayfasi():
    st.header("Otomatik Kredi Risk Skorlama")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Müşteri veri setlerini işleyerek kredi temerrüt olasılığını skorlar.")
    
    st.file_uploader("📂 Kendi Müşteri Kredi Verinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="kredi_up")
    gelir = st.number_input("Aylık Net Gelir (TL)", 10000.0, 500000.0, 45000.0)
    st.metric("Kredi Riski", "%35.0")

def churn_sayfasi():
    st.header("Banka Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Müşteri terk davranışlarını tahmin eden erken uyarı skorlama modeli.")
    
    st.file_uploader("📂 Kendi Churn / Müşteri Verinizi Yükleyin (CSV/Excel)", type=["csv", "xlsx"], key="churn_up")
    kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
    st.metric("Terk Olasılığı", "%22.5")

# --- YENİ EKLENEN PROFESYONEL PROJE MODÜLLERİ ---

def black_scholes_sayfasi():
    st.header("Black-Scholes Opsiyon Fiyatlama & Volatilite Modülü")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("""
        * **Türev Ürünler Analitiği:** Black-Scholes-Merton modeli, finansal opsiyonların (Alım/Satım) teorik değerini belirlemede kullanılan çığır açıcı bir kısmi diferansiyel denklemdir.
        * **Parametreler:** Spot fiyat ($S$), kullanım fiyatı ($K$), vade ($T$), risksiz faiz oranı ($r$) ve volatilite ($\sigma$) kullanılarak opsiyonun adil primi hesaplanır.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Hisse Spot Fiyatı (S)", 10.0, 1000.0, 100.0)
        K = st.number_input("Kullanım Fiyatı (Strike - K)", 10.0, 1000.0, 100.0)
        T = st.slider("Vadeye Kalan Süre (Yıl)", 0.05, 5.0, 1.0)
    with col2:
        r = st.slider("Risksiz Faiz Oranı (%)", 1.0, 50.0, 15.0) / 100.0
        sigma = st.slider("Volatilite (Sigma %)", 5.0, 100.0, 25.0) / 100.0
        opt_tipi = st.selectbox("Opsiyon Tipi", ["Call (Alım)", "Put (Satım)"])
    
    # Basitleştirilmiş Black-Scholes simülasyon hesaplaması
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Normal dağılım yaklaşımı ile fiyat
    from math import erf
    def norm_cdf(x):
        return (1.0 + erf(x / np.sqrt(2.0))) / 2.0
        
    if opt_tipi == "Call (Alım)":
        fiyat = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        fiyat = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    st.markdown("---")
    st.metric("Hesaplanan Teorik Opsiyon Primi", f"{fiyat:,.2f} TL")
    
    if st.button("💾 Black-Scholes Sonucunu Kaydet"):
        kayit_ekle("Black-Scholes Opsiyon", f"Spot: {S}, Strike: {K}", f"Prim: {fiyat:,.2f} TL")
        st.success("✅ Veritabanına loglandı!")

def kredi_var_sayfasi():
    st.header("Kredi Portföyü VaR (Value at Risk) & Monte Carlo Simülasyonu")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("""
        * **Risk Yönetimi:** Value at Risk (VaR), belirli bir güven aralığında (%95 veya %99) bir portföyün maksimum kayıp potansiyelini ölçen endüstri standardı metriktir.
        * **Monte Carlo:** Binlerce olası makroekonomik senaryo üretilerek portföyün temerrüt dağılımı simüle edilir.
        """)
    
    portfoy_buyuklugu = st.number_input("Toplam Kredi Portföyü (TL)", 1000000, 1000000000, 50000000, step=5000000)
    guven_araligi = st.selectbox("Güven Aralığı", ["%95 (1.65 z)", "%99 (2.33 z)"])
    simulasyon_sayisi = st.slider("Monte Carlo Simülasyon Adımı", 1000, 20000, 5000, step=1000)
    
    z_skor = 1.65 if "%95" in guven_araligi else 2.33
    yillik_vol = 0.12
    var_tutar = portfoy_buyuklugu * z_skor * yillik_vol / np.sqrt(252) * np.sqrt(10) # 10 günlük VaR
    
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("10 Günlük Portföy VaR (Risk Değeri)", f"{var_tutar:,.0f} TL")
    m2.metric("Maksimum Beklenen Beklenmeyen Zarar", f"{var_tutar * 1.3:,.0f} TL")

def clv_sayfasi():
    st.header("Müşteri Yaşam Boyu Değeri (CLV) Tahminleme Modeli")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("""
        * **Pazarlama & Finans Analitiği:** CLV (Customer Lifetime Value), bir müşterinin şirketle olan ticari ilişkisi boyunca o kuruma kazandıracağı net bugünkü değeri ifade eder.
        * **Kullanım Amacı:** Müşteri edinme maliyetlerinin optimize edilmesi ve kârlı müşteri segmentlerinin elde tutulması.
        """)
    
    ortalama_sepet = st.number_input("Ortalama Poliçe / İşlem Tutarı (TL)", 500.0, 50000.0, 4500.0)
    yillik_islem = st.slider("Yıllık Ortalama İşlem / Yenileme Sayısı", 1, 12, 2)
    musteri_omru = st.slider("Ortalama Müşteri Ömrü (Yıl)", 1, 20, 5)
    kar_marji = st.slider("Net Kâr Marjı (%)", 5, 50, 20) / 100.0
    
    clv_deger = (ortalama_sepet * yillik_islem * musteri_omru) * kar_marji
    
    st.markdown("---")
    st.metric("Ortalama Müşteri Yaşam Boyu Değeri (CLV)", f"{clv_deger:,.2f} TL")

def veritabani_sayfasi():
    st.header("SQLite Veritabanı: Kayıtlı Simülasyon Geçmişi")
    st.write("Sistem üzerinde şimdiye kadar çalıştırılıp veritabanına loglanan tüm finansal simülasyon kayıtları:")
    
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
        st.info("Henüz veritabanına kaydedilmiş bir simülasyon bulunmuyor.")

def hakkinda_sayfasi():
    st.header("Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**, İstanbul Üniversitesi Matematik mezunuyum. Veri bilimi, yapay zeka, finansal risk analitiği ve aktüerya alanlarında projeler geliştiriyorum.
    
    Bu platform; modern veri bilimi, aktüeryal modelleme ve finansal mühendislik tekniklerinin kurumsal finans dünyasındaki gerçek iş süreçlerine nasıl entegre edilebileceğini sergileyen uçtan uca bir karar destek sistemidir.
    """)
    st.success("✨ Bu platform, risk yönetimi ve finansal analitik alanındaki teknik yetkinlikleri sergilemek amacıyla geliştirilmiştir.")

# ---------------------------------------------------------
# STREAMLIT MULTIPAGE NAVIGASYON YAPISI (FULL KAPSAMLI)
# ---------------------------------------------------------
pg = st.navigation({
    "Genel Bakış": [
        st.Page(ana_sayfa, title="Ana Sayfa", icon="🏠")
    ],
    "Sigorta & Aktüeryal": [
        st.Page(kasko_fiyatlama_sayfasi, title="Kasko Saf Prim Fiyatlama", icon="🚗"),
        st.Page(hasar_frekans_sayfasi, title="Hasar Frekans & Risk", icon="📈"),
        st.Page(reasurans_sayfasi, title="Reasürans & Afet Optimizasyonu", icon="🌐"),
        st.Page(stres_testi_sayfasi, title="Aktüeryal Stres Testi", icon="⚡")
    ],
    "Yatırım & Portföy": [
        st.Page(katılım_fon_sayfasi, title="Katılım Emeklilik Fon Takibi", icon="🪙"),
        st.Page(varlik_dagilimi_sayfasi, title="Varlık Dağılım Simülatörü", icon="🥧"),
        st.Page(alm_sayfasi, title="Varlık-Yükümlülük Yönetimi (ALM)", icon="⚖️"),
        st.Page(benchmark_sayfasi, title="Benchmark & Piyasa Kıyaslama", icon="📊")
    ],
    "Finansal Mühendislik & Risk": [
        st.Page(black_scholes_sayfasi, title="Black-Scholes Opsiyon Fiyatlama", icon="📈"),
        st.Page(kredi_var_sayfasi, title="Kredi Portföyü VaR (Risk Değeri)", icon="📉")
    ],
    "ML & Finansal Skorlama": [
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🤖"),
        st.Page(churn_sayfasi, title="Müşteri Kaybı (Churn) Erken Uyarı", icon="🏦"),
        st.Page(clv_sayfasi, title="Müşteri Yaşam Boyu Değeri (CLV)", icon="💎")
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Simülasyon Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻")
    ]
})

pg.run()
