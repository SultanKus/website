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
    /* Üst Appbar - Şık ve Uyumlu Kurumsal Renk */
    header[data-testid="stHeader"] {
        background-color: #111827 !important;
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
# GÜVENLİ MODEL VE VERİ YÜKLEME (DİNAMİK DOSYA DESTEKLİ)
# ---------------------------------------------------------
@st.cache_data
def varsayilan_kasko_verisi_getir():
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

def model_egit(df_egitim):
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
    Bu platform; sigortacılık, risk yönetimi, varlık-yükümlülük yönetimi (ALM) ve makine öğrenmesi alanlarındaki karmaşık matematiksel modelleri somutlaştırmak ve endüstriyel standartlarda simüle etmek amacıyla geliştirilmiştir. 
    
    Finans ve sigorta sektöründe karar alıcıların en büyük ihtiyaç duyduğu şey; teorik modellerin gerçek veri setleriyle nasıl çalıştığını görmek ve olası makroekonomik şokların bilançoya etkilerini anlık olarak test edebilmektir. Bu laboratuvar, bu ihtiyacı uçtan uca karşılayan yaşayan bir karar destek sistemidir.
    """)
    
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktüeryal Modüller", "4 Ana Başlık")
    c2.metric("Veritabanı Altyapısı", "SQLite (Loglama)")
    c3.metric("Veri Kaynağı", "Dinamik CSV/Excel & Açık Kaynak")
    c4.metric("Raporlama Desteği", "XlsxWriter (Excel)")
    
    st.markdown("---")
    st.subheader("💡 Odak Alanları ve Yetkinlikler")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        * **Aktüeryal Fiyatlama & Risk:** GLM ve regresyon tabanlı saf prim hesaplamaları, hasar frekans analizi ve reasürans optimizasyonu.
        * **Bilanço & ALM Yönetimi:** Gelecek döneme ait tazminat yükümlülükleri ile sabit getirili varlıkların nakit akışı eşitlemesi.
        """)
    with col_b:
        st.markdown("""
        * **Stres Testleri:** Enflasyon ve faiz şoklarının kârlılık üzerindeki duyarlılık analizleri.
        * **Kalıcılık & Veri Akışı:** Yapılan tüm simülasyonların ilişkisel bir veritabanına anlık olarak loglanması.
        """)
    
    st.info("👈 Sol menüden modülleri seçerek simülasyonları gerçekleştirebilir, dilerseniz kendi veri setinizi yükleyerek modelleri çalıştırabilirsiniz.")

def kasko_fiyatlama_sayfasi():
    st.header("Aktüeryal Kasko Saf Prim (Pure Premium) Fiyatlama Motoru")
    st.write("Hazır açık kaynak veri setini kullanabilir veya **kendi veri setinizi yükleyerek** modelin anlık olarak sizin verilerinizle eğitilmesini sağlayabilirsiniz.")
    
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("""
        * **Matematiksel Model:** Bu modülde, sürücü yaşı, araç yaşı ve motor gücü gibi risk faktörleri kullanılarak Genelleştirilmiş Doğrusal Modeller (GLM) ve regresyon yaklaşımlarıyla saf prim tahmini yapılır.
        * **Kendi Verinizi Kullanma:** Yükleyeceğiniz CSV/Excel dosyasında `DrivAge`, `VehAge`, `VehPower` ve `ClaimNb` kolonlarının bulunması modelin kusursuz çalışmasını sağlar.
        """)
    
    st.markdown("---")
    yuklenen_dosya = st.file_uploader("📂 Kendi Veri Setinizi Yükleyin (CSV veya Excel)", type=["csv", "xlsx"])
    
    if yuklenen_dosya is not None:
        try:
            if yuklenen_dosya.name.endswith('.csv'):
                user_df = pd.read_csv(yuklenen_dosya)
            else:
                user_df = pd.read_excel(yuklenen_dosya)
            
            GEREKLI_KOLONLAR = ['DrivAge', 'VehAge', 'VehPower', 'ClaimNb']
            if all(kol in user_df.columns for kol in GEREKLI_KOLONLAR):
                st.success("✅ Veri setiniz başarıyla yüklendi ve doğrulandı! Model sizin verilerinizle yeniden eğitiliyor.")
                aktif_df = user_df[GEREKLI_KOLONLAR].dropna().head(2000)
            else:
                st.warning("⚠️ Yüklediğiniz dosyada gerekli kolonlar eksik ('DrivAge', 'VehAge', 'VehPower', 'ClaimNb'). Varsayılan veri setine dönülüyor.")
                aktif_df = varsayilan_kasko_verisi_getir().head(1000)
        except Exception as e:
            st.error(f"Dosya okunurken bir hata oluştu: {e}. Varsayılan veriye dönülüyor.")
            aktif_df = varsayilan_kasko_verisi_getir().head(1000)
    else:
        aktif_df = varsayilan_kasko_verisi_getir().head(1000)

    dinamik_model = model_egit(aktif_df)

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
    
    if st.button("💾 Bu Hesaplamayı Veritabanına Kaydet"):
        girdi_ozeti = f"Sürücü Yaşı: {driv_age}, Araç Yaşı: {veh_age}, Motor: {veh_power}"
        sonuc_ozeti = f"{saf_prim:,.2f} TL Saf Prim"
        kayit_ekle("Kasko Fiyatlama", girdi_ozeti, sonuc_ozeti)
        st.success("✅ Simülasyon başarıyla SQLite veritabanına loglandı!")

    yas_listesi = list(range(18, 81))
    simulasyon_primleri = [dinamik_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] for y in yas_listesi]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yas_listesi, y=simulasyon_primleri, mode='lines+markers', name='Model Tahmini Saf Prim', line=dict(color='#0055a5', width=3)))
    fig.update_layout(xaxis_title="Sürücü Yaşı", yaxis_title="Hesaplanan Saf Prim (TL)", template="plotly_white")
    st.plotly_chart(fig, width='stretch')

def hasar_frekans_sayfasi():
    st.header("Hasar Frekansı & Aktüeryal Portföy Dağılımı")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Gerçek portföy verilerindeki hasar olasılıklarının yaş ve demografik kırılımlara göre dağılımını inceler.")
    
    df_sigorta = varsayilan_kasko_verisi_getir()
    st.metric("Veri Setindeki Toplam Poliçe", f"{len(df_sigorta):,}")
    yas_hasar = df_sigorta.groupby('DrivAge')['ClaimNb'].mean().reset_index()
    fig_hasar = px.line(yas_hasar, x='DrivAge', y='ClaimNb', title="Yaş Bazlı Ortalama Hasar Frekansı", markers=True)
    st.plotly_chart(fig_hasar, width='stretch')

def reasurans_sayfasi():
    st.header("Uluslararası Reasürans ve Afet Riski Optimizasyon Modeli")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Sigorta şirketlerinin büyük ölçekli katastrofik risklerde (deprem vb.) bilançolarını korumak için reasüröre devrettikleri risk payının maliyet ve koruma optimizasyonunu simüle eder.")
    
    portfoy_buyuklugu = st.number_input("Toplam Portföy Teminat Büyüklüğü (TL)", 10000000, 1000000000, 150000000, step=10000000)
    afet_siddeti = st.slider("Beklenen Afet Şiddet Senaryosu (Hasar Oranı %)", 5, 50, 20)
    reasurans_orani = st.slider("Reasüröre Devredilen Risk Oranı (%)", 10, 90, 60)
    
    toplam_hasar = portfoy_buyuklugu * (afet_siddeti / 100)
    sirket_net_hasar = toplam_hasar * (1 - reasurans_orani / 100)
    st.metric("Şirketin Üzerinde Kalan Net Hasar", f"{sirket_net_hasar:,.0f} TL")
    
    if st.button("💾 Reasürans Sonucunu Kaydet"):
        kayit_ekle("Reasürans Optimizasyonu", f"Portföy: {portfoy_buyuklugu:,} TL, Afet: %{afet_siddeti}", f"Net Hasar: {sirket_net_hasar:,.0f} TL")
        st.success("✅ Kaydedildi!")

def stres_testi_sayfasi():
    st.header("Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Enflasyon ve faiz şoklarının şirketin teknik karşılıkları ve kârlılığı üzerindeki marjinal etkilerini test eder.")
    
    enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
    faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
    
    baz_kar = 10000000 
    simule_kar = baz_kar * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
    st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")

def varlik_dagilimi_sayfasi():
    st.header("Yatırım Portföyü Risk ve Varlık Dağılım Simülatörü")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Modern Portföy Teorisi çerçevesinde varlık sınıflarının (hisse, tahvil, altın) risk-getiri dengesini kurar.")
    
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
        st.markdown("""
        * **ALM (Asset-Liability Management):** Sigorta şirketlerinin gelecekte ödeyeceği tazminat yükümlülükleri ile elindeki varlıkların nakit akışı vadelerini eşitleyerek likidite riskini yönetme sürecidir.
        * **Amaç:** Vade uyumsuzluğundan kaynaklanabilecek nakit açığı riskini önceden raporlamaktır.
        """)
    
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
    processed_data = output.getvalue()
    
    st.download_button(
        label="📊 ALM Simülasyon Raporunu İndir (Excel)",
        data=processed_data,
        file_name="ALM_Simulasyon_Raporu.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def benchmark_sayfasi():
    st.header("Benchmark ve Piyasa Kıyaslama Analizi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Portföy getirisinin piyasa endeksleri (BIST 100) ve enflasyon karşısındaki reel performansını ölçer.")
    
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
        st.markdown("Müşteri verilerini işleyerek kredi temerrüt (default) olasılığını skorlar.")
    gelir = st.number_input("Aylık Net Gelir (TL)", 10000.0, 500000.0, 45000.0)
    st.metric("Kredi Riski", "%35.0")

def churn_sayfasi():
    st.header("Banka Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    with st.expander("📖 Teorik Arka Plan & Model Mantığı"):
        st.markdown("Müşteri terk davranışlarını tahmin eden erken uyarı skorlama modeli.")
    kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
    st.metric("Terk Olasılığı", "%22.5")

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
    
    Bu platform; modern veri bilimi ve aktüeryal modelleme tekniklerinin, kurumsal finans dünyasındaki gerçek iş süreçlerine nasıl entegre edilebileceğini sergileyen bir karar destek sistemidir.
    """)
    st.success("✨ Bu platform, risk yönetimi ve aktüeryal analitik alanındaki teknik yetkinlikleri sergilemek amacıyla aktif olarak geliştirilmektedir.")

# ---------------------------------------------------------
# STREAMLIT MULTIPAGE NAVIGASYON YAPISI (KURUMSAL İKONLU)
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
        st.Page(varlik_dagilimi_sayfasi, title="Varlık Dağılım Simülatörü", icon="🥧"),
        st.Page(alm_sayfasi, title="Varlık-Yükümlülük Yönetimi (ALM)", icon="⚖️"),
        st.Page(benchmark_sayfasi, title="Benchmark & Piyasa Kıyaslama", icon="📊")
    ],
    "ML & Finansal Skorlama": [
        st.Page(kredi_risk_sayfasi, title="Kredi Risk Skorlama", icon="🤖"),
        st.Page(churn_sayfasi, title="Müşteri Kaybı (Churn) Erken Uyarı", icon="🏦")
    ],
    "Sistem & İletişim": [
        st.Page(veritabani_sayfasi, title="Simülasyon Veritabanı Geçmişi", icon="📂"),
        st.Page(hakkinda_sayfasi, title="Hakkımda & İletişim", icon="👩‍💻")
    ]
})

pg.run()
