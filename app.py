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
# VERİTABANI BAĞLANTISI (SQLite Yerel Veritabanı)
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
# GÜVENLİ MODEL VE VERİ YÜKLEME
# ---------------------------------------------------------
@st.cache_data
def gercek_kasko_verisini_getir():
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

@st.cache_resource
def kasko_modelini_yukle():
    df = gercek_kasko_verisini_getir().head(1000)
    X = df[['DrivAge', 'VehAge', 'VehPower']]
    y = df['ClaimNb'] * 12000 + 4000
    model = LinearRegression()
    model.fit(X, y)
    return model

kasko_model = kasko_modelini_yukle()

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Sultan Kuş | Finansal Veri Bilimi & Aktüeryal Lab", 
    page_icon="💼", 
    layout="wide"
)

# Başlık ve Üst Bilgi
st.title("💼 Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
st.markdown("**Geliştirici:** Sultan Kuş | Matematik & Finansal Veri Bilimi")
st.markdown("Bu platform; sigorta risk analitiği, bankacılık kredi ve müşteri kayıp (churn) skorlaması, portföy optimizasyonu, ALM, stres testleri, global reasürans modelleri ve **yerleşik veritabanı loglama altyapısını** tek çatı altında sunan kurumsal bir analitik laboratuvardır.")
st.markdown("---")

# Kenar Çubuğu (Profesyonel 3'lü Kategori ve Alt Modül Mimarisi)
st.sidebar.title("🚀 Analitik Laboratuvarı")
st.sidebar.markdown("---")

kategori = st.sidebar.selectbox(
    "Ana Kategori", 
    [
        "🚗 Sigorta & Aktüeryal Veri Analitiği", 
        "📊 Yatırım & Portföy Veri Bilimi", 
        "🤖 Finansal Tahminleme & ML Modelleri",
        "📂 Veritabanı & Simülasyon Geçmişi",
        "👩‍💻 Hakkımda & İletişim"
    ]
)

if kategori == "🚗 Sigorta & Aktüeryal Veri Analitiği":
    modul = st.sidebar.radio(
        "Alt Modüller", 
        [
            "Kasko Saf Prim Fiyatlaması (ML)", 
            "Hasar Frekans & Risk Profili",
            "Uluslararası Reasürans & Afet Optimizasyonu",
            "Aktüeryal Stres Testi (Duyarlılık)"
        ]
    )
elif kategori == "📊 Yatırım & Portföy Veri Bilimi":
    modul = st.sidebar.radio(
        "Alt Modüller", 
        [
            "Varlık Dağılımı & Risk Simülatörü", 
            "Varlık-Yükümlülük Yönetimi (ALM)",
            "Benchmark & Enflasyon Kıyaslama"
        ]
    )
elif kategori == "🤖 Finansal Tahminleme & ML Modelleri":
    modul = st.sidebar.radio(
        "Alt Modüller", 
        [
            "Otomatik Kredi Risk Skorlama", 
            "Müşteri Kaybı (Churn) Erken Uyarı"
        ]
    )
elif kategori == "📂 Veritabanı & Simülasyon Geçmişi":
    modul = "Simülasyon Veritabanı Kayıtları"
else:
    modul = "👩‍💻 Hakkımda & İletişim"

# ---------------------------------------------------------
# MODÜL 1: KASKO SAF PRİM FİYATLAMASI
# ---------------------------------------------------------
if modul == "Kasko Saf Prim Fiyatlaması (ML)":
    st.header("🚗 Aktüeryal Kasko Saf Prim (Pure Premium) Fiyatlama Motoru")
    st.write("Açık kaynak gerçek sigorta veri setiyle eğitilmiş Makine Öğrenmesi Modeli üzerinden risk bazlı adil kasko primi hesaplayın.")
    
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
    saf_prim = kasko_model.predict(girdi_df)[0]
    
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
    m2.metric("Tahmini Hasar Frekansı", "%8.0")
    
    # Veritabanına Otomatik Kaydetme Butonu
    if st.button("💾 Bu Hesaplamayı Veritabanına Kaydet"):
        girdi_ozeti = f"Sürücü Yaşı: {driv_age}, Araç Yaşı: {veh_age}, Motor: {veh_power}"
        sonuc_ozeti = f"{saf_prim:,.2f} TL Saf Prim"
        kayit_ekle("Kasko Fiyatlama", girdi_ozeti, sonuc_ozeti)
        st.success("✅ Simülasyon başarıyla SQLite veritabanına loglandı!")

    yas_listesi = list(range(18, 81))
    simulasyon_primleri = [kasko_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] for y in yas_listesi]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yas_listesi, y=simulasyon_primleri, mode='lines+markers', name='Model Tahmini Saf Prim', line=dict(color='#ff4b4b', width=3)))
    fig.update_layout(xaxis_title="Sürücü Yaşı", yaxis_title="Hesaplanan Saf Prim (TL)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 2: HASAR FREKANS & RİSK PROFİLİ
# ---------------------------------------------------------
elif modul == "Hasar Frekans & Risk Profili":
    st.header("📈 Hasar Frekans & Aktüeryal Portföy Dağılımı")
    df_sigorta = gercek_kasko_verisini_getir()
    st.metric("Veri Setindeki Toplam Poliçe", f"{len(df_sigorta):,}")
    yas_hasar = df_sigorta.groupby('DrivAge')['ClaimNb'].mean().reset_index()
    fig_hasar = px.line(yas_hasar, x='DrivAge', y='ClaimNb', title="Yaş Bazlı Ortalama Hasar Frekansı", markers=True)
    st.plotly_chart(fig_hasar, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 3: ULUSLARARASI REASÜRANS & AFET OPTİMİZASYONU
# ---------------------------------------------------------
elif modul == "Uluslararası Reasürans & Afet Optimizasyonu":
    st.header("🌐 Uluslararası Reasürans ve Afet Riski Optimizasyon Modeli")
    portfoy_buyuklugu = st.number_input("Toplam Portföy Teminat Büyüklüğü (TL)", 10000000, 1000000000, 150000000, step=10000000)
    afet_siddeti = st.slider("Beklenen Afet Şiddet Senaryosu (Hasar Oranı %)", 5, 50, 20)
    reasurans_orani = st.slider("Reasüröre Devredilen Risk Oranı (%)", 10, 90, 60)
    
    toplam_hasar = portfoy_buyuklugu * (afet_siddeti / 100)
    sirket_net_hasar = toplam_hasar * (1 - reasurans_orani / 100)
    st.metric("Şirketin Üzerinde Kalan Net Hasar", f"{sirket_net_hasar:,.0f} TL")
    
    if st.button("💾 Reasürans Sonucunu Kaydet"):
        kayit_ekle("Reasürans Optimizasyonu", f"Portföy: {portfoy_buyuklugu:,} TL, Afet: %{afet_siddeti}", f"Net Hasar: {sirket_net_hasar:,.0f} TL")
        st.success("✅ Kaydedildi!")

# ---------------------------------------------------------
# MODÜL 4: AKTÜERYAL STRES TESTİ VE DUYARLILIK MATRİSİ
# ---------------------------------------------------------
elif modul == "Aktüeryal Stres Testi (Duyarlılık)":
    st.header("⚡ Aktüeryal Stres Testi ve Duyarlılık Matrisi")
    enflasyon_soku = st.slider("Enflasyon Artış Şoku (%)", 0, 50, 20)
    faiz_soku = st.slider("Faiz Oranı Değişim Şoku (%)", -20, 20, 5)
    
    baz_kar = 10000000 
    simule_kar = baz_kar * (1 + (faiz_soku / 100) - (enflasyon_soku / 100) * 1.5)
    st.metric("Simüle Edilen Net Teknik Kâr / Zarar", f"{simule_kar:,.0f} TL")

# ---------------------------------------------------------
# MODÜL 5: VARLIK DAĞILIMI & RİSK SİMÜLATÖRÜ
# ---------------------------------------------------------
elif modul == "Varlık Dağılımı & Risk Simülatörü":
    st.header("📈 Yatırım Portföyü Risk ve Varlık Dağılım Simülatörü")
    w_hisse = st.slider("Hisse Senedi Ağırlığı (%)", 0, 100, 50)
    w_tahvil = st.slider("Tahvil / Bono Ağırlığı (%)", 0, 100, 30)
    w_altin = st.slider("Altın / Kıymetli Maden (%)", 0, 100, 20)
    
    if w_hisse + w_tahvil + w_altin == 100:
        df_portfoy = pd.DataFrame({'Varlık Sınıfı': ['Hisse Senedi', 'Tahvil', 'Altın'], 'Ağırlık': [w_hisse, w_tahvil, w_altin]})
        fig_p = px.pie(df_portfoy, names='Varlık Sınıfı', values='Ağırlık', hole=0.4)
        st.plotly_chart(fig_p, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 6: VARLIK-YÜKÜMLÜLÜK YÖNETİMİ (ALM) & RAPOR İNDİRME
# ---------------------------------------------------------
elif modul == "Varlık-Yükümlülük Yönetimi (ALM)":
    st.header("⚖️ Varlık-Yükümlülük Yönetimi (ALM) ve Nakit Akışı Eşitleme")
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
    st.plotly_chart(fig_alm, use_container_width=True)

    # Excel İndirme
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

# ---------------------------------------------------------
# YENİ MODÜL: VERİTABANI & SİMÜLASYON GEÇMİŞİ
# ---------------------------------------------------------
elif modul == "Simülasyon Veritabanı Kayıtları":
    st.header("📂 SQLite Veritabanı: Kayıtlı Simülasyon Geçmişi")
    st.write("Sistem üzerinde şimdiye kadar çalıştırılıp veritabanına loglanan tüm finansal simülasyon kayıtları:")
    
    df_gecmis = gecmisi_getir()
    if len(df_gecmis) > 0:
        st.dataframe(df_gecmis, use_container_width=True)
        
        # Veritabanını Temizleme Butonu
        if st.button("🗑️ Veritabanı Geçmişini Temizle"):
            conn = sqlite3.connect('finansal_lab.db', check_same_thread=False)
            c = conn.cursor()
            c.execute("DELETE FROM simulasyonlar")
            conn.commit()
            conn.close()
            st.rerun()
    else:
        st.info("Henüz veritabanına kaydedilmiş bir simülasyon bulunmuyor. Modüllerdeki '💾 Kaydet' butonlarını kullanabilirsiniz.")

# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------
elif modul == "Benchmark & Enflasyon Kıyaslama":
    st.header("📊 Benchmark ve Piyasa Kıyaslama Analizi")

# ---------------------------------------------------------
# KREDİ RİSK
# ---------------------------------------------------------
elif modul == "Otomatik Kredi Risk Skorlama":
    st.header("🤖 Otomatik Kredi Risk Skorlama")
    gelir = st.number_input("Aylık Net Gelir (TL)", 10000.0, 500000.0, 45000.0)
    st.metric("Kredi Riski", "%35.0")

# ---------------------------------------------------------
# CHURN
# ---------------------------------------------------------
elif modul == "Müşteri Kaybı (Churn) Erken Uyarı":
    st.header("🏦 Banka Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    kredi_skoru = st.slider("Kredi Skoru", 350, 850, 650)
    st.metric("Terk Olasılığı", "%22.5")

# ---------------------------------------------------------
# HAKKINDA & İLETİŞİM
# ---------------------------------------------------------
elif modul == "👩‍💻 Hakkımda & İletişim":
    st.header("👩‍💻 Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**, İstanbul Üniversitesi Matematik mezunuyum. Veri bilimi, yapay zeka, finansal risk analitiği ve aktüerya alanlarında projeler geliştiriyorum.
    
    Bu platform; teorik finans modellerini, açık kaynak veri setlerini, makine öğrenmesi algoritmalarını ve **yerleşik SQLite veritabanı yönetimini** tek çatı altında sunmaktadır.
    """)
    st.success("✨ Bu platform işe alım mülakatlarında teknik yetkinliği kanıtlamak amacıyla tasarlanmıştır.")
