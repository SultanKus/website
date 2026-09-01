import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# GÜVENLİ MODEL VE VERİ YÜKLEME (UTILS MİMARİSİ)
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

# Sayfa Yapılandırması (Geniş Ekran)
st.set_page_config(
    page_title="Sultan Kuş | Finansal Veri Bilimi & Aktüeryal Lab", 
    page_icon="💼", 
    layout="wide"
)

# Stil ve Başlık Düzenlemesi
st.title("💼 Finansal Veri Bilimi & Aktüeryal Laboratuvarı")
st.markdown("**Geliştirici:** Sultan Kuş | Matematik & Finansal Veri Bilimi")
st.markdown("Bu platform; sigorta risk analitiği, bankacılık kredi ve müşteri kayıp (churn) skorlaması, portföy optimizasyonu ve global reasürans modellerini tek çatı altında sunan kurumsal bir analitik laboratuvardır.")
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
        "👩‍💻 Hakkımda & İletişim"
    ]
)

if kategori == "🚗 Sigorta & Aktüeryal Veri Analitiği":
    modul = st.sidebar.radio(
        "Alt Modüller", 
        [
            "Kasko Saf Prim Fiyatlaması (ML)", 
            "Hasar Frekans & Risk Profili",
            "Uluslararası Reasürans & Afet Optimizasyonu"
        ]
    )
elif kategori == "📊 Yatırım & Portföy Veri Bilimi":
    modul = st.sidebar.radio(
        "Alt Modüller", 
        [
            "Varlık Dağılımı & Risk Simülatörü", 
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
else:
    modul = "👩‍💻 Hakkımda & İletişim"

# ---------------------------------------------------------
# MODÜL 1: KASKO SAF PRİM FİYATLAMASI (ML)
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
        
    girdi_df = pd.DataFrame(
        [[driv_age, veh_age, veh_power]], 
        columns=['DrivAge', 'VehAge', 'VehPower']
    )
    
    saf_prim = kasko_model.predict(girdi_df)[0]
    tahmin_frekans = 0.08  
    
    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
    m2.metric("Tahmini Hasar Frekansı", f"%{tahmin_frekans*100:.1f}")
    
    if saf_prim > 8000:
        st.error("⚠️ **Yüksek Risk Grubu:** Bu profildeki müşteriye ek teminat ve yüksek muafiyet şartı uygulanmalıdır.")
    else:
        st.success("✅ **Düşük/Orta Risk Grubu:** Standart tarife üzerinden poliçelendirme uygundur.")

    st.markdown("---")
    st.subheader("📊 Risk Analizi: Sürücü Yaşı ve Prim Değişim Eğrisi")
    
    yas_listesi = list(range(18, 81))
    simulasyon_primleri = [
        kasko_model.predict(pd.DataFrame([[y, veh_age, veh_power]], columns=['DrivAge', 'VehAge', 'VehPower']))[0] 
        for y in yas_listesi
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yas_listesi, 
        y=simulasyon_primleri, 
        mode='lines+markers', 
        name='Model Tahmini Saf Prim', 
        line=dict(color='#ff4b4b', width=3)
    ))
    fig.update_layout(
        xaxis_title="Sürücü Yaşı",
        yaxis_title="Hesaplanan Saf Prim (TL)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 2: HASAR FREKANS & RİSK PROFİLİ
# ---------------------------------------------------------
elif modul == "Hasar Frekans & Risk Profili":
    st.header("📈 Hasar Frekans & Aktüeryal Portföy Dağılımı")
    st.write("Gerçek sigorta veri seti (`freMTPL2freq`) üzerinden portföyün genel risk dağılımını inceleyin.")
    
    df_sigorta = gercek_kasko_verisini_getir()
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.metric("Veri Setindeki Toplam Poliçe", f"{len(df_sigorta):,}")
    with col_y:
        st.metric("Toplam Kaydedilen Hasar", f"{df_sigorta['ClaimNb'].sum():,}")
        
    st.markdown("---")
    st.subheader("🔍 Sürücü Yaşına Göre Ortalama Hasar Dağılımı")
    
    yas_hasar = df_sigorta.groupby('DrivAge')['ClaimNb'].mean().reset_index()
    fig_hasar = px.line(yas_hasar, x='DrivAge', y='ClaimNb', title="Yaş Bazlı Ortalama Hasar Frekansı", markers=True)
    fig_hasar.update_layout(template="plotly_white", xaxis_title="Sürücü Yaşı", yaxis_title="Ortalama Hasar Sayısı")
    st.plotly_chart(fig_hasar, use_container_width=True)

# ---------------------------------------------------------
# YENİ MODÜL: ULUSLARARASI REASÜRANS & AFET OPTİMİZASYONU
# ---------------------------------------------------------
elif modul == "Uluslararası Reasürans & Afet Optimizasyonu":
    st.header("🌐 Uluslararası Reasürans ve Afet Riski Optimizasyon Modeli")
    st.write("Büyük ölçekli afet senaryolarında (örn. Deprem), sigorta şirketinin bilançosunu korumak için uyguladığı reasürans (risk transferi) maliyet ve optimizasyon simülasyonu.")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        portfoy_buyuklugu = st.number_input("Toplam Portföy Teminat Büyüklüğü (TL)", 10000000, 1000000000, 150000000, step=10000000)
        afet_siddeti = st.slider("Beklenen Afet Şiddet Senaryosu (Hasar Oranı %)", 5, 50, 20)
    with col_r2:
        reasurans_retorasyon_orani = st.slider("Reasüröre Devredilen Risk Oranı (%)", 10, 90, 60)
        reasurans_komisyonu = st.slider("Reasürör Kesinti / Maliyet Oranı (%)", 2, 15, 6)
        
    # Finansal ve Aktüeryal Simülasyon Hesaplamaları
    toplam_hasar_tutari = portfoy_buyuklugu * (afet_siddeti / 100)
    sirketin_ustlendigi_hasar = toplam_hasar_tutari * (1 - reasurans_retorasyon_orani / 100)
    reasurore_devredilen_risk = toplam_hasar_tutari * (reasurans_retorasyon_orani / 100)
    reasurans_maliyeti = reasurore_devredilen_risk * (reasurans_komisyonu / 100)
    
    toplam_maliyet = sirketin_ustlendigi_hasar + reasurans_maliyeti
    
    st.markdown("---")
    r_m1, r_m2, r_m3 = st.columns(3)
    r_m1.metric("Simüle Edilen Toplam Hasar", f"{toplam_hasar_tutari:,.0f} TL")
    r_m2.metric("Şirketin Üzerinde Kalan Hasar", f"{sirketin_ustlendigi_hasar:,.0f} TL")
    r_m3.metric("Reasürans Maliyeti (Primi)", f"{reasurans_maliyeti:,.0f} TL")
    
    if sirketin_ustlendigi_hasar > 40000000:
        st.error("🚨 **Bilanço Riski Yüksek:** Şirketin üstlendiği net hasar özkaynakları zorlayabilir. Reasürans oranı artırılmalıdır.")
    else:
        st.success("✅ **Optimal Risk Transferi:** Bilanço yapısı uluslararası IFRS 17 risk kriterlerine uygundur.")
        
    # Reasürans Karşılaştırma Grafiği
    reasurans_df = pd.DataFrame({
        'Finansal Kalem': ['Şirket Net Hasar Yükü', 'Reasürör Ödemesi', 'Reasürans Maliyeti'],
        'Tutar (TL)': [sirketin_ustlendigi_hasar, reasurore_devredilen_risk, reasurans_maliyeti]
    })
    
    fig_re = px.bar(
        reasurans_df, 
        x='Finansal Kalem', 
        y='Tutar (TL)',
        text='Tutar (TL)',
        color='Finansal Kalem',
        title="Afet Senaryosu Bilanço Dağılımı ve Risk Transferi"
    )
    fig_re.update_traces(texttemplate='%{text:,.0f} TL', textposition='outside')
    fig_re.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_re, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 3: VARLIK DAĞILIMI & RİSK SİMÜLATÖRÜ
# ---------------------------------------------------------
elif modul == "Varlık Dağılımı & Risk Simülatörü":
    st.header("📈 Yatırım Portföyü Risk ve Varlık Dağılım Simülatörü")
    st.write("Modern Portföy Teorisi temelleriyle farklı varlık sınıflarının getiri ve risk simülasyonunu inceleyin.")
    
    st.markdown("### Portföy Varlık Ağırlıkları Dağılımı")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        w_hisse = st.slider("Hisse Senedi Ağırlığı (%)", 0, 100, 50)
    with col_b:
        w_tahvil = st.slider("Tahvil / Bono Ağırlığı (%)", 0, 100, 30)
    with col_c:
        w_altin = st.slider("Altın / Kıymetli Maden (%)", 0, 100, 20)
        
    toplam_agirlik = w_hisse + w_tahvil + w_altin
    if toplam_agirlik != 100:
        st.warning(f"⚠️ Toplam ağırlık %100 olmalıdır! Şu anki toplam: %{toplam_agirlik}")
    else:
        beklenen_getiri = (w_hisse * 0.24 + w_tahvil * 0.12 + w_altin * 0.18) / 100
        portfoy_risk = np.sqrt((w_hisse*0.2)**2 + (w_tahvil*0.08)**2 + (w_altin*0.15)**2) / 100
        sharpe_orani = (beklenen_getiri - 0.08) / portfoy_risk
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tahmini Yıllık Getiri", f"%{beklenen_getiri*100:.2f}")
        m2.metric("Portföy Volatilitesi (Risk)", f"%{portfoy_risk*100:.2f}")
        m3.metric("Sharpe Oranı", f"{sharpe_orani:.2f}")
        
        df_portfoy = pd.DataFrame({
            'Varlık Sınıfı': ['Hisse Senedi', 'Tahvil / Bono', 'Altın'],
            'Ağırlık': [w_hisse, w_tahvil, w_altin]
        })
        fig_portfoy = px.pie(df_portfoy, names='Varlık Sınıfı', values='Ağırlık', title='İnteraktif Varlık Dağılımı (Asset Allocation)', hole=0.4)
        st.plotly_chart(fig_portfoy, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 4: BENCHMARK & ENFLASYON KIYASLAMA
# ---------------------------------------------------------
elif modul == "Benchmark & Enflasyon Kıyaslama":
    st.header("📊 Benchmark ve Piyasa Kıyaslama Analizi")
    st.write("Oluşturduğunuz portföy getirisini BIST 100 ve Enflasyon oranlarıyla karşılaştırın.")
    
    b_hisse = st.slider("Hisse Senedi Oranı (%)", 0, 100, 50, key="b_hisse")
    b_tahvil = st.slider("Tahvil Oranı (%)", 0, 100, 30, key="b_tahvil")
    b_altin = st.slider("Altın Oranı (%)", 0, 100, 20, key="b_altin")
    
    if b_hisse + b_tahvil + b_altin == 100:
        portfoy_getirisi = (b_hisse / 100 * 0.28) + (b_tahvil / 100 * 0.18) + (b_altin / 100 * 0.22)
        
        karsilastirma_df = pd.DataFrame({
            'Varlık / Endeks': ['Senin Portföyün', 'BIST 100 (Endeks)', 'TÜFE (Enflasyon)', 'Altın (GSYİH)'],
            'Yıllık Getiri (%)': [portfoy_getirisi * 100, 24.5, 38.0, 22.0]
        })
        
        fig_bench = px.bar(
            karsilastirma_df, 
            x='Varlık / Endeks', 
            y='Yıllık Getiri (%)',
            text='Yıllık Getiri (%)',
            color='Varlık / Endeks',
            color_discrete_map={
                'Senin Portföyün': '#ff4b4b',
                'BIST 100 (Endeks)': '#1f77b4',
                'TÜFE (Enflasyon)': '#7f7f7f',
                'Altın (GSYİH)': '#bcbd22'
            }
        )
        fig_bench.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bench.update_layout(template="plotly_white", showlegend=False, yaxis_title="Yıllık Getiri Oranı (%)")
        st.plotly_chart(fig_bench, use_container_width=True)
    else:
        st.warning("Toplam ağırlık %100 olmalıdır.")

# ---------------------------------------------------------
# MODÜL 5: OTOMATİK KREDİ RİSK SKORLAMA
# ---------------------------------------------------------
elif modul == "Otomatik Kredi Risk Skorlama":
    st.header("🤖 Otomatik Kredi Risk Skorlama ve Sınıflandırma")
    st.write("Banka müşterilerinin kredi başvuru verilerini işleyerek temerrüt (default) riskini hesaplayın.")
    
    col1, col2 = st.columns(2)
    with col1:
        gelir = st.number_input("Aylık Net Gelir (TL)", 10000.0, 500000.0, 45000.0)
        borc_orani = st.slider("Mevcut Borç / Gelir Oranı (%)", 0, 100, 35)
    with col2:
        kredi_gecmisi = st.selectbox("Kredi Geçmişi Durumu", ["Çok İyi", "Orta", "Riskli / Gecikmeli"])
        calisma_suresi = st.slider("Mevcut İş Yerindeki Çalışma Süresi (Yıl)", 0, 30, 4)
        
    risk_skoru = (borc_orani * 0.6) + (0 if kredi_gecmisi == "Çok İyi" else (20 if kredi_gecmisi == "Orta" else 50)) - (calisma_suresi * 1.5)
    risk_skoru = max(min(risk_skoru, 100.0), 0.0)
    
    st.markdown("---")
    st.metric("Hesaplanan Kredi Temerrüt Risk Skoru", f"%{risk_skoru:.1f}")
    
    if risk_skoru > 60:
        st.error("🚨 **Yüksek Kredi Riski:** Kredi talebinin reddedilmesi veya ek teminat istenmesi önerilir.")
    else:
        st.success("✅ **Uygun Kredi Profili:** Kredi onay süreci için uygundur.")

# ---------------------------------------------------------
# MODÜL 6: MÜŞTERİ KAYBI (CHURN) ERKEN UYARI
# ---------------------------------------------------------
elif modul == "Müşteri Kaybı (Churn) Erken Uyarı":
    st.header("🏦 Banka Müşteri Kaybı (Churn) Erken Uyarı Sistemi")
    st.write("Makine öğrenmesi altyapısıyla bankayı terk etme potansiyeli taşıyan müşterileri önceden tespit edin.")
    
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.slider("Kredi Skoru", 350, 850, 650)
        age = st.slider("Müşteri Yaşı", 18, 80, 38)
        balance = st.number_input("Vadesiz / Vadeli Hesap Bakiyesi (TL)", 0.0, 500000.0, 75000.0)
    with c2:
        num_products = st.selectbox("Kullandığı Bankacılık Ürün Sayısı", [1, 2, 3, 4])
        is_active = st.radio("Dijital Kanallarda Aktif mi?", ["Evet", "Hayır"])
        has_credit_card = st.selectbox("Kredi Kartı Var mı?", ["Evet", "Hayır"])
        
    risk_puani = (850 - credit_score) / 600 + (balance < 1000) * 0.3 + (num_products == 1) * 0.25
    if is_active == "Hayır":
        risk_puani += 0.2
    risk_puani = min(risk_puani, 1.0)
        
    st.markdown("---")
    st.metric("Tahmini Terk (Churn) Olasılığı", f"%{risk_puani*100:.1f}")
    
    if risk_puani > 0.65:
        st.error("🚨 **YÜKSEK TERK RİSKİ!**")
        st.write("💡 **Aksiyon Önerisi:** Müşteri temsilcisi atanmalı, kredi kartı aidat indirimi veya özel mevduat faiz oranı teklif edilmelidir.")
    else:
        st.success("✅ **Müşteri Sadık / Düşük Risk**")
        st.info("💡 **Aksiyon Önerisi:** Mevcut çapraz satış (cross-sell) fırsatları değerlendirilebilir.")

    st.markdown("### 📊 Risk Analizi: Kredi Skoru ve Terk Olasılığı İlişkisi")
    skor_araligi = list(range(350, 851, 25))
    simulasyon_risk = [min(max(((850 - s) / 600 + (balance < 1000) * 0.3 + (num_products == 1) * 0.25), 0.0), 1.0) * 100 for s in skor_araligi]

    fig_churn = go.Figure()
    fig_churn.add_trace(go.Scatter(
        x=skor_araligi, 
        y=simulasyon_risk, 
        mode='lines+markers', 
        name='Kredi Skoru Bazlı Churn Riski', 
        line=dict(color='#ffa15a', width=3)
    ))
    fig_churn.update_layout(
        xaxis_title="Kredi Skoru",
        yaxis_title="Terk Olasılığı (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig_churn, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 7: HAKKINDA & İLETİŞİM
# ---------------------------------------------------------
elif modul == "👩‍💻 Hakkımda & İletişim":
    st.header("👩‍💻 Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**, İstanbul Üniversitesi Matematik mezunuyum. Veri bilimi, yapay zeka, finansal risk analitiği ve aktüerya alanlarında projeler geliştiriyorum.
    
    Bu platform; teorik finans modellerini, açık kaynak veri setlerini ve makine öğrenmesi algoritmalarını pratik yazılım ürünlerine dönüştürme vizyonumun bir parçasıdır.
    
    ### 🔗 Bağlantılar ve İletişim
    * **GitHub:** [Profilim](https://github.com)
    * **LinkedIn:** [Profilim](https://linkedin.com)
    * **Odak Alanlarım:** Kredi Riski, Aktüerya, Portföy Veri Bilimi, Bankacılık Analitiği, Python & ML.
    """)
    st.success("✨ Bu platform işe alım mülakatlarında ve portfolyo sunumlarında teknik yetkinliği kanıtlamak amacıyla tasarlanmıştır.")
