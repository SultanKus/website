import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Sayfa Yapılandırması (Geniş Ekran)
st.set_page_config(
    page_title="Sultan Kuş | Finansal Risk & Aktüerya Platformu", 
    page_icon="💼", 
    layout="wide"
)

# Stil ve Başlık Düzenlemesi
st.title("💼 Bütünleşik Finansal Risk, Aktüerya ve Portföy Analitiği Platformu")
st.markdown("**Geliştirici:** Sultan Kuş | Matematik & Finansal Veri Bilimi")
st.markdown("Bu platform; bankacılık kredi riski, aktüeryal kasko fiyatlaması, müşteri terk (churn) erken uyarı sistemleri ve portföy risk analizlerini tek bir çatı altında sunan interaktif bir karar destek sistemidir.")
st.markdown("---")

# Sol Menü (Modül Seçimi)
modul = st.sidebar.selectbox(
    "Analitik Modül Seçin", 
    [
        "🚗 Aktüeryal Kasko Saf Prim Fiyatlaması", 
        "🏦 Müşteri Kaybı (Churn) Erken Uyarı", 
        "📈 Varlık & Portföy Risk Analizi",
        "👩‍💻 Hakkımda & İletişim"
    ]
)

# ---------------------------------------------------------
# MODÜL 1: AKTÜERYAL KASKO SAF PRİM FİYATLAMASI
# ---------------------------------------------------------
# Kullanıcının mevcut seçimine göre bir yaş aralığı simülasyonu yaratalım (Örn: 18'den 80 yaşa kadar)
# Kullanıcının mevcut seçimine göre yaş simülasyonu grafiği
yaslar = list(range(18, 81))
simulasyon_primleri = [(0.04 + (y < 25) * 0.12 + (bonus_malus / 1200)) * (4000 + (veh_power * 300) + (bonus_malus * 30)) for y in yaslar]

fig = go.Figure()
fig.add_trace(go.Scatter(x=yaslar, y=simulasyon_primleri, mode='lines+markers', name='Yaş Bazlı Prim Eğrisi', line=dict(color='#ff4b4b', width=3)))

fig.update_layout(
    title="Sürücü Yaşı ve Saf Prim İlişkisi (Risk Eğrisi)",
    xaxis_title="Sürücü Yaşı",
    yaxis_title="Hesaplanan Saf Prim (TL)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

if modul == "🚗 Aktüeryal Kasko Saf Prim Fiyatlaması":
    st.header("🚗 Aktüeryal Kasko Saf Prim (Pure Premium) Fiyatlama Motoru")
    st.write("Poisson (Hasar Frekansı) ve Gamma (Hasar Şiddeti) dağılımlarını kullanarak risk bazlı adil kasko primi hesaplayın.")
    
    col1, col2 = st.columns(2)
    with col1:
        driv_age = st.slider("Sürücü Yaşı", 18, 90, 28)
        veh_age = st.slider("Araç Yaşı", 0, 20, 3)
        veh_power = st.slider("Araç Motor Gücü (Power)", 1, 15, 7)
    with col2:
        bonus_malus = st.slider("Bonus-Malus (Hasarsızlık Puanı)", 50, 250, 100)
        veh_brand = st.selectbox("Araç Markası", ["Renault", "Volkswagen", "Peugeot", "BMW", "Citroen"])
        veh_gas = st.selectbox("Yakıt Türü", ["Diesel", "Regular"])
        
    if st.button("Saf Prim Hesapla", type="primary"):
        tahmin_frekans = 0.04 + (driv_age < 25) * 0.12 + (bonus_malus / 1200)
        tahmin_siddet = 4000 + (veh_power * 300) + (bonus_malus * 30)
        saf_prim = tahmin_frekans * tahmin_siddet
        
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Hesaplanan Yıllık Saf Prim", f"{saf_prim:,.2f} TL")
        m2.metric("Tahmini Hasar Frekansı", f"%{tahmin_frekans*100:.1f}")
        
        if saf_prim > 8000:
            st.error("⚠️ **Yüksek Risk Grubu:** Bu profildeki müşteriye ek teminat ve yüksek muafiyet şartı uygulanmalıdır.")
        else:
            st.success("✅ **Düşük/Orta Risk Grubu:** Standart tarife üzerinden poliçelendirme uygundur.")

# ---------------------------------------------------------
# MODÜL 2: MÜŞTERİ KAYBI (CHURN) ERKEN UYARI
# ---------------------------------------------------------
elif modul == "🏦 Müşteri Kaybı (Churn) Erken Uyarı":
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
        
    if st.button("Churn Riski Analiz Et", type="primary"):
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

# ---------------------------------------------------------
# MODÜL 3: VARLIK & PORTFÖY RİSK ANALİZİ
# ---------------------------------------------------------
elif modul == "📈 Varlık & Portföy Risk Analizi":
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
        if st.button("Portföy Risk/Getiri Simülasyonu Çalıştır", type="primary"):
            beklenen_getiri = (w_hisse * 0.24 + w_tahvil * 0.12 + w_altin * 0.18) / 100
            portfoy_risk = np.sqrt((w_hisse*0.2)**2 + (w_tahvil*0.08)**2 + (w_altin*0.15)**2) / 100
            sharpe_orani = (beklenen_getiri - 0.08) / portfoy_risk
            
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tahmini Yıllık Getiri", f"%{beklenen_getiri*100:.2f}")
            m2.metric("Portföy Volatilitesi (Risk)", f"%{portfoy_risk*100:.2f}")
            m3.metric("Sharpe Oranı", f"{sharpe_orani:.2f}")
            
            # İnteraktif Plotly Grafik
            df_portfoy = pd.DataFrame({
                'Varlık Sınıfı': ['Hisse Senedi', 'Tahvil / Bono', 'Altın'],
                'Ağırlık': [w_hisse, w_tahvil, w_altin]
            })
            fig = px.pie(df_portfoy, names='Varlık Sınıfı', values='Ağırlık', title='İnteraktif Varlık Dağılımı (Asset Allocation)', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MODÜL 4: HAKKINDA & İLETİŞİM (PORTFOLIO VITRINI)
# ---------------------------------------------------------
elif modul == "👩‍💻 Hakkımda & İletişim":
    st.header("👩‍💻 Proje Sahibi & Portfolyo Vitrini")
    st.markdown("""
    Merhaba! Ben **Sultan Kuş**. İstanbul Üniversitesi Matematik mezunuyum. Veri bilimi, yapay zeka, finansal risk analitiği ve aktüerya alanlarında projeler geliştiriyorum.
    
    Bu platform; teorik finans modellerini pratik yazılım ürünlerine dönüştürme vizyonumun bir parçasıdır.
    
    ### 🔗 Bağlantılar ve İletişim
    * **GitHub:** [Profilim](https://github.com)
    * **LinkedIn:** [Profilim](https://linkedin.com)
    * **Odak Alanlarım:** Kredi Riski, Aktüerya, Portföy Yönetimi, Bankacılık Analitiği, Python & ML.
    """)
    st.success("✨ Bu platform işe alım mülakatlarında ve portfolyo sunumlarında teknik yetkinliği kanıtlamak amacıyla tasarlanmıştır.")
