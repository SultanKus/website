"""Temel Analiz sayfası: mali tablo oranları ve veri doğrulama kontrolleri.

app.py içinde şu şekilde kullanılır:
    from temel_analiz import temel_analiz_sayfasi
    st.Page(temel_analiz_sayfasi, title="Temel Analiz (Mali Tablo)", icon="📑")
"""
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# yfinance tablolarındaki satır adları (ilk bulunan kullanılır)
ETIKETLER = {
    "Aktif": ["Total Assets"],
    "Yükümlülük": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "Özkaynak": ["Stockholders Equity", "Common Stock Equity"],
    "Dönen Varlık": ["Current Assets"],
    "Kısa Vadeli Borç": ["Current Liabilities"],
    "Hasılat": ["Total Revenue", "Operating Revenue"],
    "Brüt Kâr": ["Gross Profit"],
    "Net Kâr": ["Net Income", "Net Income Common Stockholders"],
}
ZORUNLU = ["Aktif", "Yükümlülük", "Özkaynak", "Hasılat", "Net Kâr"]
VARSAYILAN_SEMBOLLER = ["THYAO.IS", "ASELS.IS", "BIMAS.IS"]


# ---------------------------------------------------------
# SAF HESAP FONKSİYONLARI (Streamlit'e bağımlı değil, test edilebilir)
# ---------------------------------------------------------
def _satir(df, adaylar):
    """Tabloda adaylardan ilk bulunan satırı sayısal seri olarak döndürür."""
    if df is None or df.empty:
        return None
    for ad in adaylar:
        if ad in df.index:
            return pd.to_numeric(df.loc[ad], errors="coerce")
    return None


def _bol(a, b):
    return a / b.where(b != 0)


def ozet_tablo(bilanco, gelir):
    """Bilanço ve gelir tablosundan yıl bazlı kalemler ve oranlar üretir."""
    seriler = {}
    for ad, adaylar in ETIKETLER.items():
        kaynak = bilanco if ad in ("Aktif", "Yükümlülük", "Özkaynak", "Dönen Varlık", "Kısa Vadeli Borç") else gelir
        s = _satir(kaynak, adaylar)
        if s is not None:
            seriler[ad] = s
    if not seriler:
        return pd.DataFrame()

    df = pd.DataFrame(seriler)
    df.index = pd.to_datetime(df.index).year
    df = df[~df.index.duplicated(keep="first")].sort_index()

    for ad in ETIKETLER:
        if ad not in df.columns:
            df[ad] = float("nan")

    df["Cari Oran"] = _bol(df["Dönen Varlık"], df["Kısa Vadeli Borç"])
    df["Borç / Özkaynak"] = _bol(df["Yükümlülük"], df["Özkaynak"])
    df["ROE (%)"] = _bol(df["Net Kâr"], df["Özkaynak"]) * 100
    df["ROA (%)"] = _bol(df["Net Kâr"], df["Aktif"]) * 100
    df["Net Marj (%)"] = _bol(df["Net Kâr"], df["Hasılat"]) * 100
    df["Brüt Marj (%)"] = _bol(df["Brüt Kâr"], df["Hasılat"]) * 100
    return df


def dogrula(df, tolerans=0.01, sicrama=1.0):
    """Veri doğrulama kontrolleri. Her dönem için (Dönem, Kontrol, Sonuç, Detay) satırı üretir."""
    satirlar = []
    for yil in df.index:
        r = df.loc[yil]

        # 1) Bilanço denkliği: Aktif = Yükümlülük + Özkaynak
        if pd.notna(r["Aktif"]) and pd.notna(r["Yükümlülük"]) and pd.notna(r["Özkaynak"]) and r["Aktif"] != 0:
            fark = abs(r["Aktif"] - (r["Yükümlülük"] + r["Özkaynak"])) / abs(r["Aktif"])
            ok = fark <= tolerans
            satirlar.append((yil, "Bilanço denkliği", "✅" if ok else "⚠️", f"Fark: %{fark * 100:.2f}"))
        else:
            satirlar.append((yil, "Bilanço denkliği", "⚠️", "Hesaplanamadı (eksik kalem)"))

        # 2) Eksik zorunlu kalemler
        eksik = [k for k in ZORUNLU if pd.isna(r[k])]
        satirlar.append((yil, "Eksik kalem", "⚠️" if eksik else "✅", ", ".join(eksik) if eksik else "Yok"))

        # 3) Negatif özkaynak
        if pd.notna(r["Özkaynak"]):
            neg = r["Özkaynak"] < 0
            satirlar.append((yil, "Özkaynak işareti", "⚠️" if neg else "✅", "Negatif özkaynak" if neg else "Pozitif"))

    # 4) Dönemler arası aşırı sıçrama (Hasılat, Net Kâr)
    for kalem in ("Hasılat", "Net Kâr"):
        degisim = df[kalem].pct_change()
        for yil, d in degisim.items():
            if pd.notna(d) and abs(d) > sicrama:
                satirlar.append((yil, f"{kalem} sıçraması", "⚠️", f"Bir önceki döneme göre %{d * 100:+.0f}, kontrol edin"))

    out = pd.DataFrame(satirlar, columns=["Dönem", "Kontrol", "Sonuç", "Detay"])
    return out.sort_values(["Dönem", "Kontrol"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------
# VERİ ÇEKME
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def tablolari_getir(sembol):
    try:
        t = yf.Ticker(sembol)
        bilanco, gelir = t.balance_sheet, t.financials
        try:
            fk = t.info.get("trailingPE")
        except Exception:
            fk = None
        return {"bilanco": bilanco, "gelir": gelir, "fk": fk}
    except Exception:
        return None


# ---------------------------------------------------------
# SAYFA
# ---------------------------------------------------------
def temel_analiz_sayfasi():
    st.header("Temel Analiz: Mali Tablo Oranları ve Veri Doğrulama")
    st.caption("Veri kaynağı: Yahoo Finance (yfinance), yıllık tablolar. Borsa İstanbul şirketlerinde bazı kalemler "
               "eksik olabilir; bu durum aşağıdaki doğrulama tablosunda işaretlenir. Yatırım tavsiyesi değildir.")

    secilen = st.multiselect("Şirketler", VARSAYILAN_SEMBOLLER, default=VARSAYILAN_SEMBOLLER,
                             accept_new_options=True, key="ta_semboller",
                             help="Listeye yazarak yeni sembol ekleyebilirsiniz (örn. EREGL.IS).")
    if not secilen:
        st.info("Analiz için en az bir şirket seçin.")
        return

    sonuclar, karsilastirma = {}, []
    with st.spinner("Mali tablolar çekiliyor..."):
        for s in secilen:
            veri = tablolari_getir(s)
            if not veri:
                st.warning(f"{s}: veri alınamadı.")
                continue
            df = ozet_tablo(veri["bilanco"], veri["gelir"])
            if df.empty:
                st.warning(f"{s}: mali tablo bulunamadı.")
                continue
            sonuclar[s] = (df, veri["fk"])
            son = df.iloc[-1]
            karsilastirma.append({
                "Şirket": s, "Dönem": int(df.index[-1]), "F/K": veri["fk"],
                "Cari Oran": son["Cari Oran"], "Borç / Özkaynak": son["Borç / Özkaynak"],
                "ROE (%)": son["ROE (%)"], "ROA (%)": son["ROA (%)"], "Net Marj (%)": son["Net Marj (%)"],
            })

    if not karsilastirma:
        return

    st.subheader("Şirket Karşılaştırması (son dönem)")
    kars = pd.DataFrame(karsilastirma).set_index("Şirket")
    st.dataframe(kars.style.format("{:.2f}", subset=[c for c in kars.columns if c != "Dönem"], na_rep="-"),
                 width="stretch")

    oran = st.selectbox("Grafikte karşılaştırılacak oran",
                        ["ROE (%)", "ROA (%)", "Net Marj (%)", "Cari Oran", "Borç / Özkaynak"], key="ta_oran")
    st.plotly_chart(px.bar(kars.reset_index(), x="Şirket", y=oran, title=f"{oran} (son dönem)"),
                    width="stretch")

    st.subheader("Şirket Detayları")
    sekmeler = st.tabs(list(sonuclar.keys()))
    for sekme, (s, (df, _)) in zip(sekmeler, sonuclar.items()):
        with sekme:
            st.markdown("**Yıllık kalemler ve oranlar** (tutarlar şirketin raporlama para biriminde)")
            st.dataframe(df.sort_index(ascending=False).T.style.format("{:,.2f}", na_rep="-"), width="stretch")
            st.plotly_chart(px.line(df.reset_index(names="Yıl"), x="Yıl", y=["ROE (%)", "Net Marj (%)"],
                                    markers=True, title=f"{s}: ROE ve Net Marj"), width="stretch")
            st.markdown("**Veri doğrulama kontrolleri**")
            kontrol = dogrula(df)
            sorun = (kontrol["Sonuç"] == "⚠️").sum()
            (st.warning if sorun else st.success)(f"{sorun} uyarı bulundu." if sorun else "Tüm kontroller geçti.")
            st.dataframe(kontrol, width="stretch", hide_index=True)
