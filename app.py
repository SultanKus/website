# =========================================================
# MAKROEKONOMİ & PİYASALAR — YENİLENMİŞ BÖLÜM
#
# KULLANIM:
# 1) app.py'nin en üstündeki import bloğuna şunu ekleyin:
#        from plotly.subplots import make_subplots
# 2) Mevcut "CANLI PİYASA VERİSİ (yfinance)" bölümündeki
#    canli_piyasa_verisi_getir / hisse_ara / BIST_POPULER / hisse_secici
#    fonksiyonlarını ve finansal_bilgi_sayfasi() fonksiyonunu
#    aşağıdaki kodla DEĞİŞTİRİN. (Navigasyonda değişiklik gerekmiyor,
#    fonksiyon adı aynı kaldı.)
# 3) EVDS API anahtarınız varsa .streamlit/secrets.toml içine yazın:
#        EVDS_API_KEY = "xxxxxxxx"
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests

# ---------------------------------------------------------
# YARDIMCILAR
# ---------------------------------------------------------
def tr_sayi(x, ondalik=2):
    """1234567.89 -> '1.234.567,89' (Türkçe sayı biçimi)."""
    try:
        s = f"{float(x):,.{ondalik}f}"
    except (TypeError, ValueError):
        return "—"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


def _degisim_yuzde(seri, gun):
    """`gun` kadar geri gidip yüzde değişim hesaplar; yetersiz veri varsa None."""
    seri = seri.dropna()
    if len(seri) < 2:
        return None
    konum = max(0, len(seri) - 1 - gun)
    onceki = seri.iloc[konum]
    if onceki == 0:
        return None
    return (seri.iloc[-1] / onceki - 1) * 100


def _ybb_degisim(seri):
    """Yılbaşından bugüne değişim."""
    seri = seri.dropna()
    if seri.empty:
        return None
    bu_yil = seri[seri.index.year == seri.index[-1].year]
    if len(bu_yil) < 2 or bu_yil.iloc[0] == 0:
        return None
    return (bu_yil.iloc[-1] / bu_yil.iloc[0] - 1) * 100


def _rsi(seri, periyot=14):
    """Wilder yöntemiyle Relative Strength Index."""
    delta = seri.diff()
    kazanc = delta.clip(lower=0).ewm(alpha=1 / periyot, adjust=False).mean()
    kayip = (-delta.clip(upper=0)).ewm(alpha=1 / periyot, adjust=False).mean()
    rs = kazanc / kayip.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------
# CANLI PİYASA VERİSİ (yfinance)
# ---------------------------------------------------------
ONS_GRAM = 31.1034768  # 1 troy ons = 31.1034768 gram

# Panoda ve tabloda gösterilecek enstrümanlar
PANO_ENSTRUMANLARI = [
    {"kod": "XU100.IS",   "ad": "BIST 100",       "birim": "",   "ondalik": 0},
    {"kod": "TRY=X",      "ad": "Dolar / TL",     "birim": "₺",  "ondalik": 4},
    {"kod": "EURTRY=X",   "ad": "Euro / TL",      "birim": "₺",  "ondalik": 4},
    {"kod": "GRAM_ALTIN", "ad": "Gram Altın",     "birim": "₺",  "ondalik": 2},
    {"kod": "GC=F",       "ad": "Ons Altın",      "birim": "$",  "ondalik": 2},
    {"kod": "BZ=F",       "ad": "Brent Petrol",   "birim": "$",  "ondalik": 2},
    {"kod": "SI=F",       "ad": "Ons Gümüş",      "birim": "$",  "ondalik": 2},
    {"kod": "BTC-USD",    "ad": "Bitcoin",        "birim": "$",  "ondalik": 0},
]

PANO_SEMBOLLERI = [e["kod"] for e in PANO_ENSTRUMANLARI if e["kod"] != "GRAM_ALTIN"]

PERIYOT_SECENEKLERI = {"1 Hafta": 5, "1 Ay": 22, "3 Ay": 66, "6 Ay": 132, "1 Yıl": 252}


@st.cache_data(ttl=900, show_spinner=False)
def pano_verisi_getir():
    """
    Tüm pano enstrümanlarının 1 yıllık kapanış serilerini TEK bir yfinance
    çağrısıyla indirir (8 ayrı istek yerine 1 istek — sayfa çok daha hızlı açılır).
    Gram altın, ons altın ve USD/TRY serilerinden türetilir:
        Gram Altın (TL) = Ons Altın ($) / 31.1035 × USD/TRY
    """
    ham = yf.download(PANO_SEMBOLLERI, period="1y", progress=False,
                      auto_adjust=True, threads=True)
    if ham is None or ham.empty:
        return pd.DataFrame()

    if isinstance(ham.columns, pd.MultiIndex):
        kapanis = ham["Close"].copy()
    else:  # tek sembol dönerse
        kapanis = ham[["Close"]].copy()
        kapanis.columns = PANO_SEMBOLLERI[:1]

    # Farklı borsalar farklı saat dilimi döndürebiliyor; indeksi normalize ediyoruz.
    try:
        if getattr(kapanis.index, "tz", None) is not None:
            kapanis.index = kapanis.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    kapanis.index = pd.to_datetime(kapanis.index).normalize()

    # BIST tatilde, kripto 7/24 → boşlukları son bilinen fiyatla dolduruyoruz.
    kapanis = kapanis.ffill().dropna(how="all")

    if {"GC=F", "TRY=X"}.issubset(kapanis.columns):
        kapanis["GRAM_ALTIN"] = kapanis["GC=F"] / ONS_GRAM * kapanis["TRY=X"]

    return kapanis


@st.cache_data(ttl=3600, show_spinner=False)
def canli_piyasa_verisi_getir(sembol, periyot="1y"):
    return yf.Ticker(sembol).history(period=periyot)


@st.cache_data(ttl=1800, show_spinner=False)
def hisse_ara(sorgu, max_sonuc=8):
    """Yahoo Finance canlı arama servisiyle şirket adı/sembol eşleştirir."""
    try:
        sonuc = yf.Search(sorgu, max_results=max_sonuc)
        quotes = getattr(sonuc, "quotes", [])
        return [(q.get("symbol"), q.get("shortname") or q.get("longname") or q.get("symbol"))
                for q in quotes if q.get("symbol")]
    except Exception:
        return []


BIST_POPULER = [
    ("THYAO.IS", "Türk Hava Yolları"), ("AKBNK.IS", "Akbank"), ("KCHOL.IS", "Koç Holding"),
    ("SASA.IS", "Sasa Polyester"), ("FROTO.IS", "Ford Otosan"), ("BIMAS.IS", "BİM"),
    ("EREGL.IS", "Ereğli Demir Çelik"), ("TUPRS.IS", "Tüpraş"), ("ASELS.IS", "Aselsan"),
    ("GARAN.IS", "Garanti BBVA"), ("SISE.IS", "Şişecam"), ("PGSUS.IS", "Pegasus"),
    ("YKBNK.IS", "Yapı Kredi"), ("TCELL.IS", "Turkcell"), ("ISCTR.IS", "İş Bankası C"),
]


def hisse_secici(key_prefix, varsayilan="THYAO.IS"):
    """Kullanıcının popüler listeden, canlı aramadan veya manuel girişten hisse seçmesini sağlar."""
    mod = st.radio(
        "Hisse Seçim Yöntemi",
        ["Popüler Listeden Seç", "Şirket Adıyla Ara (canlı)", "Manuel Sembol Gir"],
        horizontal=True, key=f"{key_prefix}_mod"
    )
    if mod == "Popüler Listeden Seç":
        etiketler = [f"{ad} ({sembol})" for sembol, ad in BIST_POPULER]
        secim = st.selectbox("Hisse", etiketler, key=f"{key_prefix}_sel")
        return BIST_POPULER[etiketler.index(secim)][0]
    elif mod == "Şirket Adıyla Ara (canlı)":
        sorgu = st.text_input("Şirket adı veya sembol yazın (örn. 'Turkcell', 'Apple', 'Tesla')",
                              key=f"{key_prefix}_q")
        if not sorgu:
            st.caption("Aramak için bir şirket adı veya sembol girin.")
            return None
        sonuclar = hisse_ara(sorgu)
        if not sonuclar:
            st.warning("Sonuç bulunamadı veya Yahoo Finance arama servisine şu an ulaşılamıyor.")
            return None
        etiketler = [f"{ad} ({sembol})" for sembol, ad in sonuclar]
        secim = st.selectbox("Eşleşen Sonuçlar", etiketler, key=f"{key_prefix}_aramasel")
        return sonuclar[etiketler.index(secim)][0]
    else:
        return st.text_input("Sembol (örn. THYAO.IS, AAPL)", value=varsayilan, key=f"{key_prefix}_manuel")


# ---------------------------------------------------------
# PİYASA KARTI (değer + yüzde rozeti + mini sparkline)
# ---------------------------------------------------------
def piyasa_karti(sutun, enstruman, seri, gun_sayisi):
    seri = seri.dropna()
    with sutun:
        if len(seri) < 2:
            st.markdown(
                f'<div class="piyasa-karti"><div class="pk-ad">{enstruman["ad"]}</div>'
                f'<div class="pk-deger">—</div>'
                f'<div class="pk-alt">Veri alınamadı</div></div>',
                unsafe_allow_html=True
            )
            return

        dilim = seri.iloc[-(gun_sayisi + 1):] if len(seri) > gun_sayisi else seri
        son = dilim.iloc[-1]
        degisim = (son / dilim.iloc[0] - 1) * 100 if dilim.iloc[0] else 0.0
        artis = degisim >= 0
        renk = "#1e6b34" if artis else "#b3261e"
        arka = "rgba(30,107,52,0.10)" if artis else "rgba(179,38,30,0.10)"
        ok = "▲" if artis else "▼"

        st.markdown(f"""
        <div class="piyasa-karti" style="border-left: 4px solid {renk};">
            <div class="pk-ad">{enstruman['ad']}</div>
            <div class="pk-deger">{enstruman['birim']}{tr_sayi(son, enstruman['ondalik'])}</div>
            <div class="pk-rozet" style="color:{renk}; background:{arka};">
                {ok} %{tr_sayi(abs(degisim), 2)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Scatter(
            x=dilim.index, y=dilim.values, mode="lines",
            line=dict(color=renk, width=2),
            fill="tozeroy", fillcolor=arka,
            hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.2f}<extra></extra>"
        ))
        fig.update_layout(
            height=70, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False, range=[dilim.min() * 0.995, dilim.max() * 1.005]),
            showlegend=False
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")


# ---------------------------------------------------------
# ANA SAYFA FONKSİYONU
# ---------------------------------------------------------
def finansal_bilgi_sayfasi():
    st.markdown("""
    <style>
    .piyasa-karti { background:#ffffff; border-radius:10px; padding:14px 16px 6px 16px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.06); margin-bottom:-10px; }
    .pk-ad     { font-size:0.80rem; text-transform:uppercase; letter-spacing:0.6px;
                 color:#5a6b7b !important; font-weight:600; }
    .pk-deger  { font-size:1.65rem; font-weight:700; color:#0b1f33 !important; line-height:1.4; }
    .pk-rozet  { display:inline-block; padding:2px 8px; border-radius:6px;
                 font-size:0.82rem; font-weight:700; }
    .pk-alt    { font-size:0.85rem; color:#8a97a3 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.header("🌍 Canlı Makroekonomi & Küresel Piyasalar")
    st.markdown(
        '<div class="model-badge">✅ Tüm rakamlar Yahoo Finance üzerinden canlı çekilir '
        '(15 dk önbellekli). Gram altın, ons altın ve USD/TRY kurundan türetilir.</div>',
        unsafe_allow_html=True
    )

    egitim_notu("""
**Gram altın neden ayrı bir sembol değil?** Çünkü piyasada gram altının bağımsız bir fiyatı yoktur —
türetilmiş bir büyüklüktür. Dünya piyasasında altın **ons** cinsinden ve **dolarla** işlem görür
(`XAU/USD`, vadeli kontratı `GC=F`). Türkiye'de gördüğünüz gram fiyatı şu dönüşümün sonucudur:

`Gram Altın (TL) = Ons Altın ($) ÷ 31,1035 × USD/TRY`

Bunun pratik sonucu şu: **gram altın iki ayrı riske aynı anda maruzdur** — ons altın fiyatı ve kur.
Ons altın sabit kalsa bile dolar yükselirse gram altın yükselir. Bu yüzden "altın aldım, riskten korundum"
cümlesi Türkiye'de tam doğru değildir; aslında kısmen döviz pozisyonu taşınmaktadır. Aşağıdaki tabloda
ons altın ile gram altının farklı dönemlerdeki getirilerini yan yana karşılaştırarak bu ayrışmayı
somut olarak görebilirsiniz.

**Yüzde değişimler neden dönem seçimine bağlı?** Kartlardaki oran, seçtiğiniz periyodun **ilk gününe**
göre hesaplanır. Finans sitelerinde varsayılan olarak günlük değişim gösterilir; burada periyodu
serbest bıraktım çünkü tek günlük hareket çoğu zaman gürültüdür, trendi 1 ay/3 ay ölçeğinde görmek
daha anlamlıdır.
""", baslik="📚 Gram altın nasıl hesaplanıyor? (Yöntem notu)")

    periyot_etiket = st.radio(
        "Değişim Periyodu", list(PERIYOT_SECENEKLERI.keys()),
        index=1, horizontal=True, key="pano_periyot"
    )
    gun_sayisi = PERIYOT_SECENEKLERI[periyot_etiket]

    with st.spinner("Canlı piyasa verileri çekiliyor..."):
        pano = pano_verisi_getir()

    if pano.empty:
        st.warning("Yahoo Finance verileri şu an çekilemiyor. İnternet bağlantınızı kontrol edin.")
        return

    son_tarih = pano.index[-1].strftime("%d.%m.%Y")
    st.caption(f"Son veri tarihi: {son_tarih} · Kaynak: Yahoo Finance")

    # --- Kart paneli (2 satır × 4 sütun) ---
    mevcut = [e for e in PANO_ENSTRUMANLARI if e["kod"] in pano.columns]
    for satir_baslangic in range(0, len(mevcut), 4):
        sutunlar = st.columns(4)
        for sutun, enstruman in zip(sutunlar, mevcut[satir_baslangic:satir_baslangic + 4]):
            piyasa_karti(sutun, enstruman, pano[enstruman["kod"]], gun_sayisi)

    st.markdown("---")

    # --- Çok dönemli getiri tablosu ---
    st.subheader("📋 Dönemsel Getiri Karşılaştırması")
    st.caption("Her enstrümanın farklı zaman ölçeklerindeki yüzde değişimi — hangi varlığın hangi dönemde öne çıktığını gösterir.")

    satirlar = []
    for e in mevcut:
        seri = pano[e["kod"]].dropna()
        if seri.empty:
            continue
        satirlar.append({
            "Enstrüman": e["ad"],
            "Son Fiyat": f"{e['birim']}{tr_sayi(seri.iloc[-1], e['ondalik'])}",
            "1 Gün %": _degisim_yuzde(seri, 1),
            "1 Hafta %": _degisim_yuzde(seri, 5),
            "1 Ay %": _degisim_yuzde(seri, 22),
            "3 Ay %": _degisim_yuzde(seri, 66),
            "YBB %": _ybb_degisim(seri),
            "1 Yıl %": _degisim_yuzde(seri, 252),
        })

    tablo = pd.DataFrame(satirlar)
    yuzde_kolonlari = [k for k in tablo.columns if k.endswith("%")]

    def _renk(v):
        if pd.isna(v):
            return "color:#8a97a3;"
        return "color:#1e6b34; font-weight:600;" if v >= 0 else "color:#b3261e; font-weight:600;"

    stil = tablo.style.format({k: lambda v: "—" if pd.isna(v) else f"%{v:+.2f}" for k in yuzde_kolonlari})
    stil = (stil.map(_renk, subset=yuzde_kolonlari) if hasattr(stil, "map")
            else stil.applymap(_renk, subset=yuzde_kolonlari))
    st.dataframe(stil, width="stretch", hide_index=True)

    # --- Normalize edilmiş karşılaştırma grafiği ---
    st.subheader("📈 Bazlanmış Performans Karşılaştırması")
    egitim_notu("""
Farklı ölçekteki serileri (BIST 100 ≈ 10.000 puan, dolar ≈ 40 TL) aynı grafikte ham haliyle çizmek
anlamsızdır — büyük olan diğerlerini ezer. Bu yüzden her seri, seçilen dönemin başlangıcında
**100'e eşitlenir** (`seri / ilk_değer × 100`). Böylece grafikte okunan şey fiyat değil, **göreli
performanstır**: 100 çizgisinin üzerindeki her nokta, dönem başına göre kazanç demektir.

Yüksek enflasyon ortamında bu grafiğin asıl faydası şu: TL bazlı bir yatırımın "kazanç" sayılabilmesi
için dolar/euro ve gram altın eğrilerinin **üzerinde** kalması gerekir. Nominal getiri değil, göreli
getiri kritiktir (bkz. Piyasa Kıyaslama sayfası — reel getiri hesabı).
""", baslik="📚 Neden 100'e bazlanıyor?")

    varsayilan = [e["ad"] for e in mevcut if e["kod"] in ("XU100.IS", "TRY=X", "GRAM_ALTIN")]
    ad_kod = {e["ad"]: e["kod"] for e in mevcut}
    secilen_adlar = st.multiselect("Karşılaştırılacak Enstrümanlar", list(ad_kod.keys()),
                                   default=varsayilan, key="pano_karsilastir")

    if secilen_adlar:
        dilim = pano.iloc[-(gun_sayisi + 1):] if len(pano) > gun_sayisi else pano
        normalize = pd.DataFrame(index=dilim.index)
        for ad in secilen_adlar:
            seri = dilim[ad_kod[ad]].dropna()
            if not seri.empty and seri.iloc[0] != 0:
                normalize[ad] = seri / seri.iloc[0] * 100
        if not normalize.empty:
            fig = px.line(normalize, title=f"Göreli Performans — {periyot_etiket} (başlangıç = 100)")
            fig.add_hline(y=100, line_dash="dash", line_color="#8a97a3")
            fig.update_layout(height=420, yaxis_title="Endeks (baz 100)", xaxis_title="",
                              legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # ---------------------------------------------------------
    # TCMB EVDS
    # ---------------------------------------------------------
    st.subheader("🏛️ TCMB EVDS Veri Analizi")
    EVDS_SERILERI = {
        "USD/TRY (Alış, Günlük)": "TP.DK.USD.A",
        "EUR/TRY (Alış, Günlük)": "TP.DK.EUR.A",
        "TCMB Politika Faizi (1 Hafta Repo)": "TP.APIFON4",
        "TÜFE (2003=100)": "TP.FG.J0",
    }
    try:
        TCMB_API_KEY = st.secrets.get("EVDS_API_KEY", "")
    except Exception:
        TCMB_API_KEY = ""

    if not TCMB_API_KEY:
        demo_rozeti("EVDS API anahtarı tanımlı değil — aşağıdaki grafik simüle edilmiş örnek veridir. "
                    "Gerçek veri için evds2.tcmb.gov.tr üzerinden ücretsiz anahtar alıp "
                    ".streamlit/secrets.toml dosyasına EVDS_API_KEY olarak ekleyin.")
        df_trend = pd.DataFrame({
            'Yıl': [2020, 2021, 2022, 2023, 2024, 2025],
            'TCMB Politika Faizi': [17, 14, 9, 42.5, 50, 42.5],
            'Ortalama Hasar Maliyeti Endeksi': [118, 145, 285, 465, 540, 610]
        })
        fig_tcmb = px.line(df_trend, x='Yıl', y=['TCMB Politika Faizi', 'Ortalama Hasar Maliyeti Endeksi'],
                           title="Makro Göstergeler vs Sigorta Hasar Maliyeti (simüle)", markers=True)
        fig_tcmb.update_layout(height=380, legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig_tcmb, width="stretch")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            seri_adi = st.selectbox("EVDS Serisi", list(EVDS_SERILERI.keys()), key="evds_seri")
        with c2:
            yil_sayisi = st.slider("Kaç yıllık?", 1, 10, 3, key="evds_yil")

        if st.button("EVDS Verisini Çek"):
            bitis = pd.Timestamp.today()
            baslangic = bitis - pd.DateOffset(years=yil_sayisi)
            kod = EVDS_SERILERI[seri_adi]
            url = (f"https://evds2.tcmb.gov.tr/service/evds/series={kod}"
                   f"&startDate={baslangic.strftime('%d-%m-%Y')}&endDate={bitis.strftime('%d-%m-%Y')}"
                   f"&type=json")
            try:
                with st.spinner("TCMB EVDS sisteminden canlı veri çekiliyor..."):
                    cevap = requests.get(url, headers={"key": TCMB_API_KEY}, timeout=20)
                if cevap.status_code != 200:
                    st.error(f"EVDS yanıt vermedi (HTTP {cevap.status_code}). API anahtarını kontrol edin.")
                else:
                    kayitlar = cevap.json().get("items", [])
                    df_evds = pd.DataFrame(kayitlar)
                    deger_kolonu = kod.replace(".", "_")
                    if deger_kolonu not in df_evds.columns:
                        st.error("Beklenen seri sütunu yanıtta bulunamadı.")
                    else:
                        df_evds["Tarih"] = pd.to_datetime(df_evds["Tarih"], dayfirst=True, errors="coerce")
                        df_evds[seri_adi] = pd.to_numeric(df_evds[deger_kolonu], errors="coerce")
                        df_evds = df_evds[["Tarih", seri_adi]].dropna()
                        st.success(f"{len(df_evds)} gözlem çekildi.")
                        fig_e = px.line(df_evds, x="Tarih", y=seri_adi, title=f"TCMB EVDS — {seri_adi}")
                        fig_e.update_layout(height=400, hovermode="x unified")
                        st.plotly_chart(fig_e, width="stretch")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Son Değer", tr_sayi(df_evds[seri_adi].iloc[-1], 4))
                        m2.metric("Dönem Ortalaması", tr_sayi(df_evds[seri_adi].mean(), 4))
                        m3.metric("Dönem Değişimi",
                                  f"%{(df_evds[seri_adi].iloc[-1] / df_evds[seri_adi].iloc[0] - 1) * 100:+.1f}")
            except Exception as hata:
                st.error(f"EVDS bağlantı hatası: {hata}")

    st.markdown("---")

    # ---------------------------------------------------------
    # TEKNİK ANALİZ
    # ---------------------------------------------------------
    st.subheader("📈 Gerçek Zamanlı Hisse Teknik Analizi")
    egitim_notu("""
Bu paneldeki göstergeler fiyatın kendisinden türetilir; yani yeni bilgi eklemezler, mevcut fiyat
hareketini farklı bir açıdan özetlerler.

**SMA20 / SMA50 (hareketli ortalamalar):** Son 20 ve 50 günün ortalama kapanışı. Kısa ortalamanın
uzun ortalamayı yukarı kesmesi piyasada "golden cross", aşağı kesmesi "death cross" olarak anılır.
Bunların tahmin gücü akademik olarak tartışmalıdır — trend takibi için kullanılırlar, sinyal olarak değil.

**Bollinger Bantları:** Orta çizgi SMA20, bantlar ±2 standart sapma. Yani bant genişliği doğrudan
**volatilitenin** görsel karşılığıdır: bantlar daralıyorsa piyasa sakinleşmiş, açılıyorsa oynaklık
artmıştır. Fiyatın bandın dışına taşması "pahalı/ucuz" demek değildir; sadece son 20 günün normalinden
istatistiksel olarak uzaklaşıldığını gösterir.

**RSI (14):** Son 14 günün kazanç/kayıp oranını 0-100 arasına sıkıştırır. Geleneksel yorum 70 üstü
"aşırı alım", 30 altı "aşırı satım"dır. Güçlü trendlerde RSI haftalarca 70'in üzerinde kalabilir —
tek başına karar aracı değildir.

**Yıllık volatilite:** Günlük getirilerin standart sapmasının `√252` ile ölçeklenmesidir. Bu sayı,
sitedeki Black-Scholes sayfasındaki `σ` girdisinin ve Markowitz optimizasyonundaki risk ölçüsünün
tam olarak kendisidir — yani buradaki teknik panel ile kantitatif modüller aynı büyüklüğü kullanır.
""", baslik="📚 Bu göstergeler ne anlatıyor?")

    secilen_hisse = hisse_secici("teknik")
    c1, c2 = st.columns([1, 1])
    with c1:
        secilen_periyot = st.selectbox("Zaman Aralığı", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                       index=3, key="teknik_periyot")
    with c2:
        gostergeler = st.multiselect("Göstergeler", ["SMA20", "SMA50", "Bollinger", "Hacim", "RSI"],
                                     default=["SMA20", "SMA50", "Hacim", "RSI"], key="teknik_gosterge")

    if secilen_hisse and st.button("Teknik Analizi Getir"):
        with st.spinner("Canlı piyasa verileri çekiliyor..."):
            df = canli_piyasa_verisi_getir(secilen_hisse, secilen_periyot)

        if df.empty:
            st.error("Sembol bulunamadı (BIST hisselerinin sonuna .IS eklemeyi unutmayın, örn: KCHOL.IS).")
        else:
            df = df.copy()
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            std20 = df["Close"].rolling(20).std()
            df["BB_UST"] = df["SMA20"] + 2 * std20
            df["BB_ALT"] = df["SMA20"] - 2 * std20
            df["RSI"] = _rsi(df["Close"])
            gunluk_getiri = df["Close"].pct_change().dropna()
            yillik_vol = gunluk_getiri.std() * np.sqrt(252) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Son Kapanış", tr_sayi(df['Close'].iloc[-1], 2),
                      f"{(df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100:+.2f}%")
            m2.metric("Dönem Getirisi", f"%{(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:+.1f}")
            m3.metric("Dönem Yüksek / Düşük",
                      f"{tr_sayi(df['High'].max(), 2)} / {tr_sayi(df['Low'].min(), 2)}")
            m4.metric("Yıllık Volatilite", f"%{yillik_vol:.1f}",
                      help="Günlük getirilerin standart sapması × √252. Black-Scholes'taki σ ile aynı büyüklük.")

            rsi_var = "RSI" in gostergeler
            hacim_var = "Hacim" in gostergeler
            satir_sayisi = 1 + int(hacim_var) + int(rsi_var)
            yukseklikler = [0.62] + ([0.18] if hacim_var else []) + ([0.20] if rsi_var else [])
            yukseklikler = [y / sum(yukseklikler) for y in yukseklikler]

            fig = make_subplots(rows=satir_sayisi, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=yukseklikler)

            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                         low=df["Low"], close=df["Close"], name="Fiyat",
                                         increasing_line_color="#1e6b34",
                                         decreasing_line_color="#b3261e"), row=1, col=1)

            if "Bollinger" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_UST"], line=dict(color="rgba(11,31,51,0.25)", width=1),
                                         name="Bollinger Üst"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_ALT"], line=dict(color="rgba(11,31,51,0.25)", width=1),
                                         fill="tonexty", fillcolor="rgba(11,31,51,0.06)",
                                         name="Bollinger Alt"), row=1, col=1)
            if "SMA20" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], line=dict(color="#0055a5", width=1.5),
                                         name="SMA20"), row=1, col=1)
            if "SMA50" in gostergeler:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], line=dict(color="#e08b00", width=1.5),
                                         name="SMA50"), row=1, col=1)

            siradaki = 2
            if hacim_var:
                renkler = np.where(df["Close"] >= df["Open"], "rgba(30,107,52,0.55)", "rgba(179,38,30,0.55)")
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=renkler, name="Hacim"),
                              row=siradaki, col=1)
                fig.update_yaxes(title_text="Hacim", row=siradaki, col=1)
                siradaki += 1
            if rsi_var:
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#6a3fa0", width=1.5),
                                         name="RSI(14)"), row=siradaki, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#b3261e", row=siradaki, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#1e6b34", row=siradaki, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=siradaki, col=1)

            fig.update_layout(title=f"{secilen_hisse.upper()} — Canlı Teknik Analiz ({secilen_periyot})",
                              height=300 + 180 * (satir_sayisi - 1),
                              xaxis_rangeslider_visible=False, hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            fig.update_yaxes(title_text="Fiyat", row=1, col=1)
            st.plotly_chart(fig, width="stretch")

            if rsi_var and not np.isnan(df["RSI"].iloc[-1]):
                son_rsi = df["RSI"].iloc[-1]
                if son_rsi > 70:
                    st.caption(f"RSI {son_rsi:.0f} — geleneksel yorumla 'aşırı alım' bölgesinde. "
                               "Güçlü trendlerde bu seviyenin uzun süre korunabileceğini unutmayın.")
                elif son_rsi < 30:
                    st.caption(f"RSI {son_rsi:.0f} — 'aşırı satım' bölgesinde. Tek başına alım sinyali değildir.")
                else:
                    st.caption(f"RSI {son_rsi:.0f} — nötr bölgede (30-70).")

            kayit_ekle("Canlı Teknik Analiz", f"{secilen_hisse} / {secilen_periyot} incelendi", "Başarılı")
