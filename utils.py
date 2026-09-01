import streamlit as st
import joblib
import os
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

@st.cache_resource
def kasko_modelini_yukle():
    """Model dosyasını güvenli yükler; dosya yoksa veya bozuksa otomatik geçici model oluşturur."""
    dosya_yolu = 'models/kasko_model.pkl'
    try:
        if os.path.exists(dosya_yolu) and os.path.getsize(dosya_yolu) > 1000:
            return joblib.load(dosya_yolu)
        else:
            # Boyut küçük veya dosya yoksa acil durum modelini devreye sok
            return _acil_durum_modeli_uret()
    except Exception:
        return _acil_durum_modeli_uret()

def _acil_durum_modeli_uret():
    df = gercek_kasko_verisini_getir().head(200)
    X = df[['DrivAge', 'VehAge', 'VehPower']]
    y = df['ClaimNb'] * 12000 + 4000
    model = LinearRegression()
    model.fit(X, y)
    return model

@st.cache_data
def gercek_kasko_verisini_getir():
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()
