import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import fetch_openml

@st.cache_resource
def kasko_modelini_yukle():
    """Tüm projede ortak kullanılacak kasko modelini yükler."""
    try:
        return joblib.load('models/kasko_model.pkl')
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

@st.cache_data
def gercek_kasko_verisini_getir():
    """İleride başka modüllerde de analiz yapmak isteyeceğin gerçek sigorta veri setini çeker."""
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()
