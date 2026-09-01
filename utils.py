import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression

@st.cache_data
def gercek_kasko_verisini_getir():
    """Açık kaynak kasko veri setini çeker."""
    dataset = fetch_openml(name='freMTPL2freq', version=1, as_frame=True, parser='auto')
    df = dataset.frame
    return df[['VehPower', 'VehAge', 'DrivAge', 'ClaimNb', 'Exposure']].dropna()

@st.cache_resource
def kasko_modelini_yukle():
    """GitHub dosya bağımlılığını tamamen ortadan kaldırarak modeli anlık eğitir."""
    df = gercek_kasko_verisini_getir().head(1000)
    X = df[['DrivAge', 'VehAge', 'VehPower']]
    y = df['ClaimNb'] * 12000 + 4000
    
    model = LinearRegression()
    model.fit(X, y)
    return model
