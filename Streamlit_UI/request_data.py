import pandas as pd
from core.config import key, secret
import streamlit as st

@st.cache_data
def load_data():
    storage_options={
    "key": key,
    "secret": secret ,
    "client_kwargs": {
        "region_name": "ap-southeast-1"  # Ví dụ: khu vực Singapore
    }}
    bucket_path = "s3://united-airlines-nam-project"
    processed_s3_path = f"{bucket_path}/processed_data.parquet"
    df= pd.read_parquet(processed_s3_path, storage_options=storage_options)
    return df
