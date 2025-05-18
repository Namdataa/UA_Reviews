from ETL.Extract import extract
from ETL.Transform import transform, feature_engineer
import logging
import pandas as pd
from config import key, secret
headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}

storage_options={
    "key": key,
    "secret": secret ,
    "client_kwargs": {
        "region_name": "ap-southeast-1"  # Ví dụ: khu vực Singapore
    }}

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Extract
    df = extract(headers)

    # Save raw data
    bucket_path = "s3://united-airlines-nam-project"
    raw_s3_path = f"{bucket_path}/raw_data.parquet"
    clean_s3_path = f"{bucket_path}/clean_data.parquet"
    processed_s3_path = f"{bucket_path}/processed_data.parquet"

    df.to_parquet(raw_s3_path, engine="pyarrow", index=False, storage_options=storage_options)
    logging.info("Raw data saved to S3.")

    # Transform
    df = transform(df)
    df.to_parquet(clean_s3_path, engine="pyarrow", index=False, storage_options=storage_options)
    logging.info("Cleaned data saved to S3.")
    #Read file airport list
    airport_list = f"{bucket_path}/airport_iata.csv"
    df_airport = pd.read_csv(airport_list, storage_options=storage_options)
    # Feature Engineering
    df = feature_engineer(df,df_airport)
    df.to_parquet(processed_s3_path, engine="pyarrow", index=False, storage_options=storage_options)
    logging.info("Processed Parquet file uploaded to S3.")


if __name__ == "__main__":
    main()