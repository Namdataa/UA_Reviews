from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from core.config import key, secret
import pandas as pd

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

storage_options={
    "key": key,
    "secret": secret ,
    "client_kwargs": {
        "region_name": "ap-southeast-1"  # Ví dụ: khu vực Singapore
    }}
bucket_path = "s3://united-airlines-nam-project"
processed_s3_path = f"{bucket_path}/processed_data.parquet"
df= pd.read_parquet(processed_s3_path, storage_options=storage_options)

documents = []
ids = []

for i, row in df.iterrows():
    document = Document(
    page_content=(
        f"Customer name: {row['name']}. "
        f"Review date: {row['date_review']} (flew in {row['month_year_fly']}). "
        f"Route: {row['mapped_route']} | Origin: {row['origin']} | Destination: {row['destination']} | Transit: {row['transit']}. "
        f"Aircraft: {row['aircraft_combined']} | Type: {row['type']} | Seat type: {row['seat_type']}. "
        f"Service experience: seat comfort = {row['seat_comfort']}, cabin service = {row['cabin_serv']}, food = {row['food']}, "
        f"ground service = {row['ground_service']}, wifi = {row['wifi']}, value for money = {row['money_value']}. "
        f"Recommended: {row['recommended']}. Score: {row['score']}/10. "
        f"Customer experience summary: {row['experience']}. "
        f"Full review: {row['review']}"
    ),
    metadata={
        "id": row["id"],
        "verified": row["verified"],
        "date_review": str(row["date_review"]),
        "year_review": row["year_review"],
        "month_review": row["month_review"],
        "month_review_num": row["month_review_num"],
        "year_fly": row["year_fly"],
        "month_fly": row["month_fly"],
        "month_fly_num": row["month_fly_num"],
        "month_year_fly": row["month_year_fly"],
        "route": row["mapped_route"],
        "origin": row["origin"],
        "destination": row["destination"],
        "transit": row["transit"],
        "multi_leg": row["multi_leg"],
        "seat_type": row["seat_type"],
        "aircraft": row["aircraft_combined"],
        "score": row["score"],
        "recommended": row["recommended"]
    },
    id=str(row["id"]))
    ids.append(str(i))
    documents.append(document)
        
vector_store = Chroma(
    collection_name="airlines_reviews",
    persist_directory=r"Chatbot\Chroma",
    embedding_function=embedding_model
)
vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)