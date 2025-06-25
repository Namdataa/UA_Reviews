from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from core.config import key, secret
import streamlit as st
from Streamlit_UI.request_data import load_data

@st.cache_resource
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

df= load_data()

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
        
metadata_field_info = [
    AttributeInfo(
        name="verified",
        description="Whether the review is from a verified customer (True/False)",
        type="bool",
    ),
    AttributeInfo(
        name="date_review",
        description="The exact date when the review was written",
        type="string",
    ),
    AttributeInfo(
        name="year_review",
        description="The year when the review was written",
        type="integer",
    ),
    AttributeInfo(
        name="month_review",
        description="Month name when the review was written (e.g., January, February)",
        type="string",
    ),
    AttributeInfo(
        name="month_review_num",
        description="Month number when the review was written (e.g., 1 for January)",
        type="integer",
    ),
    AttributeInfo(
        name="year_fly",
        description="The year when the flight took place",
        type="integer",
    ),
    AttributeInfo(
        name="month_fly",
        description="The name of the month when the flight took place",
        type="string",
    ),
    AttributeInfo(
        name="month_fly_num",
        description="The number of the month when the flight took place",
        type="integer",
    ),
    AttributeInfo(
        name="month_year_fly",
        description="The combined month and year when the flight occurred (e.g., July 2022)",
        type="string",
    ),
    AttributeInfo(
        name="route",
        description="The full flight route taken (e.g., New York to Tokyo)",
        type="string",
    ),
    AttributeInfo(
        name="origin",
        description="The departure airport or city",
        type="string",
    ),
    AttributeInfo(
        name="destination",
        description="The arrival airport or city",
        type="string",
    ),
    AttributeInfo(
        name="transit",
        description="Any transit or layover airport if applicable",
        type="string",
    ),
    AttributeInfo(
        name="multi_leg",
        description="Whether the flight has multiple legs (True/False)",
        type="bool",
    ),
    AttributeInfo(
        name="seat_type",
        description="Class of the seat (e.g., Economy, Business, First)",
        type="string",
    ),
    AttributeInfo(
        name="aircraft",
        description="The aircraft model used (e.g., Boeing 777, Airbus A320)",
        type="string",
    ),
    AttributeInfo(
        name="score",
        description="Overall score of the review from 1 to 10",
        type="integer",
    ),
    AttributeInfo(
        name="recommended",
        description="Whether the customer recommends the airline (True/False)",
        type="bool",
    ),
]
document_content_description = "Detailed customer review of a flight experience with United Airlines including service, aircraft, route, and recommendation."

vector_store = Chroma(
    collection_name="airlines_reviews",
    persist_directory=r"Chatbot\Chroma",
    embedding_function=embedding_model
)
#vector_store.add_documents(documents=documents, ids=ids)
