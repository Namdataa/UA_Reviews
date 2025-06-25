from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from core.config import google_api
from Chatbot.ChromaDB import vector_store, document_content_description, metadata_field_info
import os

os.environ["GOOGLE_API_KEY"] = google_api

llm= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.5,
    max_tokens=None,  
    timeout=None,
    max_retries=2)
template = """You are an expert assistant specialized in answering questions about United Airlines, including services, flights, and customer experiences. You have access to the following customer reviews and related internal data: {reviews} Below is the user's question: {question} Guidelines: 
- If the answer to the question can be found in the reviews above, respond based solely on that information. 
- If the answer is not available in the reviews, respond based on your general knowledge and training (e.g., publicly available information from Google or reliable sources). - Based on the answer, consider the language the user is asking in and respond in that language; sometimes users may use multiple languages in one sentence, so determine which country's language is most prevalent and respond in that country's language. 
- Be clear, concise, and helpful in your response."""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

retriever = SelfQueryRetriever.from_llm(
    llm,
    vector_store,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True
)
# Tùy chọn nếu muốn chạy kiểm thử
"""
while True:
    try:
        print("\n\n-------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.lower().strip() == "q":
            break

        # Debug input
        print(">>> Question:", question)

        print(">>> Before calling retriever.invoke")
        reviews = retriever.invoke(question)
        print(">>> Retrieved Reviews:")
        print(reviews)

        result = chain.invoke({"reviews": reviews, "question": question})
        
        # Debug result type and content
        print(">>> Result Type:", type(result))
        if hasattr(result, "content"):
            print(">>> Result Content:", result.content)
        else:
            print(">>> Result:", result)

    except Exception as e:
        print("❌ Error occurred:", e)
"""