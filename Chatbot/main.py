from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import google_api
from Chatbot.ChromaDB import retriever
import os

os.environ["GOOGLE_API_KEY"] = google_api
model= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.5,
    max_tokens=None,  
    timeout=None,
    max_retries=2)
template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Tùy chọn nếu muốn chạy kiểm thử

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
