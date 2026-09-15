import os

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from llm import CHAT_MODEL, EMBEDDING_MODEL, get_chat_model, get_embeddings

load_dotenv()

NEW_REVIEWS = 500
TOP_K = 5
# Keyed by embedding model so switching models in .env can't silently reuse
# vectors produced by a different one.
CACHE_FILE = f"review_embeddings.{EMBEDDING_MODEL.replace('/', '-')}.json"

embeddings = get_embeddings()
llm = get_chat_model(temperature=0.2)


def read_reviews_from_snowflake():
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT REVIEW_ID, CITY, RATING, COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()

    df.columns = [col.lower() for col in df.columns]
    return df


def to_native(value):
    """Snowflake hands back numpy scalars, which json.dump (used by the cache) can't serialize."""
    return value.item() if hasattr(value, "item") else value


def reviews_to_documents(df):
    return [
        Document(
            page_content=row.comment,
            metadata={
                "review_id": to_native(row.review_id),
                "city": to_native(row.city),
                "rating": to_native(row.rating),
            },
        )
        for row in df.itertuples()
    ]


@st.cache_resource(show_spinner="Loading and embedding reviews...")
def load_vector_store():
    """Load the cached embeddings if we have them, otherwise embed fresh and cache them."""
    if os.path.exists(CACHE_FILE):
        return InMemoryVectorStore.load(CACHE_FILE, embedding=embeddings)

    df = read_reviews_from_snowflake()
    store = InMemoryVectorStore.from_documents(reviews_to_documents(df), embeddings)
    store.dump(CACHE_FILE)
    return store


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY using the customer reviews provided. "
            "Be concise. If the reviews don't cover it, say so.",
        ),
        ("user", "Question: {question}\n\nReviews:\n{context}"),
    ]
)

answer_chain = ANSWER_PROMPT | llm | StrOutputParser()


def format_reviews(documents):
    return "\n".join(
        f" ({doc.metadata['city']}, {doc.metadata['rating']} stars) {doc.page_content}"
        for doc in documents
    )


def ask_llm(question, documents):
    return answer_chain.invoke(
        {"question": question, "context": format_reviews(documents)}
    )


st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {NEW_REVIEWS} reviews, answering with {CHAT_MODEL} model")

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

question = st.text_input("Ask a question about your reviews:",
                         placeholder="e.g. What are the most common complaints about delivery?")

if question:
    top_reviews = retriever.invoke(question)
    answer = ask_llm(question, top_reviews)

    st.markdown(f"**Answer:**")
    st.write(answer)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(
            pd.DataFrame([
                {
                    "city": doc.metadata["city"],
                    "rating": doc.metadata["rating"],
                    "comment": doc.page_content,
                }
                for doc in top_reviews
            ]),
            hide_index=True,
        )
