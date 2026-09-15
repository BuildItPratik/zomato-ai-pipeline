import os
import re

import pandas as pd
import snowflake.connector
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import CHAT_MODEL, get_chat_model

load_dotenv()

FORBIDDEN_WORDS = ['drop', 'delete', 'truncate', 'alter', 'update', 'insert', 'create', 'replace', 'grant', 'revoke']

# Word boundaries, not substrings: a plain `in` check rejects any query that
# merely mentions a column like created_at or updated_at.
FORBIDDEN_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b")

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisin has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method"
]

SCHEMA = """
Tables available (Snowflake). Use bare table names, no database or schema prefix.

FACT_ORDERS(order_id, order_date, customer_id, restaurant_id, city, cuisine,
           payment_method, order_status, is_delivered, sales_amount, discount,
           delivery_fee, gst, customer_rating, delivery_time_min)
DIM_RESTAURANT(restaurant_id, restaurant_name, city, cuisine, rating, cost_for_two)
DIM_CUSTOMER(customer_id, customer_name, age, age_segment, gender, city)
MART_DAILY_CITY_REVENUE(order_date, city, orders, cancel_rate, gmv, aov)
MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine,
                            orders, revenue, avg_customer_rating, cancel_rate)
MART_DELIVERY_SLA(city, order_hour, delivered_orders, p50_delivery_min, late_rate)


Note: gmv means delivered revenue. Prefer the MART_ tables when they fit the question.
"""

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert. Write ONE SELECT query that answers the question.

Rules:
- SELECT queries only, never modify data.
- Use bare table names (FACT_ORDERS, not ZOMATO.MARTS.FACT_ORDERS).
- Add a LIMIT of 100 or less, unless the question asks for a single total.
- Only reference columns that exist in the schema below — never assume a
  column exists on a table just because a similar table has it.
- Prefer MART_ tables ONLY when they contain every column the question needs.
  If the question needs a dimension (e.g. payment_method) that isn't on any
  MART_ table, query FACT_ORDERS directly and compute the metric yourself
  (e.g. cancel_rate = AVG(CASE WHEN order_status = 'Cancelled' THEN 1.0 ELSE 0.0 END)).

{SCHEMA}
"""


class SqlQuery(BaseModel):
    """A single read-only Snowflake query that answers the user's question."""

    sql: str = Field(description="One SELECT statement, using bare table names")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{question}"),
    ]
)

# temperature=0 keeps generated SQL stable; with_structured_output replaces the
# old "reply as JSON" instruction with a schema the model has to fill in.
llm = get_chat_model(temperature=0).with_structured_output(SqlQuery)

sql_chain = prompt | llm


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema="MARTS",
        role = "DBT_ROLE"
    )


def generate_sql(question):
    sql = sql_chain.invoke({"question": question}).sql

    sql = sql.replace("ZOMATO.MARTS.", "").replace("ZOMATO.", "")
    return sql.strip().rstrip(";")


def is_safe(sql):
    lowered = sql.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False

    return FORBIDDEN_PATTERN.search(lowered) is None

def run_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    return cursor.execute(sql).fetch_pandas_all()


st.title("Chat with your Zomato Data")
st.caption(f"Ask in English, {CHAT_MODEL} writes the SQL, Snowflake runs it")

with st.sidebar:
    st.header("Example Questions")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f" - {q}")

question = st.text_input("Enter your question here",
                         placeholder="e.g. Top 10 restaurants by revenune in Banglore")


if question:
    sql = generate_sql(question)
    st.code(sql, language="sql")

    if not is_safe(sql):
        st.error("The generated SQL is not safe to run. Please modify your question.")

    else:
        try:
            df = run_query(sql)
            st.success(f"{len(df)} rows returned")
            st.dataframe(df, hide_index=True)

            if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                st.bar_chart(df, x=df.columns[0], y=df.columns[1])

        except Exception as e:
            st.error(f"Error running query: {e}")