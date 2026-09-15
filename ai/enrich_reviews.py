import os
from typing import Literal, get_args

import snowflake.connector
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import CHAT_MODEL, get_chat_model

load_dotenv()

SAMPLE_N = 5

# The Literal is the source of truth: it becomes an enum in the tool schema the
# model has to pick from, and TOPICS is derived from it for the prompt text.
Topic = Literal["food quality", "delivery", "pricing", "service", "packaging", "other"]
TOPICS = list(get_args(Topic))


class ReviewLabels(BaseModel):
    """The classification we want back for a single customer review."""

    sentiment_label: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment strength from -1.0 (very negative) to 1.0 (very positive)",
    )
    topic: Topic = Field(description="Main topic the review is about")
    key_issue: str | None = Field(
        default=None,
        description="Short phrase of 6 words or less describing the main issue, or null if there is none",
    )


SYSTEM_PROMPT = f"""
You classify customer reviews for a food delivery app.

Read the review you are given and work out:
- sentiment_label: positive, negative, or neutral
- sentiment_score: a number between -1.0 and 1.0
- topic: the main topic, one of {TOPICS}
- key_issue: a short phrase of 6 words or less that describes the main issue in the review, if any. If there is no issue, return null
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{comment}"),
    ]
)

# temperature=0 keeps the labels stable across runs; with_structured_output makes
# the model return a ReviewLabels object instead of a JSON blob we parse by hand.
llm = get_chat_model(temperature=0).with_structured_output(ReviewLabels)

classify_chain = prompt | llm


def get_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

def create_output_table(cursor):
    cursor.execute("CREATE SCHEMA IF NOT EXISTS ZOMATO.AI")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ZOMATO.AI.REVIEW_ENRICHED (
            REVIEW_ID STRING,
            SENTIMENT_LABEL STRING,
            SENTIMENT_SCORE FLOAT,
            TOPIC STRING,
            KEY_ISSUE STRING,
            MODEL STRING,
            ENRICHED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

def get_reviews_to_enrich(cursor):
    cursor.execute(f"""
        SELECT REVIEW_ID, COMMENT
        FROM ZOMATO.RAW.REVIEWS
        WHERE REVIEW_ID NOT IN (SELECT REVIEW_ID FROM ZOMATO.AI.REVIEW_ENRICHED)
        LIMIT {SAMPLE_N}
    """)
    return cursor.fetchall()

def classify_review(comment):
    return classify_chain.invoke({"comment": comment})

def save_results(cursor, results):
    """Insert all the enriched rows into Snowflake in one go."""
    print(f"Saving {len(results)} enriched reviews to Snowflake...")
    cursor.executemany(
        """
        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
            (review_id, sentiment_label, sentiment_score, topic, key_issue, model)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        results,
    )


def main():
    conn = get_connection()
    cursor = conn.cursor()
    create_output_table(cursor)
    reviews = get_reviews_to_enrich(cursor)

    if len(reviews) == 0:
        print("No new reviews to enrich.")
        return

    print(f"Enriching {len(reviews)} reviews...")

    # .batch() runs the chain over every review and return_exceptions keeps one bad
    # review from sinking the whole run.
    labels = classify_chain.batch(
        [{"comment": comment} for _, comment in reviews],
        return_exceptions=True,
    )

    results = []
    for (review_id, _), label in zip(reviews, labels):
        if isinstance(label, Exception):
            print(f"Error occurred while classifying review {review_id}: {label}")
            continue

        print(f"Labels for review {review_id}: {label}")
        results.append((
            review_id,
            label.sentiment_label,
            label.sentiment_score,
            label.topic,
            label.key_issue,
            CHAT_MODEL
        ))

    save_results(cursor, results)
    print(f"Saved {len(results)} enriched reviews to Snowflake.")
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
