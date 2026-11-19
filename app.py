import os
import json
import re
import time
from typing import List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Product Review Sentiment Analyzer")
st.write(
    "Upload a CSV of product reviews to classify each review as "
    "Positive, Negative, or Neutral and summarize common complaints."
)


def get_api_key():
    try:
        return st.secrets.get(
            "GEMINI_API_KEY",
            os.getenv("GEMINI_API_KEY", "")
        )
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def generate_with_retry(client, prompt):
    """Retry temporarily unavailable Gemini API requests."""

    for attempt in range(4):
        try:
            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception as e:
            error_text = str(e).lower()

            temporary_error = any(
                word in error_text
                for word in [
                    "503",
                    "unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "500",
                    "502",
                    "504",
                    "timeout"
                ]
            )

            if not temporary_error or attempt == 3:
                raise

            wait_time = 2 ** (attempt + 1)

            st.warning(
                f"Gemini API is temporarily busy. "
                f"Retrying in {wait_time} seconds "
                f"(attempt {attempt + 2}/4)..."
            )

            time.sleep(wait_time)


def extract_json(text: str):
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def classify_batch(client, reviews: List[str]) -> List[Dict]:

    numbered = "\n".join(
        f"{i+1}. {review}"
        for i, review in enumerate(reviews)
    )

    prompt = f"""
You are a careful product-review sentiment classifier.

Classify every review as exactly one of:
Positive, Negative, Neutral.

Do not invent information.
Keep each reason short and grounded in the review.

Return ONLY a valid JSON array.

Each object must have:
{{"index": 1, "sentiment": "Positive", "reason": "short reason"}}

Include one object for every input review and preserve its index.

REVIEWS:
{numbered}
"""

    response = generate_with_retry(client, prompt)

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    data = extract_json(response.text)

    results = []

    for item in data:

        sentiment = str(
            item.get("sentiment", "Neutral")
        ).capitalize()

        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"

        results.append({
            "index": int(item.get("index", len(results) + 1)),
            "sentiment": sentiment,
            "reason": str(item.get("reason", ""))
        })

    return results


def summarize_complaints(client, negative_reviews: List[str]):

    if not negative_reviews:
        return "No negative reviews were identified."

    prompt = (
        "Summarize the common complaints in these negative product reviews. "
        "Use concise bullet points, mention recurring themes, "
        "and do not invent facts.\n\n"
        + "\n".join(f"- {r}" for r in negative_reviews)
    )

    response = generate_with_retry(client, prompt)

    return response.text or "No summary was returned."


with st.sidebar:

    st.header("Settings")

    api_key = st.text_input(
        "Gemini API key",
        value=get_api_key(),
        type="password",
        help="You can also set GEMINI_API_KEY in a .env file."
    )

    st.caption(
        "Your API key is used locally and should never be uploaded to GitHub."
    )


uploaded = st.file_uploader(
    "Upload product reviews CSV",
    type=["csv"]
)

st.caption(
    "Your CSV must contain a column with review text. "
    "Common column names: review, reviews, text, comment."
)


if uploaded:

    try:
        df = pd.read_csv(uploaded)

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    candidates = [
        c for c in df.columns
        if c.strip().lower() in [
            "review",
            "reviews",
            "text",
            "comment",
            "content",
            "review_text"
        ]
    ]

    default_col = candidates[0] if candidates else df.columns[0]

    review_col = st.selectbox(
        "Select the review text column",
        options=list(df.columns),
        index=list(df.columns).index(default_col)
    )

    reviews = df[review_col].dropna().astype(str).tolist()

    st.write(f"**{len(reviews)} reviews** found.")

    if st.button(
        "Analyze Reviews",
        type="primary",
        disabled=not reviews
    ):

        if not api_key:
            st.error(
                "Enter your Gemini API key in the sidebar "
                "or add it to a .env file."
            )
            st.stop()

        client = genai.Client(api_key=api_key)

        all_results = []

        progress = st.progress(0)
        status = st.empty()

        batches = [
            reviews[i:i+20]
            for i in range(0, len(reviews), 20)
        ]

        try:

            for batch_no, batch in enumerate(batches, start=1):

                status.info(
                    f"Analyzing batch {batch_no} of "
                    f"{len(batches)} ({len(batch)} reviews)..."
                )

                batch_results = classify_batch(client, batch)

                by_index = {
                    r["index"]: r
                    for r in batch_results
                }

                for idx, review in enumerate(batch, start=1):

                    result = by_index.get(
                        idx,
                        {
                            "sentiment": "Neutral",
                            "reason": "No classification returned"
                        }
                    )

                    all_results.append({
                        "Review": review,
                        "Sentiment": result["sentiment"],
                        "Reason": result["reason"]
                    })

                progress.progress(batch_no / len(batches))

            # Save sentiment results before generating the summary.
            result_df = pd.DataFrame(all_results)

            st.session_state["analysis_df"] = result_df

            negative_reviews = result_df.loc[
                result_df["Sentiment"] == "Negative",
                "Review"
            ].tolist()

            status.info("Summarizing common complaints...")

            try:

                complaint_summary = summarize_complaints(
                    client,
                    negative_reviews
                )

                st.session_state["complaint_summary"] = complaint_summary

                status.success("Analysis completed successfully!")

            except Exception as summary_error:

                st.session_state["complaint_summary"] = (
                    "Sentiment analysis completed successfully, "
                    "but the complaint summary could not be generated. "
                    "Please try again later.\n\n"
                    f"Details: {summary_error}"
                )

                status.warning(
                    "Sentiment analysis completed. "
                    "Complaint summary is temporarily unavailable."
                )

        except Exception as e:

            st.error(f"Analysis failed: {e}")

            st.info(
                "Gemini API may be temporarily unavailable. "
                "Please wait and try again. "
                "Also check your API key and API quota."
            )


if "analysis_df" in st.session_state:

    result_df = st.session_state["analysis_df"]

    st.divider()

    st.subheader("📊 Sentiment Analysis Results")

    counts = result_df["Sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"],
        fill_value=0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Positive", int(counts["Positive"]))
    c2.metric("Neutral", int(counts["Neutral"]))
    c3.metric("Negative", int(counts["Negative"]))

    chart_df = counts.rename_axis(
        "Sentiment"
    ).reset_index(name="Reviews")

    fig = px.pie(
        chart_df,
        names="Sentiment",
        values="Reviews",
        title="Review Sentiment Distribution",
        hole=0.35
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📝 Common Complaints from Negative Reviews")

    st.write(
        st.session_state.get(
            "complaint_summary",
            "Summary is not available yet."
        )
    )

    st.subheader("All Classified Reviews")

    filter_sentiment = st.multiselect(
        "Filter by sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

    st.dataframe(
        result_df[
            result_df["Sentiment"].isin(filter_sentiment)
        ],
        use_container_width=True
    )

    csv_bytes = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download analyzed reviews CSV",
        data=csv_bytes,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Upload a CSV file to get started. "
        "A sample dataset is included in the project ZIP."
    )