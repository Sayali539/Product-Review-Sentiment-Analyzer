# Product Review Sentiment Analyzer

**Roll No.: 61**  
**Project category:** Business, Finance & E-commerce  
**Project title:** Product Review Sentiment Analyzer

## Problem statement
Sellers cannot manually read thousands of product reviews. This application uses an LLM to classify reviews as **Positive**, **Negative**, or **Neutral**, visualize the sentiment distribution, and summarize common complaints from negative reviews.

## Features
- Upload a product reviews CSV file.
- Choose the column containing review text.
- Process reviews in batches of 20 per Gemini API call.
- Classify each review as Positive, Negative, or Neutral.
- Display a sentiment pie chart and totals.
- Summarize common complaints from negative reviews.
- Filter results and download the analyzed CSV.

## Technology stack
- Python
- Streamlit
- Google Gemini API (`gemini-3.8-flash`)
- Pandas
- Plotly

## Setup

1. Install Python 3.10 or later.
2. Extract this ZIP and open a terminal in the project folder.
3. Create and activate a virtual environment (recommended):

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Get a Gemini API key from Google AI Studio.
6. Create a file named `.env` in this folder by copying `.env.example`, then replace the placeholder with your API key:
   ```env
   GEMINI_API_KEY=your_actual_key
   ```
   Keep `.env` private. Never upload it to GitHub.
7. Run the application:
   ```bash
   streamlit run app.py
   ```
8. Open the local URL shown in the terminal (usually `http://localhost:8501`).
9. Upload `sample_reviews.csv` to test the application.

You can also paste the API key into the password field in the app sidebar instead of using `.env`.

## CSV format
The uploaded CSV must contain a review-text column. The app automatically looks for common names such as `review`, `reviews`, `text`, `comment`, or `review_text`; otherwise, choose the correct column from the dropdown.

## Suggested testing
Test with positive, negative, and neutral reviews, including a CSV with more than 20 reviews to verify batch processing. Review the generated labels and complaint summary for accuracy.

## Security
- API keys are not included in this ZIP.
- Do not commit `.env` or Streamlit secrets to GitHub.
- Do not upload private customer data without permission.

## Academic note
This is a student mini-project demonstration. LLM classifications may be imperfect and should be reviewed before business decisions are made.
