import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef4ff;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Metric cards */
    .metric-card {
        padding: 22px;
        border-radius: 18px;
        background: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
        min-height: 130px;
    }

    .metric-title {
        font-size: 17px;
        color: #334155;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #172554;
        margin-top: 5px;
    }

    .metric-subtitle {
        font-size: 14px;
        color: #64748b;
    }

    /* Sentiment result */
    .positive-result {
        background-color: #ecfdf5;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        border: 1px solid #bbf7d0;
    }

    .negative-result {
        background-color: #fef2f2;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        border: 1px solid #fecaca;
    }

    .neutral-result {
        background-color: #fffbeb;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        border: 1px solid #fde68a;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    .result-review {
        font-size: 17px;
        font-style: italic;
        color: #475569;
        margin-top: 15px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #172554;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Quote */
    .quote-box {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
        color: #475569;
        font-style: italic;
        font-size: 17px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel("P652-Dataset.xlsx")

    # Create review text
    df["review_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )

    # Create sentiment from rating
    def rating_to_sentiment(rating):

        if rating <= 2:
            return "Negative"

        elif rating == 3:
            return "Neutral"

        else:
            return "Positive"

    df["sentiment"] = df["rating"].apply(
        rating_to_sentiment
    )

    return df


df = load_dataset()


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "sentiment_model.joblib"
    )

    return (
        model_package["tfidf"],
        model_package["model"]
    )


tfidf, model = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    review_tfidf = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_tfidf
    )

    return prediction[0]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🛒 Customer Review<br>
        Sentiment Analysis</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "💬 Single Review",
            "📄 Batch Prediction",
            "📊 Visualizations",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
        text-align:center;
        color:#64748b;
        font-size:15px;
        padding-top:40px;
        ">
        <i>
        “Turning Customer<br>
        Feedback into<br>
        Valuable Insights”
        </i>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
        text-align:center;
        margin-top:100px;
        color:#475569;
        ">
        ❤️ Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CALCULATE DASHBOARD VALUES
# =========================================================

total_reviews = len(df)

positive_reviews = (
    df["sentiment"] == "Positive"
).sum()

neutral_reviews = (
    df["sentiment"] == "Neutral"
).sum()

negative_reviews = (
    df["sentiment"] == "Negative"
).sum()

average_rating = df["rating"].mean()


positive_percentage = (
    positive_reviews / total_reviews * 100
)

neutral_percentage = (
    neutral_reviews / total_reviews * 100
)

negative_percentage = (
    negative_reviews / total_reviews * 100
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    # Header

    st.markdown(
        '<div class="main-title">🛍️ Customer Review Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Analyze customer reviews and understand what your customers feel about your products.</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # METRIC CARDS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            😊 Positive Reviews
            </div>

            <div class="metric-value">
            {positive_reviews}
            </div>

            <div class="metric-subtitle">
            {positive_percentage:.1f}% of total
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            😐 Neutral Reviews
            </div>

            <div class="metric-value">
            {neutral_reviews}
            </div>

            <div class="metric-subtitle">
            {neutral_percentage:.1f}% of total
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            ☹️ Negative Reviews
            </div>

            <div class="metric-value">
            {negative_reviews}
            </div>

            <div class="metric-subtitle">
            {negative_percentage:.1f}% of total
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # =====================================================
    # SINGLE REVIEW + PIE CHART
    # =====================================================

    left_col, right_col = st.columns(
        [1.5, 1]
    )


    # -----------------------------------------------------
    # SINGLE REVIEW
    # -----------------------------------------------------

    with left_col:

        st.markdown(
            '<div class="section-title">💬 Analyze a Single Review</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Enter a customer review below and get the predicted sentiment."
        )


        review = st.text_area(
            "Customer Review",
            placeholder=(
                "Example: The product quality is excellent "
                "and I really love it!"
            ),
            height=150,
            label_visibility="collapsed"
        )


        if st.button(
            "🔍 Analyze Sentiment",
            use_container_width=True
        ):

            if review.strip() == "":

                st.warning(
                    "Please enter a customer review."
                )

            else:

                sentiment = predict_sentiment(
                    review
                )


                if sentiment == "Positive":

                    st.markdown(
                        f"""
                        <div class="positive-result">

                        <div class="result-title">
                        😊 Positive Sentiment
                        </div>

                        <p>
                        <b>Predicted Sentiment:</b>
                        Positive
                        </p>

                        <div class="result-review">
                        "{review}"
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                elif sentiment == "Negative":

                    st.markdown(
                        f"""
                        <div class="negative-result">

                        <div class="result-title">
                        ☹️ Negative Sentiment
                        </div>

                        <p>
                        <b>Predicted Sentiment:</b>
                        Negative
                        </p>

                        <div class="result-review">
                        "{review}"
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                else:

                    st.markdown(
                        f"""
                        <div class="neutral-result">

                        <div class="result-title">
                        😐 Neutral Sentiment
                        </div>

                        <p>
                        <b>Predicted Sentiment:</b>
                        Neutral
                        </p>

                        <div class="result-review">
                        "{review}"
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    with right_col:

        st.markdown(
            '<div class="section-title">📊 Dataset Sentiment Distribution</div>',
            unsafe_allow_html=True
        )


        sentiment_data = pd.DataFrame({

            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],

            "Reviews": [
                positive_reviews,
                neutral_reviews,
                negative_reviews
            ]

        })


        fig = px.pie(
            sentiment_data,
            names="Sentiment",
            values="Reviews",
            hole=0.0
        )


        fig.update_traces(
            textinfo="percent",
            textfont_size=15
        )


        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=20
            ),
            legend=dict(
                orientation="v"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # RATING DISTRIBUTION
    # =====================================================

    st.markdown(
        '<div class="section-title">⭐ Rating Distribution</div>',
        unsafe_allow_html=True
    )


    rating_data = (
        df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_data.columns = [
        "Rating",
        "Reviews"
    ]


    fig_rating = px.bar(
        rating_data,
        x="Rating",
        y="Reviews",
        text="Reviews"
    )


    fig_rating.update_traces(
        textposition="outside"
    )


    fig_rating.update_layout(
        height=400,
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )


    st.plotly_chart(
        fig_rating,
        use_container_width=True
    )


    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📄 Total Reviews
            </div>

            <div class="metric-value">
            {total_reviews:,}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            ⭐ Average Rating
            </div>

            <div class="metric-value">
            {average_rating:.2f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-title">
            🗄️ Data Source
            </div>

            <div class="metric-value"
            style="font-size:24px;">
            Amazon Reviews
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # =====================================================
    # QUOTE
    # =====================================================

    st.markdown(
        """
        <div class="quote-box">
        💡 “Customer feedback helps businesses understand
        what customers really think.”
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SINGLE REVIEW PAGE
# =========================================================

elif page == "💬 Single Review":

    st.markdown(
        '<div class="main-title">💬 Single Review Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter any customer review to predict its sentiment."
    )

    review = st.text_area(
        "Customer Review",
        height=200,
        placeholder="Type your customer review here..."
    )


    if st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    ):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            sentiment = predict_sentiment(
                review
            )

            st.subheader(
                "Prediction Result"
            )


            if sentiment == "Positive":

                st.success(
                    "😊 Positive Sentiment"
                )

            elif sentiment == "Negative":

                st.error(
                    "☹️ Negative Sentiment"
                )

            else:

                st.warning(
                    "😐 Neutral Sentiment"
                )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📄 Batch Prediction":

    st.markdown(
        '<div class="main-title">📄 Batch Sentiment Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing a column named "
        "`review`."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        batch_df = pd.read_csv(
            uploaded_file
        )

        st.subheader(
            "Uploaded Data"
        )

        st.dataframe(
            batch_df,
            use_container_width=True
        )


        if "review" not in batch_df.columns:

            st.error(
                "CSV must contain a column named 'review'."
            )

        else:

            batch_df["sentiment"] = (
                batch_df["review"]
                .apply(predict_sentiment)
            )


            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                batch_df,
                use_container_width=True
            )


            csv = batch_df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                "⬇️ Download Predictions",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================

elif page == "📊 Visualizations":

    st.markdown(
        '<div class="main-title">📊 Data Visualizations</div>',
        unsafe_allow_html=True
    )


    # Sentiment pie chart

    sentiment_data = pd.DataFrame({

        "Sentiment": [
            "Positive",
            "Neutral",
            "Negative"
        ],

        "Reviews": [
            positive_reviews,
            neutral_reviews,
            negative_reviews
        ]

    })


    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Reviews",
        title="Customer Sentiment Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Rating chart

    rating_data = (
        df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_data.columns = [
        "Rating",
        "Reviews"
    ]


    fig = px.bar(
        rating_data,
        x="Rating",
        y="Reviews",
        text="Reviews",
        title="Customer Rating Distribution"
    )


    fig.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Review length

    df["word_count"] = (
        df["review_text"]
        .str.split()
        .str.len()
    )


    fig = px.histogram(
        df,
        x="word_count",
        color="sentiment",
        title="Review Length Distribution",
        nbins=30
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        ## NLP Sentiment Analysis

        ### Business Objective

        Extract sentiment from customer reviews
        of a product.

        ### Dataset

        The project uses customer review data
        containing:

        - Review Title
        - Rating
        - Review Body

        ### Sentiment Classification

        | Rating | Sentiment |
        |--------|-----------|
        | 1–2 ⭐ | Negative |
        | 3 ⭐ | Neutral |
        | 4–5 ⭐ | Positive |

        ### NLP Pipeline

        **Data Collection → Data Cleaning → EDA →
        Text Preprocessing → TF-IDF → Model Training →
        Model Evaluation → Deployment**

        ### Machine Learning Model

        **Logistic Regression**

        ### Feature Engineering

        **TF-IDF (Term Frequency–Inverse Document Frequency)**

        ### Deployment

        **Streamlit**

        """
    )