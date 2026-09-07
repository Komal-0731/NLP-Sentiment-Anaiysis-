import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛒 Customer Review Sentiment Analysis")

st.write(
    "Analyze customer reviews and understand what "
    "customers feel about products."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("P652-Dataset.xlsx")

    def get_sentiment(rating):

        if rating <= 2:
            return "Negative"

        elif rating == 3:
            return "Neutral"

        else:
            return "Positive"

    df["sentiment"] = df["rating"].apply(get_sentiment)

    return df


df = load_data()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "sentiment_model.joblib"
    )

    tfidf = model_package["tfidf"]
    model = model_package["model"]

    return tfidf, model


tfidf, model = load_model()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# PREDICTION
# =========================================================

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    review_vector = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    return prediction


# =========================================================
# DATASET STATISTICS
# =========================================================

total_reviews = len(df)

positive_count = (
    df["sentiment"] == "Positive"
).sum()

neutral_count = (
    df["sentiment"] == "Neutral"
).sum()

negative_count = (
    df["sentiment"] == "Negative"
).sum()

average_rating = df["rating"].mean()


positive_percent = (
    positive_count / total_reviews * 100
)

neutral_percent = (
    neutral_count / total_reviews * 100
)

negative_percent = (
    negative_count / total_reviews * 100
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛒 ReviewSense")

st.sidebar.write(
    "Customer Review Sentiment Analysis"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💬 Single Review",
        "📄 Batch Prediction",
        "📊 Visualizations",
        "ℹ️ About"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("📊 Dashboard")

    st.write(
        "Overview of customer reviews in the dataset."
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )


    with col2:

        st.metric(
            "😊 Positive",
            f"{positive_count:,}"
        )


    with col3:

        st.metric(
            "😐 Neutral",
            f"{neutral_count:,}"
        )


    with col4:

        st.metric(
            "😞 Negative",
            f"{negative_count:,}"
        )


    st.divider()


    # -----------------------------------------------------
    # SINGLE REVIEW
    # -----------------------------------------------------

    left, right = st.columns(
        [1, 1]
    )


    with left:

        st.subheader(
            "💬 Analyze a Single Review"
        )

        review = st.text_area(
            "Enter customer review",
            placeholder=
            "Example: The product quality is excellent and I really love it!",
            height=150
        )


        if st.button(
            "🔍 Analyze Sentiment",
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


                if sentiment == "Positive":

                    st.success(
                        "😊 Positive Sentiment"
                    )

                elif sentiment == "Negative":

                    st.error(
                        "😞 Negative Sentiment"
                    )

                else:

                    st.warning(
                        "😐 Neutral Sentiment"
                    )


    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    with right:

        st.subheader(
            "📊 Sentiment Distribution"
        )


        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reindex(
                [
                    "Positive",
                    "Negative",
                    "Neutral"
                ]
            )
        )


        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.25
        )


        fig.update_traces(
            textinfo="percent",
            texttemplate="%{percent:.1%}"
        )


        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------------------
    # RATING DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "⭐ Rating Distribution"
    )


    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )


    rating_df = pd.DataFrame({
        "Rating": rating_counts.index,
        "Reviews": rating_counts.values
    })


    rating_fig = px.bar(
        rating_df,
        x="Rating",
        y="Reviews",
        text="Reviews"
    )


    rating_fig.update_traces(
        textposition="outside"
    )


    rating_fig.update_layout(
        height=400
    )


    st.plotly_chart(
        rating_fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "📌 Dataset Summary"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Average Rating",
            f"{average_rating:.2f}"
        )


    with c2:

        st.metric(
            "Positive Ratio",
            f"{positive_percent:.1f}%"
        )


    with c3:

        st.metric(
            "Negative Ratio",
            f"{negative_percent:.1f}%"
        )


# =========================================================
# SINGLE REVIEW
# =========================================================

elif page == "💬 Single Review":

    st.header(
        "💬 Single Review Analysis"
    )

    st.write(
        "Enter a customer review and predict its sentiment."
    )


    review = st.text_area(
        "Customer Review",
        placeholder=
        "Example: I am very happy with this product!",
        height=200
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


            if sentiment == "Positive":

                st.success(
                    "😊 Positive Sentiment"
                )

            elif sentiment == "Negative":

                st.error(
                    "😞 Negative Sentiment"
                )

            else:

                st.warning(
                    "😐 Neutral Sentiment"
                )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "📄 Batch Prediction":

    st.header(
        "📄 Batch Prediction"
    )

    st.write(
        "Upload a CSV file containing customer reviews."
    )


    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )


    if uploaded_file is not None:

        batch_df = pd.read_csv(
            uploaded_file
        )


        if "review" not in batch_df.columns:

            st.error(
                "Your CSV must contain a column named 'review'."
            )

        else:

            predictions = []


            for review in batch_df["review"]:

                prediction = predict_sentiment(
                    review
                )

                predictions.append(
                    prediction
                )


            batch_df["sentiment"] = predictions


            st.success(
                "✅ Prediction completed!"
            )


            st.dataframe(
                batch_df,
                use_container_width=True
            )


            csv = batch_df.to_csv(
                index=False
            )


            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "📊 Visualizations":

    st.header(
        "📊 Data Visualizations"
    )


    # -----------------------------------------------------
    # SENTIMENT PIE
    # -----------------------------------------------------

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
    )


    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Customer Sentiment Distribution"
    )


    fig1.update_traces(
        textinfo="percent",
        texttemplate="%{percent:.1%}"
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # -----------------------------------------------------
    # RATING BAR
    # -----------------------------------------------------

    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )


    rating_df = pd.DataFrame({
        "Rating": rating_counts.index,
        "Reviews": rating_counts.values
    })


    fig2 = px.bar(
        rating_df,
        x="Rating",
        y="Reviews",
        text="Reviews",
        title="Rating Distribution"
    )


    fig2.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About the Project"
    )


    st.write(
        """
        ### 🛒 Customer Review Sentiment Analysis

        This project uses **Natural Language Processing (NLP)**
        and **Machine Learning** to analyze customer reviews.

        The application classifies reviews into three sentiment
        categories:

        😊 Positive

        😐 Neutral

        😞 Negative
        """
    )


    st.divider()


    st.subheader(
        "📊 Dataset"
    )


    st.write(
        """
        The dataset contains customer review information
        including:

        - Review title
        - Rating
        - Review body
        """
    )


    st.subheader(
        "⭐ Sentiment Mapping"
    )


    mapping_df = pd.DataFrame({
        "Rating": [
            "1–2",
            "3",
            "4–5"
        ],

        "Sentiment": [
            "Negative",
            "Neutral",
            "Positive"
        ]
    })


    st.table(
        mapping_df
    )


    st.subheader(
        "🧠 Machine Learning"
    )


    st.write(
        """
        The project uses TF-IDF Vectorization to convert
        review text into numerical features.

        Machine learning models evaluated include:

        - Logistic Regression
        - Multinomial Naive Bayes
        - Linear SVM

        Logistic Regression was selected as the final model
        based on the highest F1 Score.
        """
    )


    st.subheader(
        "🛠️ Technologies Used"
    )


    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn •
        TF-IDF • Joblib • Plotly • Streamlit
        """
    )


    st.divider()


    st.success(
        "🎯 Goal: Turn customer feedback into valuable insights."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "❤️ Built with Streamlit | Customer Review Sentiment Analysis"
)