import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReviewSense | Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

# ============================================================
# MAIN BACKGROUND
# ============================================================
.stApp {
    background:
        radial-gradient(circle at 85% 5%, #f0eaff 0%, transparent 25%),
        radial-gradient(circle at 10% 90%, #e8edff 0%, transparent 30%),
        linear-gradient(135deg, #f7f9ff, #faf8ff);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


# ============================================================
# SIDEBAR
# ============================================================
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #edf3ff 0%,
        #f1efff 55%,
        #f8f2ff 100%
    );

    border-right: 1px solid #dce3f5;
}

section[data-testid="stSidebar"] h1 {
    color: #172b61;
    font-size: 27px;
    font-weight: 800;
}

section[data-testid="stSidebar"] p {
    color: #63708e;
}


# ============================================================
# HEADINGS
# ============================================================
h1 {
    font-size: 45px !important;
    font-weight: 850 !important;

    background: linear-gradient(
        90deg,
        #173d91,
        #345bd9,
        #7025d5
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2 {
    color: #172b61 !important;
    font-weight: 800 !important;
}

h3 {
    color: #1c3470 !important;
    font-weight: 750 !important;
}


# ============================================================
# NORMAL TEXT
# ============================================================
p {
    color: #5d6b89;
}


# ============================================================
# METRIC CARDS
# ============================================================
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid #e1e6f4;
    border-radius: 20px;
    padding: 20px;

    box-shadow:
        0 8px 25px rgba(45,65,120,0.08);

    min-height: 120px;
}

div[data-testid="stMetric"] label {
    color: #5d6a87 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #172b61 !important;
    font-weight: 850 !important;
}


# ============================================================
# TEXT AREA
# ============================================================
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #d4dced !important;
    border-radius: 14px !important;
    color: #27375f !important;
    font-size: 15px !important;
    padding: 15px !important;
}

.stTextArea textarea:focus {
    border-color: #7166ee !important;
    box-shadow: 0 0 0 1px #7166ee !important;
}


# ============================================================
# BUTTON
# ============================================================
.stButton > button {
    background: linear-gradient(
        90deg,
        #625cf5,
        #7952e8
    );

    color: white;

    border: none;
    border-radius: 12px;

    min-height: 48px;

    font-size: 16px;
    font-weight: 700;

    box-shadow:
        0 7px 18px rgba(91,82,220,0.25);

    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);

    background: linear-gradient(
        90deg,
        #554ee0,
        #6942d3
    );

    color: white;
}


# ============================================================
# FILE UPLOADER
# ============================================================
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 15px;
    padding: 10px;
}


# ============================================================
# DATAFRAME
# ============================================================
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}


# ============================================================
# DIVIDER
# ============================================================
hr {
    border: none;
    border-top: 1px solid #dfe4f2;
    margin: 25px 0;
}


# ============================================================
# SUCCESS / ERROR / WARNING
# ============================================================
div[data-testid="stAlert"] {
    border-radius: 14px;
}


# ============================================================
# FOOTER
# ============================================================
.footer-text {
    text-align: center;
    color: #73809b;
    font-size: 14px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel("P652-Dataset.xlsx")

    def convert_rating(rating):

        if rating <= 2:
            return "Negative"

        elif rating == 3:
            return "Neutral"

        else:
            return "Positive"

    df["sentiment"] = df["rating"].apply(convert_rating)

    return df


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    package = joblib.load(
        "sentiment_model.joblib"
    )

    tfidf = package["tfidf"]
    model = package["model"]

    return tfidf, model


tfidf, model = load_model()


# ============================================================
# TEXT CLEANING
# ============================================================

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


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    review_vector = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    return prediction


# ============================================================
# DATASET STATISTICS
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 ReviewSense")

st.sidebar.caption(
    "Understand • Improve • Grow"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "💬 Single Review",
        "📄 Batch Prediction",
        "📊 Visualizations",
        "🏆 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "💡 Turning customer feedback "
    "into valuable insights."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "❤️ Built with Streamlit"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # --------------------------------------------------------
    # HERO TITLE
    # --------------------------------------------------------

    st.title(
        "🛒 Customer Review Sentiment Analysis"
    )

    st.write(
        "Turn customer feedback into meaningful insights "
        "with the power of NLP and Machine Learning."
    )

    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Total Reviews",
            f"{total_reviews:,}"
        )

    with col2:

        st.metric(
            "😊 Positive Reviews",
            f"{positive_count:,}",
            f"{positive_percent:.1f}% of total"
        )

    with col3:

        st.metric(
            "😐 Neutral Reviews",
            f"{neutral_count:,}",
            f"{neutral_percent:.1f}% of total"
        )

    with col4:

        st.metric(
            "😞 Negative Reviews",
            f"{negative_count:,}",
            f"{negative_percent:.1f}% of total"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # ANALYSIS + SENTIMENT CHART
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [1.05, 0.95]
    )


    # ========================================================
    # LEFT - REVIEW ANALYSIS
    # ========================================================

    with left_col:

        st.subheader(
            "💬 Analyze a Single Review"
        )

        st.write(
            "Enter a customer review below and let "
            "the machine learning model predict its sentiment."
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

        st.caption(
            f"{len(review)}/1000 characters"
        )

        if st.button(
            "🔍  Analyze Sentiment",
            use_container_width=True
        ):

            if not review.strip():

                st.warning(
                    "Please enter a customer review."
                )

            else:

                sentiment = predict_sentiment(
                    review
                )

                st.markdown("---")

                if sentiment == "Positive":

                    st.success(
                        "😊 Positive Sentiment"
                    )

                    st.write(
                        "The review expresses a positive opinion."
                    )

                elif sentiment == "Negative":

                    st.error(
                        "😞 Negative Sentiment"
                    )

                    st.write(
                        "The review expresses a negative opinion."
                    )

                else:

                    st.warning(
                        "😐 Neutral Sentiment"
                    )

                    st.write(
                        "The review expresses a relatively neutral opinion."
                    )


    # ========================================================
    # RIGHT - SENTIMENT DISTRIBUTION
    # ========================================================

    with right_col:

        st.subheader(
            "📊 Customer Sentiment Distribution"
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

        pie_fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            hole=0.45
        )

        pie_fig.update_traces(
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            textposition="inside"
        )

        pie_fig.update_layout(
            height=410,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5
            ),
            legend_title="Sentiment",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )


    st.divider()


    # ========================================================
    # RATING DISTRIBUTION
    # ========================================================

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
        "Number of Reviews": rating_counts.values
    })

    rating_fig = px.bar(
        rating_df,
        x="Rating",
        y="Number of Reviews",
        text="Number of Reviews"
    )

    rating_fig.update_traces(
        textposition="outside"
    )

    rating_fig.update_layout(
        height=390,
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        rating_fig,
        use_container_width=True
    )


    # ========================================================
    # DATASET INSIGHTS
    # ========================================================

    st.subheader(
        "📌 Dataset Insights"
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "⭐ Average Rating",
            f"{average_rating:.2f}"
        )

    with i2:

        st.metric(
            "📈 Positive Ratio",
            f"{positive_percent:.1f}%"
        )

    with i3:

        st.metric(
            "📉 Negative Ratio",
            f"{negative_percent:.1f}%"
        )


    # ========================================================
    # GOAL
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(
        "🎯 Our Goal"
    )

    st.info(
        "Help businesses understand customer opinions, "
        "identify areas for improvement, and build better "
        "products through data-driven insights."
    )


# ============================================================
# SINGLE REVIEW PAGE
# ============================================================

elif page == "💬 Single Review":

    st.title(
        "💬 Single Review Analysis"
    )

    st.write(
        "Enter a customer review and predict its sentiment."
    )

    review = st.text_area(
        "Customer Review",
        placeholder=(
            "Example: I am extremely happy with this "
            "product. The quality is excellent!"
        ),
        height=220
    )

    if st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    ):

        if not review.strip():

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


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📄 Batch Prediction":

    st.title(
        "📄 Batch Prediction"
    )

    st.write(
        "Upload multiple customer reviews in CSV format."
    )

    st.info(
        "Your CSV file must contain a column named "
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
            "📋 Uploaded Data"
        )

        st.dataframe(
            batch_df,
            use_container_width=True,
            hide_index=True
        )

        if "review" not in batch_df.columns:

            st.error(
                "❌ Column 'review' was not found."
            )

        else:

            if st.button(
                "🚀 Predict All Reviews",
                use_container_width=True
            ):

                predictions = []

                for review in batch_df["review"]:

                    predictions.append(
                        predict_sentiment(review)
                    )

                batch_df["sentiment"] = predictions

                st.success(
                    "✅ Prediction completed!"
                )

                st.dataframe(
                    batch_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = batch_df.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download Predictions",
                    csv_data,
                    "sentiment_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )


# ============================================================
# VISUALIZATIONS PAGE
# ============================================================

elif page == "📊 Visualizations":

    st.title(
        "📊 Data Visualizations"
    )

    st.write(
        "Explore the sentiment and rating patterns "
        "in the customer review dataset."
    )


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    st.subheader(
        "😊 Sentiment Distribution"
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

    fig1 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        hole=0.4
    )

    fig1.update_traces(
        textinfo="percent",
        texttemplate="%{percent:.1%}"
    )

    fig1.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

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

    fig2 = px.bar(
        rating_df,
        x="Rating",
        y="Reviews",
        text="Reviews"
    )

    fig2.update_traces(
        textposition="outside"
    )

    fig2.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🏆 Model Performance":

    st.title(
        "🏆 Model Performance"
    )

    st.write(
        "Comparison of the machine learning models "
        "evaluated for sentiment classification."
    )


    model_results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Linear SVM"
        ],

        "Accuracy": [
            80.21,
            71.88,
            80.90
        ],

        "Precision": [
            78.84,
            63.79,
            78.59
        ],

        "Recall": [
            80.21,
            71.88,
            80.90
        ],

        "F1 Score": [
            79.37,
            65.94,
            79.06
        ]
    })


    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )


    chart_data = model_results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )


    performance_fig = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score"
    )


    performance_fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    performance_fig.update_layout(
        height=500,
        yaxis_title="Score (%)",
        paper_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        performance_fig,
        use_container_width=True
    )


    st.success(
        "🏆 Final Selected Model: Logistic Regression "
        "with an F1 Score of 79.37%."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About ReviewSense"
    )

    st.write(
        "NLP-Based Customer Review Sentiment Analysis"
    )

    st.divider()


    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        """
        The objective of this project is to analyze customer
        reviews using Natural Language Processing and Machine
        Learning.

        The system identifies whether a customer review is:

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
        f"""
        The dataset contains **{total_reviews:,} customer reviews**.

        The main columns are:

        • **Title** – Review title

        • **Rating** – Customer rating from 1 to 5

        • **Body** – Customer review text
        """
    )


    st.subheader(
        "⭐ Sentiment Mapping"
    )

    mapping_df = pd.DataFrame({

        "Rating": [
            "1 – 2",
            "3",
            "4 – 5"
        ],

        "Sentiment": [
            "😞 Negative",
            "😐 Neutral",
            "😊 Positive"
        ]
    })


    st.table(
        mapping_df
    )


    st.divider()


    st.subheader(
        "🧠 NLP Workflow"
    )

    st.write(
        """
        **1. Data Preparation**

        Customer review data is loaded and checked.

        **2. Text Preprocessing**

        Review text is cleaned before modelling.

        **3. Feature Engineering**

        TF-IDF Vectorization converts text into numerical
        features.

        **4. Model Building**

        Logistic Regression, Multinomial Naive Bayes and
        Linear SVM are evaluated.

        **5. Model Evaluation**

        Models are compared using Accuracy, Precision,
        Recall and F1 Score.

        **6. Prediction**

        The trained model predicts sentiment for new reviews.

        **7. Deployment**

        The application is deployed using Streamlit.
        """
    )


    st.divider()


    st.subheader(
        "🛠️ Technologies"
    )

    tech_cols = st.columns(5)

    technologies = [
        ("🐍", "Python"),
        ("🧠", "NLP"),
        ("📊", "Pandas"),
        ("🤖", "Scikit-learn"),
        ("🌐", "Streamlit")
    ]

    for col, (icon, name) in zip(
        tech_cols,
        technologies
    ):

        with col:

            st.info(
                f"{icon}\n\n{name}"
            )


    st.divider()


    st.subheader(
        "💡 Project Goal"
    )

    st.success(
        "Turn customer feedback into valuable insights "
        "that can help businesses understand customers "
        "and improve their products."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        ❤️ ReviewSense | Customer Review Sentiment Analysis
        <br>
        Built with Python • NLP • Machine Learning • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)