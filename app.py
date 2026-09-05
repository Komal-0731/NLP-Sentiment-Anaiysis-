import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ReviewSense | Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   MAIN PAGE
===================================================== */

.stApp {
    background:
        radial-gradient(circle at 85% 5%, #f5eaff 0%, transparent 25%),
        radial-gradient(circle at 20% 20%, #e9f2ff 0%, transparent 30%),
        linear-gradient(135deg, #f7f9ff, #fbfaff);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #edf3ff 0%,
        #f2efff 55%,
        #f8f4ff 100%
    );

    border-right: 1px solid #e0e6f4;
}

.sidebar-brand {
    padding: 10px 5px 20px 5px;
}

.sidebar-logo {
    font-size: 30px;
}

.sidebar-brand-name {
    font-size: 25px;
    font-weight: 800;
    color: #172b61;
}

.sidebar-brand-sub {
    font-size: 13px;
    color: #687594;
    margin-left: 3px;
}

.sidebar-line {
    height: 1px;
    background: #d6ddec;
    margin: 12px 0 25px 0;
}

.sidebar-section {
    color: #697592;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}

.sidebar-quote {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 22px 15px;
    margin-top: 35px;
    text-align: center;
    color: #596789;
    font-size: 15px;
    font-style: italic;
    line-height: 1.7;
    box-shadow: 0 5px 20px rgba(60,70,120,0.05);
}


/* =====================================================
   SIDEBAR BUTTONS
===================================================== */

section[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: #172b61;
    border: none;
    text-align: left;
    box-shadow: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 500;
    padding: 10px 14px;
    min-height: 42px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,91,255,0.10);
    color: #4d46d9;
}


/* =====================================================
   HERO
===================================================== */

.welcome-text {
    color: #667399;
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 3px;
}

.hero-title {
    font-size: 44px;
    line-height: 1.12;
    font-weight: 850;

    background: linear-gradient(
        90deg,
        #163b91,
        #315bd5,
        #7228dc
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #66728f;
    font-size: 17px;
    line-height: 1.6;
    margin-top: 8px;
}

.hero-art {
    background:
        radial-gradient(circle at 30% 30%, #ffffff 0%, transparent 28%),
        linear-gradient(135deg, #eeeaff, #fceaf5);
    border-radius: 25px;
    min-height: 170px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.hero-art-main {
    font-size: 80px;
    z-index: 2;
}

.hero-art-text {
    position: absolute;
    top: 17px;
    right: 18px;
    color: #34427b;
    font-style: italic;
    font-size: 15px;
    text-align: center;
}

.hero-art-bottom {
    position: absolute;
    bottom: -20px;
    right: 20px;
    font-size: 80px;
}


/* =====================================================
   METRIC CARDS
===================================================== */

.metric-card {
    border-radius: 21px;
    padding: 20px;
    min-height: 138px;
    border: 1px solid rgba(255,255,255,0.9);
    box-shadow: 0 8px 25px rgba(48,61,110,0.08);
}

.positive {
    background: linear-gradient(135deg,#e8fff2,#f8fffb);
}

.neutral {
    background: linear-gradient(135deg,#fff8df,#fffdf5);
}

.negative {
    background: linear-gradient(135deg,#ffedf1,#fff9fa);
}

.metric-icon {
    font-size: 39px;
    float: left;
    margin-right: 13px;
}

.metric-label {
    color: #26385f;
    font-size: 16px;
    font-weight: 600;
}

.metric-number {
    color: #142452;
    font-size: 33px;
    font-weight: 850;
    line-height: 1.15;
    margin-top: 4px;
}

.metric-percent {
    color: #6a7692;
    font-size: 14px;
    margin-top: 5px;
}


/* =====================================================
   GENERAL CARDS
===================================================== */

.card {
    background: rgba(255,255,255,0.93);
    border: 1px solid #e9edf7;
    border-radius: 22px;
    padding: 23px;
    box-shadow: 0 8px 28px rgba(43,55,100,0.065);
}

.section-title {
    color: #172b61;
    font-size: 24px;
    font-weight: 800;
}

.section-subtitle {
    color: #687593;
    font-size: 15px;
    margin-top: 5px;
    margin-bottom: 15px;
}


/* =====================================================
   INPUT
===================================================== */

.stTextArea textarea {
    background: #f7f9fd !important;
    border: 1px solid #dbe1ef !important;
    border-radius: 13px !important;
    color: #27365a !important;
    font-size: 15px !important;
}


/* =====================================================
   BUTTON
===================================================== */

.stButton > button {
    border-radius: 13px;
    min-height: 48px;
    border: none;
    font-size: 16px;
    font-weight: 650;

    background: linear-gradient(
        90deg,
        #625cff,
        #7450e8
    );

    color: white;

    box-shadow:
        0 7px 18px rgba(92,83,220,0.22);
}

.stButton > button:hover {
    color: white;
    background: linear-gradient(
        90deg,
        #5148eb,
        #663dd3
    );
}


/* =====================================================
   RESULT
===================================================== */

.result-card {
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    text-align: center;
}

.result-positive {
    background: #ecfff5;
    border: 1px solid #afe8c9;
}

.result-negative {
    background: #fff0f2;
    border: 1px solid #ffb9c3;
}

.result-neutral {
    background: #fff9e8;
    border: 1px solid #f1d77d;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
    color: #172b61;
}

.review-box {
    background: white;
    border-radius: 11px;
    padding: 13px;
    color: #65718b;
    margin-top: 12px;
    font-style: italic;
}


/* =====================================================
   INSIGHT CARDS
===================================================== */

.insight-card {
    background: white;
    border: 1px solid #e6eaf4;
    border-radius: 16px;
    padding: 17px;
    min-height: 115px;
    box-shadow: 0 5px 18px rgba(40,55,100,0.05);
}

.insight-icon {
    font-size: 31px;
}

.insight-title {
    color: #1c3470;
    font-size: 15px;
    font-weight: 700;
    margin-top: 5px;
}

.insight-value {
    color: #172b61;
    font-size: 25px;
    font-weight: 850;
}


/* =====================================================
   GOAL
===================================================== */

.goal-banner {
    background:
        radial-gradient(
            circle at 85% 50%,
            rgba(255,255,255,0.7),
            transparent 25%
        ),
        linear-gradient(
            110deg,
            #eee8ff,
            #e9f1ff,
            #fff0fa
        );

    border-radius: 22px;
    padding: 24px 28px;
    margin-top: 20px;
    border: 1px solid #e0daf5;
}

.goal-title {
    color: #172b61;
    font-size: 25px;
    font-weight: 850;
}

.goal-text {
    color: #596782;
    font-size: 16px;
    line-height: 1.6;
}


/* =====================================================
   PAGE HEADER
===================================================== */

.page-title {
    font-size: 39px;
    font-weight: 850;
    color: #172b61;
}

.page-subtitle {
    color: #687593;
    font-size: 16px;
    margin-bottom: 25px;
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    text-align: center;
    color: #71809e;
    font-size: 14px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("P652-Dataset.xlsx")

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


df = load_data()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    package = joblib.load(
        "sentiment_model.joblib"
    )

    return (
        package["tfidf"],
        package["model"]
    )


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

    cleaned = clean_text(review)

    vector = tfidf.transform(
        [cleaned]
    )

    prediction = model.predict(
        vector
    )[0]

    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            vector
        )[0]

        confidence = (
            max(probabilities) * 100
        )

    return prediction, confidence


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

positive_pct = (
    positive_count / total_reviews * 100
)

neutral_pct = (
    neutral_count / total_reviews * 100
)

negative_pct = (
    negative_count / total_reviews * 100
)

average_rating = df["rating"].mean()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

if "page" not in st.session_state:

    st.session_state.page = "Home"


with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <span class="sidebar-logo">🛒</span>

            <span class="sidebar-brand-name">
            ReviewSense
            </span>

            <div class="sidebar-brand-sub">
            Understand • Improve • Grow
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Navigation</div>',
        unsafe_allow_html=True
    )


    if st.button(
        "🏠  Home",
        use_container_width=True
    ):
        st.session_state.page = "Home"


    if st.button(
        "💬  Single Review",
        use_container_width=True
    ):
        st.session_state.page = "Single Review"


    if st.button(
        "📄  Batch Prediction",
        use_container_width=True
    ):
        st.session_state.page = "Batch Prediction"


    if st.button(
        "📊  Visualizations",
        use_container_width=True
    ):
        st.session_state.page = "Visualizations"


    if st.button(
        "🏆  Model Performance",
        use_container_width=True
    ):
        st.session_state.page = "Model Performance"


    if st.button(
        "ℹ️  About",
        use_container_width=True
    ):
        st.session_state.page = "About"


    st.markdown(
        """
        <div class="sidebar-quote">

        💬<br><br>

        “Every review is a story.<br>
        Let's turn them into insights.”

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br><br><br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
        text-align:center;
        color:#687594;
        font-size:14px;">

        ❤️ Built with Streamlit<br>
        <small>Made for a better tomorrow</small>

        </div>
        """,
        unsafe_allow_html=True
    )


page = st.session_state.page


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    # -----------------------------------------------------
    # HERO
    # -----------------------------------------------------

    hero_left, hero_right = st.columns(
        [3.4, 1]
    )

    with hero_left:

        st.markdown(
            """
            <div class="welcome-text">
            Welcome to
            </div>

            <div class="hero-title">
            Customer Review Sentiment Analysis
            </div>

            <div class="hero-subtitle">
            Turn customer feedback into meaningful insights
            with the power of NLP and Machine Learning.
            </div>
            """,
            unsafe_allow_html=True
        )


    with hero_right:

        st.markdown(
            """
            <div class="hero-art">

                <div class="hero-art-text">
                “Happy Customers<br>
                Brighter Businesses”
                </div>

                <div class="hero-art-main">
                👩‍💻
                </div>

                <div class="hero-art-bottom">
                📈
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card positive">

                <div class="metric-icon">
                😊
                </div>

                <div class="metric-label">
                Positive Reviews
                </div>

                <div class="metric-number">
                {positive_count}
                </div>

                <div class="metric-percent">
                {positive_pct:.1f}% of total
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card neutral">

                <div class="metric-icon">
                😐
                </div>

                <div class="metric-label">
                Neutral Reviews
                </div>

                <div class="metric-number">
                {neutral_count}
                </div>

                <div class="metric-percent">
                {neutral_pct:.1f}% of total
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card negative">

                <div class="metric-icon">
                😞
                </div>

                <div class="metric-label">
                Negative Reviews
                </div>

                <div class="metric-number">
                {negative_count}
                </div>

                <div class="metric-percent">
                {negative_pct:.1f}% of total
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------------------------------
    # REVIEW + PIE
    # -----------------------------------------------------

    left, right = st.columns(
        [1.05, 0.95]
    )


    # REVIEW CARD
    with left:

        st.markdown(
            """
            <div class="card">

                <div class="section-title">
                💬 Analyze a Single Review
                </div>

                <div class="section-subtitle">
                Enter a customer review below and get
                the predicted sentiment.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        review = st.text_area(
            "review",
            placeholder=
            "Example: The product quality is excellent and I really love it!",
            height=135,
            label_visibility="collapsed"
        )


        if st.button(
            "🔍  Analyze Sentiment",
            key="home_analysis",
            use_container_width=True
        ):

            if not review.strip():

                st.warning(
                    "Please enter a customer review."
                )

            else:

                sentiment, confidence = (
                    predict_sentiment(review)
                )


                if sentiment == "Positive":

                    st.markdown(
                        f"""
                        <div class="result-card result-positive">

                            <div class="result-title">
                            😊 Positive Sentiment
                            </div>

                            <p>
                            <b>Predicted Sentiment:</b>
                            Positive
                            </p>

                            {
                                f"<p><b>Confidence:</b> {confidence:.1f}%</p>"
                                if confidence is not None else ""
                            }

                            <div class="review-box">
                            "{review}"
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                elif sentiment == "Negative":

                    st.markdown(
                        f"""
                        <div class="result-card result-negative">

                            <div class="result-title">
                            😞 Negative Sentiment
                            </div>

                            <p>
                            <b>Predicted Sentiment:</b>
                            Negative
                            </p>

                            {
                                f"<p><b>Confidence:</b> {confidence:.1f}%</p>"
                                if confidence is not None else ""
                            }

                            <div class="review-box">
                            "{review}"
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                else:

                    st.markdown(
                        f"""
                        <div class="result-card result-neutral">

                            <div class="result-title">
                            😐 Neutral Sentiment
                            </div>

                            <p>
                            <b>Predicted Sentiment:</b>
                            Neutral
                            </p>

                            {
                                f"<p><b>Confidence:</b> {confidence:.1f}%</p>"
                                if confidence is not None else ""
                            }

                            <div class="review-box">
                            "{review}"
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # PIE CHART
    with right:

        st.markdown(
            """
            <div class="section-title">
            📊 Customer Review Sentiment Distribution
            </div>
            """,
            unsafe_allow_html=True
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


        pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index
        )


        pie.update_traces(
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            textposition="inside",
            pull=[
                0.025,
                0.025,
                0.025
            ]
        )


        pie.update_layout(
            height=430,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5
            ),
            legend_title="Sentiment",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )


        st.plotly_chart(
            pie,
            use_container_width=True
        )


    # -----------------------------------------------------
    # RATING + DATASET INSIGHTS
    # -----------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    rating_col, insight_col = st.columns(
        [1.45, 0.85]
    )


    # RATING
    with rating_col:

        st.markdown(
            """
            <div class="section-title">
            ⭐ Rating Distribution
            </div>
            """,
            unsafe_allow_html=True
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
            height=365,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Rating",
            yaxis_title="Number of Reviews"
        )


        st.plotly_chart(
            rating_fig,
            use_container_width=True
        )


    # INSIGHTS
    with insight_col:

        st.markdown(
            """
            <div class="section-title">
            🗄️ Dataset Insights
            </div>
            """,
            unsafe_allow_html=True
        )


        i1, i2 = st.columns(2)


        with i1:

            st.markdown(
                f"""
                <div class="insight-card">

                    <div class="insight-icon">
                    📄
                    </div>

                    <div class="insight-title">
                    Total Reviews
                    </div>

                    <div class="insight-value">
                    {total_reviews:,}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with i2:

            st.markdown(
                f"""
                <div class="insight-card">

                    <div class="insight-icon">
                    ⭐
                    </div>

                    <div class="insight-title">
                    Average Rating
                    </div>

                    <div class="insight-value">
                    {average_rating:.2f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)


        i3, i4 = st.columns(2)


        with i3:

            st.markdown(
                f"""
                <div class="insight-card">

                    <div class="insight-icon">
                    📈
                    </div>

                    <div class="insight-title">
                    Positive Ratio
                    </div>

                    <div class="insight-value">
                    {positive_pct:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with i4:

            st.markdown(
                f"""
                <div class="insight-card">

                    <div class="insight-icon">
                    📉
                    </div>

                    <div class="insight-title">
                    Negative Ratio
                    </div>

                    <div class="insight-value">
                    {negative_pct:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # GOAL
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="goal-banner">

            <div class="goal-title">
            🎯 Our Goal
            </div>

            <div class="goal-text">
            Help businesses understand customer opinions,
            identify areas for improvement, and build better
            products through data-driven insights.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SINGLE REVIEW PAGE
# =========================================================

elif page == "Single Review":

    st.markdown(
        """
        <div class="page-title">
        💬 Single Review Analysis
        </div>

        <div class="page-subtitle">
        Analyze an individual customer review using our NLP model.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <div class="section-title">
        🔍 Enter Customer Review
        </div>

        <div class="section-subtitle">
        The trained machine learning model will classify
        the review as Positive, Neutral or Negative.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    review = st.text_area(
        "Customer Review",
        placeholder=
        "Example: I am very happy with this product. The quality is excellent!",
        height=200
    )


    if st.button(
        "🔍  Predict Sentiment",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Please enter a review."
            )

        else:

            sentiment, confidence = (
                predict_sentiment(review)
            )


            if sentiment == "Positive":

                st.success(
                    f"😊 Positive Sentiment"
                )

            elif sentiment == "Negative":

                st.error(
                    f"😞 Negative Sentiment"
                )

            else:

                st.warning(
                    f"😐 Neutral Sentiment"
                )


            if confidence is not None:

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )

                st.caption(
                    f"Prediction confidence: {confidence:.1f}%"
                )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.markdown(
        """
        <div class="page-title">
        📄 Batch Prediction
        </div>

        <div class="page-subtitle">
        Analyze multiple customer reviews at the same time.
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )


    if uploaded_file:

        batch_df = pd.read_csv(
            uploaded_file
        )


        if "review" not in batch_df.columns:

            st.error(
                "Your CSV must contain a column named 'review'."
            )

        else:

            predictions = []

            confidences = []


            for review in batch_df["review"]:

                sentiment, confidence = (
                    predict_sentiment(review)
                )

                predictions.append(
                    sentiment
                )

                confidences.append(
                    confidence
                )


            batch_df["sentiment"] = predictions


            if any(
                c is not None
                for c in confidences
            ):

                batch_df["confidence"] = [
                    round(c, 2)
                    if c is not None
                    else None
                    for c in confidences
                ]


            st.success(
                "✅ Prediction completed successfully!"
            )


            st.dataframe(
                batch_df,
                use_container_width=True,
                hide_index=True
            )


            csv = batch_df.to_csv(
                index=False
            )


            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "sentiment_predictions.csv",
                "text/csv",
                use_container_width=True
            )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "Visualizations":

    st.markdown(
        """
        <div class="page-title">
        📊 Visualizations
        </div>

        <div class="page-subtitle">
        Explore customer sentiment and rating patterns.
        </div>
        """,
        unsafe_allow_html=True
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
        title="Customer Review Sentiment Distribution"
    )


    fig1.update_traces(
        textinfo="percent",
        texttemplate="%{percent:.1%}"
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
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
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        """
        <div class="page-title">
        🏆 Model Performance
        </div>

        <div class="page-subtitle">
        Comparison of the machine learning models evaluated
        for sentiment classification.
        </div>
        """,
        unsafe_allow_html=True
    )


    model_df = pd.DataFrame({
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
        model_df.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        """
        <div class="goal-banner">

            <div class="goal-title">
            🏆 Selected Model: Logistic Regression
            </div>

            <div class="goal-text">

            Logistic Regression was selected as the final model
            because it achieved the highest F1 Score among the
            evaluated models.

            <br><br>

            <b>F1 Score: 79.37%</b>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.markdown(
        """
        <div class="page-title">
        ℹ️ About ReviewSense
        </div>

        <div class="page-subtitle">
        NLP-based Customer Review Sentiment Analysis
        </div>
        """,
        unsafe_allow_html=True
    )


    # PROJECT OVERVIEW
    st.markdown(
        """
        <div class="card">

            <div class="section-title">
            🎯 Project Overview
            </div>

            <p style="color:#5f6d89;font-size:16px;line-height:1.7;">

            Customer Review Sentiment Analysis is an NLP and
            Machine Learning project designed to analyze customer
            feedback and identify the sentiment expressed in reviews.

            </p>

            <p style="font-size:18px;">

            😊 <b>Positive</b>
            &nbsp;&nbsp;&nbsp;
            😐 <b>Neutral</b>
            &nbsp;&nbsp;&nbsp;
            😞 <b>Negative</b>

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # DATASET
    st.markdown(
        """
        <div class="section-title">
        📊 Dataset Information
        </div>
        """,
        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:
        st.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )


    with b:
        st.metric(
            "Features",
            "3"
        )


    with c:
        st.metric(
            "Sentiment Classes",
            "3"
        )


    st.markdown(
        """
        ### Dataset Features

        | Feature | Description |
        |---|---|
        | `title` | Customer review title |
        | `rating` | Customer rating from 1 to 5 |
        | `body` | Detailed customer review |

        ### Sentiment Mapping

        ⭐ **1–2 → Negative**  
        ⭐ **3 → Neutral**  
        ⭐ **4–5 → Positive**
        """
    )


    st.divider()


    # WORKFLOW
    st.markdown(
        """
        <div class="section-title">
        🔄 NLP & Machine Learning Workflow
        </div>
        """,
        unsafe_allow_html=True
    )


    workflow = [
        ("📥", "Data Collection"),
        ("🧹", "Data Preprocessing"),
        ("⚙️", "Feature Engineering"),
        ("🔤", "TF-IDF"),
        ("🤖", "Model Training"),
        ("📈", "Evaluation"),
        ("🔮", "Prediction"),
        ("🌐", "Deployment")
    ]


    workflow_cols = st.columns(4)


    for i, (icon, title) in enumerate(
        workflow
    ):

        with workflow_cols[i % 4]:

            st.markdown(
                f"""
                <div class="insight-card"
                     style="margin-bottom:15px;">

                    <div class="insight-icon">
                    {icon}
                    </div>

                    <div class="insight-title">
                    {title}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.divider()


    # TECHNOLOGIES
    st.markdown(
        """
        <div class="section-title">
        🛠️ Technologies Used
        </div>
        """,
        unsafe_allow_html=True
    )


    techs = [
        ("🐍", "Python"),
        ("🧠", "NLP"),
        ("🤖", "Scikit-learn"),
        ("📊", "Plotly"),
        ("🌐", "Streamlit")
    ]


    tech_cols = st.columns(5)


    for col, (icon, name) in zip(
        tech_cols,
        techs
    ):

        with col:

            st.markdown(
                f"""
                <div class="insight-card"
                     style="text-align:center;">

                    <div style="font-size:35px;">
                    {icon}
                    </div>

                    <div class="insight-title">
                    {name}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        """
        <br>

        **Libraries:** Pandas • NumPy • Scikit-learn •
        Joblib • Plotly • OpenPyXL
        """
    )


    # BUSINESS OBJECTIVE
    st.markdown(
        """
        <div class="goal-banner">

            <div class="goal-title">
            💡 Business Objective
            </div>

            <div class="goal-text">

            Transform unstructured customer feedback into
            meaningful insights.

            <br><br>

            ✅ Understand customer satisfaction<br>
            ✅ Identify negative customer experiences<br>
            ✅ Monitor customer feedback<br>
            ✅ Improve products and services<br>
            ✅ Support data-driven decisions

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    ❤️ ReviewSense | Customer Review Sentiment Analysis

    <br>

    NLP • Machine Learning • Data Science • Streamlit

    <br>

    <i>
    “Turning Customer Feedback into Valuable Insights”
    </i>

    </div>
    """,
    unsafe_allow_html=True
)