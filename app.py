import streamlit as st
import pandas as pd
import joblib
import re
import textwrap
import html as html_escape
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ReviewSense | Customer Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HTML HELPER
# =========================================================
# This prevents Streamlit from displaying HTML as plain text.

def render_html(content):
    st.markdown(
        textwrap.dedent(content),
        unsafe_allow_html=True
    )


# =========================================================
# CUSTOM CSS
# =========================================================

render_html("""
<style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(226, 218, 255, 0.65),
                transparent 28%
            ),
            radial-gradient(
                circle at 15% 15%,
                rgba(218, 235, 255, 0.60),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f7f9ff 0%,
                #fbfaff 50%,
                #fff9fc 100%
            );
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #edf3ff 0%,
                #f1efff 50%,
                #f8f4ff 100%
            );

        border-right: 1px solid #dfe5f3;
    }

    .brand-container {
        padding: 8px 5px 18px 5px;
    }

    .brand-icon {
        font-size: 30px;
        vertical-align: middle;
    }

    .brand-name {
        color: #142a60;
        font-size: 25px;
        font-weight: 850;
        vertical-align: middle;
    }

    .brand-tagline {
        color: #697694;
        font-size: 12px;
        margin-left: 38px;
        margin-top: -2px;
    }

    .sidebar-divider {
        height: 1px;
        background: #d6ddec;
        margin: 10px 0 24px 0;
    }

    .nav-heading {
        color: #687593;
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    section[data-testid="stSidebar"] .stButton {
        margin-bottom: 4px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: transparent;
        color: #1b2e5c;
        border: none;
        border-radius: 12px;
        text-align: left;
        font-size: 15px;
        font-weight: 550;
        min-height: 43px;
        box-shadow: none;
        padding-left: 14px;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 91, 255, 0.12);
        color: #5148d9;
    }

    .sidebar-quote {
        margin-top: 35px;
        padding: 22px 14px;
        border-radius: 20px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 7px 22px rgba(50,65,110,0.05);
        text-align: center;
        color: #596887;
        font-size: 15px;
        font-style: italic;
        line-height: 1.75;
    }

    .sidebar-footer {
        text-align: center;
        color: #687594;
        font-size: 14px;
        margin-top: 50px;
        line-height: 1.7;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .welcome {
        color: #687593;
        font-size: 19px;
        font-weight: 500;
        margin-bottom: 2px;
    }

    .hero-title {
        font-size: 43px;
        font-weight: 850;
        line-height: 1.12;

        background: linear-gradient(
            90deg,
            #153a8f 0%,
            #315bd4 45%,
            #7025d8 100%
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #65728f;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 8px;
    }

    .hero-art {
        height: 170px;
        border-radius: 25px;
        background:
            radial-gradient(
                circle at 20% 30%,
                rgba(255,255,255,0.9),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #eee9ff,
                #eaf1ff,
                #fdebf7
            );
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 8px 25px rgba(60,70,120,0.05);
    }

    .hero-person {
        font-size: 76px;
        margin-top: 30px;
    }

    .hero-review-one,
    .hero-review-two,
    .hero-review-three {
        position: absolute;
        background: white;
        border-radius: 12px;
        padding: 7px 12px;
        box-shadow: 0 5px 15px rgba(50,60,100,0.10);
        font-size: 13px;
    }

    .hero-review-one {
        top: 12px;
        left: 10px;
    }

    .hero-review-two {
        top: 42px;
        right: 10px;
    }

    .hero-review-three {
        bottom: 12px;
        left: 14px;
    }


    /* =====================================================
       METRIC CARDS
    ===================================================== */

    .metric-card {
        border-radius: 20px;
        padding: 20px;
        min-height: 138px;
        border: 1px solid rgba(255,255,255,0.95);
        box-shadow: 0 8px 26px rgba(48,60,110,0.08);
    }

    .positive-card {
        background: linear-gradient(
            135deg,
            #e6fff1,
            #f8fffb
        );
    }

    .neutral-card {
        background: linear-gradient(
            135deg,
            #fff8df,
            #fffdf7
        );
    }

    .negative-card {
        background: linear-gradient(
            135deg,
            #ffedf1,
            #fff9fa
        );
    }

    .metric-icon {
        float: left;
        font-size: 40px;
        margin-right: 14px;
    }

    .metric-label {
        color: #26375e;
        font-size: 16px;
        font-weight: 650;
    }

    .metric-number {
        color: #122554;
        font-size: 34px;
        font-weight: 850;
        line-height: 1.1;
        margin-top: 5px;
    }

    .metric-percent {
        color: #687594;
        font-size: 14px;
        margin-top: 5px;
    }


    /* =====================================================
       SECTION TITLES
    ===================================================== */

    .section-title {
        color: #172b61;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 3px;
    }

    .section-subtitle {
        color: #687593;
        font-size: 15px;
        margin-bottom: 15px;
    }


    /* =====================================================
       REVIEW CARD
    ===================================================== */

    .review-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #e8edf7;
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 26px rgba(45,55,100,0.065);
    }

    .stTextArea textarea {
        background: #f7f9fd !important;
        border: 1px solid #d9e0ef !important;
        border-radius: 13px !important;
        color: #26375c !important;
        font-size: 15px !important;
    }

    .stTextArea textarea:focus {
        border-color: #7469ed !important;
        box-shadow: 0 0 0 1px #7469ed !important;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {
        min-height: 48px;
        border-radius: 13px;
        border: none;

        background: linear-gradient(
            90deg,
            #625cff,
            #7351e8
        );

        color: white;
        font-size: 16px;
        font-weight: 650;

        box-shadow:
            0 7px 18px rgba(91,83,220,0.23);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #5148e8,
            #643ed2
        );

        color: white;
    }


    /* =====================================================
       RESULT CARDS
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
        border: 1px solid #ffb9c2;
    }

    .result-neutral {
        background: #fff9e8;
        border: 1px solid #f0d477;
    }

    .result-title {
        color: #172b61;
        font-size: 27px;
        font-weight: 800;
    }

    .review-box {
        background: white;
        border-radius: 11px;
        padding: 13px;
        margin-top: 12px;
        color: #64708b;
        font-style: italic;
    }


    /* =====================================================
       INSIGHT CARDS
    ===================================================== */

    .insight-card {
        background: white;
        border: 1px solid #e6ebf5;
        border-radius: 16px;
        padding: 17px;
        min-height: 108px;
        box-shadow: 0 5px 18px rgba(45,55,100,0.05);
    }

    .insight-icon {
        font-size: 30px;
    }

    .insight-title {
        color: #1d3470;
        font-size: 14px;
        font-weight: 650;
        margin-top: 4px;
    }

    .insight-value {
        color: #172b61;
        font-size: 25px;
        font-weight: 850;
        margin-top: 2px;
    }


    /* =====================================================
       GOAL BANNER
    ===================================================== */

    .goal-banner {
        background:
            radial-gradient(
                circle at 88% 50%,
                rgba(255,255,255,0.8),
                transparent 25%
            ),
            linear-gradient(
                110deg,
                #eee8ff,
                #eaf1ff,
                #fff0fa
            );

        border-radius: 22px;
        padding: 23px 28px;
        border: 1px solid #e0daf5;
        margin-top: 20px;
    }

    .goal-title {
        color: #172b61;
        font-size: 24px;
        font-weight: 850;
    }

    .goal-text {
        color: #596782;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 5px;
    }


    /* =====================================================
       PAGE HEADERS
    ===================================================== */

    .page-title {
        color: #172b61;
        font-size: 40px;
        font-weight: 850;
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
        line-height: 1.7;
        padding: 28px 0 5px 0;
    }

</style>
""")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_excel(
        "P652-Dataset.xlsx"
    )

    def rating_to_sentiment(rating):

        if rating <= 2:
            return "Negative"

        elif rating == 3:
            return "Neutral"

        else:
            return "Positive"

    data["sentiment"] = data["rating"].apply(
        rating_to_sentiment
    )

    return data


df = load_data()


# =========================================================
# LOAD TRAINED MODEL
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
# TEXT PREPROCESSING
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
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(review):

    cleaned_review = clean_text(
        review
    )

    review_vector = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    confidence = None

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            review_vector
        )[0]

        confidence = (
            max(probabilities) * 100
        )

    return (
        prediction,
        confidence,
        probabilities
    )


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
    positive_count /
    total_reviews *
    100
)

neutral_pct = (
    neutral_count /
    total_reviews *
    100
)

negative_pct = (
    negative_count /
    total_reviews *
    100
)

average_rating = df["rating"].mean()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:

    st.session_state.page = "Home"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html("""
    <div class="brand-container">

        <span class="brand-icon">🛒</span>

        <span class="brand-name">
        ReviewSense
        </span>

        <div class="brand-tagline">
        Understand • Improve • Grow
        </div>

    </div>
    """)

    render_html(
        '<div class="sidebar-divider"></div>'
    )

    render_html(
        '<div class="nav-heading">Navigation</div>'
    )


    # HOME
    if st.button(
        "🏠  Home",
        key="nav_home",
        use_container_width=True
    ):

        st.session_state.page = "Home"


    # SINGLE REVIEW
    if st.button(
        "💬  Single Review",
        key="nav_single",
        use_container_width=True
    ):

        st.session_state.page = "Single Review"


    # BATCH
    if st.button(
        "📄  Batch Prediction",
        key="nav_batch",
        use_container_width=True
    ):

        st.session_state.page = "Batch Prediction"


    # VISUALIZATIONS
    if st.button(
        "📊  Visualizations",
        key="nav_visual",
        use_container_width=True
    ):

        st.session_state.page = "Visualizations"


    # MODEL
    if st.button(
        "🏆  Model Performance",
        key="nav_model",
        use_container_width=True
    ):

        st.session_state.page = "Model Performance"


    # ABOUT
    if st.button(
        "ℹ️  About",
        key="nav_about",
        use_container_width=True
    ):

        st.session_state.page = "About"


    render_html("""
    <div class="sidebar-quote">

        💬<br><br>

        “Every review is a story.<br>
        Let's turn them into insights.”

    </div>
    """)


    render_html("""
    <div class="sidebar-footer">

        ❤️ Built with Streamlit<br>

        <small>
        Made for better decisions
        </small>

    </div>
    """)


page = st.session_state.page


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    # -----------------------------------------------------
    # HERO
    # -----------------------------------------------------

    hero_left, hero_right = st.columns(
        [3.2, 1]
    )


    with hero_left:

        render_html("""
        <div class="welcome">
        Welcome to
        </div>

        <div class="hero-title">
        Customer Review Sentiment Analysis
        </div>

        <div class="hero-description">
        Turn customer feedback into meaningful insights
        with the power of NLP and Machine Learning.
        </div>
        """)


    with hero_right:

        render_html("""
        <div class="hero-art">

            <div class="hero-review-one">
            😊 ⭐⭐⭐⭐⭐
            </div>

            <div class="hero-review-two">
            😐 ⭐⭐⭐
            </div>

            <div class="hero-review-three">
            😞 ⭐⭐
            </div>

            <div class="hero-person">
            👩‍💻
            </div>

        </div>
        """)


    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        render_html(f"""
        <div class="metric-card positive-card">

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
        """)


    with col2:

        render_html(f"""
        <div class="metric-card neutral-card">

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
        """)


    with col3:

        render_html(f"""
        <div class="metric-card negative-card">

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
        """)


    st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------------------------------
    # REVIEW + PIE CHART
    # -----------------------------------------------------

    left_col, right_col = st.columns(
        [1.05, 0.95]
    )


    # REVIEW
    with left_col:

        render_html("""
        <div class="review-card">

            <div class="section-title">
            💬 Analyze a Single Review
            </div>

            <div class="section-subtitle">
            Enter a customer review below and get
            the predicted sentiment.
            </div>

        </div>
        """)


        review = st.text_area(
            "review_home",
            placeholder=
            "Example: The product quality is excellent and I really love it!",
            height=135,
            label_visibility="collapsed"
        )


        if st.button(
            "🔍  Analyze Sentiment",
            key="home_analyze",
            use_container_width=True
        ):

            if not review.strip():

                st.warning(
                    "Please enter a customer review."
                )

            else:

                sentiment, confidence, probabilities = (
                    predict_sentiment(review)
                )


                safe_review = html_escape.escape(
                    review
                )


                if sentiment == "Positive":

                    render_html(f"""
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
                            if confidence is not None
                            else ""
                        }

                        <div class="review-box">
                        "{safe_review}"
                        </div>

                    </div>
                    """)


                elif sentiment == "Negative":

                    render_html(f"""
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
                            if confidence is not None
                            else ""
                        }

                        <div class="review-box">
                        "{safe_review}"
                        </div>

                    </div>
                    """)


                else:

                    render_html(f"""
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
                            if confidence is not None
                            else ""
                        }

                        <div class="review-box">
                        "{safe_review}"
                        </div>

                    </div>
                    """)


    # PIE
    with right_col:

        render_html("""
        <div class="section-title">
        📊 Customer Review Sentiment Distribution
        </div>
        """)


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
            names=sentiment_counts.index
        )


        pie_fig.update_traces(
            textinfo="percent",
            texttemplate="%{percent:.1%}",
            textposition="inside",
            pull=[
                0.025,
                0.025,
                0.025
            ]
        )


        pie_fig.update_layout(
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
            pie_fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # RATING + INSIGHTS
    # -----------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)


    rating_col, insight_col = st.columns(
        [1.45, 0.85]
    )


    # RATING
    with rating_col:

        render_html("""
        <div class="section-title">
        ⭐ Rating Distribution
        </div>
        """)


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
            height=350,
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

        render_html("""
        <div class="section-title">
        🗄️ Dataset Insights
        </div>
        """)


        a, b = st.columns(2)


        with a:

            render_html(f"""
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
            """)


        with b:

            render_html(f"""
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
            """)


        st.markdown("<br>", unsafe_allow_html=True)


        c, d = st.columns(2)


        with c:

            render_html(f"""
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
            """)


        with d:

            render_html(f"""
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
            """)


    # -----------------------------------------------------
    # GOAL
    # -----------------------------------------------------

    render_html("""
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
    """)


# =========================================================
# SINGLE REVIEW PAGE
# =========================================================

elif page == "Single Review":

    render_html("""
    <div class="page-title">
    💬 Single Review Analysis
    </div>

    <div class="page-subtitle">
    Understand how customers feel about a product using NLP.
    </div>
    """)


    render_html("""
    <div class="review-card">

        <div class="section-title">
        🔍 Enter Customer Review
        </div>

        <div class="section-subtitle">
        The trained machine learning model will classify
        your review as Positive, Neutral or Negative.
        </div>

    </div>
    """)


    review = st.text_area(
        "customer_review",
        placeholder=
        "Example: I am very happy with this product. The quality is excellent!",
        height=190
    )


    if st.button(
        "🔍  Predict Sentiment",
        key="single_predict",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Please enter a review."
            )

        else:

            sentiment, confidence, probabilities = (
                predict_sentiment(review)
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


            if probabilities is not None:

                probability_df = pd.DataFrame({
                    "Sentiment": model.classes_,
                    "Probability": probabilities
                })

                probability_df["Probability"] *= 100

                fig = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Probability",
                    text="Probability"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="",
                    height=330
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =========================================================
# BATCH PREDICTION PAGE
# =========================================================

elif page == "Batch Prediction":

    render_html("""
    <div class="page-title">
    📄 Batch Prediction
    </div>

    <div class="page-subtitle">
    Upload multiple customer reviews and analyze them together.
    </div>
    """)


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

            confidences = []


            for review in batch_df["review"]:

                sentiment, confidence, _ = (
                    predict_sentiment(review)
                )

                predictions.append(
                    sentiment
                )

                confidences.append(
                    confidence
                )


            batch_df["sentiment"] = predictions


            batch_df["confidence"] = [
                round(c, 2)
                if c is not None
                else None
                for c in confidences
            ]


            st.success(
                "✅ Predictions completed successfully!"
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


# =========================================================
# VISUALIZATIONS PAGE
# =========================================================

elif page == "Visualizations":

    render_html("""
    <div class="page-title">
    📊 Visualizations
    </div>

    <div class="page-subtitle">
    Explore customer sentiment and rating patterns.
    </div>
    """)


    # SENTIMENT
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


    # RATING
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
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    render_html("""
    <div class="page-title">
    🏆 Model Performance
    </div>

    <div class="page-subtitle">
    Comparison of the machine learning models evaluated
    for sentiment classification.
    </div>
    """)


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


    # MODEL CHART
    chart_df = model_df[
        ["Model", "Accuracy", "F1 Score"]
    ].melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )


    fig = px.bar(
        chart_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score"
    )


    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig.update_layout(
        yaxis_title="Score (%)",
        xaxis_title="",
        height=430
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    render_html("""
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
    """)


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    render_html("""
    <div class="page-title">
    ℹ️ About ReviewSense
    </div>

    <div class="page-subtitle">
    NLP-based Customer Review Sentiment Analysis
    </div>
    """)


    # OVERVIEW
    render_html("""
    <div class="review-card">

        <div class="section-title">
        🎯 Project Overview
        </div>

        <p style="
        color:#5f6d89;
        font-size:16px;
        line-height:1.7;">

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
    """)


    st.markdown("<br>", unsafe_allow_html=True)


    # DATASET
    render_html("""
    <div class="section-title">
    📊 Dataset Information
    </div>
    """)


    d1, d2, d3 = st.columns(3)


    with d1:
        st.metric(
            "Total Reviews",
            f"{total_reviews:,}"
        )


    with d2:
        st.metric(
            "Features",
            "3"
        )


    with d3:
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
    render_html("""
    <div class="section-title">
    🔄 NLP & Machine Learning Workflow
    </div>
    """)


    workflow = [
        ("📥", "Data Collection"),
        ("🧹", "Data Preprocessing"),
        ("⚙️", "Feature Engineering"),
        ("🔤", "TF-IDF Vectorization"),
        ("🤖", "Model Training"),
        ("📈", "Model Evaluation"),
        ("🔮", "Prediction"),
        ("🌐", "Deployment")
    ]


    workflow_cols = st.columns(4)


    for i, (icon, title) in enumerate(
        workflow
    ):

        with workflow_cols[i % 4]:

            render_html(f"""
            <div class="insight-card"
                 style="margin-bottom:15px;">

                <div class="insight-icon">
                {icon}
                </div>

                <div class="insight-title">
                {title}
                </div>

            </div>
            """)


    st.divider()


    # TECHNOLOGIES
    render_html("""
    <div class="section-title">
    🛠️ Technologies Used
    </div>
    """)


    technologies = [
        ("🐍", "Python"),
        ("🧠", "NLP"),
        ("🤖", "Scikit-learn"),
        ("📊", "Plotly"),
        ("🌐", "Streamlit")
    ]


    tech_cols = st.columns(5)


    for col, (icon, title) in zip(
        tech_cols,
        technologies
    ):

        with col:

            render_html(f"""
            <div class="insight-card"
                 style="text-align:center;">

                <div style="font-size:35px;">
                {icon}
                </div>

                <div class="insight-title">
                {title}
                </div>

            </div>
            """)


    st.markdown(
        """
        <br>

        **Libraries:** Pandas • NumPy • Scikit-learn •
        Joblib • Plotly • OpenPyXL
        """
    )


    st.divider()


    # BUSINESS OBJECTIVE
    render_html("""
    <div class="goal-banner">

        <div class="goal-title">
        💡 Business Objective
        </div>

        <div class="goal-text">

        Transform unstructured customer feedback into
        meaningful sentiment insights.

        <br><br>

        ✅ Understand customer satisfaction<br>
        ✅ Identify negative customer experiences<br>
        ✅ Monitor customer feedback<br>
        ✅ Improve products and services<br>
        ✅ Support data-driven decisions

        </div>

    </div>
    """)


    # FINAL
    render_html("""
    <div style="
        text-align:center;
        padding:35px 10px 10px 10px;
        color:#65718e;">

        <div style="
            font-size:25px;
            font-weight:850;
            color:#172b61;">

            🛒 ReviewSense

        </div>

        <p>
        NLP • Machine Learning • Data Science • Streamlit
        </p>

        <i>
        “Turning Customer Feedback into Valuable Insights”
        </i>

    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">

❤️ ReviewSense | Customer Review Sentiment Analysis

<br>

NLP • Machine Learning • Data Science • Streamlit

<br>

<i>
Turning Customer Feedback into Valuable Insights
</i>

</div>
""")