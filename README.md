# 🛒 ReviewSense – Customer Review Sentiment Analysis

## 📌 Project Overview
ReviewSense is an NLP and Machine Learning project that analyzes customer product reviews and classifies them into:
- 😊 Positive
- 😐 Neutral
- 😞 Negative

The project includes an interactive Streamlit dashboard for single-review analysis, batch prediction, visualizations, and model performance.

## 🎯 Objective
To extract useful sentiment insights from customer reviews and help businesses understand customer opinions, identify areas for improvement, and make better product decisions.

## 📊 Dataset
The dataset contains:
- `title` – Review title
- `rating` – Customer rating from 1 to 5
- `body` – Review text

### Sentiment Mapping
| Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

### Dataset Summary
- Total reviews: **1,440**
- Positive: **729 (50.6%)**
- Neutral: **199 (13.8%)**
- Negative: **512 (35.6%)**
- Average rating: **3.34**

> The project brief recommends e-commerce reviews, preferably from Amazon. This README does not claim a specific source for the supplied dataset unless independently verified.

## 🧠 NLP Workflow
1. Data loading
2. Exploratory Data Analysis
3. Data quality checking
4. Sentiment creation from ratings
5. Text preprocessing
6. TF-IDF feature extraction
7. Train-test split
8. Model training
9. Model evaluation
10. Sentiment prediction
11. Streamlit deployment

## 🧹 Text Preprocessing
The review text is:
- converted to lowercase
- cleaned of URLs
- cleaned of HTML tags
- cleaned of unwanted characters
- normalized for extra spaces

The review title and body are combined for modelling.

## 🔤 Feature Engineering
TF-IDF (Term Frequency–Inverse Document Frequency) converts review text into numerical features. Word-level unigram and bigram features are used.

## 🤖 Machine Learning Models
The project evaluates:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM

### Model Performance
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.21% | 78.84% | 80.21% | **79.37%** |
| Multinomial Naive Bayes | 71.88% | 63.79% | 71.88% | 65.94% |
| Linear SVM | **80.90%** | 78.59% | **80.90%** | 79.06% |

### 🏆 Selected Model
**Logistic Regression** was selected because it achieved the highest F1 Score of **79.37%**.

The trained model and TF-IDF vectorizer are stored in:
```text
sentiment_model.joblib
```

## 🌐 Streamlit Application
### 🏠 Home
Shows review statistics, sentiment distribution, rating distribution, dataset insights, and single-review analysis.

### 💬 Single Review
Enter a customer review and receive a predicted sentiment.

### 📄 Batch Prediction
Upload a CSV containing a `review` column. The application predicts sentiment for each review and provides a downloadable CSV.

### 📊 Visualizations
Includes sentiment and rating distribution charts.

### 🏆 Model Performance
Compares Accuracy, Precision, Recall, and F1 Score for the trained models.

### ℹ️ About
Provides project, dataset, NLP workflow, and technology information.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing
- TF-IDF
- Joblib
- Plotly
- Streamlit
- Jupyter Notebook

## 📁 Project Structure
```text
NLP Sentiment Analysis/
│
├── app.py
├── P652-Dataset.xlsx
├── sentiment_model.joblib
├── requirements.txt
├── README.md
├── NLP_Sentiment_Analysis.ipynb
├── EDA.py
├── train_model.py
│
└── reports/
    ├── model_results.json
    ├── rating_distribution.png
    └── sentiment_distribution.png
```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder
```bash
cd NLP-Sentiment-Analysis
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python -m streamlit run app.py
```

## 📦 Requirements
```text
streamlit
pandas
openpyxl
joblib
scikit-learn
plotly
numpy
```

## 🚀 Deployment
The application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Ensure `app.py`, `requirements.txt`, the dataset, and model are in the repository.
3. Connect the GitHub repository to Streamlit Community Cloud.
4. Select branch `main`.
5. Select `app.py` as the main file.
6. Deploy.

## 📌 Example Reviews

### 😊 Positive
```text
This product is excellent. I really love the quality.
```

### 😞 Negative
```text
Very bad product. It stopped working after two days.
```

### 😐 Neutral
```text
The product is okay. It works as expected.
```

## 🔮 Future Scope
- Larger and more diverse review datasets
- Better multilingual support
- Sentiment confidence scores
- Keyword and topic analysis
- Aspect-based sentiment analysis
- Real-time review collection
- Advanced NLP models such as transformers

## 🎓 Project Deliverables
- Dataset
- EDA
- Text preprocessing
- Feature engineering
- Machine Learning models
- Model evaluation
- Trained model
- Jupyter Notebook
- Streamlit application
- Documentation
- GitHub repository

## ❤️ Conclusion
ReviewSense demonstrates how Natural Language Processing and Machine Learning can transform unstructured customer reviews into useful sentiment insights.

**Made with ❤️ using Python, NLP, Machine Learning and Streamlit.**
