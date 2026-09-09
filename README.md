# 📱 SMS Spam Detection using Machine Learning & Google Gemini

A machine learning project that classifies SMS messages as *Spam* or *Ham (Not Spam)* using *TF-IDF Vectorization* and *Logistic Regression*.

The project also integrates *Google Gemini* to provide simple, human-readable explanations of the model's predictions.

---

## 🚀 Project Overview

Spam messages are unwanted messages that may contain advertisements, scams, fraudulent offers, or malicious links.

This project builds a text classification system that can automatically identify whether an SMS message is:

* 🟢 *Ham* — Normal / legitimate message
* 🔴 *Spam* — Unwanted or suspicious message

The machine learning pipeline converts text messages into numerical features using *TF-IDF* and then uses *Logistic Regression* to classify them.

After prediction, Google Gemini is used to generate a simple explanation for each input message and its predicted class.

---

## 🎯 Objectives

The main objectives of this project are:

1. Load and understand an SMS spam dataset.
2. Separate SMS text and target labels.
3. Split the dataset into training and testing data.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classification model.
6. Evaluate the model using classification metrics.
7. Predict classes for new/unseen SMS messages.
8. Use Google Gemini to generate simple explanations for the predictions.

---

## 🛠️ Technologies Used

| Technology             | Purpose                       |
| ---------------------- | ----------------------------- |
| Python                 | Programming language          |
| Pandas                 | Data loading and manipulation |
| Scikit-learn           | Machine learning              |
| TF-IDF                 | Text feature extraction       |
| Logistic Regression    | Spam classification           |
| LangChain Google GenAI | Gemini integration            |
| Google Gemini          | Prediction explanation        |
| Jupyter Notebook       | Development environment       |

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

text
SMS Dataset
     ↓
Load Dataset
     ↓
Separate Features & Target
     ↓
Train-Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Prediction
     ↓
Classification Report
     ↓
New SMS Prediction
     ↓
Google Gemini Explanation


---

## 📊 Dataset

The project uses an SMS spam dataset containing two important columns:

* v1 → Target label
* v2 → SMS message

The target column contains categories such as:

text
ham
spam


The SMS text is used as the input feature.

---

## 🔍 Step-by-Step Implementation

### 1. Import Required Libraries

The project uses Pandas and several Scikit-learn components:

python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


Google Gemini is accessed through:

python
from langchain_google_genai import ChatGoogleGenerativeAI


---

### 2. Load the Dataset

The dataset is loaded using Pandas:

python
df = pd.read_csv("naive bayes spam dataset.csv")


> Update the file path according to your local/project directory.

---

### 3. Separate Input and Target

The SMS messages are stored in v2 and the labels are stored in v1.

python
x = df['v2']
y = df['v1']


Therefore:

text
X → SMS message
Y → Spam/Ham label


---

### 4. Split the Dataset

The dataset is divided into training and testing data:

python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=42,
    train_size=0.8
)


Approximately:

text
80% → Training data
20% → Testing data


The random_state=42 makes the split reproducible.

---

## 🔢 5. TF-IDF Vectorization

Machine learning models cannot directly understand raw text.

For example:

text
"Congratulations! You won a free prize"


needs to be converted into numerical features.

The project uses:

python
TfidfVectorizer()


TF-IDF stands for:

*Term Frequency – Inverse Document Frequency*

It assigns higher importance to words that are useful for distinguishing between messages.

---

## 🤖 6. Logistic Regression Model

The classification algorithm used is:

python
LogisticRegression()


It learns patterns in the TF-IDF features and predicts whether a message belongs to:

text
ham


or

text
spam


---

## 🔗 7. Machine Learning Pipeline

TF-IDF and Logistic Regression are combined into a single Scikit-learn Pipeline:

python
main_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('model', LogisticRegression())
])


The model is then trained:

python
main_pipeline.fit(x_train, y_train)


### Why use a Pipeline?

The pipeline ensures that the same preprocessing steps are applied consistently during training and prediction.

Instead of manually performing:

text
Text
 ↓
TF-IDF
 ↓
Model


we can simply use:

python
main_pipeline.predict(new_messages)


---

## 📈 8. Model Evaluation

The project evaluates both training and testing predictions using:

python
classification_report()


Example:

python
train_score = classification_report(
    y_train,
    main_pipeline.predict(x_train)
)

test_score = classification_report(
    y_test,
    main_pipeline.predict(x_test)
)


The classification report provides metrics such as:

* Precision
* Recall
* F1-score
* Support

### Important Metrics

*Precision*

Out of the messages predicted as spam, how many were actually spam?

*Recall*

Out of all actual spam messages, how many did the model correctly identify?

*F1-score*

The harmonic mean of precision and recall.

---

## 🧪 9. Prediction on New Messages

The trained model can classify previously unseen messages.

Example:

python
new_data = pd.DataFrame({
    'v2': [
        'Go until jurong point, crazy.. Available only in bugis...',
        'Ok lar... Joking wif u oni...',
        'Free entry in a wkly comp to win FA Cup final tkts...'
    ]
})


Predictions are generated using:

python
result = main_pipeline.predict(new_data['v2'])


The output can contain predictions such as:

text
ham
ham
spam


---

## 🤖 10. Google Gemini Integration

Google Gemini is used to generate a simple explanation of the model's prediction.

The project uses:

python
ChatGoogleGenerativeAI


and sends the input message along with the predicted output to Gemini.

The goal is not to make Gemini perform the classification itself.

Instead:

text
Machine Learning Model
        ↓
Predict Spam/Ham
        ↓
Gemini
        ↓
Generate Human-readable Explanation


This separates *prediction* from *explanation*.

---

## 🔐 API Key Security

### ⚠️ Important

Never hard-code your Google API key in a public GitHub repository.

Avoid:

python
ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key="YOUR_API_KEY"
)


Instead, store the API key as an environment variable.

Example:

python
import os

api_key = os.getenv("GOOGLE_API_KEY")


Then:

python
llm_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=api_key
)


Create a .env file locally:

text
GOOGLE_API_KEY=your_api_key_here


Add .env to .gitignore:

text
.env


*If an API key has already been committed to GitHub, revoke/rotate it rather than simply deleting it from the latest commit.*

---

## 📁 Project Structure

A recommended GitHub structure is:

text
SMS-Spam-Detection/
│
├── project_1.ipynb
├── naive bayes spam dataset.csv
├── README.md
├── requirements.txt
├── .gitignore
└── .env


Do not upload the real .env file.

---

## 📦 Requirements

Create a requirements.txt file:

text
pandas
scikit-learn
jupyter
langchain-google-genai


Install dependencies using:

bash
pip install -r requirements.txt


---

## ▶️ How to Run the Project

### Step 1 — Clone the repository

bash
git clone <your-github-repository-url>


### Step 2 — Navigate to the project

bash
cd SMS-Spam-Detection


### Step 3 — Install dependencies

bash
pip install -r requirements.txt


### Step 4 — Add your dataset

Place the SMS dataset inside the project directory.

### Step 5 — Configure the Gemini API key

Set your API key as an environment variable.

### Step 6 — Open the notebook

bash
jupyter notebook


Open:

text
project_1.ipynb


Run the cells sequentially.

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Natural Language Processing (NLP)
* Text Classification
* TF-IDF
* Logistic Regression
* Train-Test Split
* Scikit-learn Pipelines
* Model Evaluation
* Classification Reports
* Prediction on Unseen Data
* Generative AI Integration
* LangChain
* Google Gemini API
* API Key Security

---

## 🔮 Future Improvements

The project can be improved by adding:

* Text preprocessing and cleaning
* Stop-word removal
* Stemming/Lemmatization
* Hyperparameter tuning
* Cross-validation
* Confusion matrix visualization
* ROC-AUC evaluation
* Comparison with Naive Bayes, SVM, Random Forest and other models
* Streamlit web application
* REST API deployment
* Better prompt engineering for Gemini
* Explainable AI techniques
* Model persistence using joblib

---

## 🌐 Possible Real-World Applications

The same approach can be adapted for:

* SMS spam filtering
* Email spam detection
* Comment moderation
* Fraud message detection
* Phishing message identification
* Customer-support message classification
* Social-media content filtering

---

## 📌 Project Highlights

### Machine Learning

text
TF-IDF + Logistic Regression


### NLP

text
Text → Numerical Features → Classification


### Generative AI

text
Prediction → Gemini → Simple Explanation


### Evaluation

text
Precision + Recall + F1-score

## 👨‍💻 Author

*Basava Khandale*

Data Science | Machine Learning | Python | SQL | Generative AI

## ⭐ If you found this project useful

Feel free to ⭐ star the repository and explore the project.


## 📄 License

This project is intended for educational and learning purposes.
