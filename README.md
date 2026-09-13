# SMS Spam Detection with Machine Learning and Gemini

This project classifies SMS messages as **spam** or **ham** (legitimate messages) using a scikit-learn pipeline: TF-IDF vectorization plus Logistic Regression. Google Gemini then turns the model's labels into a simple explanation.

## Features

- Loads an SMS spam dataset from CSV
- Splits data into training and test sets
- Trains and evaluates a TF-IDF + Logistic Regression classifier
- Predicts spam/ham for new messages
- Uses Gemini to write a readable prediction summary

## Project structure

```text
SMS-Spam-Detection/
├── project_1.ipynb
├── naive bayes spam dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

The CSV must contain `v1` (label: `ham` or `spam`) and `v2` (SMS text).

## Installation

```bash
pip install pandas scikit-learn jupyter langchain-google-genai
jupyter notebook
```

Place `naive bayes spam dataset.csv` in the same folder as `project_1.ipynb`, then open the notebook and run all cells in order.

## Gemini setup

The notebook expects a `GEMINI_API_KEY` environment variable. In PowerShell, for the current session:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
```

Never commit an API key. Add this to `.gitignore`:

```gitignore
.env
.ipynb_checkpoints/
__pycache__/
```

If a key has ever been pushed to GitHub, revoke/rotate it in Google AI Studio.

## Workflow

```text
SMS data → train/test split → TF-IDF → Logistic Regression
        → evaluation → new-message prediction → Gemini summary
```

Gemini explains the machine-learning prediction; it is not used as the spam classifier.

## Author

Basava Khandale  
Data Science | Machine Learning | Python | SQL | Generative AI
