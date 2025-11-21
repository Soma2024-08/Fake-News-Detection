# 📰 Fake News Classifier using NLP

## Project Overview
This project implements a **Fake News Classifier** using Natural Language Processing (NLP) techniques and multiple machine learning models. The primary goal is to classify news articles as either **"Real" (0)** or **"Fake" (1)** based on their `title` and `text` content using the `WELFake_Dataset.csv`.

The notebook demonstrates a complete machine learning pipeline: data loading, cleaning, feature engineering, model training (Naive Bayes, Random Forest, KNN, Logistic Regression, and SVM), and comprehensive performance evaluation.

---

## 💾 Dataset
The project utilizes the **WELFake_Dataset**, which contains over 72,000 news articles and is a merge of four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political).

**Dataset Source:**
[https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

| Column | Description |
| :--- | :--- |
| `title` | The headline of the news article. |
| `text` | The full body text of the news article. |
| `label` | The target variable: **0** for Real News, **1** for Fake News. |

### Class Distribution
The dataset exhibits a good balance between the two classes:
* **Fake News (Label 1):** 37,106 samples
* **Real News (Label 0):** 35,028 samples

---

## 🛠️ Setup and Installation

### Prerequisites
Ensure you have Python installed (version 3.7+ recommended).

### Dependencies
You can install all necessary libraries using the following command:

```bash
pip install pandas numpy scikit-learn nltk seaborn nlp_utils matplotlib

### Methodology
 -Data Preprocessing & Cleaning
-Missing Values: All missing values in the title and text columns were filled with an empty string ("").

-Text Cleaning: The raw text was converted to lowercase and punctuation/numbers were removed.

-Tokenization & Stemming: Stop words (English) were removed, and the Porter Stemmer was applied to reduce words to their base form (e.g., 'running' to 'run').

-Data Split: The dataset was split into 70% training and 30% testing sets (test_size=0.30, random_state=40).
