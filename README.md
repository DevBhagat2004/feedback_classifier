# feedback_classifier

This project is a simple machine learning model to classify free-response student feedback about their use and perception of AI tools in education. The goal is to categorize text responses into **Positive**, **Neutral**, or **Negative** sentiment classes.

---

## Features

- Text preprocessing with **TF-IDF Vectorization**
- Classification using **Random Forest** algorithm
- Model evaluation with accuracy, classification report, and confusion matrix visualization
- Save/load trained model and vectorizer using **joblib**
- Command-line interface for classifying new input text

---

## Getting Started

### Prerequisites

Make sure you have Python 3 installed along with the following packages:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib


