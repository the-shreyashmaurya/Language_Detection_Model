# Language Detection Model

## Overview
This project is a machine learning-based **Language Detection Model** that classifies text into different languages. It is built using **Scikit-learn**, **TF-IDF Vectorization**, and **Logistic Regression**.

## Dataset
The dataset is obtained from **Kaggle** ("basilb2s/language-detection") and contains text samples from **17 different languages**:

- English, Malayalam, Hindi, Tamil, Portuguese, French, Dutch, Spanish, Greek, Russian, Danish, Italian, Turkish, Swedish, Arabic, German, Kannada.

## Features
- Text Preprocessing (lowercasing, punctuation removal, number removal, etc.)
- TF-IDF vectorization with character n-grams (1 to 3)
- **Logistic Regression** model for classification
- Model persistence using **Pickle**

---

## Installation & Setup
### 1. Clone the Repository
```sh
$ git clone https://github.com/the-shreyashmaurya/Language_Detection_Model.git
$ cd Language_Detection_Model
```

### 2. Install Dependencies
```sh
$ pip install -r requirements.txt
```

### 3. Download the Dataset
```python
import kagglehub
path = kagglehub.dataset_download("basilb2s/language-detection")
```

### 4. Load Data
```python
import pandas as pd
df = pd.read_csv(f"{path}/Language Detection.csv")
```

---

## Training the Model
```python
from sklearn import feature_extraction, linear_model, pipeline
from sklearn.model_selection import train_test_split

# Splitting Data
X, y = df["Text"], df["Language"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building Pipeline
vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), analyzer="char")
pipe_lr = pipeline.Pipeline([
    ("vectorizer", vectorizer),
    ("clf", linear_model.LogisticRegression())
])

# Training Model
pipe_lr.fit(X_train, y_train)
```

---

## Model Evaluation
```python
from sklearn import metrics

y_pred = pipe_lr.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```
Expected Output:
```sh
Accuracy: ~98.3%
```

---

## Saving & Loading the Model
### Save the Model
```python
import pickle
with open("lrModel.pkl", "wb") as file:
    pickle.dump(pipe_lr, file)
```

### Load the Model
```python
with open("lrModel.pkl", "rb") as file:
    lrModel = pickle.load(file)
```

---

## Predicting Language
To detect the language of a given text:
```python
pred = lrModel.predict(["नहीं, हम नहीं जानते, जिमी ने कहा."])
print(pred)  # Output: ['Hindi']
```

---

## Contributing
Feel free to open issues or submit pull requests to improve the model or add new features!

---

## License
This project is open-source and available under the **MIT License**.

---

## Author
Developed by **Shreyash Maurya**.

For any queries, reach out via **[GitHub](https://github.com/the-shreyashmaurya/)**.

