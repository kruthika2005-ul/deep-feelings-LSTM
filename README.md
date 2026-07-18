# 🎭 Deep Feelings: Sentiment Analysis using LSTM

Deep Feelings is a Deep Learning project that predicts whether a movie review expresses a **Positive** or **Negative** sentiment using an **LSTM (Long Short-Term Memory)** neural network.

The project includes data preprocessing, model training, evaluation, and a user-friendly Streamlit web application for real-time sentiment prediction.

---

## 📌 Features

- 🎬 Movie Review Sentiment Prediction
- 🧹 Text Preprocessing
- 🧠 LSTM Deep Learning Model
- 📊 Model Evaluation
- 💾 Saved Model & Tokenizer
- 🌐 Interactive Streamlit Web App
- ⚡ Real-Time Predictions

---

## 📂 Project Structure

```
Deep-Feelings/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── IMDB_Dataset_CLEANED.csv
│
├── models/
│   ├── sentiment_model.keras
│   └── tokenizer.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
```

---

## 📊 Dataset

**Dataset:** IMDb Movie Reviews Dataset

Columns:

| Column | Description |
|---------|-------------|
| review | Movie Review |
| sentiment | Positive / Negative |

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- LSTM
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## 🧠 Model Architecture

```
Input Review
      │
      ▼
Text Cleaning
      │
      ▼
Tokenizer
      │
      ▼
Padding
      │
      ▼
Embedding Layer
      │
      ▼
LSTM Layer
      │
      ▼
Dropout
      │
      ▼
Dense Layer
      │
      ▼
Sigmoid
      │
      ▼
Prediction
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/deep-feelings.git
```

Move into the project folder

```bash
cd deep-feelings
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python -m src.train
```

This generates:

```
models/
├── sentiment_model.keras
└── tokenizer.pkl
```

---

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Example Predictions

### Positive Review

```
This movie was amazing.
The acting was brilliant and the story was fantastic.
```

Prediction

```
😊 Positive
Confidence : 98%
```

---

### Negative Review

```
Worst movie ever.
Completely boring and a waste of time.
```

Prediction

```
😞 Negative
Confidence : 99%
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Typical performance on the IMDb dataset is around **88–92% accuracy**, depending on preprocessing and training settings.

---

## 📦 Requirements

```
streamlit
tensorflow
numpy
pandas
scikit-learn
matplotlib
joblib
```

---

## 🔮 Future Improvements

- Multi-class sentiment classification
- Attention-based LSTM
- Bidirectional LSTM
- Transformer-based models (BERT)
- Explainable AI visualizations
- Deploy on Streamlit Cloud

---

## 👩‍💻 Author

**Kothapeta Kruthika**

B.Tech Data Science Student

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub!