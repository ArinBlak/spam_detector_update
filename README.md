# spam_detector_update
this is n update to the previous spam_detector repository
# ✉️ Spam Mail Detector API (FastAPI)

A production-ready machine learning pipeline and API that classifies email content as **Spam** or **Ham (not spam)** using traditional NLP techniques, TF-IDF vectorization, and a logistic regression or random forest classifier.

This project is served using **FastAPI** and supports interactive testing via Swagger UI and a simple HTML form on the homepage.

---

## 🚀 Features

- 🧠 Trained using the [SpamAssassin Public Corpus](https://spamassassin.apache.org/publiccorpus/)
- 📊 Uses TF-IDF with n-grams for email feature extraction
- 📉 Logistic Regression and Random Forest models supported
- ✅ Custom preprocessing pipeline: header removal, lemmatization, stopword removal, POS tagging
- 🌐 Interactive Swagger UI at `/docs`
- 📝 Built-in HTML form with a “Use Sample Email” button at `/`
- 📦 Easily deployable to Hugging Face Spaces

---

## 🏗 Project Structure

spam-detector_update/
├── app/
│ ├── main.py # FastAPI app with HTML + Swagger UI
│ ├── text_preprocessor.py # Cleans, tokenizes, lemmatizes email text
│ └── spam_detector_pipeline.pkl # Trained ML pipeline (TF-IDF + model)
├── training/
│ ├── train_model.ipynb # Jupyter notebook for model training
│ └── load_dataset.py # Reads raw spam/ham files and builds dataset
├── sample_data/
│ └── sample.txt #sample of the type of data i used
├── requirements.txt
└── README.md

---

## 💻 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector/app
```

### 2. Install requirements

```bash
pip install -r ../requirements.txt
```

### 3. Run FastAPI server
```bash
uvicorn main:app --reload
```

Then open your browser to:
-http://127.0.0.1:8000/ → HTML interface
-http://127.0.0.1:8000/docs → Swagger UI

