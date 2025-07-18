# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:00:24 2025

@author: Aries
"""

from fastapi import FastAPI
import joblib
from app.text_preprocessor import TextPreprocessor 
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import Form

class EmailRequest(BaseModel):
    text: str

app = FastAPI()

import os

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "spam_detector_pipeline.pkl")
pipeline = joblib.load(model_path)

@app.get("/", response_class=HTMLResponse)
def interactive_home():
    return """
    <html>
        <head>
            <title>Spam Detector</title>
            <script>
                function useSample() {
                    document.getElementById('email').value = 
                        "Congratulations! You've been selected to win a free iPhone. Click the link to claim now!";
                }
            </script>
        </head>
        <body>
            <h2>📧 Welcome to the Spam Mail Detector</h2>
            <p>Use the form below to test your email content:</p>
            <form action="/predict_form" method="post">
                <textarea id="email" name="text" rows="10" cols="80" placeholder="Paste your email content here..."></textarea><br><br>
                <button type="button" onclick="useSample()">Use Sample Email</button>
                <button type="submit">Check for Spam</button>
            </form>
            <p>Or try the <a href="/docs">Swagger UI</a> for structured testing.</p>
        </body>
    </html>
    """
    
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(text: str = Form(...)):
    proba = pipeline.predict_proba([text])[0]
    label = "Spam" if proba[1] > 0.65 else "Ham"
    confidence = round(proba[1 if label == "Spam" else 0] * 100, 2)

    return f"""
    <html>
        <body>
            <h2>📨 Prediction Result</h2>
            <p><strong>Label:</strong> {label}</p>
            <p><strong>Confidence:</strong> {confidence}%</p>
            <a href="/">⏪ Go back</a>
        </body>
    </html>
    """
def classify_email(request: EmailRequest):
    text = request.text
    proba = pipeline.predict_proba([text])[0]
    label = "Spam" if proba[1] > 0.65 else "Ham"
    confidence = round(proba[1 if label == "Spam" else 0] * 100, 2)

    return {
        "label": label,
        "confidence": f"{confidence}%"
    }
