# 🔍 Search Autocomplete & Query Prediction Engine

This project implements a **real-time search autocomplete system** that predicts and ranks **multi-word query suggestions** as a user types.  
The primary focus is on **backend system design**, **NLP-based sequence modeling**, and **production-style integration**, rather than isolated model experimentation.

---

## 🚀 Features

- Real-time search query autocomplete  
- **Top-K ranked suggestions** with confidence scores  
- NLP-based sequence modeling (**LSTM / Transformer concepts**)  
- **REST API–driven backend** for low-latency inference  
- **Explainability & visualization** (token highlighting, prediction flow)  
- Clean separation of **training, inference, and frontend layers**

---

## 🔄 System Workflow

```
User Query
   ↓
Text Tokenization
   ↓
Sequence Modeling (LSTM / Transformer-based Language Model)
   ↓
Top-K Next-Token Prediction
   ↓
Confidence Scoring & Ranking
   ↓
Autocomplete Suggestions (API Response)
```

The model predicts the **next most probable tokens**, which are combined to form meaningful query completions similar to modern search engine autocomplete systems.

---

## 🛠️ Tech Stack

### Backend & Machine Learning
- Python  
- Hugging Face Transformers  
- PyTorch  
- NLP Tokenization (Byte Pair Encoding - BPE)  
- Sequence Modeling (LSTM / Transformer concepts)

### Backend Services
- Flask (REST APIs)  
- JSON-based inference responses  

### Frontend
- HTML, CSS, JavaScript  
- Live autocomplete UI  
- Visualization of prediction flow and confidence scores  

---

## 📂 Project Structure

```
nlp_project/
│
├── model/
│   ├── train.py          # Model training pipeline
│   ├── inference.py      # Autocomplete inference logic
│   ├── final_model/      # Saved trained model & tokenizer
│   └── data/
│       └── queries.txt   # Search-style training corpus
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py                # Flask application
├── metrics.json          # Training metrics (loss curve)
└── README.md
```

---

## 🧪 Training Details

- **Dataset:** Search-style query corpus (`queries.txt`)  
- **Model Type:** Causal Language Model  
- **Training Objective:** Next-token prediction  
- **Evaluation:** Training loss tracking  
- **Output:** Trained model and tokenizer saved locally  

The training pipeline is modular and can be extended to larger datasets or alternative language models.

---

## 📊 Explainability & Visualization

To improve transparency and interpretability, the system includes:

- **Token highlighting** to show influential query tokens  
- **Prediction flow visualization** explaining how suggestions are generated  
- **Confidence-based ranking** for each autocomplete suggestion  

These features help bridge the gap between **machine learning predictions** and **real-world system behavior**.

---

## ▶️ How to Run

### 1️⃣ Train the Model
```bash
python model/train.py
```

### 2️⃣ Start the Backend Server
```bash
python app.py
```

### 3️⃣ Open in Browser
```
http://127.0.0.1:5000
```

---

## 🎯 Project Goals

- Demonstrate how **NLP models integrate into backend systems**  
- Build a **search-style product**, not just a standalone ML model  
- Emphasize **system design, APIs, and scalability**  
- Showcase **explainable AI** in a user-facing application  

---
