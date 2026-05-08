# 🧠 Smart Document Analyzer

A full-featured AI-powered document analysis tool built with Python (Flask).
Supports PDF, DOCX, TXT, and image files with OCR.

## Features

| Feature | Description |
|---------|-------------|
| 📄 Document Parsing | PDF, DOCX, TXT, PNG/JPG (OCR via Tesseract) |
| 🧠 Summarization | TL;DR, Detailed, Bullet Points, ELI5 mode |
| 🔍 Keyword Extraction | TF-IDF keywords, bigrams, frequency analysis |
| ❓ Chat with Doc | Q&A using TF-IDF cosine similarity |
| 💭 Sentiment Analysis | Positive/Negative/Neutral + Tone detection |
| 🏷️ Named Entity Recognition | Dates, Emails, Phones, Money, Orgs, Names |
| 🗂️ Auto Classification | Resume, Legal, Invoice, Research, Medical, etc. |
| 📑 Table Extraction | From PDFs/DOCX → export as CSV |
| 📊 Document Stats | Word count, reading time, structure detection |

## Installation

```bash
# Install system dependencies (for OCR)
sudo apt-get install tesseract-ocr  # Linux/Ubuntu
# brew install tesseract              # macOS

# Install Python dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open: http://localhost:5050

## Project Structure

```
smart_doc_analyzer/
├── app.py              # Flask backend (all analysis logic)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Dark-mode UI frontend
└── uploads/            # Temporary file storage (auto-cleaned)
```

## How It Works

### Text Extraction
- **PDF** → pdfplumber (text + tables)
- **DOCX** → python-docx
- **TXT** → direct read
- **Images** → pytesseract OCR

### Summarization (Extractive)
Uses TF-IDF scoring to rank sentences by importance, then selects the top N sentences.

### Keyword Extraction
Custom TF-IDF implementation — no external NLP libraries needed.

### Q&A (Chat with Document)
TF-IDF vectorization + cosine similarity to find the most relevant passages for any question.

### Sentiment Analysis
Rule-based lexicon matching with ~200 positive/negative signal words.

### Named Entity Recognition
Regex-based extraction for dates, emails, phones, URLs, monetary values, percentages, organizations, and person names.

### Auto Classification
Keyword frequency matching against 8 category lexicons (Resume, Legal, Research, etc.).

## No External NLP Libraries Required!
This project intentionally avoids heavy dependencies like spaCy, NLTK, or transformers.
Everything runs on: Flask + pdfplumber + python-docx + scikit-learn + numpy + pandas.
