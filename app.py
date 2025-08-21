import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load model and tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("./Model_dir", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("./Model_dir")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="BART Text Summarizer API")

class TextRequest(BaseModel):
    text: str

# -----------------------------
# Summarizer function
# -----------------------------
def summarizer(text: str, max_input_length=1024, max_summary_length=150, min_summary_length=40):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_summary_length,
        min_length=min_summary_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=False
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/summarize")
def summarize_text(request: TextRequest):
    summary = summarizer(request.text)
    return {"summary": summary}

# -----------------------------
# Optional root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "BART Text Summarizer API is running"}
