# 📝 Text Summarization with BART Transformer

---

## 🌟 Project Overview

This project delivers a **robust solution** for *abstractive text summarization* using the **BART (Bidirectional and Auto-Regressive Transformer)** model from Hugging Face's Transformers library. It specializes in summarizing *conversational text*—think meeting notes, chat logs, or medical dialogues—producing concise, human-like summaries.

The project is split into two core components:
1. **Training Notebook** (`Text_Summarization_using_BERT_Transformer.ipynb`): A Jupyter notebook that fine-tunes the `facebook/bart-large-cnn` model on the SAMSum dataset for dialogue-specific summarization.
2. **API Deployment** (`app.py`): A *FastAPI-based web service* that serves the fine-tuned model via a RESTful API for real-time summarization.

### ✨ Key Features
- **Abstractive Summarization**: Crafts natural, paraphrased summaries instead of extracting raw text. 📜
- **Dialogue-Optimized**: Fine-tuned for conversational data, capturing context and intent. 💬
- **Scalable API**: Easily integrates into apps with HTTP endpoints. 🌐
- **GPU Acceleration**: Supports CUDA for lightning-fast training and inference. ⚡
- **Developer-Friendly**: Includes interactive Swagger UI for API testing. 🛠️

Perfect for developers, data scientists, or teams building *NLP-powered applications*!

---

## 📚 Dataset

The project leverages the **SAMSum dataset** from Hugging Face:
- **Size**:
  - *Training*: ~12,460 dialogue-summary pairs
  - *Validation*: 500 pairs
  - *Test*: 1,500 pairs
- **Content**: Real-world dialogues (e.g., casual chats, doctor-patient exchanges) with concise summaries.
- **Example**:
  - *Dialogue*: A conversation about a child’s vaccination.
  - *Summary*: "Mrs. Parker takes Ricky for his vaccines. Dr. Peters checks the record and administers a vaccine."

Load it with: `datasets.load_dataset("samsum")`.

---

## 🧠 Model Details

- **Base Model**: `facebook/bart-large-cnn`, pre-trained on CNN/DailyMail for news summarization.
- **Fine-Tuning**:
  - **Tokenizer**: Truncates inputs to 1024 tokens (BART’s max). ✂️
  - **Generation Parameters**:
    - Max summary length: 150 tokens
    - Min summary length: 40 tokens
    - Beam search: 4 beams
    - Length penalty: 2.0
    - No-repeat n-gram size: 3
  - **Training Setup**:
    - Epochs: 2
    - Batch size: 8
    - Hardware: GPU (CUDA) or CPU fallback
  - **Preprocessing**: Tokenizes dialogues and summaries, masks padding for training.
- **Output**: Fine-tuned model saved to `./Model_dir` for deployment. 💾

*Fine-tuning enhances dialogue-specific context, outperforming the base model.*

---

## 🛠️ Requirements

- **Python**: 3.8 or higher
- **Dependencies** (install via `pip install -r requirements.txt`):
  - `torch`: Model training/inference (CUDA recommended) 🔥
  - `fastapi`: API server
  - `uvicorn`: ASGI server for FastAPI
  - `pydantic`: Request validation
  - `transformers`: Hugging Face library for BART 🤗
  - `datasets`: SAMSum dataset access
  - `accelerate`: Optional, for distributed training
  - `wandb`: Training metric logging (API key needed) 📊
- **Hardware**: GPU (e.g., NVIDIA with CUDA) for optimal speed; CPU supported but slower.

### Sample `requirements.txt`
```plaintext
torch
fastapi
uvicorn
pydantic
transformers
datasets
accelerate
wandb
```

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/text-summarization-bart.git
   cd text-summarization-bart
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up the model:
   - Unzip a pre-trained model to `./Model_dir` if provided.
   - Alternatively, run the training notebook to generate the model.

---

## 🏋️ Training the Model

1. Open `Text_Summarization_using_BERT_Transformer.ipynb` in Jupyter Notebook or Google Colab (GPU recommended).
2. Execute cells in order:
   - Load libraries and SAMSum dataset.
   - Initialize `facebook/bart-large-cnn` model and tokenizer.
   - Preprocess data (tokenize dialogues and summaries).
   - Configure training arguments (2 epochs, batch size=8).
   - Train with Hugging Face `Trainer`, logging metrics to Weights & Biases.
3. **Training Time**: ~1 hour on a single GPU (e.g., NVIDIA T4) for 2 epochs.
4. **Output**: Model and tokenizer saved to `./Model_dir`.
5. **Evaluation**: Compare pre- and post-fine-tuning summaries on test dialogues.

**Note**: For Weights & Biases, provide an API key during training to track loss and metrics.

---

## 🌐 Deploying the API

### Starting the Server

1. Ensure the fine-tuned model is in `./Model_dir`.
2. Launch the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   - Default URL: `http://127.0.0.1:8000`

### API Endpoints

- **GET /**: Returns a welcome message: `"BART Text Summarizer API is running"`.
- **POST /summarize**: Generates a summary for input text.
  - **Request Body** (JSON):
    ```json
    {"text": "Your dialogue or text here"}
    ```
  - **Response**:
    ```json
    {"summary": "Generated summary text"}
    ```
  - **Example** (using `curl`):
    ```bash
    curl -X POST "http://127.0.0.1:8000/summarize" -H "Content-Type: application/json" -d '{"text": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I will bring you tomorrow :)"}'
    ```
    - **Response**:
      ```json
      {"summary": "Amanda baked cookies and will bring some to Jerry tomorrow."}
      ```

3. **Interactive Testing**: Access Swagger UI at `http://127.0.0.1:8000/docs`.

### Example Client Code

```python
import requests

url = "http://127.0.0.1:8000/summarize"
data = {"text": "Your long dialogue here..."}
response = requests.post(url, json=data)
print(response.json()["summary"])
```

---

## 📊 Performance Insights

- **Training Metrics**: Cross-entropy loss decreases from ~1.59 to ~1.00 over 2 epochs.
- **Evaluation**: Use ROUGE scores for quantitative assessment (add `rouge_score` library for implementation).
- **Qualitative Improvement**: Fine-tuned model better captures dialogue context, speaker roles, and key actions.
- **Limitations**:
  - May generate inaccurate details (hallucinations) for very long or out-of-domain text.
  - Adjust generation parameters (e.g., `max_length`, `length_penalty`) to balance brevity and detail.

---

## 🤝 Contributing

- Fork the repository and submit pull requests for enhancements (e.g., adding ROUGE evaluation, supporting new datasets).
- Report bugs or suggest features via GitHub Issues.

---

## 📜 License

Licensed under the **MIT License**. Free to use, modify, and distribute.

---

## 🙌 Acknowledgments

- **Hugging Face**: For the Transformers library and SAMSum dataset.
- **Weights & Biases**: For training visualization and logging.
- **SAMSum Dataset**: For providing high-quality dialogue data.

---

This project showcases the power of fine-tuned transformers for practical NLP applications. For questions or support, open a GitHub issue or contact the maintainers. Happy summarizing! 🎉
