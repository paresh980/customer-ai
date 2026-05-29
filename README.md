# Smartnode Customer Support – Call Transcription & Classification POC

## Overview

This project demonstrates a proof of concept (POC) for automating customer support call analysis. The solution transcribes customer interactions and classifies each call into one of the following categories:

* **Closed** – Issue resolved, no further action required
* **Open** – Issue reported and awaiting resolution
* **Urgent** – Critical issue requiring immediate attention

The objective is to reduce manual effort, improve response prioritization, and provide structured insights from customer communications.

---

## Project Structure

```text
smartnode-customer-ai/
│
├── presentation.html
│   └── Business use cases and solution overview
│
├── Smartnode_Call_Transcription_POC.ipynb
│   └── Jupyter Notebook version of the POC
│
├── smartnode_call_transcription_poc.py
│   └── Standalone Python implementation
│
├── requirements.txt
│   └── Project dependencies
│
├── sample_audio/
│   └── Generated sample audio recordings
│
└── results/
    └── Classification output and generated reports
```

---

## Features

* Automatic generation of sample customer support calls
* Speech-to-text transcription using Whisper
* Rule-based call classification
* Zero-shot classification using Hugging Face Transformers
* JSON export of classification results
* Accuracy comparison between classification approaches

---

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd smartnode-customer-ai
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install FFmpeg

Whisper requires FFmpeg for audio processing.

**Windows**

```bash
winget install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
```

---

## Running the Project

### Option 1: Jupyter Notebook

Open the notebook in Jupyter or Google Colab and execute all cells:

```text
Smartnode_Call_Transcription_POC.ipynb
```

### Option 2: Python Script

Run the standalone implementation:

```bash
python smartnode_call_transcription_poc.py
```

---

## Output

After execution, the project generates:

* Transcribed call content
* Predicted call category
* Classification confidence scores
* Accuracy summary
* Consolidated JSON results file

Output files are stored in the `results/` directory.

---

## Technologies Used

* Python
* OpenAI Whisper
* Hugging Face Transformers
* PyTorch
* gTTS (Google Text-to-Speech)

---

## Assumptions

* Sample customer calls are used for demonstration purposes.
* Audio samples are generated automatically from predefined scripts.
* English language transcription is assumed.
* Classification is limited to three categories: Closed, Open, and Urgent.
* The project is intended as a proof of concept and not as a production-ready deployment.

---

## Author

Developed as part of the Smartnode Customer Support AI/ML Proof of Concept project.
