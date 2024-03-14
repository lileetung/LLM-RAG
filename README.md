
# LLM RAG

## Overview

This project integrates a language model (LLM) with a Retrieval-Augmented Generation (RAG) approach to enhance question-answering capabilities. It leverages the power of the Llama 2-7B model for generating responses and combines it with a retrieval system for fetching relevant information from a database created from PDF documents. The project demonstrates how to split, embed, and retrieve document data to support the LLM in generating accurate and informative answers.

## Prerequisites

Before running this project, ensure you have the following prerequisites installed or available:

- Python 3.9 or later
- Llama-2-7B-Chat model. Download the `llama-2-7b-chat.Q4_K_M.gguf` model file from [HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main).

## How to Run

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Clone this repository

```bash
git clone https://github.com/lileetung/LLM-RAG.git
```

3. Navigate to the project directory:

```bash
cd LLM-RAG
```

4. Run the main script:

```bash
python RAG.py
```

## Usage

After running the `RAG.py` script, the system will prompt you to input a question. You can use the suggested questions or input your own. The system operates in two modes:

- **LLMChain mode:** Directly utilizes the LLM for answering questions.
- **RetrievalQA mode:** Uses a retrieval system to fetch relevant information which is then passed to the LLM for generating answers.

## GUI Development

I'm currently developing a GUI for the project using Tkinter. Here's a screenshot:

<img width="717" alt="LLMGUI" src="https://github.com/lileetung/LLM-RAG/assets/83776772/1d752b34-6179-4fb0-8036-01f34ee12625">

