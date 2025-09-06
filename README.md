📚 Flask RAG Server

A lightweight Retrieval-Augmented Generation (RAG) server built with Flask and LangChain, containerized with Docker for easy deployment.

🚀 Features

Upload and index documents (PDFs, text files, etc.)

Query the documents using LLMs (OpenAI / Google GenAI)

Switch backend models via API

Run fully containerized with Docker

🛠️ Tech Stack

Flask (Python web framework)

LangChain Core & Community

LangChain OpenAI & Google GenAI integrations

Docker (for deployment)

📂 Project Structure
flask_rag_app/
│── app.py              # Flask entrypoint
│── requirements.txt    # Dependencies
│── Dockerfile          # Docker build file
│── data/               # Store uploaded documents
│── README.md           # Project documentation


⚙️ Installation
1️⃣ Clone the repo
git clone https://github.com/your-username/flask-rag-server.git
cd flask-rag-server/flask_rag_app

2️⃣ Build Docker image
docker build -t rag-app .

3️⃣ Run the container
docker run -p 5000:5000 rag-app


Server runs at:
👉 http://localhost:5000

📡 API Endpoints
🔹 Upload documents
POST /upload

🔹 Query documents
POST /query

🔹 Switch backend model
POST /switch_model

📦 Dependencies

Your requirements.txt includes:

flask
langchain-core==0.3.75
langchain-community==0.3.29
langchain-openai==0.3.32
langchain-google-genai==2.1.10

🛳️ Deployment

To run on a different machine:

Copy the project folder

Run docker build -t rag-app .

Run docker run -p 5000:5000 rag-app

No extra setup needed 🚀
