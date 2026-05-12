# 🚀 Project Vega: Coaction Underwriting Assistant

Welcome to **Project Vega**! This is a modern, AI-powered Underwriting Assistant built for Coaction. It uses advanced language models and your internal company manuals to answer complex underwriting questions accurately, complete with direct citations.

If you are new to the project, this guide will walk you through how the system works and how the different pieces fit together.

---

## 🏗️ Architecture Overview

The project is built using a **modular, serverless-ready architecture** powered by AWS AgentCore. Here is a simplified breakdown of the flow:

1. **The User Interface (Frontend):** 
   - Built with **Gradio** (`ui/gradio_app.py`), providing a beautiful, glassmorphism-style chat interface.
   - Users can sign up, log in, manage their chat history, and chat with the AI.
   
2. **The Brain (Backend API):**
   - Built with **FastAPI** (`main.py` and `coaction_agent_platform/`). 
   - This is the middleman that safely handles requests from the UI, verifies who you are, and talks to the AI.

3. **Authentication:**
   - Powered by **AWS Cognito**. When a user logs in via the UI, Cognito provides a secure token (JWT) which is used to verify their identity for all subsequent requests.

4. **Memory & Storage:**
   - Powered by **Amazon DynamoDB**. We use a fast, single-table design (`dynamodb.py`) to remember user profiles, save chat histories so you can resume previous conversations, and store Knowledge Base metadata.

5. **The AI & Knowledge Retrieval (RAG):**
   - Powered by **Amazon Bedrock**. We use the **Nova Pro** model to generate intelligent responses.
   - **Knowledge Bases (KB):** Instead of guessing answers, the AI searches through your uploaded company manuals (like the Property or General Liability manual) using Bedrock Knowledge Bases. It retrieves the exact paragraphs needed and cites them at the end of its answer.

---

## 🌊 How the Data Flows (The "Chat" Lifecycle)

When a user types a message like *"What is class code 10040?"*, here is what happens behind the scenes:

1. **Request:** The Gradio UI sends the user's message + their secure login token to the backend API.
2. **Security Check:** The `identity.py` dependency checks the token with AWS Cognito to ensure the user is allowed to make the request.
3. **Session Load:** The `AgentService` checks DynamoDB to see what the user was talking about previously, loading the chat history.
4. **Retrieval (RAG):** The `UnderwritingAgent` triggers the `retriever.py` tool. This tool asks Amazon Bedrock to search the Knowledge Base (your uploaded PDFs/Manuals) for relevant information about "class code 10040".
5. **Generation:** The retrieved paragraphs + the user's question are sent to the **Nova Pro** AI model. The model is given strict instructions to only use the provided manuals to answer the question.
6. **Citation Formatting:** The backend formats the sources the AI used into a strict citation protocol (`Source Manual`, `Section`, `Link`).
7. **Response:** The formatted answer is saved back to DynamoDB (so it remembers it next time) and sent back to the Gradio UI for the user to read!

---

## 📂 Folder Structure

Here is where everything lives:

```text
d:\project-vega\
│
├── coaction_agent_platform/      # The core backend logic
│   ├── adapters/aws/             # Code that talks to AWS (Cognito, DynamoDB, Bedrock)
│   ├── agents/                   # The AI logic (Underwriting Agent, Prompts, Retriever tools)
│   ├── app/                      # FastAPI setup (Routers, Middleware, Dependencies)
│   ├── domain/                   # Data models (Shapes of our data like AgentInvocationResponse)
│   └── services/                 # Orchestration (AgentService connects DB, Auth, and AI)
│
├── config/                       # Agent Execution Profiles (YAML configs)
├── ui/                           # Frontend code
│   ├── gradio_app.py             # The Gradio UI application
│   └── Dockerfile                # Docker instructions for the UI
│
├── .env.example                  # Template for required environment variables
├── main.py                       # The entry point to start the FastAPI backend
└── README.md                     # You are here!
```

---

## 💻 How to Run Locally

### 1. Set up Environment Variables
1. Copy `.env.example` to a new file named `.env`.
2. Fill in your AWS credentials, Cognito IDs, and Bedrock IDs. **Never commit your `.env` file to Git!**

### 2. Start the Backend API
Open a terminal, activate your virtual environment, and run:
```bash
python main.py
```
*The API will start on `http://localhost:8000`.*

### 3. Start the UI
Open a second terminal, activate your virtual environment, and run:
```bash
python ui/gradio_app.py
```
*The UI will start on `http://localhost:7860`.*

Open your browser to `http://localhost:7860`, sign up/login, and start chatting!
