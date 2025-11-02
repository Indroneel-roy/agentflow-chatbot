
# 🧠 LangGraph Chatbot — Multi-Tool Conversational AI (Groq + Streamlit)

A powerful AI assistant built using LangGraph, LangChain, and Groq LLM, integrated with Streamlit for a modern web-based chat interface.
This chatbot supports tool-calling, multi-threaded conversations, and persistent chat history using SQLite.


## Features

✅ Multi-Tool Support — Built-in tools for real-world tasks:

* 🔍 Web search (DuckDuckGo)

* ➗ Calculator (add, sub, mul, div)

* 💹 Stock price lookup (Alpha Vantage)

* 🌦️ Weather info (WTTR API)

* 🕒 Timezone-based current time

* 💱 Currency converter

* 🎲 Random number generator

* 🧾 Text analyzer

* 🪙 Cryptocurrency prices (CoinGecko)

* 😂 Joke generator

✅ Persistent Chat Threads
* Each conversation is stored in chatbot.db using LangGraph’s SQLite checkpointer.
* You can switch between previous chats in the sidebar.

✅ Tool-Aware LLM
* The model (llama-3.3-70b-versatile via Groq) can automatically decide when to call tools and how to interpret their results.

✅ Beautiful UI with Streamlit
* Chat interface with message streaming, dynamic tool execution indicators, and multiple conversation threads.

✅ Error Handling & Stability
* Every node and tool is wrapped with try/except for robust performance.

## Tech Stack

| Component        | Technology                                                 |
| ---------------- | ---------------------------------------------------------- |
| **Frontend**     | Streamlit                                                  |
| **Backend**      | LangGraph + LangChain                                      |
| **LLM Provider** | Groq (llama-3.3-70b-versatile)                                       |
| **Database**     | SQLite (for checkpoints)                                   |
| **APIs Used**    | Alpha Vantage, CoinGecko, WTTR, ExchangeRate-API, Joke API |
| **Language**     | Python 3.10+                                               |

## 📦 Installation
1️⃣ Clone the Repository

```bash
  git clone https://github.com/Indroneel-roy/agentflow-chatbot.git
  cd agentflow-chatbot

```


2️⃣ Create Virtual Environment
```bash
uv venv
On Windows use: .venv\Scripts\activate

```


3️⃣ Install Dependencies
```bash
uv add -r requirements.txt

```


4️⃣ Setup Environment Variables

Create a .env file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here or any others llm api key 

```
## Workflow
