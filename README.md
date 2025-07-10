# 🧠 Medical Symptom Checker

This is an AI-powered medical assistant that takes user-input symptoms and provides a possible diagnosis, advice, and treatment suggestions using **Retrieval-Augmented Generation (RAG)** powered by **OpenAI**. It has an interactive **Streamlit UI** that makes the system easy to use for general users, medical students, or healthcare providers.

---

# Features

- ✅ Input symptoms like "fever, chest pain, nausea"
- 🔍 Searches similar medical cases from a CSV/Excel dataset
- 🤖 Uses LangChain + FAISS + OpenAI GPT-3.5 Turbo
- 💊 Returns possible diagnosis, medications, severity, and lifestyle advice
- 📦 Simple UI built with Streamlit
- 🧠 Supports over 10,000+ patient records

---

# Technologies Used

| Component | Library / Tool |
|----------|----------------|
| UI       | Streamlit      |
| Backend  | Python         |
| AI Model | OpenAI GPT-3.5 |
| Vector DB| FAISS          |
| RAG      | LangChain      |
| Dataset  | Excel (medical_symptom_dataset.xlsx) |

---
## 📦 Setup Instructions
1.Clone the repository
bash
git clone https://github.com/your-username/Medical-Symptom-Checker.git
cd Medical-Symptom-Checker 
2.Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate
3.Install dependencies
pip install -r requirements.txt
4.OPENAI_API_KEY=sk-xxxx
5.streamlit run main.py

# Use Cases
Health education
First-aid suggestions
Symptom tracking for rural clinics

# How Streamlit Works

Streamlit is a Python library to build web apps for machine learning and data science.

# How it Works:

1. You write Python code with `st.title()`, `st.text_area()`, etc.
2. Streamlit runs a local web server.
3. The script **re-runs from top to bottom** on every user interaction.
4. Output is shown instantly in the browser.

We used Streamlit to create a form where the user enters symptoms. On clicking a button, the AI generates advice.
