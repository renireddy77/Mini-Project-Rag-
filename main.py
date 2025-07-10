import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
api_key = None

# ✅ First: Try from Streamlit Cloud (secrets)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    # ❌ If not found in cloud, try from local .env
    api_key = os.getenv("OPENAI_API_KEY")

# ✅ Use the key
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in Streamlit secrets or .env")
else:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("✅ API key loaded successfully.")
# UI Title
st.set_page_config(page_title="🧠 Symptom Checker (RAG)")
st.title("🧠 Medical Symptom Checker")
st.markdown("Enter your symptoms to get condition prediction and suggestions (powered by OpenAI + RAG).")

# Load medical dataset
@st.cache_resource
def load_and_prepare_docs():
    df = pd.read_excel("medical_symptom_dataset.xlsx")

    text_data = []
    for _, row in df.iterrows():
        sentence = (
            f"This is case for Patient {row['Patient_ID']}. "
            f"The patient experienced symptoms like {row['Reported_Symptoms']}. "
            f"The suspected condition was {row['Suspected_Condition']} "
            f"with severity score {row['Severity_Score']}. "
            f"Medications advised: {row['Medications_Used']}."
        )
        text_data.append(sentence)

    documents = [Document(page_content=txt) for txt in text_data]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings, collection_name="medical_cases")
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=vectorstore.as_retriever())

# Load RAG chain
rag_chain = load_and_prepare_docs()

# Prompt Template
prompt_template = """
You are a trusted AI-powered medical assistant. Based on the following symptoms described by a user, analyze the case carefully using relevant past cases. 

Symptoms:
{symptoms}

Please provide a detailed response that includes:
1. A possible medical condition or diagnosis based on the symptoms.
2. An explanation of why this diagnosis might be relevant.
3. Recommended over-the-counter or prescribed medications.
4. Suggested lifestyle changes or precautions.
5. A severity estimate if possible (e.g., mild, moderate, severe).
6. Whether or not the user should seek immediate medical attention.

Use clear and non-technical language, but remain medically accurate.

If insufficient information is provided, ask clarifying questions.
"""

# Input box
symptoms = st.text_area("Enter symptoms (e.g. fever, cough, chest pain)", height=100)

# Process user input
if st.button("Get Medical Advice"):
    if symptoms.strip():
        with st.spinner("Analyzing symptoms..."):
            final_prompt = prompt_template.format(symptoms=symptoms.strip())
            response = rag_chain.run(final_prompt)
        st.success("✅ Medical Advice:")
        st.markdown(response)
    else:
        st.warning("⚠ Please enter some symptoms.")