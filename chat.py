# chatbot_embeddings.py
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

nltk.download('punkt')

with open("alice.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = sent_tokenize(raw_text)

model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good for general QA
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

def chatbot(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
    best_idx = torch.argmax(similarities).item()
    return sentences[best_idx]

def main():
    st.title("📚 Alice Chatbot (Semantic)")
    st.write("Ask me anything about Alice's Adventures in Wonderland!")

    user_input = st.text_input("Your question:")

    if user_input:
        response = chatbot(user_input)
        st.write("🤖 Chatbot:", response)

if __name__ == "__main__":
    main()
