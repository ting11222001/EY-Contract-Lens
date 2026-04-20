import os
import anthropic
import chromadb
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load ANTHROPIC_API_KEY from your .env file into the environment
load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create the Anthropic client using the key from .env
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Create an in-memory ChromaDB instance (data is lost on restart)
chroma_client = chromadb.Client()


def extract_text(pdf_file):
    text = ""
    # Open the PDF and loop through every page
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Add page text to the result; use "" if a page has no text
            text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    # Split the full text into individual words
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        # Join the next 500 words into one chunk
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        # Move forward by 450 words so the next chunk shares 50 words with this one
        i += chunk_size - overlap
    return chunks


def embed_and_store(chunks, collection):
    for i, chunk in enumerate(chunks):
        # Convert chunk text to a vector using the local model
        vector = embed_model.encode(chunk).tolist()
        # Store the text and its vector in ChromaDB with a unique ID
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[f"chunk_{i}"]
        )


def search_chunks(query, collection, n=3):
    # Convert the user's question to a vector using the same model
    vector = embed_model.encode(query).tolist()
    # Find the 3 chunks whose vectors are closest in meaning to the question
    results = collection.query(
        query_embeddings=[vector],
        n_results=n
    )
    # Return the matched chunk texts (not the vectors)
    return results["documents"][0]


def ask_claude(question, context_chunks):
    # Join the 3 retrieved chunks into one block of text
    context = "\n\n".join(context_chunks)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                # Tell Claude to answer only from the contract text, not general knowledge
                "content": f"""You are a contract review assistant.
                Answer the question using only the contract text below.
                If the answer is not in the text, say so.
                Contract text:
                {context}
                Question: {question}"""
            }
        ]
    )
    # Return only the text part of Claude's response
    return message.content[0].text


# Render the page title and subtitle
st.title("Contract Lens")
st.caption("Upload a PDF contract and ask questions about it.")

# Show a file upload button that only accepts PDFs
uploaded_file = st.file_uploader("Upload a contract PDF", type="pdf")

if uploaded_file:
    # Only index the PDF once; skip if already stored in session state
    if "collection" not in st.session_state:
        with st.spinner("Reading and indexing contract..."):
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)
            # Create a new ChromaDB collection for this contract
            collection = chroma_client.create_collection("contract")
            embed_and_store(chunks, collection)
            # Save the collection to session state so it survives Streamlit reruns
            st.session_state.collection = collection
            st.session_state.chunks_loaded = True
        st.success(f"Indexed {len(chunks)} chunks.")

    # Initialise chat history in session state if this is the first question
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render all previous messages in the chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Show the chat input box
    question = st.chat_input("Ask about the contract...")

    if question:
        # Save and display the user's question
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            # Find the most relevant chunks, then ask Claude
            chunks = search_chunks(question, st.session_state.collection)
            answer = ask_claude(question, chunks)

        # Save and display Claude's answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)