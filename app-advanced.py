import os                           # lets Python read environment variables like ANTHROPIC_API_KEY
import json                         # used to parse the JSON response from Claude
import anthropic                    # the official Anthropic Python SDK
import chromadb                     # vector database for storing and searching embedded text chunks
import pdfplumber                   # reads PDF files and extracts text from each page
import streamlit as st              # builds the web UI
from sentence_transformers import SentenceTransformer   # converts text into numeric vectors (embeddings)
from dotenv import load_dotenv      # reads the .env file so os.getenv() can find your API key

load_dotenv()   # load environment variables from .env before anything else runs

# create the Anthropic client using the key from your .env file
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# create an in-memory ChromaDB instance (no disk storage, resets when the app restarts)
chroma_client = chromadb.Client()

# load a small, fast embedding model that runs locally, no API call needed
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(pdf_file):
    text = ""                                       # start with an empty string
    with pdfplumber.open(pdf_file) as pdf:          # open the uploaded PDF file
        for page in pdf.pages:                      # loop through every page
            text += page.extract_text() or ""       # add the page text, or "" if the page has no text (e.g. a blank page)
    return text                                     # return the full document as one long string


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()        # split the full text into individual words
    chunks = []                 # list to collect each chunk
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])     # take up to 500 words starting at position i, join them back into a sentence
        chunks.append(chunk)                        # add this chunk to the list
        i += chunk_size - overlap                   # move forward by 450 words (500 minus 50 overlap), so the next chunk shares 50 words with this one, giving Claude some context at the boundaries
    return chunks               # chunks is a list of strings, each roughly 500 words long e.g. ["Payment is due...", "Either party may terminate...", "Contractor shall not..."]


def embed_and_store(chunks, collection):
    embeddings = embedder.encode(chunks).tolist()   # convert each chunk into a vector. embedder.encode() returns one vector per chunk. .tolist() converts from numpy array to plain Python list, which ChromaDB expects. e.g. [[0.1, 0.2, ...], [0.05, 0.3, ...], [0.2, 0.1, ...]]
    # print("Generated embeddings for chunks:", embeddings[0])    # print the first vector to the terminal so you can confirm encoding worked e.g. Generated embeddings for chunks: [-0.0668252632021904, 0.04720580205321312,...]
    pairs = zip(chunks, embeddings)     # zip() pairs each chunk with its vector: (chunk0, embedding0), (chunk1, embedding1), ...
    for i, pair in enumerate(pairs):    # enumerate() adds a counter: (0, (chunk0, embedding0)), (1, (chunk1, embedding1)), ...
        chunk, embedding = pair         # unpack the pair into two separate variables
        collection.add(
            documents=[chunk],          # the raw text, stored so we can return it later when searching
            embeddings=[embedding],     # the vector, used to find the closest match to a query
            ids=[f"chunk_{i}"]          # a unique ID like "chunk_0", "chunk_1", required by ChromaDB
        )


def search_chunks(query, collection, n=3):
    embedding = embedder.encode([query]).tolist()[0]    # convert the user's question into a vector. encode() always returns a list, so [0] unwraps it to a single vector
    results = collection.query(
        query_embeddings=[embedding],   # compare this question vector against all stored chunk vectors
        n_results=n                     # return the 3 closest matches
    )
    # print("Search results from ChromaDB:", results)     # print the raw ChromaDB response to the terminal for debugging, includes matched chunk texts and distance scores
    return results["documents"][0]      # ChromaDB wraps results in a nested list. [0] gets the first query's results, which is a plain list of 3 chunk strings


def flag_risks(chunks):
    # join the first 20 chunks into one big block of text
    # we cap at 20 to stay within Claude's context window and keep the API call fast
    contract_text = "\n\n".join(chunks[:20])

    # send the contract text to Claude and ask it to find risky clauses
    message = client.messages.create(
        model="claude-sonnet-4-6",      # the Claude model to use
        max_tokens=2048,                # maximum length of Claude's reply
        messages=[
            {
                "role": "user",
                "content": f"""You are a contract risk analyst.

                Read the contract text below and identify clauses that carry legal or financial risk.

                Return ONLY a JSON array. No explanation outside the JSON. Each item must have:
                - "clause": short name of the clause
                - "risk_level": one of "high", "medium", or "low"
                - "explanation": one or two sentences explaining the risk

                Contract text:
                {contract_text}"""      # the actual contract text is inserted here via the f-string
            }
        ]
    )

    # Claude's reply is in message.content, which is a list of content blocks
    # [0].text gets the text from the first (and only) block
    raw = message.content[0].text
    # print("Raw response from Claude for risk flagging:", raw)             # See NOTES.md

    # Claude sometimes wraps JSON in markdown code fences like ```json ... ```
    # .strip() removes leading and trailing whitespace
    # .removeprefix() removes the opening fence if present
    # .removesuffix() removes the closing fence if present
    # the second .strip() cleans up any remaining whitespace
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # print("Cleaned response from Claude for risk flagging:", clean)       # See NOTES.md

    # parse the clean string into a Python list of dicts, e.g. [{"clause": "Payment", "risk_level": "high", "explanation": "..."}]
    return json.loads(clean)


def ask_claude(question, context_chunks):
    # join the 3 retrieved chunks into one block of text to give Claude as context
    context = "\n\n".join(context_chunks)

    # send the question and the relevant contract text to Claude
    message = client.messages.create(
        model="claude-sonnet-4-6",          # the Claude model to use
        max_tokens=1024,                    # maximum length of Claude's reply
        messages=[
            {
                "role": "user",
                "content": f"""You are a contract review assistant.
                Answer the question using only the contract text below.
                If the answer is not in the text, say so.

                Contract text:
                {context}               

                Question: {question}"""     # the user's question is inserted here
            }
        ]
    )

    # return only the text from Claude's reply
    return message.content[0].text


# map each risk level to a coloured circle emoji for display in the UI
BADGE_COLOURS = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢"
}

# set the page title shown at the top of the Streamlit app
st.title("Contract Lens")

# show a small subtitle below the title
st.caption("Upload a PDF contract and ask questions about it.")

# show a file uploader widget that only accepts PDF files
uploaded_file = st.file_uploader("Upload a contract PDF", type="pdf")

if uploaded_file:
    # only index the contract if we have not done it yet this session
    # st.session_state persists data across Streamlit reruns (which happen on every user action)
    if "collection" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        with st.spinner("Reading and indexing contract..."):    # show a loading message while this runs
            text = extract_text(uploaded_file)                  # extract all text from the PDF
            chunks = chunk_text(text)                           # split text into overlapping chunks
            collection = chroma_client.create_collection("contract")    # create a new ChromaDB collection named "contract"
            embed_and_store(chunks, collection)                 # embed all chunks and store them in ChromaDB
            st.session_state.collection = collection            # save the collection to session state so it survives reruns
            st.session_state.chunks = chunks                    # save the raw chunks too, used later for risk flagging
            st.session_state.last_file = uploaded_file.name     # remember which file we indexed
            st.session_state.risks = None                       # reset risks so they get re-scanned
            st.session_state.messages = []                      # reset chat history
        st.success(f"Indexed {len(chunks)} chunks.")            # show a green success message when done

    # only run risk flagging if we have not done it yet this session
    if st.session_state.get("risks") is None:
        with st.spinner("Scanning for risks..."):               # show a loading message while Claude analyses the contract
            st.session_state.risks = flag_risks(st.session_state.chunks)    # call Claude and save the results

    # display the risk flags section
    st.subheader("Risk flags")
    for risk in st.session_state.risks:                         # loop through each risk item returned by Claude
        level = risk.get("risk_level", "low")                   # get the risk level, default to "low" if missing
        icon = BADGE_COLOURS.get(level, "🟢")                   # get the matching emoji, default to green if unknown level
        with st.expander(f"{icon} {risk['clause']} — {level.upper()}"):    # create a collapsible section with the clause name and level
            st.write(risk["explanation"])                       # show the explanation text inside the expander

    # add a horizontal line between the risk section and the chat section
    st.divider()

    st.subheader("Ask a question")

    # initialise the chat history list if this is the first run
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # replay all previous messages so the chat history is visible after each rerun
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])      # "role" is either "user" or "assistant"

    # show the chat input box at the bottom of the page
    question = st.chat_input("Ask about the contract...")

    if question:
        # save the user's question and show it in the chat
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):                                         # show a loading message while Claude generates a reply
            chunks = search_chunks(question, st.session_state.collection)       # find the 3 most relevant chunks for this question
            answer = ask_claude(question, chunks)                               # send the question and chunks to Claude

        # save Claude's answer and show it in the chat
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)