# Notes

## Create a virtual environment

Go to the project root, and run:
```
python -m venv venv

venv\Scripts\Activate.ps1
```

## Install Dependencies

Create a requirements.txt file in the project root with this content:
```
anthropic
streamlit
chromadb
pdfplumber
sentence-transformers
```

Then install everything in one command:
```
pip install -r requirements.txt
```

This will take a few minutes because sentence-transformers pulls in PyTorch.

To verify it worked:
```
python -c "import anthropic, streamlit, chromadb, pdfplumber; print('all good')"
```

Then it will print `all good`.

Also, install:
```
pip install python-dotenv
```

On Windows, Install C++ Build Tools.

Inside Visual Studio, install Desktop development with C++.

ChromaDB uses a library called hnswlib to do fast vector similarity search. That library is written in C++, so Python needs a C++ compiler to build it on your machine.
On Mac and Linux this compiler comes pre-installed. On Windows it does not, so you have to add it manually.

When you run pip install chromadb, pip needs to compile the C++ code in hnswlib. It looks for a C++ compiler on your machine, and on Windows that compiler comes from Visual Studio's C++ tools.

## Testing questions

- "Who are the parties in this contract?"
- "What insurance is required?"
- "What is the contract value?"
- "What are the payment terms?"
- "When does this contract expire?"
- "What are the termination conditions?"
- "Who owns the work produced?"
- "What are the insurance requirements?"
- "What happens in a dispute?"
- "Who owns the deliverables?"

## The idea of `app.py`

### Phase 1

The app splits the PDF into small pieces of 500 words each, called chunks, because Claude cannot read the whole document at once. 

Each chunk is then converted into a list of numbers called a vector, where text with similar meaning produces similar numbers. 

These vectors are stored in ChromaDB, which is a vector database. 

When user asks a question, it is converted into a vector the same way, and ChromaDB finds the chunks whose vectors are closest to user's question. Those chunks are then sent to Claude to answer from.

### Phase 2

chunks is a list of strings, where each string is about 500 words joined together.

```
chunks = [
    "Payment is due within 30 days of invoice. Late fees apply...",  # 500 words
    "Either party may terminate this agreement with 30 days notice...",  # 500 words
    "The contractor shall not disclose confidential information..."   # 500 words
]
```

embeddings work on the whole chunk, not each word. One chunk produces one vector.

```
chunks     = ["Payment is due...",  "Either party may terminate...",  "Contractor shall not..."]
embeddings = [[0.12, -0.34, ...],   [0.88, 0.01, ...],               [0.44, -0.21, ...]]
              one vector             one vector                        one vector
```

The model reads all 500 words together and produces one vector that represents the overall meaning of that chunk.

So when you search "When is payment due?", ChromaDB compares that question's vector to each chunk's vector, and returns the chunk about payment, because their meanings are close.

##### The print results in `flag_risks()`

`print("Raw response from Claude for risk flagging:", raw)`:

```
Raw response from Claude for risk flagging: ```json
[
  {
    "clause": "Unilateral Termination for Convenience",
    "risk_level": "high",
    "explanation": "The COMMISSION can terminate the Agreement at any time for convenience with only 30 days' notice, leaving the CONSULTANT exposed to lost anticipated revenue and unrecoverable mobilization or overhead costs. The CONSULTANT has no reciprocal right to terminate for convenience on equivalent terms."
  },
  {
    "clause": "CONSULTANT Termination Notice Period (120 Days)",
    "risk_level": "high",
    "explanation": "The CONSULTANT must provide 120 days' advance written notice to terminate, which is disproportionately long compared to the 30-day notice the COMMISSION requires. Additionally, the CONSULTANT bears liability for all reprocurement costs even when exercising a voluntary exit right."
  },
  {
    "clause": "Broad Indemnification",
    "risk_level": "high",
    "explanation": "The indemnification clause requires the CONSULTANT to defend and hold harmless the COMMISSION for claims arising out of CONSULTANT's negligence, recklessness, or willful misconduct, which is standard, but also extends to all federal, state, and local taxes and payroll contributions, creating broad and potentially costly financial exposure."
  },
 ...
]
```


`print("Cleaned response from Claude for risk flagging:", clean)`:

```
Cleaned response from Claude for risk flagging: [
  {
    "clause": "Unilateral Termination for Convenience",
    "risk_level": "high",
    "explanation": "The COMMISSION can terminate the Agreement at any time for convenience with only 30 days' notice, leaving the CONSULTANT exposed to lost anticipated revenue and unrecoverable mobilization or overhead costs. The CONSULTANT has no reciprocal right to terminate for convenience on equivalent terms."
  },
  {
    "clause": "CONSULTANT Termination Notice Period (120 Days)",
    "risk_level": "high",
    "explanation": "The CONSULTANT must provide 120 days' advance written notice to terminate, which is disproportionately long compared to the 30-day notice the COMMISSION requires. Additionally, the CONSULTANT bears liability for all reprocurement costs even when exercising a voluntary exit right."
  },
  {
    "clause": "Broad Indemnification",
    "risk_level": "high",
    "explanation": "The indemnification clause requires the CONSULTANT to defend and hold harmless the COMMISSION for claims arising out of CONSULTANT's negligence, recklessness, or willful misconduct, which is standard, but also extends to all federal, state, and local taxes and payroll contributions, creating broad and potentially costly financial exposure."
  },
  ...
]
```