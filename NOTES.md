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