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