import os

# Importing `main` builds an OpenAI-free app, but data_loader reads env at import.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
