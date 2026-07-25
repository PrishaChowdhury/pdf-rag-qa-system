"""
Diagnostic script: lists every Gemini model your API key can access, and flags
which ones support embedContent (i.e. which embedding model names are valid
for YOUR key/account/API version).

Run with:  python list_models.py
"""

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY", "")

if not api_key:
    raise SystemExit("GOOGLE_API_KEY not found in .env")

from google import genai

client = genai.Client(api_key=api_key)

print("Models available to your API key:\n")
embedding_models = []

for model in client.models.list():
    actions = getattr(model, "supported_actions", None) or []
    print(f"{model.name:45s} actions={actions}")
    if any("embed" in a.lower() for a in actions):
        embedding_models.append(model.name)

print("\n" + "=" * 60)
if embedding_models:
    print("Models that support embedding (use one of these in eval_ndcg.py):")
    for m in embedding_models:
        print(f"  - {m}")
else:
    print("No models on your key report embedding support.")
    print("This likely means the Generative Language API embedding endpoint")
    print("isn't enabled for this key/project, or your key is scoped to a")
    print("different API version than v1beta.")