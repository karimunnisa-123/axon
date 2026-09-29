"""Try each candidate Gemini model and report which ones respond."""
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Ordered by preference: fast + stable first, then fallbacks
CANDIDATES = [
    "gemini-3.8-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
]

for model in CANDIDATES:
    print(f"Trying {model} ...", end=" ", flush=True)
    try:
        r = client.models.generate_content(
            model=model,
            contents="Reply with exactly: AXON online",
        )
        print(f"OK -> {r.text.strip()!r}")
        print(f"\n*** Use this model: {model} ***")
        break
    except errors.ServerError as e:
        print(f"503 (busy)")
    except errors.ClientError as e:
        # 404 = model name wrong / no access. 429 = rate limited.
        code = getattr(e, "code", None) or "?"
        print(f"client error {code}")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
    time.sleep(1)
else:
    print("\nNo model responded. Google may be having a broad outage.")