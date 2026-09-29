# AXON

**Perception that acts.** An agentic visual inspection system: OpenCV 5 perceives defects, a language-model agent decides what to do about them, and every decision is traceable.

## What it does

AXON watches video footage, detects defects with OpenCV 5, then runs an **agentic loop** — the perception output drives the next tool call, which changes what the system does next.

This is not a detector with a chat interface. The visual evidence changes the system's behavior.

## Quick start

```powershell
# Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,agent]"

# Set up your Gemini API key
notepad .env
# Add: GEMINI_API_KEY=AIza...
#      AXON_LLM_MODELS=gemini-flash-lite-latest,gemini-flash-latest,...

# Generate a test video
python scripts\make_test_video.py

# Run the agent
python scripts\run_agent_demo.py
