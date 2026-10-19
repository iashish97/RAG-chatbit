# University Helpdesk RAG Chatbot
Setup: `pip install -r requirements.txt` then `py -m streamlit run app.py` (or `streamlit run app.py`)
Optional: put your Groq key in `.env` (`GROQ_API_KEY=...`), paste it in the sidebar, or set `GROQ_API_KEY` in your environment. Model defaults to `openai/gpt-oss-120b` (or auto-detects available Groq chat models; override with `GROQ_MODEL`). Without a key the bot works offline, quoting the retrieved FAQ.
Files: knowledge_base.json (15 FAQs) · rag.py (retrieval, memory, grounding) · app.py (Streamlit UI)
Try: "How many books can I borrow?" → "And for how long?" → "What are the opening hours?" → "Do you offer parking permits?" (not in KB → refusal)
