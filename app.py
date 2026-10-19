import streamlit as st
from rag import SupportBot, GROQ_MODEL

st.set_page_config(page_title="University Helpdesk Bot", page_icon="🎓")
st.title("🎓 University Helpdesk Assistant")
st.caption("RAG chatbot · answers only from the helpdesk FAQ · remembers this conversation")

WELCOME_MSG = ("assistant", "Hi! I'm the University Helpdesk assistant. Ask me about fees, registration, library, Wi-Fi, hostels and more.")

if "bot" not in st.session_state:
    st.session_state.bot = SupportBot()  # uses GROQ_API_KEY env var if set
    st.session_state.messages = [WELCOME_MSG]
bot = st.session_state.bot

with st.sidebar:
    st.subheader("Settings & memory")
    key = st.text_input("Groq API key", type="password", help="Get one free at console.groq.com")
    if key and st.session_state.get("key") != key:
        st.session_state.key = key
        st.session_state.bot = SupportBot(api_key=key); bot = st.session_state.bot
    st.write("Mode:", f"🤖 Groq ({bot.model})" if bot.client else "📄 Offline extractive")
    if bot.last_error: st.warning(f"Groq error, used fallback: {bot.last_error[:150]}")
    st.write("Remembered facts:", bot.memory.facts or "none yet")
    if st.button("🗑️ New conversation"):
        bot.reset(); st.session_state.messages = [WELCOME_MSG]; st.rerun()
    with st.expander("Browse knowledge base"):
        for d in bot.retriever.docs: st.markdown(f"**{d['id']}. {d['question']}**")

for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

if q := st.chat_input("Type your question..."):
    st.chat_message("user").write(q)
    st.session_state.messages.append(("user", q))
    answer, sources = bot.ask(q)
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Retrieved sources"):
                for score, d in sources:
                    st.markdown(f"**FAQ {d['id']}** (similarity {score:.2f}) – {d['question']}")
    st.session_state.messages.append(("assistant", answer))
