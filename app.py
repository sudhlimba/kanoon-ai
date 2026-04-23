"""
app.py
-------
KanoonAI — Indian Legal Document Analyzer
Streamlit frontend. Run with:  streamlit run app.py
"""

import os
import streamlit as st
from document_downloader import get_available_document_paths, check_documents_exist
from rag_pipeline import initialize_rag_system, ask_question, rebuild_vector_store

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "KanoonAI – Indian Legal Assistant",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── CSS: Warm saffron + deep navy theme — editorial, trustworthy, Indian ──────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Noto+Sans:wght@300;400;500;600&family=Noto+Sans+Devanagari:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans', 'Noto Sans Devanagari', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0d1117;
    background-image:
        radial-gradient(ellipse at 10% 10%, rgba(255,153,51,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 90%, rgba(19,136,8,0.06) 0%, transparent 50%);
    color: #e2e8f0;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem;
    position: relative;
}
.hero-flag {
    display: flex;
    justify-content: center;
    gap: 0;
    margin-bottom: 1rem;
    border-radius: 4px;
    overflow: hidden;
    width: 80px;
    height: 14px;
    margin: 0 auto 1rem auto;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.flag-saffron { background: #FF9933; flex: 1; }
.flag-white   { background: #FFFFFF; flex: 1; }
.flag-green   { background: #138808; flex: 1; }

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #FF9933;
    letter-spacing: -0.5px;
    margin: 0.3rem 0 0.2rem;
    text-shadow: 0 0 40px rgba(255,153,51,0.3);
}
.hero .tagline {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}
.hero .tagline-hindi {
    font-size: 0.9rem;
    color: #64748b;
    font-family: 'Noto Sans Devanagari', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,153,51,0.15) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Document status badges ── */
.doc-badge {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    font-weight: 500;
}
.doc-badge.ready {
    background: rgba(19,136,8,0.12);
    border: 1px solid rgba(19,136,8,0.3);
    color: #4ade80;
}
.doc-badge.missing {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    color: #f87171;
}

/* ── Suggested questions ── */
.sq-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin: 1rem 0;
}
.sq-card {
    background: rgba(255,153,51,0.06);
    border: 1px solid rgba(255,153,51,0.18);
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    font-size: 0.82rem;
    color: #cbd5e1;
    cursor: pointer;
    transition: all 0.2s;
    line-height: 1.4;
}
.sq-card:hover {
    background: rgba(255,153,51,0.12);
    border-color: rgba(255,153,51,0.4);
    color: #FF9933;
}

/* ── Chat messages ── */
.user-msg {
    background: linear-gradient(135deg, #FF9933 0%, #e07b10 100%);
    color: #0d1117;
    font-weight: 500;
    padding: 1rem 1.2rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.6rem 0 0.6rem 3rem;
    font-size: 0.93rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(255,153,51,0.25);
}
.bot-msg {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    padding: 1rem 1.2rem;
    border-radius: 16px 16px 16px 4px;
    margin: 0.6rem 3rem 0.6rem 0;
    font-size: 0.93rem;
    line-height: 1.7;
}
.msg-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    opacity: 0.55;
}

/* ── Source reference chips ── */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255,153,51,0.1);
    border: 1px solid rgba(255,153,51,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.74rem;
    color: #FF9933;
    margin: 0.2rem 0.2rem 0 0;
    font-weight: 500;
}
.source-text {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid rgba(255,153,51,0.4);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin: 0.35rem 0;
    font-size: 0.79rem;
    color: #94a3b8;
    line-height: 1.55;
    font-style: italic;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #FF9933, #e07b10) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Status/info ── */
.info-box {
    background: rgba(255,153,51,0.08);
    border: 1px solid rgba(255,153,51,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #fbd38d;
    margin: 0.6rem 0;
    line-height: 1.6;
}
.success-box {
    background: rgba(19,136,8,0.1);
    border: 1px solid rgba(19,136,8,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #4ade80;
    margin: 0.6rem 0;
}

hr { border-color: rgba(255,153,51,0.12) !important; }

/* ── Section headers ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FF9933;
    margin: 1rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Suggested Questions ───────────────────────────────────────────────────────
SUGGESTED_QUESTIONS = [
    {"en": "What are my Fundamental Rights?",             "label": "Fundamental Rights"},
    {"en": "How do I file an RTI application?",           "label": "File RTI"},
    {"en": "What is the punishment for cybercrime?",      "label": "Cybercrime"},
    {"en": "What are my rights as a consumer?",           "label": "Consumer Rights"},
    {"en": "Can the government take away my freedom of speech?", "label": "Free Speech"},
    {"en": "What is the Right to Education?",             "label": "Right to Education"},
    {"en": "How can I complain against a company?",       "label": "File Complaint"},
    {"en": "मेरे मौलिक अधिकार क्या हैं?",                "label": "मौलिक अधिकार (Hindi)"},
]


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "rag_chain"     : None,
    "chat_history"  : [],
    "system_ready"  : False,
    "loading"       : False,
    "suggested_q"   : None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙️ Setup</div>', unsafe_allow_html=True)

    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if not api_key:
       st.error("⚠️ OpenAI API key not found. Add it in Streamlit Secrets.")
       st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown('<div class="section-title">📚 Documents</div>', unsafe_allow_html=True)

    # Show document status
    doc_status = check_documents_exist()
    all_ready  = all(info["exists"] for info in doc_status.values())

    for info in doc_status.values():
        icon  = "✅" if info["exists"] else "❌"
        cls   = "ready" if info["exists"] else "missing"
        st.markdown(
            f'<div class="doc-badge {cls}">{icon} {info["name"]}</div>',
            unsafe_allow_html=True
        )

    if not all_ready:
        st.markdown("""
        <div class="info-box">
            ⚠️ Some documents are missing.<br>
            Run in terminal:<br>
            <code>python document_downloader.py</code>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Initialize / Re-index button
    available_paths = get_available_document_paths()

    if available_paths and not st.session_state.system_ready:
        if st.button("🚀 Initialize KanoonAI", use_container_width=True):
          if st.session_state.rag_chain is None:
            if not api_key:
                st.error("⚠️ Enter your OpenAI API Key first.")
            else:
                with st.spinner("⚙️ Loading legal documents..."):
                    try:
                        chain = initialize_rag_system(available_paths)
                        st.session_state.rag_chain   = chain
                        st.session_state.system_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    if st.session_state.system_ready:
        st.markdown('<div class="success-box">✅ System ready! Ask your question.</div>',
                    unsafe_allow_html=True)

        if st.button("🔄 Re-index Documents", use_container_width=True):
            with st.spinner("Re-building index..."):
                chain = rebuild_vector_store(available_paths)
                st.session_state.rag_chain    = chain
                st.session_state.chat_history = []
                st.rerun()

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.76rem; color:#475569; line-height:1.8;'>
    <strong style='color:#64748b;'>How it works:</strong><br>
    1️⃣ 4 Indian legal acts loaded<br>
    2️⃣ Split into text chunks<br>
    3️⃣ Embedded via OpenAI<br>
    4️⃣ Stored locally in ChromaDB<br>
    5️⃣ Your question → retrieve chunks<br>
    6️⃣ GPT answers in Hindi/English<br><br>
    <em>Supports Hindi & English questions</em>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <div class="hero-flag">
        <div class="flag-saffron"></div>
        <div class="flag-white"></div>
        <div class="flag-green"></div>
    </div>
    <h1>⚖️ KanoonAI</h1>
    <div class="tagline">Indian Legal Document Analyzer · Know Your Rights</div>
    <div class="tagline-hindi">अपने अधिकार जानें · हिंदी और अंग्रेज़ी में पूछें</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Not initialized yet ───────────────────────────────────────────────────────
if not st.session_state.system_ready:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:2.5rem 0; color:#475569;'>
            <div style='font-size:3.5rem; margin-bottom:1rem;'>📜</div>
            <div style='font-size:1.15rem; font-weight:600; color:#64748b; margin-bottom:0.5rem;'>
                System not initialized yet
            </div>
            <div style='font-size:0.88rem; line-height:1.7;'>
                1. Run <code>python document_downloader.py</code> to get legal PDFs<br>
                2. Add your OpenAI API key in the sidebar<br>
                3. Click <strong>Initialize KanoonAI</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── System ready — show chat ──────────────────────────────────────────────────
else:
    # Suggested questions (shown only if chat is empty)
    if not st.session_state.chat_history:
        st.markdown('<div class="section-title">💡 Try asking:</div>', unsafe_allow_html=True)

        cols = st.columns(4)
        for i, sq in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 4]:
                if st.button(sq["label"], key=f"sq_{i}", use_container_width=True):
                    st.session_state.suggested_q = sq["en"]
                    st.rerun()

        st.markdown("---")

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <div class="msg-label">You</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-msg">
                <div class="msg-label">⚖️ KanoonAI</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)

            # Source references
            if msg.get("sources"):
                unique_sources = {}
                for doc in msg["sources"]:
                    name = doc.metadata.get("document_name", "Legal Document")
                    page = doc.metadata.get("page", 0)
                    key  = f"{name} · Page {int(page)+1}"
                    if key not in unique_sources:
                        unique_sources[key] = doc.page_content[:250].replace("\n", " ")

                chips_html = "".join(
                    f'<span class="source-chip">📄 {src}</span>'
                    for src in unique_sources
                )
                st.markdown(
                    f'<div style="margin:0.4rem 3rem 0 0">{chips_html}</div>',
                    unsafe_allow_html=True
                )

                with st.expander("📖 View exact text used from documents"):
                    for src, text in unique_sources.items():
                        st.markdown(f"""
                        <div style="margin-bottom:0.3rem; font-size:0.8rem;
                                    font-weight:600; color:#FF9933;">{src}</div>
                        <div class="source-text">{text}...</div>
                        """, unsafe_allow_html=True)

    # Handle suggested question click
    if st.session_state.suggested_q:
        question = st.session_state.suggested_q
        st.session_state.suggested_q = None

        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching legal documents..."):
            try:
                result = ask_question(st.session_state.rag_chain, question)
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                })
        st.rerun()

    # Chat input
    user_input = st.chat_input(
        "Ask any legal question in Hindi or English... / कोई भी कानूनी सवाल पूछें...",
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching legal documents..."):
            try:
                result = ask_question(st.session_state.rag_chain, user_input)
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                })
        st.rerun()
