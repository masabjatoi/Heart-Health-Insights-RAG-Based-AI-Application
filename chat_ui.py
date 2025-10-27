import streamlit as st
from streamlit_chat import message
import requests
import uuid
import time

# --- Page config ---
st.set_page_config(
    page_title="CardioRAG 💬",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Narrow sidebar */
    [data-testid="stSidebar"] {
        width: 280px;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding-left: 10px;
        padding-right: 10px;
    }

    /* Narrow, centered chat input like ChatGPT */
    div[data-testid="stChatInput"] {
        max-width: 600px;
        margin: 0 auto;
    }

    /* Scrollable sidebar for questions */
    .sidebar-scrollable {
        max-height: 70vh;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title section ---
st.markdown(
    """
    <h1 style='text-align:center; color:#00ADB5;'>💬 CardioRAG</h1>
    <p style='text-align:center; color:gray; font-size:18px;'>Ask anything about heart disease</p>
    <hr style="border:1px solid #00ADB5;">
    """,
    unsafe_allow_html=True
)

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

# --- Chat input ---
query = st.chat_input("Type your question...", key="chat_input")
if query:
    st.session_state.current_input = query
    st.session_state.history.append({"role": "user", "content": query})

    # Call backend
    with st.spinner("🤖 Thinking..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/rag/search",
                json={"query": query},
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("summary", "No response received.")
            else:
                answer = f"⚠️ Server Error: {response.status_code}"
        except Exception as e:
            answer = f"❌ Error connecting to backend: {e}"

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.current_input = ""

# --- Sidebar ---
with st.sidebar:
    # Styled CardioRAG title
    st.markdown(
        """
        <h2 style='font-family: "Courier New", monospace; color:#00ADB5; text-align:center;'>
            CardioRAG
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.subheader("💬 Your Questions")

    # Make questions scrollable
    st.markdown('<div class="sidebar-scrollable">', unsafe_allow_html=True)
    # Display previous user questions
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.markdown(chat['content'])
    # Display current typing question
    if st.session_state.current_input:
        st.markdown(st.session_state.current_input)
    st.markdown('</div>', unsafe_allow_html=True)

    # Spacer
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    

# --- Main chat display ---
for msg in st.session_state.history:
    unique_key = str(uuid.uuid4())
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=unique_key)
    else:
        message(msg["content"], key=unique_key)




# # --- Footer ---
# st.markdown(
#     """
#     <hr>
#     <p style='text-align:center; color:gray; font-size:14px;'>
#     ⚡ Built with <b>FastAPI + Streamlit + FAISS</b>
#     </p>
#     """,
#     unsafe_allow_html=True
# )
