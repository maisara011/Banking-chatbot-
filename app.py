import streamlit as st
import os
import json
import sys
import subprocess
import time
from pathlib import Path
from database.db import init_db, get_conn
from database.bank_crud import create_account
from dialogue_manager.dialogue_handler import DialogueManager
from nlu_engine.nlu_router import NLUProcessor
from llm.llm_handler import LLMHandler
from nlu_engine.domain_gate import is_banking_query
from datetime import datetime
import pandas as pd
import plotly.express as px

# NEW: Import query analytics logging
from query_analytics import log_nlu_query

# ---------------- INIT ----------------
init_db()
nlu = NLUProcessor()

st.set_page_config(page_title="BankBot AI", layout="wide")

# LLM init (once per session)
if "llm" not in st.session_state:
    st.session_state.llm = LLMHandler()

# ---------------- PATHS FOR USER QUERY ----------------
INTENTS_PATH = "nlu_engine/intents.json"
MODEL_DIR = "models/intent_model"
LOG_PATH = os.path.join("models", "training.log")
os.makedirs("models", exist_ok=True)

# ---------------- CSS FOR CHAT UI ----------------
st.markdown("""
<style>
/* User message (green) */
[data-testid="chat-message-user"] {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Bot message (light green) */
[data-testid="chat-message-assistant"] {
    background-color: #E8F5E9;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Remove Streamlit footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* NLU Visualizer Styles */
.title { font-size:26px; font-weight:700; color:#0b5cff; margin-bottom:8px; }
.query-area { background:#f3f7fb; border-radius:8px; padding:12px; border:1px solid #e4eefc; }
.intent-card { background: #fff; border-radius:10px; padding:10px 14px; margin-bottom:10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); display:flex; align-items:center; justify-content:space-between; }
.intent-name { font-weight:600; font-size:15px; color:#16325c; }
.intent-score { background:#eef6ff; color:#0b5cff; padding:6px 12px; border-radius:999px; font-weight:700; min-width:56px; text-align:center; }
.entity-card { background:#f7fff7; border-radius:10px; padding:10px; margin-bottom:8px; border:1px solid #e0f6e0; display:flex; align-items:center; }
.entity-icon { width:36px; height:36px; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; margin-right:10px; background:#fff; }
.small-muted { color:#6b7280; font-size:13px; }
.model-badge { display:inline-block; padding:6px 10px; border-radius:8px; background:#e6ffed; color:#097a3b; font-weight:700; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

# ---------------- UTILITY FUNCTIONS FOR USER QUERY ----------------
def load_intents_file():
    if not os.path.exists(INTENTS_PATH):
        return {"intents": []}
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_intents_file(data):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def model_exists():
    return os.path.isdir(MODEL_DIR) and any(Path(MODEL_DIR).iterdir())

def start_training_subprocess(epochs, batch_size, lr):
    python_exe = sys.executable
    script_path = os.path.join("nlu_engine", "train_intent.py")
    cmd = [
        python_exe, script_path,
        "--intents", INTENTS_PATH,
        "--output_dir", MODEL_DIR,
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--lr", str(lr)
    ]
    logf = open(LOG_PATH, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return proc

def extract_entities_safe(text):
    try:
        from nlu_engine.entity_extractor import EntityExtractor
        ee = EntityExtractor()
        return ee.extract(text)
    except Exception as e:
        import re
        ents = []
        m = re.search(r'(\b(?:₹|rs\.?|Rs\.?|\$)\s*\d[\d,\,\.]*)', text)
        if m:
            ents.append({"entity": "AMOUNT", "value": m.group(1)})
        m2 = re.search(r'\b\d{4,16}\b', text)
        if m2:
            ents.append({"entity": "ACCOUNT_NUMBER", "value": m2.group(0)})
        if "savings" in text.lower():
            ents.append({"entity": "ACCOUNT_TYPE", "value": "savings"})
        if "checking" in text.lower() or "current" in text.lower():
            ents.append({"entity": "ACCOUNT_TYPE", "value": "checking"})
        return ents

def predict_with_trained_model(text, top_k=3):
    if not model_exists():
        raise FileNotFoundError("Trained model not found. Train the model first.")
    from nlu_engine.infer_intent import IntentClassifier
    ic = IntentClassifier(model_dir=MODEL_DIR)
    return ic.predict(text, top_k=top_k)

# Session state for training
if "train_proc" not in st.session_state:
    st.session_state.train_proc = None
    st.session_state.proc_start_time = None

# ---------------- SIDEBAR (REMOVED USER QUERY & QUERY ANALYTICS) ----------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Chatbot", "Database", "Admin Panel"]
)

# ---------------- HOME ----------------
if menu == "Home":
    st.title("🏦 BankBot AI")
    st.markdown("""
    ### Intelligent Banking Chatbot

    **Milestone 1**
    - Intent Classification
    - Entity Extraction

    **Milestone 2**
    - Secure Multi-turn Conversations
    - Authentication
    - Database-backed Transfers
    - Admin Analytics Dashboard

    **Tech Stack**
    - Python, Streamlit
    - spaCy, Transformers
    - SQLite + bcrypt
    - Plotly for Analytics
    """)

# ---------------- CHATBOT ----------------
elif menu == "Chatbot":
    st.markdown("### 🏦 Bank Assistant  \n🟢 Online")

    if "dm" not in st.session_state:
        st.session_state.dm = DialogueManager()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your virtual banking assistant. How can I help you today?"
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        intent, confidence, entities = nlu.process(user_input)

        if st.session_state.dm.in_flow:
            bot_reply = st.session_state.dm.handle(
                intent=intent,
                entities=entities,
                user_text=user_input
            )

            if bot_reply.startswith("__ERROR__:"):
                error_msg = bot_reply.replace("__ERROR__:", "")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {error_msg}"}
                )
                st.error(error_msg)
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply}
                )
                with st.chat_message("assistant"):
                    st.write(bot_reply)

        elif not is_banking_query(user_input, entities):
            st.session_state.dm.reset()

            with st.spinner("🔎 Searching & generating answer..."):
                llm_answer = st.session_state.llm.generate(user_input)

            llm_reply = (
                "🌐 **General Information (LLM Generated)**\n\n"
                "_Source: Web / LLM knowledge_\n\n"
                f"{llm_answer}"
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": llm_reply}
            )

            with st.chat_message("assistant"):
                st.write(llm_reply)

        else:
            bot_reply = st.session_state.dm.handle(
                intent=intent,
                entities=entities,
                user_text=user_input
            )

            if bot_reply.startswith("__ERROR__:"):
                error_msg = bot_reply.replace("__ERROR__:", "")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {error_msg}"}
                )
                st.error(error_msg)
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": bot_reply}
                )
                with st.chat_message("assistant"):
                    st.write(bot_reply)

# ---------------- DATABASE ----------------
elif menu == "Database":
    st.header("🗄️ Database Management")

    name = st.text_input("User Name")
    acc_no = st.text_input("Account Number")
    acc_type = st.selectbox("Account Type", ["savings", "current"])
    balance = st.number_input("Initial Balance", value=50000)
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):
        create_account(name, acc_no, acc_type, balance, password)
        st.success("Account created successfully")

# ---------------- ADMIN PANEL (WITH TABS: CHAT ANALYTICS + USER QUERY + QUERY ANALYTICS) ----------------
elif menu == "Admin Panel":
    st.title("🔧 Admin Dashboard")

    # NO BACKGROUND COLOR - Removed gradient styling
    st.markdown("""
    <style>
    /* Remove background gradient from Admin Panel */
    .stApp {
        background: #ffffff !important;
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Database connection
    conn = get_conn()
    
    # Top-level Metrics
    total = pd.read_sql("SELECT COUNT(*) FROM chat_history", conn).iloc[0, 0]
    success_rate = pd.read_sql(
        "SELECT AVG(CASE WHEN success=1 THEN 1 ELSE 0 END)*100 FROM chat_history", 
        conn
    ).iloc[0, 0] or 0
    low_conf = pd.read_sql(
        "SELECT COUNT(*) FROM chat_history WHERE confidence < 0.7", 
        conn
    ).iloc[0, 0]

    # Display top metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Queries", total)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Low Confidence", low_conf)

    st.markdown("---")

    # TABS: Chat Analytics, User Query, Query Analytics, Export Logs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chat Analytics", "🔍 User Query", "📈 Query Analytics", "📤 Export Logs"])

    # ========== TAB 1: CHAT ANALYTICS ==========
    with tab1:
        st.subheader("📊 Chat Analytics")

        intent_df = pd.read_sql("""
            SELECT predicted_intent, COUNT(*) as count
            FROM chat_history 
            WHERE predicted_intent IS NOT NULL
            GROUP BY predicted_intent 
            ORDER BY count DESC
        """, conn)

        if not intent_df.empty:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_pie = px.pie(
                    intent_df, 
                    values='count', 
                    names='predicted_intent',
                    title="🎯 Intent Distribution"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, width='stretch')

            with col_chart2:
                success_df = pd.read_sql("""
                    SELECT predicted_intent,
                           ROUND(AVG(CASE WHEN success=1 THEN 100 ELSE 0 END), 1) as success_pct
                    FROM chat_history
                    WHERE predicted_intent IS NOT NULL
                    GROUP BY predicted_intent
                """, conn)
                
                fig_bar = px.bar(
                    success_df,
                    x='predicted_intent',
                    y='success_pct',
                    title="✅ Success Rate by Intent",
                    color='success_pct',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_bar, width='stretch')

        st.markdown("### 💬 Recent Chat Activity")
        recent_df = pd.read_sql("""
            SELECT user_query, 
                   predicted_intent, 
                   ROUND(confidence*100,0)||'%' as confidence,
                   CASE WHEN success=1 THEN '✅' ELSE '❌' END as success,
                   substr(timestamp, 1, 16) as timestamp
            FROM chat_history 
            ORDER BY id DESC 
            LIMIT 50
        """, conn)
        st.dataframe(recent_df, width='stretch')

    # ========== TAB 2: USER QUERY (NLU VISUALIZER) ==========
    with tab2:
        st.markdown("## 🔍 User Query - NLU Visualizer")
        
        # Layout: Left (Intents Editor) + Right (NLU Visualizer)
        left_col, right_col = st.columns([1.05, 1.45])

        # LEFT: INTENTS EDITOR + TRAINING
        with left_col:
            st.subheader("📝 Intents (edit & add)")
            intents_data = load_intents_file()

            filtered_intents = [it for it in intents_data.get("intents", []) if it.get("name") != "greet"]
            
            for i, intent in enumerate(filtered_intents):
                with st.expander(f"{intent['name']} ({len(intent.get('examples', []))} examples)"):
                    st.write("**Examples:**")
                    for ex in intent.get("examples", []):
                        st.write("-", ex)
                    new_ex = st.text_input(f"Add example to {intent['name']}", key=f"add_{i}")
                    if st.button(f"➕ Add example", key=f"btn_add_{i}"):
                        if new_ex.strip():
                            for o in intents_data["intents"]:
                                if o.get("name") == intent["name"]:
                                    o.setdefault("examples", []).append(new_ex.strip())
                                    break
                            save_intents_file(intents_data)
                            st.success("✅ Example added!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Enter example text first.")

            st.markdown("---")
            st.subheader("➕ Create New Intent")
            new_intent_name = st.text_input("Intent name", key="new_intent_name")
            new_intent_examples = st.text_area("Examples (one per line)", key="new_intent_examples", height=120)
            if st.button("🚀 Create Intent"):
                if new_intent_name.strip() and new_intent_examples.strip():
                    new_obj = {
                        "name": new_intent_name.strip(), 
                        "examples": [e.strip() for e in new_intent_examples.splitlines() if e.strip()]
                    }
                    intents_data.setdefault("intents", []).append(new_obj)
                    save_intents_file(intents_data)
                    st.success(f"✅ Intent '{new_intent_name}' created!")
                    st.rerun()
                else:
                    st.warning("⚠️ Provide intent name and examples.")

            st.markdown("---")
            st.subheader("🎯 Train Model")
            if model_exists():
                st.markdown("<span style='color:green;font-weight:bold;'>✅ Trained model found</span>", unsafe_allow_html=True)
            else:
                st.info("ℹ️ No trained model found")

            epochs = st.number_input("Epochs", value=2, step=1, min_value=1)
            batch_size = st.number_input("Batch size", value=8, step=1, min_value=1)
            lr = st.number_input("Learning rate", value=2e-5, format="%.8f")
            
            if st.button("🚀 Start Training"):
                proc = st.session_state.get("train_proc")
                if proc is not None and proc.poll() is None:
                    st.warning("⚠️ Training already running.")
                else:
                    try:
                        try: 
                            open(LOG_PATH, "w", encoding="utf-8").close()
                        except: 
                            pass
                        proc = start_training_subprocess(epochs, batch_size, lr)
                        st.session_state.train_proc = proc
                        st.session_state.proc_start_time = time.time()
                        st.success(f"✅ Training started (pid={proc.pid}).")
                    except Exception as e:
                        st.error("❌ Failed to start training.")
                        st.exception(e)

        # RIGHT: NLU VISUALIZER
        with right_col:
            st.subheader("🔍 NLU Visualizer")
            query = st.text_area(
                "User Query:", 
                height=120, 
                value='I want to transfer $500 from savings to checking'
            )

            top_k = st.number_input("Top intents", min_value=1, max_value=6, value=3, step=1)

            if st.button("🔎 Analyze"):
                if not model_exists():
                    st.error("❌ No trained model. Train first.")
                else:
                    try:
                        with st.spinner("🔄 Running model..."):
                            preds = predict_with_trained_model(query, top_k=top_k)
                            
                            if preds:
                                top_intent = preds[0].get("intent")
                                top_confidence = preds[0].get("score", 0.0)
                                log_nlu_query(query, top_intent, top_confidence)
                                st.success("✅ Logged to Query Analytics!")
                                
                    except Exception as e:
                        st.error("❌ Prediction failed.")
                        st.exception(e)
                        preds = []

                    entities = extract_entities_safe(query)

                    res_left, res_right = st.columns([1, 1])
                    
                    with res_left:
                        st.markdown("#### 🎯 Intent Recognition")
                        if not preds:
                            st.info("ℹ️ No predictions.")
                        else:
                            for p in preds:
                                name = p.get("intent")
                                score = float(p.get("score", 0.0))
                                display_score = max(0.01, min(1.0, score))
                                st.markdown(
                                    f"""
                                    <div class="intent-card">
                                      <div class="intent-name">{name.replace('_',' ').title()}</div>
                                      <div style="min-width:88px;display:flex;justify-content:flex-end;">
                                        <div class="intent-score">{display_score:.2f}</div>
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    with res_right:
                        st.markdown("#### 🏷️ Entities")
                        if not entities:
                            st.info("ℹ️ No entities.")
                        else:
                            for e in entities:
                                ent_name = e.get("entity")
                          
