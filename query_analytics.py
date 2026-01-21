# query_analytics.py
"""
Query Analytics Page - Dynamically tracks NLU predictions in real-time
Shows metrics and recent queries from nlu_history table
"""

import streamlit as st
import pandas as pd
from database.db import get_conn
from datetime import datetime


def page_query_analytics():
    """Query Analytics Page with live metrics and table"""
    
    # Page Title with Icon
    st.markdown("## 🔍 Query Analytics")
    
    # CSS Styling for gradient background and cards
    st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%) !important;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        padding: 10px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Section Headers */
    h3 {
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stButton button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        border-color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Database Connection
    conn = get_conn()
    
    # Fetch Metrics from nlu_history
    total_queries = pd.read_sql(
        "SELECT COUNT(*) as count FROM nlu_history", 
        conn
    ).iloc[0, 0]
    
    intents_detected = pd.read_sql(
        "SELECT COUNT(DISTINCT predicted_intent) as count FROM nlu_history WHERE predicted_intent IS NOT NULL", 
        conn
    ).iloc[0, 0]
    
    low_confidence = pd.read_sql(
        "SELECT COUNT(*) as count FROM nlu_history WHERE confidence < 0.7", 
        conn
    ).iloc[0, 0]
    
    # Display Metrics in 3 Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total NLU Queries", total_queries)
    
    with col2:
        st.metric("Intents Detected", intents_detected)
    
    with col3:
        st.metric("Low Confidence", low_confidence)
    
    st.markdown("---")
    
    # Recent NLU Queries Section
    st.markdown("### Recent NLU Queries")
    
    # Refresh Button
    col_refresh, col_export = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Fetch Recent Queries with better date formatting
    recent_queries = pd.read_sql("""
        SELECT 
            user_query as "Query",
            predicted_intent as "Intent",
            ROUND(confidence * 100, 0) || '%' as "Confidence",
            strftime('%Y-%m-%d %H:%M', timestamp) as "Date"
        FROM nlu_history
        ORDER BY id DESC
        LIMIT 50
    """, conn)
    
    conn.close()
    
    # Display Table
    if not recent_queries.empty:
        st.dataframe(
            recent_queries, 
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("📊 No queries yet. Use the NLU Visualizer in 'User Query' to generate predictions!")
    
    # Footer
    st.markdown("---")
    st.caption("👨‍💼 Admin Panel | BankBot AI")


def log_nlu_query(user_query: str, predicted_intent: str, confidence: float):
    """
    Log NLU prediction to nlu_history table
    Call this function after every NLU prediction in User Query page
    """
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO nlu_history (user_query, predicted_intent, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_query, predicted_intent, confidence, datetime.now().isoformat()))
        conn.commit()
    except Exception as e:
        print(f"Error logging NLU query: {e}")
    finally:
        conn.close()
