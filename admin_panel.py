# admin_panel.py
"""
Admin Panel - Analytics and Management Dashboard
Add this as a new menu item in app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from database.db import get_conn


def page_admin_panel():
    st.title("🔧 Admin Dashboard")

    # CSS Styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 25%, #10b981 50%, #f59e0b 75%, #f97316 100%) !important;
    }
    
    [data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.95) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 24px !important;
        padding: 24px !important;
    }
    
    [data-testid="stMetricLabel"] { 
        color: #f1f5f9 !important; 
        font-weight: 600 !important; 
    }
    
    [data-testid="stMetricValue"] { 
        color: #ffffff !important; 
        text-shadow: 0 2px 12px rgba(255,255,255,0.9) !important;
        font-size: 2.4em !important; 
        font-weight: 800 !important; 
    }
    </style>
    """, unsafe_allow_html=True)

    # Database connection
    conn = get_conn()
    
    # Metrics
    total = pd.read_sql("SELECT COUNT(*) FROM chat_history", conn).iloc[0, 0]
    success_rate = pd.read_sql(
        "SELECT AVG(CASE WHEN success=1 THEN 1 ELSE 0 END)*100 FROM chat_history", 
        conn
    ).iloc[0, 0] or 0
    low_conf = pd.read_sql(
        "SELECT COUNT(*) FROM chat_history WHERE confidence < 0.7", 
        conn
    ).iloc[0, 0]
    num_intents = pd.read_sql(
        "SELECT COUNT(DISTINCT predicted_intent) FROM chat_history WHERE predicted_intent IS NOT NULL", 
        conn
    ).iloc[0, 0]

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", total)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Low Confidence", low_conf)
    col4.metric("Unique Intents", num_intents)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Chat Analytics", "🔍 Query Analytics", "📤 Export Logs"])

    with tab1:
        st.subheader("📊 Chat Analytics")

        # Intent Distribution Pie Chart
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
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                # Success Rate by Intent
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
                st.plotly_chart(fig_bar, use_container_width=True)

        # Recent Chat Activity
        st.markdown("### 💬 Recent Chat Activity")
        recent_df = pd.read_sql("""
            SELECT user_query, 
                   predicted_intent, 
                   ROUND(confidence*100,0)||'%' as confidence,
                   CASE WHEN success=1 THEN '✅' ELSE '❌' END as success,
                   substr(timestamp, 1, 16) as timestamp
            FROM chat_history 
            ORDER BY id DESC 
            LIMIT 100
        """, conn)
        st.dataframe(recent_df, use_container_width=True)

    with tab2:
        st.subheader("🔍 Query Analytics")

        # NLU History Metrics
        col1, col2, col3 = st.columns(3)
        nlu_total = pd.read_sql("SELECT COUNT(*) FROM nlu_history", conn).iloc[0, 0]
        nlu_intents = pd.read_sql(
            "SELECT COUNT(DISTINCT predicted_intent) FROM nlu_history WHERE predicted_intent IS NOT NULL", 
            conn
        ).iloc[0, 0]
        nlu_low_conf = pd.read_sql(
            "SELECT COUNT(*) FROM nlu_history WHERE confidence < 0.8", 
            conn
        ).iloc[0, 0]

        col1.metric("Total NLU Queries", nlu_total)
        col2.metric("Intents Detected", nlu_intents)
        col3.metric("Low Confidence", nlu_low_conf)

        # Confidence Distribution
        conf_df = pd.read_sql("""
            SELECT ROUND(confidence*100,0) as conf_pct, COUNT(*) as count
            FROM nlu_history 
            WHERE confidence IS NOT NULL
            GROUP BY conf_pct 
            ORDER BY conf_pct
        """, conn)

        if not conf_df.empty:
            fig_conf = px.bar(
                conf_df,
                x='conf_pct',
                y='count',
                title="📊 Confidence Distribution",
                labels={'conf_pct': 'Confidence %', 'count': 'Queries'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Recent NLU Queries
        st.markdown("### Recent NLU Queries")
        nlu_recent = pd.read_sql("""
            SELECT user_query as "Query", 
                   predicted_intent as "Intent", 
                   ROUND(confidence*100,0)||'%' as "Confidence",
                   substr(timestamp, 1, 10) as "Date"
            FROM nlu_history 
            WHERE predicted_intent IS NOT NULL
            ORDER BY id DESC 
            LIMIT 20
        """, conn)
        st.dataframe(nlu_recent, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("📤 Export Data")
        
        # Export Chat History
        chat_csv = pd.read_sql("SELECT * FROM chat_history", conn).to_csv(index=False)
        st.download_button(
            "📥 Export Chat History",
            chat_csv,
            "chat_history.csv",
            "text/csv"
        )

        # Export NLU History
        nlu_csv = pd.read_sql("SELECT * FROM nlu_history", conn).to_csv(index=False)
        st.download_button(
            "📥 Export NLU History",
            nlu_csv,
            "nlu_history.csv",
            "text/csv"
        )

    conn.close()
    st.markdown("---")
    st.caption("👨‍💼 Admin Panel | BankBot AI")
