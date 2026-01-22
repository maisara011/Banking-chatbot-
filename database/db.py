
# database/db.py

import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "bankbot.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Existing users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)

    # Existing accounts table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_number TEXT PRIMARY KEY,
        user_name TEXT,
        account_type TEXT,
        balance INTEGER,
        password_hash BLOB,
        FOREIGN KEY(user_name) REFERENCES users(name)
    )
    """)

    # Existing transactions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_account TEXT,
        to_account TEXT,
        amount INTEGER,
        timestamp TEXT
    )
    """)

    # NEW: Admin Panel - Chat History Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        user_query TEXT,
        predicted_intent TEXT,
        confidence REAL,
        entities TEXT,
        success INTEGER DEFAULT 1
    )
    """)

    # NEW: Admin Panel - NLU History Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS nlu_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        user_query TEXT,
        predicted_intent TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()
