import streamlit as st
import pandas as pd
import sqlite3
import os
import random
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# DATABASE SETUP
# =========================

DB_PATH = "app.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def setup_database():
    conn = get_conn()
    cur = conn.cursor()

    # USERS TABLE
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT
                )''')

    # WORKSPACES TABLE
    cur.execute('''CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )''')

    # DATASETS TABLE
    cur.execute('''CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    workspace_id INTEGER,
                    filename TEXT,
                    format TEXT,
                    uploaded_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
                )''')

    # CHAT_HISTORY TABLE
    cur.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    workspace_id INTEGER,
                    user_message TEXT,
                    bot_response TEXT,
                    intent TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
                )''')

    conn.commit()
    conn.close()

setup_database()

# =========================
# AUTHENTICATION
# =========================

def login_user(username, password):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()
    conn.close()
    return user

def register_user(username, password):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# =========================
# STREAMLIT UI SETUP
# =========================

st.set_page_config(page_title="Adaptive Chatbot", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center; color:#00FFFF;'>🤖 Adaptive Chatbot System</h1>", unsafe_allow_html=True)

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Select Option", menu)

# =========================
# LOGIN / REGISTER
# =========================
if choice == "Register":
    st.subheader("Create a new account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_pass):
            st.success("✅ Registration successful! You can now login.")
        else:
            st.error("❌ Username already exists.")

elif choice == "Login":
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# =========================
# MAIN DASHBOARD
# =========================
if "user" in st.session_state:
    user = st.session_state["user"]
    uid = user[0]
    st.sidebar.success(f"Logged in as {user[1]}")

    # Workspaces
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM workspaces WHERE user_id=?", (uid,))
    workspaces = cur.fetchall()
    conn.close()

    workspace_names = [w[1] for w in workspaces]
    selected_ws = st.sidebar.selectbox("Select Workspace", ["+ Create New"] + workspace_names)

    if selected_ws == "+ Create New":
        new_ws_name = st.sidebar.text_input("Enter new workspace name")
        if st.sidebar.button("Create"):
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO workspaces (user_id, name, created_at) VALUES (?, ?, ?)",
                        (uid, new_ws_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            st.success("✅ Workspace created successfully!")
            st.rerun()
    else:
        # Get workspace id
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id FROM workspaces WHERE name=? AND user_id=?", (selected_ws, uid))
        workspace_id = cur.fetchone()[0]
        conn.close()

        st.markdown(f"### 📂 Workspace: `{selected_ws}`")

        # Upload dataset
        uploaded_file = st.file_uploader("Upload training dataset (CSV with 'text', 'intent', 'response')", type=["csv"])

        model = None
        vectorizer = None
        df = None

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if not {'text', 'intent', 'response'}.issubset(df.columns):
                st.error("❌ Dataset must have columns: 'text', 'intent', 'response'")
                st.stop()

            # Show first 5 rows of dataset
            st.markdown("### 📊 Preview of Uploaded Dataset (First 5 Rows)")
            st.dataframe(df.head())

            conn = get_conn()
            cur = conn.cursor()
            cur.execute("INSERT INTO datasets (user_id, workspace_id, filename, format, uploaded_at) VALUES (?, ?, ?, ?, ?)",
                        (uid, workspace_id, uploaded_file.name, "csv", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()

            X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], test_size=0.2, random_state=42)
            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            model = MultinomialNB()
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(vectorizer.transform(X_test))
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Model trained successfully with accuracy: {acc:.2f}")

        # Chat area
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 💬 Chat with your Bot")
            user_input = st.text_input("🧑 You:", placeholder="Ask me something...")

            if user_input and model is not None:
                user_vec = vectorizer.transform([user_input])
                intent = model.predict(user_vec)[0]
                possible_responses = df[df["intent"] == intent]["response"].tolist()
                bot_reply = random.choice(possible_responses) if possible_responses else "I'm still learning, can you try rephrasing?"

                conn = get_conn()
                cur = conn.cursor()
                cur.execute("""INSERT INTO chat_history 
                            (user_id, workspace_id, user_message, bot_response, intent, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (uid, workspace_id, user_input, bot_reply, intent, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()

                st.markdown(f"**🧑 You:** {user_input}**")
                st.markdown(f"**🤖 Bot:** {bot_reply}**")

        # Sidebar - Recent Chats
        with col2:
            st.markdown("### 🕑 Recent Chats")
            conn = get_conn()
            chats = pd.read_sql("""
                SELECT user_message, bot_response, intent, created_at
                FROM chat_history
                WHERE user_id=? AND workspace_id=?
                ORDER BY id DESC LIMIT 10
            """, conn, params=(uid, workspace_id))
            conn.close()

            if not chats.empty:
                for _, row in chats.iterrows():
                    st.markdown(f"🕒 **{row['created_at']}**")
                    st.markdown(f"🧑 **You:** {row['user_message']}**")
                    st.markdown(f"🤖 **Bot:** {row['bot_response']}  (_{row['intent']}_)**")
                    st.markdown("---")
            else:
                st.info("No chats yet. Start chatting!")

