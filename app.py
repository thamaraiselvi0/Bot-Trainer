import streamlit as st
import sqlite3
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import ast

# ============================
# DATABASE INIT
# ============================

conn = sqlite3.connect("nlu_app.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS workspaces(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS annotations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ws_id INTEGER,
    text TEXT,
    intent TEXT,
    entities TEXT
)
""")

conn.commit()

# ============================
# LOAD SPACY MODEL
# ============================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None


# ============================
# HELPERS
# ============================

def register_user(username, password):
    try:
        cur.execute("INSERT INTO users(username, password) VALUES (?,?)", (username, password))
        conn.commit()
        return True
    except:
        return False


def login_user(username, password):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cur.fetchone()


def create_workspace(user_id, name):
    cur.execute("INSERT INTO workspaces(user_id, name) VALUES (?,?)", (user_id, name))
    conn.commit()


def save_annotation(ws_id, text, intent, entities):
    cur.execute("INSERT INTO annotations(ws_id, text, intent, entities) VALUES (?,?,?,?)",
                (ws_id, text, intent, entities))
    conn.commit()


def get_workspace_data(ws_id):
    return pd.read_sql(f"SELECT text, intent, entities FROM annotations WHERE ws_id={ws_id}", conn)


def spacy_extract(text):
    if nlp is None:
        return "[]"
    doc = nlp(text)
    ents = {ent.text: ent.label_ for ent in doc.ents}
    return str(ents)


def nltk_dummy_extract(text):
    words = text.split()
    return str({w: "WORD" for w in words[:2]})


def rasa_dummy_extract(text):
    return str({"dummy": "rasa"})


def train_intent_model(df):
    if len(df) < 2:
        return None, None, 0, "Not enough data"

    X = df["text"]
    y = df["intent"]

    vectorizer = TfidfVectorizer()
    Xv = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(Xv, y)

    pred = model.predict(Xv)
    acc = accuracy_score(y, pred)

    return model, vectorizer, acc, classification_report(y, pred)


# ============================
# STREAMLIT UI
# ============================

st.title("Simple NLU Trainer")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# -------------------------------
# REGISTER
# -------------------------------

if choice == "Register":
    st.subheader("Create Account")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Register"):
        if register_user(u, p):
            st.success("Registered! Go to login.")
        else:
            st.error("Username already exists!")

# -------------------------------
# LOGIN + MAIN APP
# -------------------------------

if choice == "Login":
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(u, p)
        if user:
            st.success(f"Welcome {u}")
            user_id = user[0]

            # ============================
            # WORKSPACE SELECTION
            # ============================

            st.header("Workspaces")
            ws_df = pd.read_sql(f"SELECT * FROM workspaces WHERE user_id={user_id}", conn)

            ws_option = st.selectbox("Select Workspace", ["-- Select --"] + list(ws_df["name"]))

            new_ws = st.text_input("Create new workspace")

            if st.button("Create Workspace"):
                if new_ws:
                    create_workspace(user_id, new_ws)
                    st.success("Workspace created. Reload app.")
                else:
                    st.error("Enter a workspace name.")

            if ws_option != "-- Select --":
                ws_id = int(ws_df[ws_df["name"] == ws_option]["id"].iloc[0])
                st.success(f"Workspace: {ws_option}")

                # ============================
                # MODEL SELECTION
                # ============================

                model_choice = st.radio("Choose Pretrained Model", 
                                        ["SpaCy", "NLTK", "Rasa-like"])

                # ============================
                # UPLOAD DATASET
                # ============================

                st.subheader("Upload CSV (text,intent,entities)")

                uploaded = st.file_uploader("Upload CSV", type="csv")

                if uploaded:
                    df = pd.read_csv(uploaded)

                    for _, row in df.iterrows():
                        save_annotation(ws_id, row["text"], row["intent"], row["entities"])

                    st.success("Dataset uploaded & saved!")

                # ============================
                # ADD MANUAL TEXT SAMPLE
                # ============================

                st.subheader("Add sample text")

                txt = st.text_area("Enter text")

                if st.button("Auto Annotate"):
                    if model_choice == "SpaCy":
                        ents = spacy_extract(txt)
                    elif model_choice == "NLTK":
                        ents = nltk_dummy_extract(txt)
                    else:
                        ents = rasa_dummy_extract(txt)

                    st.json(ents)

                intent = st.text_input("Intent")
                ents_input = st.text_input("Entities (Python dict)")

                if st.button("Save Annotation"):
                    save_annotation(ws_id, txt, intent, ents_input)
                    st.success("Saved!")

                # ============================
                # TRAIN & EVALUATE
                # ============================

                st.header("Train & Evaluate")

                ws_data = get_workspace_data(ws_id)

                if len(ws_data) > 1:
                    model, vec, acc, report = train_intent_model(ws_data)

                    st.write("Accuracy:", acc)
                    st.text(report)
                else:
                    st.warning("Need at least 2 samples to train.")

        else:
            st.error("Invalid login")



