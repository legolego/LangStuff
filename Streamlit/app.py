"""Skeleton: ask an Oracle (FreeSQL) database questions in English.

Every element is a placeholder. Nothing is connected yet; wire in the
oracledb connection and the LangGraph agent where the TODOs are.
"""
import importlib
import os
from pathlib import Path

import oracledb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import agent
from history_nav import history_nav_html

# Streamlit does not always re-import an edited helper module while the app keeps
# running, so reload agent.py on every run; edits to it take effect without a restart.
agent = importlib.reload(agent)
from agent import MODEL_NAME, ROW_LIMIT, build_graph  # noqa: E402

st.set_page_config(
    page_title="Ask the database",
    page_icon=str(Path(__file__).resolve().parent / "assets" / "favicon.svg"),  # the graph icon
    layout="wide",
)

# FreeSQL sample schemas: owner code -> display name
SCHEMAS = {
    "OLYM": "Olympic data",
    "CO": "Customer orders",
    "HR": "Human resources",
}

# Three example questions per schema; the buttons on the page ask them.
SAMPLE_QUESTIONS = {
    "OLYM": [
        "How many gold medals did the USA win in 1984?",
        "Which athletes have won the most gold medals?",
        "Which cities have hosted the Olympics, and in which years?",
    ],
    "CO": [
        "What are the five best-selling products by revenue?",
        "How many orders are there in each status?",
        "Which customers have placed the most orders?",
    ],
    "HR": [
        "Who are the five highest paid employees?",
        "What is the average salary in each department?",
        "Which employees earn more than their manager?",
    ],
}

# Local dev: .env sits in the parent folder. On Streamlit Cloud: st.secrets.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()
    try:
        return st.secrets[name]
    except Exception:
        return None


@st.cache_resource(show_spinner="Connecting to FreeSQL...")
def get_connection() -> oracledb.Connection:
    return oracledb.connect(
        user=_setting("FREE_SQL_USER"),
        password=_setting("FREE_SQL_PASSWORD"),
        dsn=_setting("FREE_SQL_DSN"),
    )


RECONNECT_CODES = {"DPY-4011", "DPY-1001"}  # connection closed / not connected


def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run SQL and return a DataFrame; reconnect once if the cached connection died."""
    for attempt in (1, 2):
        try:
            with get_connection().cursor() as cur:
                cur.execute(sql, params or {})
                cols = [c[0] for c in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=cols)
        except oracledb.Error as e:
            code = getattr(e.args[0], "full_code", "") if e.args else ""
            if attempt == 2 or code not in RECONNECT_CODES:
                raise  # real SQL errors (ORA-...) still surface normally
            get_connection.clear()  # drop the dead connection, loop and reconnect


@st.cache_data(ttl=600)
def list_tables(schema: str) -> list[str]:
    df = run_query(
        "SELECT table_name FROM all_tables WHERE owner = :o ORDER BY table_name",
        {"o": schema},
    )
    return df["TABLE_NAME"].tolist()

AGENT_FILE = Path(__file__).with_name("agent.py")


@st.cache_resource
def get_graph(agent_mtime: float):
    # agent_mtime is only a cache key: editing agent.py rebuilds the graph.
    # On Streamlit Cloud the key comes from st.secrets, not .env
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            pass
    return build_graph(run_query)

# ------------------------------------------------------- one answer, drawn as tabs
def summarise_step(node: str, update: dict) -> dict:
    """The part of one agent step worth keeping for the Agent trace tab.

    Kept small on purpose: the result table is stored once per answer, not once per step.
    """
    step = {"node": node}
    if node == "write_sql":
        step.update(sql=update["sql"], attempts=update["attempts"])
    elif node == "run_sql":
        step.update(
            error=update.get("error", ""),
            no_rows=update.get("no_rows", False),
            rows=len(update["df"]) if "df" in update else None,
        )
    elif node == "lookup_values":
        step.update(hints=update.get("hints", ""))
    return step


def render_turn(turn: dict) -> None:
    """Draw one answer as four tabs: answer, SQL, results and agent trace.

    Used for the answer being worked out now and for every earlier answer in the
    conversation, so they all look the same. A turn is a dict with the keys
    answer, sql, df, error and steps; missing keys are tolerated.
    """
    tab_answer, tab_sql, tab_data, tab_trace = st.tabs(
        ["💬 Answer", "🧾 SQL", "📊 Results", "🕸️ Agent trace"]
    )
    with tab_answer:
        st.write(turn["answer"])
    with tab_sql:
        st.code(turn.get("sql", ""), language="sql")
    with tab_data:
        if turn.get("error"):
            st.error(turn["error"])
        elif turn.get("df") is not None:
            st.dataframe(turn["df"], width="stretch")
        else:
            st.caption("No result table was kept for this answer.")
    with tab_trace:
        steps = turn.get("steps", [])
        st.write(" → ".join(s["node"] for s in steps) or "No trace was kept for this answer.")
        for s in steps:
            if s["node"] == "write_sql":
                st.caption(f"write_sql, attempt {s['attempts']}")
                st.code(s["sql"], language="sql")
            elif s["node"] == "run_sql":
                if s["error"]:
                    st.error(s["error"])
                elif s["no_rows"]:
                    st.warning("The query ran but found nothing (0 rows, or a count of 0).")
                else:
                    st.caption(f"run_sql: {s['rows']} rows")
            elif s["node"] == "lookup_values" and s["hints"]:
                st.caption("lookup_values: real values from the database")
                st.code(s["hints"], language="text")


# ---------------------------------------------------------------- session state
if "histories" not in st.session_state:
    # schema code -> list of turns: {"question", "answer", "sql", "df", "error", "steps"}
    st.session_state.histories = {}

# ------------------------------------------------------------------ banner
st.image(str(Path(__file__).resolve().parent / "assets" / "header.svg"), width="stretch")

# ------------------------------------------------------------------ page header
# The schema dropdown sits in the heading: "Ask your database about [Olympic data]".
# It is created before the sidebar so the sidebar's table list follows the selection.
with st.container(horizontal=True, vertical_alignment="center"):
    st.title(":material/database: Ask your database about", width="content")
    schema = st.selectbox(
        "Schema",
        options=list(SCHEMAS),
        format_func=lambda code: SCHEMAS[code],
        label_visibility="collapsed",
        width=260,
    )

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")

    st.subheader("Database")
    if st.button("Test connection"):
        try:
            get_connection.clear()
            run_query("SELECT 1 FROM dual")
            st.success("Connected to FreeSQL")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.subheader("Model")
    st.markdown(f":material/smart_toy: `{MODEL_NAME}`")

    st.subheader("Safety")
    st.toggle("Read-only (SELECT only)", value=True, disabled=True)
    st.slider("Max rows returned", 10, 1000, ROW_LIMIT, disabled=True)

    st.subheader("Tables")
    with st.expander(f"{SCHEMAS[schema]} ({schema})", expanded=True):
        try:
            for t in list_tables(schema):
                st.write(f"• {t}")
        except Exception as e:
            st.error(f"Could not list tables: {e}")

# ------------------------------------------------------------------- main page
st.caption("Type a question in plain English. The app writes SQL, runs it, and explains the result.")

# This schema's conversation; switching the dropdown switches the conversation.
history = st.session_state.histories.setdefault(schema, [])

# Example questions for the selected schema; clicking one asks it.
clicked = None
st.write("**Try:**")
with st.container(horizontal=True):  # chips sized to their text; they wrap on narrow screens
    for i, example in enumerate(SAMPLE_QUESTIONS[schema]):
        if st.button(example, key=f"example_{schema}_{i}", width="content"):
            clicked = example

st.divider()

# Past turns
for turn in history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_turn(turn)  # the same four tabs as when the answer first appeared

# Input: the chat box with the clear button to its right, pinned to the bottom of the page.
with st.bottom:
    with st.container(horizontal=True, vertical_alignment="center"):
        question = st.chat_input(f"Ask a question about {SCHEMAS[schema].lower()}...", width="stretch") or clicked
        clear = st.button(
            "Clear",
            icon=":material/delete_sweep:",
            help="Clear this conversation",
            # the new turn is only added to history at the end of the run, so a question
            # being asked right now also counts as something to clear
            disabled=not (history or question),
            width="content",
        )
if clear:
    st.session_state.histories[schema] = []  # only this schema's conversation
    st.rerun()

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        final, steps = {}, []
        with st.status("Thinking...", expanded=True) as status:
            for step in get_graph(AGENT_FILE.stat().st_mtime).stream({"question": question, "schema": schema}, stream_mode="updates"):
                for node, update in step.items():
                    steps.append(summarise_step(node, update))  # every step, for the trace tab
                    st.write(f"✅ {node}")
                    final.update(update)  # merge each node's changes into one dict
            status.update(label="Done", state="complete")

        # Everything the four tabs need, kept with the conversation so earlier answers
        # keep their tabs after later questions (and after switching schema and back).
        turn = {
            "question": question,
            "answer": final["answer"],
            "sql": final.get("sql", ""),
            "df": final.get("df"),
            "error": final.get("error", ""),
            "steps": steps,
        }
        render_turn(turn)

    history.append(turn)

# Up/down arrows in the chat box recall earlier questions (this schema's only).
# Rendered last so the list includes the question that was just asked.
st.html(history_nav_html([t["question"] for t in history]), unsafe_allow_javascript=True)
