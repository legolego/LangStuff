"""LangGraph agent: English question -> SQL -> result -> answer."""
from typing import Callable, TypedDict

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_RETRIES = 2
ROW_LIMIT = 100


# 1. STATE: the notebook every node reads from and writes to.
class State(TypedDict, total=False):
    question: str
    schema: str  # database schema (owner) to query, e.g. "OLYM"
    schema_text: str
    sql: str
    error: str
    attempts: int
    df: pd.DataFrame
    answer: str


def build_graph(run_query: Callable[..., pd.DataFrame]):
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    # 2. NODES: plain functions. Take the state, return ONLY the keys you change.
    def get_schema(state: State) -> dict:
        df = run_query(
            "SELECT table_name, column_name, data_type FROM all_tab_columns "
            "WHERE owner = :o ORDER BY table_name, column_id",
            {"o": state["schema"]},
        )
        lines = {}
        for t, c, d in df.itertuples(index=False):
            lines.setdefault(t, []).append(f"{c} {d}")
        text = "\n".join(f"{state['schema']}.{t}({', '.join(cols)})" for t, cols in lines.items())
        return {"schema_text": text, "attempts": 0}

    def write_sql(state: State) -> dict:
        prompt = (
            "You write Oracle SQL. Use ONLY these tables and columns:\n"
            f"{state['schema_text']}\n\n"
            "Rules: one SELECT statement, no semicolon, qualify tables with the "
            f"{state['schema']}. prefix, no explanation, no markdown.\n"
            f"Question: {state['question']}\n"
        )
        if state.get("error"):  # we are retrying after a failure
            prompt += (
                f"\nYour previous SQL failed.\nSQL: {state['sql']}\n"
                f"Error: {state['error']}\nFix it.\n"
            )
        sql = llm.invoke(prompt).text.strip()
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip().rstrip(";")
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1}

    def run_sql(state: State) -> dict:
        sql = state["sql"]
        if not sql.lstrip().lower().startswith(("select", "with")):
            return {"error": "Only SELECT queries are allowed."}
        try:
            df = run_query(f"SELECT * FROM ({sql}) FETCH FIRST {ROW_LIMIT} ROWS ONLY")
            return {"df": df, "error": ""}
        except Exception as e:
            return {"error": str(e)}

    def answer(state: State) -> dict:
        if state.get("error"):
            return {"answer": f"Sorry, I couldn't get a result. Last error: {state['error']}"}
        prompt = (
            f"Question: {state['question']}\nSQL used: {state['sql']}\n"
            f"Result (first rows):\n{state['df'].head(20).to_string()}\n\n"
            "Answer the question in 1-3 plain sentences using the result."
        )
        return {"answer": llm.invoke(prompt).text.strip()}

    # 3. CONDITIONAL EDGE: a function that returns the NAME of the next node.
    def after_run(state: State) -> str:
        if state.get("error") and state["attempts"] <= MAX_RETRIES:
            return "write_sql"  # loop back and try again
        return "answer"

    # 4. WIRING: nodes + edges, then compile.
    g = StateGraph(State)
    g.add_node("get_schema", get_schema)
    g.add_node("write_sql", write_sql)
    g.add_node("run_sql", run_sql)
    g.add_node("answer", answer)

    g.add_edge(START, "get_schema")
    g.add_edge("get_schema", "write_sql")
    g.add_edge("write_sql", "run_sql")
    g.add_conditional_edges("run_sql", after_run, ["write_sql", "answer"])
    g.add_edge("answer", END)
    return g.compile()