"""LangGraph agent: English question -> SQL -> result -> answer."""
import numbers
import re
from typing import Callable, TypedDict

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_RETRIES = 3  # a failed query and an empty result each use one retry
ROW_LIMIT = 200
VALUE_HINT_LIMIT = 400  # max distinct values shown per column when a query finds nothing
CATEGORY_LIMIT = 30  # other text columns are listed in full only if they have this few values
TEXT_TYPES = ("VARCHAR", "CHAR", "NVARCHAR", "NCHAR")
MAX_OUTPUT_TOKENS = 4096  # cap on the SQL-writing call, so a repetition loop cannot run away

# What the model cannot work out from table and column names alone, by schema.
SCHEMA_NOTES = {
    "OLYM": (
        "Each row in the medals tables and view is ONE ATHLETE'S medal, so a team event "
        "(a relay, rowing eight, basketball, water polo) has one row per team member for a "
        "single medal. When asked how many medals a country or person won, count one per "
        "EVENT, not one per athlete row: count DISTINCT events (include sport, discipline or "
        "event gender columns if the event name alone is ambiguous, e.g. COUNT(DISTINCT ...) "
        "or SELECT DISTINCT in a subquery). Only when asked how many ATHLETES won something, "
        "count distinct athletes. The data contains only Summer Games, so a question about a "
        "Winter Games has no answer."
    ),
}
MAX_SQL_CHARS = 4000  # longer than this is not a real query
PROMPT_SQL_CHARS = 1500  # how much of a failed SQL is shown back to the model
ANSWER_CHARS = 12000  # how much of the result the answer step sees (about 3,000 tokens)


def check_sql(sql: str, schema_df: pd.DataFrame, owner: str) -> str:
    """Return a description of what is wrong with the SQL, or '' if it looks sane.

    Catches runaway or hallucinated output before it reaches Oracle, and gives the
    model something specific to fix.
    """
    if len(sql) > MAX_SQL_CHARS:
        return (
            f"The SQL is {len(sql):,} characters long, which is far too long. It looks like "
            "repeated or runaway text. Write a short, simple query."
        )
    tables = {str(t).upper() for t in schema_df.iloc[:, 0]}
    used = re.findall(rf'\b{re.escape(owner)}\.\"?([A-Za-z0-9_$#]+)\"?', sql, re.I)
    unknown = sorted({u for u in used if u.upper() not in tables})
    if unknown:
        return (
            f"Unknown table(s): {', '.join(unknown)[:300]}. "
            f"The only tables and views in {owner} are: {', '.join(sorted(tables))}."
        )
    return ""


def clip(text: str, limit: int = PROMPT_SQL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " ...(cut off)"


def result_for_prompt(df: pd.DataFrame, max_chars: int = ANSWER_CHARS) -> tuple[str, int]:
    """The result as compact CSV that fits in the prompt, plus how many rows made it in."""
    shown = len(df)
    text = df.head(shown).to_csv(index=False)
    while len(text) > max_chars and shown > 1:
        shown = max(1, shown // 2)
        text = df.head(shown).to_csv(index=False)
    return text, shown


def looks_empty(df: pd.DataFrame) -> bool:
    """True if the query found nothing: no rows, or one aggregate row that is all 0/NULL.

    SELECT COUNT(*) never returns zero rows; it returns a single row holding 0, so a
    bad filter value shows up as that, not as an empty result.
    """
    if df.empty:
        return True
    if len(df) == 1:
        return all(
            pd.isna(v) or (isinstance(v, numbers.Number) and v == 0)
            for v in df.iloc[0].tolist()
        )
    return False


# 1. STATE: the notebook every node reads from and writes to.
class State(TypedDict, total=False):
    question: str
    schema: str  # database schema (owner) to query, e.g. "OLYM"
    schema_df: pd.DataFrame  # table_name, column_name, data_type for the schema
    schema_text: str
    sql: str
    error: str
    no_rows: bool  # the query ran fine but returned 0 rows
    hints: str  # real column values, looked up after a 0-row result
    short_hints: str  # the compact part of hints (columns with few values), for the answer
    attempts: int
    df: pd.DataFrame
    answer: str


def build_graph(run_query: Callable[..., pd.DataFrame]):
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    sql_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, max_output_tokens=MAX_OUTPUT_TOKENS)

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
        return {"schema_df": df, "schema_text": text, "attempts": 0, "error": "", "no_rows": False,
                "hints": "", "short_hints": ""}

    def write_sql(state: State) -> dict:
        prompt = (
            "You write Oracle SQL. Use ONLY these tables and columns:\n"
            f"{state['schema_text']}\n\n"
            "Rules: one SELECT statement, no semicolon, qualify tables with the "
            f"{state['schema']}. prefix, no explanation, no markdown. "
            "You cannot see the data, so a text value may be spelled differently from the "
            "question (codes vs names, abbreviations, capitalization); when unsure of the "
            "exact value, compare with UPPER(column) LIKE '%VALUE%'.\n"
            + (f"Notes about this data: {SCHEMA_NOTES[state['schema']]}\n" if state["schema"] in SCHEMA_NOTES else "")
            + f"Question: {state['question']}\n"
        )
        if state.get("error"):  # we are retrying after a failure
            prompt += (
                f"\nYour previous SQL failed.\nSQL: {clip(state['sql'])}\n"
                f"Error: {state['error']}\nFix it.\n"
            )
        elif state.get("no_rows"):  # ran fine, but found nothing
            prompt += (
                f"\nYour previous SQL ran but returned 0 rows, which is probably wrong.\n"
                f"SQL: {clip(state['sql'])}\n"
                "A filter value was probably spelled or formatted differently from the data "
                "(for example a country code vs a country name, or 'Gold' vs 'G').\n"
            )
            if state.get("hints"):
                prompt += (
                    "Real values from the database:\n"
                    f"{state['hints']}\n"
                    "Use these exact values. The concept you filtered on (a season, a country, "
                    "a medal type) may live in a different column or table than the one you "
                    "used, so check the other columns listed above, and join if needed, "
                    "before reusing the same filter.\n"
                )
            prompt += "Fix the query. If 0 rows really is the right answer, return the same query.\n"
        sql = sql_llm.invoke(prompt).text.strip()
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip().rstrip(";")
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1}

    def run_sql(state: State) -> dict:
        sql = state["sql"]
        if not sql.lstrip().lower().startswith(("select", "with")):
            return {"error": "Only SELECT queries are allowed.", "no_rows": False}
        problem = check_sql(sql, state["schema_df"], state["schema"])
        if problem:  # never send runaway or made-up SQL to the database
            return {"error": problem, "no_rows": False}
        try:
            df = run_query(f"SELECT * FROM ({sql}) FETCH FIRST {ROW_LIMIT} ROWS ONLY")
            return {"df": df, "error": "", "no_rows": looks_empty(df)}
        except Exception as e:
            return {"error": str(e), "no_rows": False}

    def distinct_values(owner: str, table: str, col: str, limit: int) -> list:
        # identifiers come from the database catalog, not from the model
        return run_query(
            f'SELECT DISTINCT "{col}" FROM {owner}."{table}" '
            f'WHERE "{col}" IS NOT NULL ORDER BY 1 FETCH FIRST {limit} ROWS ONLY'
        ).iloc[:, 0].tolist()

    def lookup_values(state: State) -> dict:
        """After a query finds nothing, show the model the real values it was guessing at.

        1. Text columns the query compared to a literal: all their values (up to a cap).
        2. Every other text column in the schema that has only a few values (like a
           season or medal-type column). The concept being filtered on may live in a
           different table than the one the model picked.
        """
        sql, owner = state["sql"], state["schema"]
        filtered, others, short, seen = [], [], [], set()
        all_text_cols = [
            (t, c) for t, c, d in state["schema_df"].itertuples(index=False)
            if str(d).startswith(TEXT_TYPES)
        ]
        text_cols = [
            (t, c) for t, c in all_text_cols if re.search(rf"\b{re.escape(t)}\b", sql, re.I)
        ]
        for table, col in text_cols:  # 1. columns the query filtered on
            if not re.search(rf"\b{re.escape(col)}\b\W*(?:=|!=|<>|LIKE|IN)\s*\(?\s*'", sql, re.I):
                continue
            seen.add((table, col))
            try:
                vals = distinct_values(owner, table, col, VALUE_HINT_LIMIT)
            except Exception:
                continue
            more = " ... (more values exist)" if len(vals) >= VALUE_HINT_LIMIT else ""
            line = f"{owner}.{table}.{col}: " + ", ".join(f"'{v}'" for v in vals) + more
            filtered.append(line)
            if len(vals) <= CATEGORY_LIMIT:
                short.append(line)
        for table, col in all_text_cols:  # 2. other low-variety columns, any table in the schema
            if (table, col) in seen:
                continue
            try:
                vals = distinct_values(owner, table, col, CATEGORY_LIMIT + 1)
            except Exception:
                continue
            if len(vals) <= CATEGORY_LIMIT:
                line = f"{owner}.{table}.{col}: " + ", ".join(f"'{v}'" for v in vals)
                others.append(line)
                short.append(line)
        parts = []
        if filtered:
            parts.append("Columns you filtered on, with all their values:\n" + "\n".join(filtered))
        if others:
            parts.append("Other columns in the schema with few values:\n" + "\n".join(others))
        return {"hints": "\n\n".join(parts), "short_hints": "\n".join(short)}

    def answer(state: State) -> dict:
        if state.get("error"):
            return {"answer": f"Sorry, I couldn't get a result. Last error: {state['error']}"}
        df = state["df"]
        csv_text, shown = result_for_prompt(df)
        prompt = (
            f"Question: {state['question']}\nSQL used: {state['sql']}\n"
            f"The query returned {len(df)} rows; the first {shown} are shown as CSV:\n{csv_text}\n"
        )
        if shown < len(df):
            prompt += (
                f"Only the first {shown} of {len(df)} rows are shown. Do NOT say anything about "
                "the rows not shown (for example the last, largest or total values); say the "
                "list is longer instead.\n"
            )
        if len(df) >= ROW_LIMIT:
            prompt += f"The result was capped at {ROW_LIMIT} rows, so there may be more.\n"
        prompt += (
            "\nAnswer the question in 1-3 plain sentences using the result. "
            "Say exactly what was counted (for example medal events, not individual athletes)."
        )
        if state.get("no_rows"):  # still empty/zero after the retries
            prompt += (
                "\nThe result is empty or zero even after retrying. Say what the query looked "
                "for (its filters) and that it found no matching rows. Do not present this as "
                "proof that something never happened: the cause may be a filter that does not "
                "match how the data is stored, or a real-world reason. If you are confident of "
                "a real-world reason (for example the country boycotted or did not take part "
                "that year, or the country did not exist then), mention it as a possibility, "
                "clearly labelled as background knowledge and not as something from the database."
            )
            if state.get("short_hints"):
                prompt += (
                    "\nThese are all the values that actually exist in the database for columns "
                    f"with few values:\n{state['short_hints']}\n"
                    "If the question asks for a value that is not among them (for example a "
                    "season), say plainly that the database does not contain it, and say what "
                    "it does contain."
                )
        return {"answer": llm.invoke(prompt).text.strip()}

    # 3. CONDITIONAL EDGE: a function that returns the NAME of the next node.
    def after_run(state: State) -> str:
        can_retry = state["attempts"] <= MAX_RETRIES
        if state.get("error") and can_retry:
            return "write_sql"  # SQL failed: loop back and try again
        if state.get("no_rows") and can_retry:
            return "lookup_values"  # SQL ran but found nothing: check the real values first
        return "answer"

    # 4. WIRING: nodes + edges, then compile.
    g = StateGraph(State)
    g.add_node("get_schema", get_schema)
    g.add_node("write_sql", write_sql)
    g.add_node("run_sql", run_sql)
    g.add_node("lookup_values", lookup_values)
    g.add_node("answer", answer)

    g.add_edge(START, "get_schema")
    g.add_edge("get_schema", "write_sql")
    g.add_edge("write_sql", "run_sql")
    g.add_conditional_edges("run_sql", after_run, ["write_sql", "lookup_values", "answer"])
    g.add_edge("lookup_values", "write_sql")
    g.add_edge("answer", END)
    return g.compile()