"""LangGraph agent: English question -> SQL -> result -> answer.

HOW IT WORKS
============
The agent is a small LangGraph "graph". Each box below is a NODE (a plain Python
function). Nodes do not call each other; they read and write a shared dictionary
called the STATE (see the State class), and the graph decides which node runs next.

        START
          |
          v
   +--------------+
   |  get_schema  |  read the table and column names of the chosen schema
   +--------------+
          |
          v
   +--------------+
   |  write_sql   |  ask the model for ONE SELECT statement
   +--------------+
          |
          v
   +--------------+
   |   run_sql    |  check the SQL, then run it against Oracle
   +--------------+
          |
          |  after_run() picks the next node:
          |
          +--- the SQL failed, retries left ---------> write_sql   (try again)
          |
          +--- it ran but found nothing, retries left
          |           |
          |           v
          |    +---------------+
          |    | lookup_values |  fetch the real values of the columns it filtered on
          |    +---------------+
          |           |
          |           +------------------------------> write_sql   (try again, with hints)
          |
          +--- a good result, or no retries left
                      |
                      v
               +--------------+
               |    answer    |  turn the result into 1-3 plain sentences
               +--------------+
                      |
                      v
                     END

EDGES (the arrows)
------------------
  START          -> get_schema      always
  get_schema     -> write_sql       always
  write_sql      -> run_sql         always
  run_sql        -> write_sql       if the SQL failed and there are retries left
  run_sql        -> lookup_values   if it found nothing and there are retries left
  run_sql        -> answer          otherwise (a good result, or out of retries)
  lookup_values  -> write_sql       always
  answer         -> END             always

WHO READS AND WRITES WHAT (the State)
-------------------------------------
  node           writes                                          reads
  get_schema     schema_df, schema_text, attempts, error,        schema
                 no_rows, hints, short_hints   (resets all of them for a new question)
  write_sql      sql, attempts                                   schema, schema_text, question,
                                                                 attempts, and, when retrying:
                                                                 sql, error, no_rows, hints
  run_sql        df, error, no_rows                              sql, schema, schema_df
  lookup_values  hints, short_hints                              sql, schema, schema_df
  answer         answer                                          question, sql, df, error, no_rows,
                                                                 and short_hints (only when the
                                                                 result is still empty)

WHY THERE ARE TWO KINDS OF RETRY
--------------------------------
  * The SQL failed (a mistake, made-up table, runaway text): the model is shown the
    error and fixes it.
  * The SQL ran but found nothing: usually the model guessed a value wrong (for
    example 'USA' vs 'United States', or 'Winter' in a database with only Summer
    Games). lookup_values shows it the real values, and it tries again.
  Both kinds share one budget (MAX_RETRIES), so the agent always stops.

HOW THE APP USES IT
-------------------
  graph = build_graph(run_query)
  for step in graph.stream({"question": ..., "schema": "OLYM"}, stream_mode="updates"):
      ...   # each step is {node_name: the keys that node changed}

The file is organised top to bottom as: settings, helper functions (no model and no
database involved), the State, then build_graph() which defines the nodes and wires
them together.
"""
import numbers
import re
from typing import Callable, TypedDict

import pandas as pd
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# =============================================================================
# SETTINGS
# =============================================================================

# The Gemini model used both to write the SQL and to write the final answer.
# The app shows this name in the sidebar.
MODEL_NAME = "gemini-3.5-flash-lite"

# Models to switch to, in order, when the one in use runs out of its requests per DAY
# (500 a day for MODEL_NAME on the free tier). Google answers that with a "quota exceeded,
# per day" error; see is_daily_limit_error. The switch lasts until the agent is rebuilt,
# which the app does when it restarts. Use [] to turn the fallback off.
FALLBACK_MODELS = ["gemini-3.1-flash-lite"]

# The peak number of requests per minute your API key allows for each model. The agent
# spaces its model calls so it never goes over the limit of MODEL_NAME (see make_rate_limiter).
# To switch models, change MODEL_NAME and make sure the new model has a line here; the
# limit for your key is on Google's rate-limits page for the Gemini API.
MODEL_REQUESTS_PER_MINUTE = {
    "gemini-3.5-flash-lite": 15,
    "gemini-3.1-flash-lite": 15,  # check this against Google's rate-limits page for your key
}

# How many times the agent may go back to write_sql after a bad result (an SQL error,
# or a result that found nothing). 3 retries means up to 4 attempts in total.
MAX_RETRIES = 3  # a failed query and an empty result each use one retry

# The most rows fetched from Oracle for any one query. The app shows this too.
ROW_LIMIT = 200

# When a query finds nothing, lookup_values lists the real values of the text columns
# it filtered on, up to this many per column.
VALUE_HINT_LIMIT = 400  # max distinct values shown per column when a query finds nothing

# lookup_values also lists OTHER text columns, but only ones with this few distinct
# values (for example a season or medal-type column), because those are the ones a
# filter might really belong to.
CATEGORY_LIMIT = 30  # other text columns are listed in full only if they have this few values

# Oracle column types that hold text (the only columns we look up values for).
TEXT_TYPES = ("VARCHAR", "CHAR", "NVARCHAR", "NCHAR")

# Upper limit on the length of what the model writes when it writes SQL. Without it, a
# small model can get stuck repeating one word for tens of thousands of characters.
MAX_OUTPUT_TOKENS = 4096  # cap on the SQL-writing call, so a repetition loop cannot run away

# Extra guidance that is added to the SQL-writing prompt for one schema. These are
# facts about the DATA that the model cannot work out from table and column names.
# To teach the agent a quirk of another schema, add an entry keyed by its owner code
# (for example "HR" or "CO").
SCHEMA_NOTES = {
    "OLYM": (
        "Each row in the medals tables and view is ONE ATHLETE'S medal, so a team event "
        "(a relay, rowing eight, basketball, water polo) has one row per team member for a "
        "single medal. When asked how many medals a country or person won, count one per "
        "medal, not one per athlete row: count DISTINCT medals, where one medal is identified "
        "by the Games (the edition/year column or the games id), the sport or discipline, the "
        "event and the event gender TOGETHER, e.g. COUNT(DISTINCT edition || '|' || discipline "
        "|| '|' || event || '|' || event_gender) using the columns that exist, or SELECT "
        "DISTINCT in a subquery. ALWAYS include the Games year or id in that distinct key: the "
        "same event is held at every Games, so leaving it out merges the same event won at "
        "different Games into one medal and undercounts anyone who won it more than once. "
        "Only when asked how many ATHLETES won something, count distinct athletes. The data "
        "contains only Summer Games, so a question about a Winter Games has no answer."
    ),
    "HR": (
        "Employee names are split into FIRST_NAME and LAST_NAME; to match a full name use "
        "UPPER(FIRST_NAME || ' ' || LAST_NAME) LIKE '%...%'. SALARY is a MONTHLY amount in this "
        "data, so say 'per month' when quoting it, and multiply by 12 only if asked for annual "
        "pay. MANAGER_ID points at another row of EMPLOYEES (use a self-join to reach a "
        "manager). DEPARTMENT_ID and COMMISSION_PCT can be NULL, so use a LEFT JOIN when "
        "employees without a department matter."
    ),
    "CO": (
        "ORDERS.ORDER_STATUS is COMPLETE, CANCELLED or REFUNDED. Unless the question asks "
        "about cancelled or refunded orders, count only COMPLETE orders for sales, revenue and "
        "order counts. An order's revenue is the sum of ORDER_ITEMS.UNIT_PRICE * "
        "ORDER_ITEMS.QUANTITY (the price paid at the time), not PRODUCTS.UNIT_PRICE. Customer "
        "names are in CUSTOMERS.FULL_NAME. This is sample data from a fixed period, so do not "
        "use SYSDATE for 'recent' or 'last year'."
    ),
}

# Size limits that keep prompts and queries sane.
MAX_SQL_CHARS = 4000  # longer than this is not a real query
PROMPT_SQL_CHARS = 1500  # how much of a failed SQL is shown back to the model
ANSWER_CHARS = 12000  # how much of the result the answer step sees (about 3,000 tokens)


# =============================================================================
# HELPER FUNCTIONS
# Plain functions with no model and no database, so they are easy to test.
# =============================================================================

def check_sql(sql: str, schema_df: pd.DataFrame, owner: str) -> str:
    """Return a description of what is wrong with the SQL, or '' if it looks sane.

    Catches runaway or hallucinated output before it reaches Oracle, and gives the
    model something specific to fix.
    """
    # 1. Runaway text. A model stuck in a loop can produce tens of thousands of characters.
    if len(sql) > MAX_SQL_CHARS:
        return (
            f"The SQL is {len(sql):,} characters long, which is far too long. It looks like "
            "repeated or runaway text. Write a short, simple query."
        )
    # 2. Made-up tables. Every "OWNER.NAME" in the SQL must be a real table or view.
    #    schema_df's first column is the table name (it has one row per column).
    tables = {str(t).upper() for t in schema_df.iloc[:, 0]}
    used = re.findall(rf'\b{re.escape(owner)}\.\"?([A-Za-z0-9_$#]+)\"?', sql, re.I)
    unknown = sorted({u for u in used if u.upper() not in tables})
    if unknown:
        return (
            f"Unknown table(s): {', '.join(unknown)[:300]}. "
            f"The only tables and views in {owner} are: {', '.join(sorted(tables))}."
        )
    # 3. Rules the model tends to forget (see check_rules).
    return check_rules(sql, owner)


# Used by check_rules. _AGGREGATE matches COUNT( SUM( AVG( MIN( MAX( .
# _ORDER_BY matches an "ORDER BY ..." clause, up to the next closing bracket, the word
# FETCH, or the end of the SQL, so the clause can be cut out of the query.
_AGGREGATE = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
_ORDER_BY = re.compile(r"ORDER\s+BY[^)]*?(?=\)|FETCH|$)", re.I | re.S)


def check_rules(sql: str, schema: str) -> str:
    """Rules the model is told but often skips; checked in code so they are enforced.

    Returns a description of the problem for the model to fix, or ''.
    """
    # Rule 1: no "WHERE ROWNUM = 1". It returns one arbitrary row, hides ties, and
    # usually means the number being ranked by was never selected.
    if re.search(r"\bROWNUM\s*=\s*1\b", sql, re.I):
        return (
            "Do not use ROWNUM = 1: it returns one arbitrary row and hides ties. Select the "
            "number you rank by as a named column, ORDER BY that column DESC, and use "
            "FETCH FIRST 5 ROWS ONLY."
        )
    # Rule 2: a count or total used in ORDER BY must also be selected, otherwise the
    # number never reaches the result and the answer cannot quote it. We find the
    # ORDER BY text, then check that an aggregate also appears everywhere EXCEPT there.
    ordered_by = " ".join(m.group(0) for m in _ORDER_BY.finditer(sql))
    if _AGGREGATE.search(ordered_by) and not _AGGREGATE.search(_ORDER_BY.sub("", sql)):
        return (
            "You ORDER BY a count or total that is not in the SELECT list, so the number is "
            "missing from the result. Select it as a named column (for example "
            "COUNT(...) AS GOLD_MEDALS), ORDER BY that column DESC, and return the top 5 rows."
        )
    # Rule 3 (Olympic data only): a distinct count over events must include the Games
    # year. The same event is held at every Games, so without the year a person who
    # won it three times is counted once. A year anywhere in the SQL (a filter on one
    # Games, for example) is enough to pass.
    if (
        schema == "OLYM"
        and re.search(r"COUNT\s*\(\s*DISTINCT[^)]*\bEVENT(?:_ID)?\b", sql, re.I)
        and not re.search(r"\b(?:EDITION|YEAR|GAMES?_ID|GAME_ID)\b", sql, re.I)
    ):
        return (
            "COUNT(DISTINCT ...) over events must include the Games year (the EDITION column "
            "or the games id) in the distinct key, for example COUNT(DISTINCT edition || '|' "
            "|| sport || '|' || event || '|' || event_gender). Without it the same event won "
            "at different Games is counted once, and people who won it several times are "
            "undercounted."
        )
    return ""


def make_rate_limiter(model_name: str = MODEL_NAME) -> InMemoryRateLimiter:
    """A limiter that holds each model call back until it is safe to send.

    The limit comes from MODEL_REQUESTS_PER_MINUTE. At 15 a minute that is one call every
    4 seconds. The bucket holds a single call, so there are no bursts, and the first call
    after the agent starts also waits one interval. A call that has to wait simply blocks
    until its turn.
    """
    try:
        per_minute = MODEL_REQUESTS_PER_MINUTE[model_name]
    except KeyError:
        raise ValueError(
            f"No requests-per-minute limit is set for the model {model_name!r}. Add a line for it to "
            "MODEL_REQUESTS_PER_MINUTE in Streamlit/agent.py (the limit for your key is on Google's "
            "rate-limits page for the Gemini API)."
        ) from None
    return InMemoryRateLimiter(requests_per_second=per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=1)


def is_daily_limit_error(error: Exception) -> bool:
    """True if Google refused a call because the DAILY request quota is used up.

    That error is an HTTP 429 whose text names the quota, for example
    "GenerateRequestsPerDayPerProjectPerModel". A per-minute 429 does not say "per day",
    so it is not mistaken for this one (the client retries those itself).
    """
    text = str(error).lower()
    return ("429" in text or "resource_exhausted" in text) and ("perday" in text or "per day" in text)


class ModelChain:
    """The models to use in order: MODEL_NAME first, then FALLBACK_MODELS.

    All the clients made from one chain use the same model at any moment. When one of them
    hits the daily limit, the whole chain moves on to the next model. Each model has its own
    rate limiter, since each has its own requests-per-minute limit.
    """

    def __init__(self):
        self.names = [MODEL_NAME, *FALLBACK_MODELS]
        self.limiters = {name: make_rate_limiter(name) for name in self.names}
        self.active = 0

    @property
    def active_name(self) -> str:
        return self.names[self.active]

    def client(self, **settings) -> "ChainClient":
        return ChainClient(self, settings)


class ChainClient:
    """Works like ChatGoogleGenerativeAI.invoke(prompt), but follows the chain's active model."""

    def __init__(self, chain: ModelChain, settings: dict):
        self.chain = chain
        self.clients = [
            ChatGoogleGenerativeAI(model=name, rate_limiter=chain.limiters[name], **settings)
            for name in chain.names
        ]

    def invoke(self, prompt: str):
        while True:
            used = self.chain.active
            try:
                return self.clients[used].invoke(prompt)
            except Exception as error:
                if not is_daily_limit_error(error):
                    raise
                if used + 1 >= len(self.chain.names):
                    raise RuntimeError("Every model has used up its requests for today: " + ", ".join(self.chain.names)) from error
                if self.chain.active == used:  # another client may have switched already
                    self.chain.active = used + 1
                    print(f"[agent] {self.chain.names[used]} is out of requests for today; switching to {self.chain.active_name}")


def clip(text: str, limit: int = PROMPT_SQL_CHARS) -> str:
    """Shorten text to `limit` characters, so a huge failed SQL cannot bloat the next prompt."""
    return text if len(text) <= limit else text[:limit] + " ...(cut off)"


def result_for_prompt(df: pd.DataFrame, max_chars: int = ANSWER_CHARS) -> tuple[str, int]:
    """The result as compact CSV that fits in the prompt, plus how many rows made it in."""
    shown = len(df)
    text = df.head(shown).to_csv(index=False)
    # Too big? Keep halving the number of rows until it fits.
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
        # One row: is every value either missing (NULL) or the number zero?
        return all(
            pd.isna(v) or (isinstance(v, numbers.Number) and v == 0)
            for v in df.iloc[0].tolist()
        )
    return False


# =============================================================================
# 1. STATE: the notebook every node reads from and writes to.
# =============================================================================

class State(TypedDict, total=False):
    """The shared dictionary passed between nodes.

    A node returns ONLY the keys it changes, and LangGraph merges them in.
    `total=False` means a key may be missing until some node has set it.
    """

    # Input, supplied by the app when it starts the graph.
    question: str  # the user's question, in English
    schema: str  # database schema (owner) to query, e.g. "OLYM"

    # Set by get_schema.
    schema_df: pd.DataFrame  # table_name, column_name, data_type for the schema
    schema_text: str  # the same, formatted as text for the prompt

    # Set by write_sql and run_sql.
    sql: str  # the SQL the model last wrote
    df: pd.DataFrame  # the rows the query returned
    error: str  # why the last attempt failed ('' if it did not)
    no_rows: bool  # the query ran fine but returned 0 rows
    attempts: int  # how many times write_sql has run (decides when to stop retrying)

    # Set by lookup_values.
    hints: str  # real column values, looked up after a 0-row result
    short_hints: str  # the compact part of hints (columns with few values), for the answer

    # Set by answer.
    answer: str  # the final reply shown to the user


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def build_graph(run_query: Callable[..., pd.DataFrame]):
    """Create the agent.

    `run_query(sql, params=None)` runs SQL against Oracle and returns a DataFrame. It is
    passed in (not imported) so the agent does not depend on how the app connects, and
    so tests can pass a fake.
    """
    # Two clients for the same model. The SQL writer has a cap on how much it may
    # write (see MAX_OUTPUT_TOKENS); the answer writer does not need one.
    # Both come from ONE chain, so they share the rate limiter of whichever model is in
    # use, and both move to the fallback model together if the daily limit is hit.
    chain = ModelChain()
    llm = chain.client()
    sql_llm = chain.client(max_output_tokens=MAX_OUTPUT_TOKENS)

    # -------------------------------------------------------------------------
    # 2. NODES: plain functions. Take the state, return ONLY the keys you change.
    # -------------------------------------------------------------------------

    def get_schema(state: State) -> dict:
        """Node 1. Read the table and column names of the chosen schema.

        Reads: schema.   Writes: schema_df, schema_text, and resets the retry state.
        """
        df = run_query(
            "SELECT table_name, column_name, data_type FROM all_tab_columns "
            "WHERE owner = :o ORDER BY table_name, column_id",
            {"o": state["schema"]},
        )
        # Turn the rows into one line per table, which is what the prompt shows the model:
        #   OLYM.OLYM_MEDALS(ID NUMBER, MEDAL VARCHAR2, ...)
        lines = {}
        for t, c, d in df.itertuples(index=False):
            lines.setdefault(t, []).append(f"{c} {d}")
        text = "\n".join(f"{state['schema']}.{t}({', '.join(cols)})" for t, cols in lines.items())
        # This is the first node, so it also resets everything left over from the
        # previous question.
        return {"schema_df": df, "schema_text": text, "attempts": 0, "error": "", "no_rows": False,
                "hints": "", "short_hints": ""}

    def write_sql(state: State) -> dict:
        """Node 2. Ask the model for one SELECT statement.

        Reads: schema, schema_text, question, and on a retry: sql, error, no_rows, hints.
        Writes: sql, attempts.
        """
        # The first part of the prompt is the same every time: the tables, the rules,
        # any notes about this schema, and the question.
        prompt = (
            "You write Oracle SQL. Use ONLY these tables and columns:\n"
            f"{state['schema_text']}\n\n"
            "Rules: one SELECT statement, no semicolon, qualify tables with the "
            f"{state['schema']}. prefix, no explanation, no markdown. "
            "You cannot see the data, so a text value may be spelled differently from the "
            "question (codes vs names, abbreviations, capitalization); when unsure of the "
            "exact value, compare with UPPER(column) LIKE '%VALUE%'.\n"
            "For 'most', 'top', 'highest' or 'who/which' ranking questions, SELECT the number "
            "you rank by as a column with a clear alias (for example GOLD_MEDALS) and return "
            "the top 5 rows ordered by it descending, so the number, any ties and the "
            "runners-up are visible. Never rank by something that is not in the SELECT list, "
            "and do not use ROWNUM = 1 to return a single row.\n"
            + (f"Notes about this data: {SCHEMA_NOTES[state['schema']]}\n" if state["schema"] in SCHEMA_NOTES else "")
            + f"Question: {state['question']}\n"
        )
        # If we are here because the last attempt was bad, say how it was bad.
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
            # lookup_values has already fetched the real values; show them.
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
        # Models sometimes wrap SQL in ```sql fences or end it with a semicolon; remove both.
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip().rstrip(";")
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1}

    def run_sql(state: State) -> dict:
        """Node 3. Check the SQL, then run it.

        Reads: sql, schema, schema_df.   Writes: df, error, no_rows.
        An SQL problem is not raised as an exception; it is stored in `error`, and
        after_run() sends the agent back to write_sql with that message.
        """
        sql = state["sql"]
        # Safety: only read queries. (The app also shows this as the "read-only" setting.)
        if not sql.lstrip().lower().startswith(("select", "with")):
            return {"error": "Only SELECT queries are allowed.", "no_rows": False}
        # Our own checks (runaway text, made-up tables, ranking rules) before Oracle sees it.
        problem = check_sql(sql, state["schema_df"], state["schema"])
        if problem:  # never send runaway or made-up SQL to the database
            return {"error": problem, "no_rows": False}
        try:
            # Wrap the model's query so it can never return more than ROW_LIMIT rows.
            df = run_query(f"SELECT * FROM ({sql}) FETCH FIRST {ROW_LIMIT} ROWS ONLY")
            return {"df": df, "error": "", "no_rows": looks_empty(df)}
        except Exception as e:
            # Oracle rejected it (bad column, bad syntax...). Keep the message for the retry.
            return {"error": str(e), "no_rows": False}

    def distinct_values(owner: str, table: str, col: str, limit: int) -> list:
        """Up to `limit` distinct non-null values of one column, sorted. Used by lookup_values."""
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

        Reads: sql, schema, schema_df.   Writes: hints (full text for the SQL writer) and
        short_hints (only the compact lists, for the answer writer).
        """
        sql, owner = state["sql"], state["schema"]
        # filtered / others hold the lines for the two groups above; short holds the
        # ones that are small enough to also show to the answer writer.
        filtered, others, short, seen = [], [], [], set()
        # Every text column in the schema, as (table, column) pairs.
        all_text_cols = [
            (t, c) for t, c, d in state["schema_df"].itertuples(index=False)
            if str(d).startswith(TEXT_TYPES)
        ]
        # Only the ones whose table is mentioned in the SQL.
        text_cols = [
            (t, c) for t, c in all_text_cols if re.search(rf"\b{re.escape(t)}\b", sql, re.I)
        ]
        # Group 1: columns the query compared to a quoted value, e.g.  NATION = 'USA'
        # or  UPPER(NATION) IN ('URS', 'RUS')  . Show all their real values.
        for table, col in text_cols:  # 1. columns the query filtered on
            if not re.search(rf"\b{re.escape(col)}\b\W*(?:=|!=|<>|LIKE|IN)\s*\(?\s*'", sql, re.I):
                continue
            seen.add((table, col))
            try:
                vals = distinct_values(owner, table, col, VALUE_HINT_LIMIT)
            except Exception:
                continue  # a column we cannot read is simply left out
            more = " ... (more values exist)" if len(vals) >= VALUE_HINT_LIMIT else ""
            line = f"{owner}.{table}.{col}: " + ", ".join(f"'{v}'" for v in vals) + more
            filtered.append(line)
            if len(vals) <= CATEGORY_LIMIT:
                short.append(line)
        # Group 2: every other text column in the whole schema, listed only if it has
        # few values. We ask for one more than the limit to tell "exactly 30" from "more".
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
        """Node 5. Turn the result into 1-3 plain sentences for the user.

        Reads: question, sql, df, error, no_rows, short_hints.   Writes: answer.
        """
        # We only get here with an error when the retries ran out; apologise honestly.
        if state.get("error"):
            return {"answer": f"Sorry, I couldn't get a result. Last error: {state['error']}"}
        df = state["df"]
        # Show the model the whole result (up to a size limit), so it cannot guess about
        # rows it was not given.
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
            "Always include the actual numbers from the result: for a count or total, state the "
            "number; for a ranking or 'who/which' question, name the top results (up to 5) each "
            "with its number, for example 'Michael Phelps (23)'. "
            "Write numbers without currency symbols (no $, €, £), even for prices, salaries or "
            "revenue; you may name a unit in words, for example 'per month'. "
            "Say exactly what was counted (for example medal events, not individual athletes)."
        )
        # Still nothing after all the retries: do not report "zero" as a fact.
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
            # The real values let the model say "the database has no Winter Games" plainly.
            if state.get("short_hints"):
                prompt += (
                    "\nThese are all the values that actually exist in the database for columns "
                    f"with few values:\n{state['short_hints']}\n"
                    "If the question asks for a value that is not among them (for example a "
                    "season), say plainly that the database does not contain it, and say what "
                    "it does contain."
                )
        return {"answer": llm.invoke(prompt).text.strip()}

    # -------------------------------------------------------------------------
    # 3. CONDITIONAL EDGE: a function that returns the NAME of the next node.
    # -------------------------------------------------------------------------

    def after_run(state: State) -> str:
        """Decide what follows run_sql. This is the only branching point in the graph.

        `attempts` counts how many times write_sql has run. We allow a retry while
        attempts <= MAX_RETRIES, so the loop always ends.
        """
        can_retry = state["attempts"] <= MAX_RETRIES
        if state.get("error") and can_retry:
            return "write_sql"  # SQL failed: loop back and try again
        if state.get("no_rows") and can_retry:
            return "lookup_values"  # SQL ran but found nothing: check the real values first
        return "answer"  # a good result, or no retries left

    # -------------------------------------------------------------------------
    # 4. WIRING: nodes + edges, then compile.
    #
    # This is the graph the calls below build. The numbers (1) to (6) match the
    # comments on the add_edge calls, so you can read the picture and the code side
    # by side.
    #
    #          START
    #            |
    #            | (1)
    #            v
    #     +--------------+
    #     |  get_schema  |
    #     +--------------+
    #            |
    #            | (2)
    #            v
    #     +--------------+
    #     |  write_sql   |
    #     +--------------+
    #            |
    #            | (3)
    #            v
    #     +--------------+
    #     |   run_sql    |
    #     +--------------+
    #            |
    #            | (4)  after_run() returns the name of the next node:
    #            |
    #            +--- "write_sql" ----------> back to write_sql
    #            |                            (the SQL failed, retries left)
    #            |
    #            +--- "lookup_values"          (it found nothing, retries left)
    #            |           |
    #            |           v
    #            |    +---------------+
    #            |    | lookup_values |
    #            |    +---------------+
    #            |           |
    #            |           | (5)  back to write_sql
    #            |
    #            +--- "answer"                 (a good result, or no retries left)
    #                        |
    #                        v
    #                 +--------------+
    #                 |    answer    |
    #                 +--------------+
    #                        |
    #                        | (6)
    #                        v
    #                       END
    #
    # In all: 5 nodes and 5 fixed arrows, (1) (2) (3) (5) (6), plus the one branch (4)
    # with three possible targets, which makes 8 arrows. The only loops are the two
    # that lead back to write_sql. They share the retry budget (MAX_RETRIES), so the
    # graph always reaches END.
    # -------------------------------------------------------------------------
    g = StateGraph(State)

    # Register each node under the name used in the diagrams.
    g.add_node("get_schema", get_schema)
    g.add_node("write_sql", write_sql)
    g.add_node("run_sql", run_sql)
    g.add_node("lookup_values", lookup_values)
    g.add_node("answer", answer)

    # Fixed arrows: the next node is always the same.
    g.add_edge(START, "get_schema")  # (1)
    g.add_edge("get_schema", "write_sql")  # (2)
    g.add_edge("write_sql", "run_sql")  # (3)
    # The branching arrow: after run_sql, after_run() returns the name of one of the
    # three listed nodes. (The list is only so LangGraph can draw and validate the graph.)
    g.add_conditional_edges("run_sql", after_run, ["write_sql", "lookup_values", "answer"])  # (4)
    g.add_edge("lookup_values", "write_sql")  # (5)
    g.add_edge("answer", END)  # (6)
    # compile() checks the graph is well formed (for example that every edge points at a
    # node that exists) and returns the runnable agent: the picture above.
    return g.compile()
