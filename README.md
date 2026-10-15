# LangStuff

A learning project for LangChain and LangGraph. It starts with the basics in a
notebook and builds up to a Streamlit app where you ask an Oracle database
questions in plain English.

## What's in here

| Path | What it is |
|------|------------|
| `LangChain01.ipynb` | LangChain 101 with Google Gemini: models, messages, prompt templates, LCEL chains (`prompt \| llm \| parser`), streaming, and structured output with Pydantic. |
| `FreeSQLTest.ipynb` | A minimal connection test for Oracle FreeSQL using `oracledb`. |
| `Streamlit/app.py` | The Streamlit front end: sidebar settings, chat input, and answer / SQL / results / agent-trace tabs. |
| `Streamlit/agent.py` | The LangGraph agent that turns a question into SQL, runs it, and explains the result. |
| `Streamlit/requirements.txt` | Python dependencies for the app. |

## The Streamlit app: "Ask your database"

Type a question in plain English. The app writes SQL, runs it against the
selected schema, and answers in a few sentences. Every answer comes with the SQL
that was used, the result table, and a trace of the agent's steps.

### How the agent works

The agent is a small LangGraph graph with a retry loop:

```
get_schema -> write_sql -> run_sql -> answer
                  ^           |
                  +-- error --+   (up to 2 retries)
```

1. **get_schema**: reads the table and column names of the chosen schema from Oracle.
2. **write_sql**: asks the model for a single `SELECT` using only those tables and columns.
3. **run_sql**: runs it. If it fails, the error goes back to `write_sql` so the model can fix its own query.
4. **answer**: summarises the result in 1 to 3 sentences.

### Schemas

A sidebar dropdown picks which FreeSQL sample schema to query. Each schema keeps
its own conversation.

- Olympic Data (`OLYM`), the default
- Customer Orders (`CO`)
- Human Resources (`HR`)

To add another, add a line to the `SCHEMAS` dictionary in `Streamlit/app.py`.

### Safety

- Only statements starting with `SELECT` or `WITH` are run.
- Results are capped at 100 rows.
- The model sees table and column names and types when writing SQL. When writing the answer it also sees up to the first 20 rows of the result.

### Model

Google Gemini, `gemini-3.5-flash-lite`. It's set by `MODEL_NAME` in `Streamlit/agent.py`.

## Setup

You need Python 3.10 or newer (developed on 3.14).

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell shown; on macOS or Linux use `source .venv/bin/activate`):

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
pip install -r Streamlit/requirements.txt
```

### The `.env` file

Secrets are kept out of git. Create a file named `.env` in the project root (the
same folder as this README) with these four keys:

```
GOOGLE_API_KEY=your-google-ai-studio-key
FREE_SQL_USER=your-freesql-username
FREE_SQL_PASSWORD=your-freesql-password
FREE_SQL_DSN=your-freesql-connection-string
```

| Key | What it's for | Where to get it |
|-----|---------------|-----------------|
| `GOOGLE_API_KEY` | Calls to Gemini | A free key from <https://aistudio.google.com/apikey> (no billing needed) |
| `FREE_SQL_USER` | Oracle FreeSQL login | Your FreeSQL account's connection details |
| `FREE_SQL_PASSWORD` | Oracle FreeSQL password | Same |
| `FREE_SQL_DSN` | Oracle connection string | Same (an Oracle "easy connect" string, for example `host:port/service_name`) |

Notes:

- `.env` is listed in `.gitignore`. Never commit it.
- `LangChain01.ipynb` prompts for `GOOGLE_API_KEY` the first time and saves it to `.env` for you.
- On Streamlit Community Cloud there is no `.env`. Put the same four keys in the app's **Secrets** settings instead; the app reads `st.secrets` as a fallback.

## Run the app

From the project root:

```bash
python -m streamlit run Streamlit/app.py
```

Streamlit prints a local address, normally <http://localhost:8501>. In the sidebar,
click **Test connection** to confirm the database login works, then ask a question.

To run the notebooks, open them in VS Code or Jupyter with the `.venv` environment selected.

## Ideas for next steps

- Per-schema example questions on the buttons in the app.
- Follow-up questions that remember earlier turns (a LangGraph checkpointer).
- A tool-calling agent that chooses its own steps, such as listing tables or describing a table, instead of a fixed pipeline.
- A stricter SQL safety check, or a read-only database user.
