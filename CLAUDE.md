# Notes for Claude

A LangGraph text-to-SQL agent (`Streamlit/agent.py`) with a Streamlit front end
(`Streamlit/app.py`). See `README.md` for setup and how to run it.

## Keep the graph documentation in sync with the code

The agent's graph is documented by hand in four places. Whenever you add, remove or
rename a node or an arrow, add a `State` key, or change what a node reads or writes,
update **all** of them in the same change:

1. **`Streamlit/agent.py`, top docstring**: the diagram, the "EDGES (the arrows)" list,
   and the "WHO READS AND WRITES WHAT" table.
2. **`Streamlit/agent.py`, the wiring section of `build_graph()`**: the numbered diagram
   above `g = StateGraph(State)`, the `# (n)` number on each `add_edge` /
   `add_conditional_edges` call, the branch targets, and the sentence with the counts
   ("N nodes and N fixed arrows ... which makes N arrows").
3. **`README.md`, "How the agent works"**: the diagram, the numbered list of steps, and
   the retry count if `MAX_RETRIES` changed.
4. **`Streamlit/assets/header.svg`**: the banner draws the graph (the five nodes, START
   and END, the main flow, and the two retry loops). Change the nodes and arrows there
   too. Each label's `textLength` is its width measured in Segoe UI, so if you rename a
   node or change a font size, measure the new text (for example with `getComputedTextLength()`
   in the browser) and resize the box to fit, then check nothing overlaps the title.

Then check the pictures against the graph that is really built, and report which places
you updated. From the project root:

```bash
cd Streamlit && GOOGLE_API_KEY=dummy ../.venv/Scripts/python.exe -c "import agent, pandas as pd; g = agent.build_graph(lambda sql, params=None: pd.DataFrame()).get_graph(); print(sorted(g.nodes)); print(sorted((e.source, e.target, bool(e.conditional)) for e in g.edges))"
```

It prints the real nodes and arrows (`__start__` and `__end__` are START and END; arrows
marked `True` are the ones out of the `run_sql` branch). Compare them with the four
places above. If you changed what a node reads or writes, read its code to check the
table, since the command does not show that.

The owner decided **not** to keep a test file for this. Do not add one. The check above
is a manual step, so do not skip it.

## Other things to remember

- `Evaluation.ipynb` holds a question bank whose reference SQL follows the schema notes in
  `agent.py` (`SCHEMA_NOTES`): sales and order counts use COMPLETE orders only, and an Olympic
  medal is one event at one Games (edition, discipline, event and event gender together). If you
  change those notes, update the matching reference SQL in the bank so it still agrees.

- `.env` holds secrets and is git-ignored. Never commit it or print its values.
- `.vscode/` is the owner's editor setting and is deliberately left untracked.
- The app reloads `agent.py` on every run, so edits to it take effect without restarting.
