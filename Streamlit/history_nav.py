"""Up/down-arrow recall of earlier questions in st.chat_input.

Streamlit's chat box has no built-in input history, so a small script listens
for the arrow keys on the chat textarea and swaps in earlier questions, like a
terminal. Render it with st.html(..., unsafe_allow_javascript=True).
"""
import json

_SCRIPT = r"""
<script>
(function () {
  var questions = __QUESTIONS__;
  var key = questions.join("\u0001");

  // New history (a question was asked, schema switched, conversation cleared):
  // point just past the newest entry and forget any half-typed draft.
  if (window.__qHistKey !== key) {
    window.__qHistKey = key;
    window.__qHist = questions;
    window.__qIdx = questions.length;
    window.__qDraft = "";
  }

  // Install the key listener once; it reads the latest history from window.
  if (window.__qHistInstalled) { return; }
  window.__qHistInstalled = true;

  var SELECTOR = 'textarea[data-testid="stChatInputTextArea"], [data-testid="stChatInput"] textarea';

  function setValue(el, value) {
    // The textarea is controlled by React, so use the native setter and fire
    // an input event; assigning el.value directly would be ignored.
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    setter.call(el, value);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.setSelectionRange(value.length, value.length);
  }

  document.addEventListener("keydown", function (e) {
    var el = e.target;
    if (!el || !el.matches || !el.matches(SELECTOR)) { return; }
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.isComposing) { return; }
    var hist = window.__qHist || [];

    if (e.key === "ArrowUp") {
      // Only on the first line, so arrows still move the cursor in multi-line text.
      var onFirstLine = el.value.slice(0, el.selectionStart).indexOf("\n") === -1;
      if (!onFirstLine || window.__qIdx <= 0) { return; }
      if (window.__qIdx === hist.length) { window.__qDraft = el.value; }
      window.__qIdx -= 1;
      setValue(el, hist[window.__qIdx]);
      e.preventDefault();
    } else if (e.key === "ArrowDown") {
      var onLastLine = el.value.slice(el.selectionEnd).indexOf("\n") === -1;
      if (!onLastLine || window.__qIdx >= hist.length) { return; }
      window.__qIdx += 1;
      setValue(el, window.__qIdx === hist.length ? window.__qDraft : hist[window.__qIdx]);
      e.preventDefault();
    }
  }, true);
})();
</script>
"""


def history_nav_html(questions: list[str]) -> str:
    """HTML for st.html(): arrow-key recall over `questions` (oldest first)."""
    # json.dumps escapes non-ASCII; also escape <, > and & so a question can
    # never close the script tag or inject markup.
    payload = (
        json.dumps(questions)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return _SCRIPT.replace("__QUESTIONS__", payload)
