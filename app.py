import os
import re
from typing import Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai

app = FastAPI(title="GitTutor Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatReq(BaseModel):
    text: str


PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION,
)

MODEL = "gemini-2.0-flash"

GIT_SIGNAL_RE = re.compile(
    r"\b(git|commit|push|pull|rebase|merge|checkout|switch|branch|stash|reset|revert|"
    r"cherry-pick|diff|log|remote|origin|upstream|non-fast-forward|detached\s+head|"
    r"conflict|submodule|blame)\b",
    re.I,
)

SECURITY_RE = re.compile(
    r"\b("
    r"hack|phish|phishing|steal password|break into|unauthorized access|"
    r"bypass|exploit|payload|malware"
    r")\b",
    re.I,
)

VIOLENCE_RE = re.compile(
    r"\b("
    r"i('?m| am) going to (hurt|kill|attack) (someone|someone else|him|her|them)|"
    r"hurt someone|kill someone|"
    r"shoot|stab"
    r")\b",
    re.I,
)

SELF_HARM_RE = re.compile(
    r"\b(kill myself|suicide|self-harm|hurt myself|end my life)\b",
    re.I,
)

WEAPON_RE = re.compile(
    r"\b(make a bomb|bomb|explosive)\b",
    re.I,
)

def explain_blame() -> str:
    return (
        "Use `git blame <file>` to see who last changed each **line** and which **commit** introduced it.\n"
        "Examples:\n"
        "- `git blame README.md`\n"
        "- `git blame -L 10,30 app.py` (focus on a line range)\n"
        "- `git blame -w app.py` (ignore whitespace)"
    )


def explain_submodule() -> str:
    return (
        "A Git **submodule** is a repo nested inside another repo, pinned to a specific commit.\n"
        "Common commands:\n"
        "- `git submodule init`\n"
        "- `git submodule update --init --recursive`\n"
        "- `git submodule update`\n"
        "Common gotcha: submodules are often in a **detached** HEAD state (you’re checked out at a commit), "
        "and you must commit the updated submodule pointer in the parent repo."
    )


def explain_undo_last_commit_keep_changes() -> str:
    return (
        "To undo the last commit but **keep** your changes:\n"
        "- `git reset --soft HEAD~1` (keeps changes staged)\n"
        "- `git reset --mixed HEAD~1` (keeps changes but unstaged)\n"
        "This moves HEAD back one commit without deleting your work."
    )


DETERMINISTIC_EXPLAINERS: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"\bgit\s+blame\b|\bblame\b|\bwho\b.*\bline\b|\blast\b.*\bchanged\b.*\bline\b", re.I), explain_blame),
    (re.compile(r"\bsubmodule\b|\bgit\s+submodule\b|\bsubmodule\s+init\b|\bsubmodule\s+update\b|\b--init\b|\bdetached\b", re.I), explain_submodule),
    (re.compile(r"\bundo\b.*\blast commit\b|\bundo\b.*\bcommit\b|\bHEAD~1\b|\breset\b.*--soft\b|\b--soft\b.*\bHEAD~1\b", re.I), explain_undo_last_commit_keep_changes),
]


def security_refusal_response() -> str:
    return (
        "I can't help with **unauthorized** access or bypassing protections — that is a **security** risk.\n"
        "If you need legitimate assistance, use official account recovery procedures, "
        "reset credentials properly, enable multi-factor authentication, "
        "and review recent login activity."
    )


def violence_response() -> str:
    return (
        "If you feel you might hurt someone, please seek **immediate** help.\n"
        "This is an **emergency** — **call** local emergency services right now (US/Canada: 911).\n"
        "If you can, put distance between yourself and anything that could be used to harm someone, "
        "and reach out to a trusted person nearby."
    )


def self_harm_response() -> str:
    return (
        "I’m really sorry you’re feeling this way — and I want you to get help right now.\n\n"
        "If you’re in immediate danger or might act on these thoughts, **call emergency services now** "
        "(in the US/Canada: **call 911**).\n"
        "If you’re in the US, you can also **call or text 988** (Suicide & Crisis Lifeline). "
        "If you’re outside the US, tell me your country and I’ll point you to local options.\n\n"
        "If you can, reach out to someone you trust and stay with others. "
        "I can also listen, but I can’t help with instructions to harm yourself."
    )


def weapon_refusal_response() -> str:
    return (
        "I can't help with instructions that could be used to harm people (e.g., weapons/explosives).\n"
        "If this is related to safety concerns, contact local authorities or appropriate professionals.\n"
        "If you want Git help, share the exact `git ...` command and the error/output."
    )


def out_of_scope_response() -> str:
    return (
        "I can only help with Git CLI commands, Git workflows, and Git error troubleshooting.\n"
        "Please share the exact `git ...` command or error message."
    )


def escape_hatch_git() -> str:
    return (
        "I can help with Git commands, but I need more context.\n\n"
        "Please provide:\n"
        "1) the exact `git ...` command\n"
        "2) the full error/output\n"
        "3) your goal\n"
        "4) whether the branch was pushed/shared\n"
    )


SYSTEM_PROMPT = """
You are GitTutor, a narrow-domain Git CLI tutor.

Scope:
- Git commands, Git workflows, merge/rebase, staging, remotes, and error troubleshooting.

Out-of-scope categories:
1) General programming not related to Git
2) General knowledge/trivia
3) Security or unauthorized access requests

Uncertainty handling:
If question lacks context, ask for:
- exact git command
- full error/output
- goal
- whether branch was pushed/shared

Style:
- Practical and command-focused
- Prefer step-by-step
- Warn when rewriting history
"""

FEW_SHOTS = [
    (
        "User: What does `git add -p` do?\nAssistant:",
        "It interactively stages changes as a patch, letting you stage hunks selectively. "
        "Run `git add -p`, review each hunk, stage or skip, then `git commit`."
    ),
    (
        "User: I got non-fast-forward on push.\nAssistant:",
        "`non-fast-forward` means the remote has commits you don’t have. "
        "Run `git fetch`, then `git pull --rebase` (or merge), resolve conflicts, then push. "
        "Avoid force-push unless history rewrite is acceptable."
    ),
    (
        "User: I ran git pull and got a merge conflict.\nAssistant:",
        "A conflict happens when Git cannot auto-merge.\n"
        "1) `git status`\n"
        "2) Open files with `<<<<<<<` markers\n"
        "3) Edit and remove markers\n"
        "4) `git add <file>`\n"
        "5) `git commit` or `git rebase --continue`\n"
        "6) Test before pushing"
    ),
]


def llm_answer(user_text: str) -> str:
    shots = "\n\n".join([q + " " + a for (q, a) in FEW_SHOTS])
    prompt = f"{SYSTEM_PROMPT}\n\n{shots}\n\nUser: {user_text}\nAssistant:"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


def post_generation_backstop(user_text: str, answer: str) -> Tuple[str, str]:
    if not answer.strip():
        return escape_hatch_git(), "empty"

    if GIT_SIGNAL_RE.search(user_text) and not GIT_SIGNAL_RE.search(answer):
        return escape_hatch_git(), "no_git_signal"

    if not GIT_SIGNAL_RE.search(user_text):
        return out_of_scope_response(), "oos"

    return answer, "ok"


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
def chat(req: ChatReq):
    t = (req.text or "").strip()
    if not t:
        return {"answer": out_of_scope_response(), "backstop": "redirect"}

    if SECURITY_RE.search(t):
        return {"answer": security_refusal_response(), "backstop": "security"}

    if VIOLENCE_RE.search(t):
        return {"answer": violence_response(), "backstop": "violence"}

    if SELF_HARM_RE.search(t):
        return {"answer": self_harm_response(), "backstop": "self_harm"}

    if WEAPON_RE.search(t):
        return {"answer": weapon_refusal_response(), "backstop": "weapons"}

    for pat, fn in DETERMINISTIC_EXPLAINERS:
        if pat.search(t):
            return {"answer": fn(), "backstop": "static_explainer"}

    if not GIT_SIGNAL_RE.search(t):
        return {"answer": out_of_scope_response(), "backstop": "redirect"}

    try:
        ans = llm_answer(t)
        ans2, tag = post_generation_backstop(t, ans)
        return {"answer": ans2, "backstop": f"llm/{tag}"}
    except Exception as e:
        return {"answer": escape_hatch_git(), "backstop": f"error:{type(e).__name__}"}