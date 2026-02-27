import sqlite3
import uuid
import os
import json
import re
import threading
import requests
import time
from typing import List, Tuple
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse, parse_qs
import imaplib
import email
from html import unescape
from email.utils import parseaddr
import smtplib
import ssl
from email.message import EmailMessage

# ==============================================================================
# AI (Mistral) setup
# ==============================================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}" if MISTRAL_API_KEY else "",
    "Content-Type": "application/json"
}


def run_local_ai_pipeline(message_text: str) -> dict:
    """
    Classify, summarise, and draft a reply using Mistral's chat API.
    Returns strict JSON-compatible dict. Falls back gracefully if API fails.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant for a small business.\n"
                "Return STRICT JSON with EXACT keys:\n"
                '"classification": one of ["lead","booking_enquiry", "reschedule","general_query"],\n'
                '"summary": one short sentence,\n'
                '"draft_reply": a helpful reply draft.\n'
                "Return ONLY JSON."
            )
        },
        {
            "role": "user",
            "content": f"Customer message:\n{message_text}"
        }
    ]

    if not MISTRAL_API_KEY:
        # No key available; degrade gracefully
        trimmed = message_text[:140] + "..." if len(message_text) > 140 else message_text
        return {
            "classification": "general_query",
            "summary": trimmed,
            "draft_reply": "Thanks — I'll get back to you shortly."
        }

    try:
        payload = {
            "model": "mistral-small-latest",
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
            "temperature": 0.2
        }

        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)

        return {
            "classification": data.get("classification", "general_query"),
            "summary": data.get("summary", message_text[:120]),
            "draft_reply": data.get("draft_reply", "Thanks — I'll get back to you shortly.")
        }

    except Exception as e:
        print(">>> Mistral JSON parse or request failed:", e, flush=True)
        trimmed = message_text[:140] + "..." if len(message_text) > 140 else message_text
        return {
            "classification": "general_query",
            "summary": trimmed,
            "draft_reply": "Thanks — I'll get back to you shortly."
        }


# ==============================================================================
# App + DB setup
# ==============================================================================

app = FastAPI(title="DEAP DEMO")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f">>> {request.method} {request.url.path}", flush=True)
    return await call_next(request)


# Ensure DB schema exists on boot
@app.on_event("startup")
async def _startup():
    init_db()


app.mount("/static", StaticFiles(directory="static"), name="static")

# Store the DB right beside this file for predictability
DB_PATH = os.path.join(os.path.dirname(__file__), "drafts.db")

print(">>> RUNNING FILE:", __file__)
print(">>> DB PATH:", os.path.abspath(DB_PATH))


# ==============================================================================
# DB setup
# ==============================================================================

def get_db_connection() -> sqlite3.Connection:
    """
    Create a short-lived SQLite connection in autocommit mode.
    """
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the SQLite database:
    - Enable WAL once.
    - Create drafts table if it doesn't exist.
    - Ensure 'status' column exists.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drafts (
                id TEXT PRIMARY KEY,
                content TEXT
            )
        """)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(drafts)").fetchall()]
        if "status" not in cols:
            print(">>> Adding 'status' column to drafts table")
            conn.execute("ALTER TABLE drafts ADD COLUMN status TEXT DEFAULT 'pending'")


def set_draft_error(draft_id: str, error: str) -> None:
    """
    Store the last error inside the JSON 'content' and set status='failed'.
    """
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if not row:
                return
            data = json.loads(row[0] or "{}")
            data["last_error"] = (error or "")[:2000]
            conn.execute(
                "UPDATE drafts SET content=?, status=? WHERE id=?",
                (json.dumps(data, ensure_ascii=False, indent=2), "failed", draft_id),
            )
            conn.commit()
    except Exception as e:
        print(">>> set_draft_error failed:", repr(e), flush=True)


# ==============================================================================
# Helpers
# ==============================================================================

def extract_sender_address(payload: dict) -> str:
    """
    Prefer 'reply_to' if present, else fallback to 'from'. Returns just the email.
    """
    reply_to = payload.get("reply_to") or payload.get("Reply-To")
    frm = payload.get("from") or payload.get("From")
    candidate = reply_to or frm or ""
    _, addr = parseaddr(candidate)
    return (addr or "").strip()


def save_draft(payload: dict) -> str:
    """
    Insert a new draft with AI-enriched data and capture sender email.
    """
    raw_text = (
        payload.get("text")            # chats, simple webhooks
        or payload.get("body")         # many email parsers use 'body'
        or payload.get("message")      # generic catch-all
        or json.dumps(payload, ensure_ascii=False)  # last resort
    )

    ai = run_local_ai_pipeline(raw_text)
    sender_email = extract_sender_address(payload)

    enriched = {
        "raw_text": raw_text,
        "classification": ai.get("classification"),
        "summary": ai.get("summary"),
        "draft_reply": ai.get("draft_reply"),
        "payload": payload,
        "sender_email": sender_email,
    }

    draft_id = str(uuid.uuid4())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO drafts (id, content, status) VALUES (?, ?, ?)",
        (draft_id, json.dumps(enriched, ensure_ascii=False, indent=2), "pending"),
    )
    conn.commit()
    conn.close()
    return draft_id


def update_draft_status(draft_id: str, new_status: str) -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE drafts SET status = ? WHERE id = ?", (new_status, draft_id))
        conn.commit()


def fetch_all_drafts() -> List[Tuple[str, str, str]]:
    conn = get_db_connection()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, content, status FROM drafts ORDER BY ROWID DESC"
    ).fetchall()
    conn.close()
    return rows


# ==============================================================================
# HTML
# ==============================================================================

def html_page_start(title: str = "Drafts") -> str:
    return f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>{title}</title>

<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{
  theme: {{
    extend: {{
      fontFamily: {{
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }},
    }}
  }}
}}
</script>

</head>
<body class="bg-gray-50 font-sans">
<header class="text-white py-4 px-8 shadow flex items-center space-x-4"
        style="background-color: rgb(29, 93, 169);">
  <img src="/static/logo.png"
       class="h-20 w-auto rounded-lg border-2 border-white shadow"
       alt="Company Logo">
  <h1 class="text-3xl font-semibold">Executive AI</h1>
</header>
<main class="max-w-4xl mx-auto p-6">
"""


def html_page_end() -> str:
    return "</main></body></html>"


def build_draft_card_html(did: str, content_json: str, status: str, root_path: str = "") -> str:
    try:
        data = json.loads(content_json)
    except Exception:
        data = {"raw_text": content_json, "payload": {}}

    ai_class = data.get("classification", "general_query")
    ai_summary = data.get("summary", "")
    ai_reply = data.get("draft_reply", "")
    raw_text = data.get("raw_text", "")
    payload_pretty = json.dumps(data.get("payload", {}), indent=2, ensure_ascii=False)

    status_color = {
        "pending": "bg-amber-500",
        "approved": "bg-blue-600",
        "executed": "bg-emerald-600",
        "failed": "bg-red-600",
    }.get(status, "bg-gray-600")

    html = []
    last_error = data.get("last_error")
    if last_error:
        html.append(f"""
        <div class="mt-4 p-3 rounded bg-red-50 border border-red-200 text-red-800">
          <div class="font-semibold text-sm mb-1">Last error</div>
          <pre class="whitespace-pre-wrap text-sm">{last_error}</pre>
        </div>
        """)

    html.append(f"""
<div class="bg-white shadow-sm rounded-xl p-6 mb-6 border border-gray-200">
  <div class="flex justify-between items-center">
    <div class="text-xs text-gray-500">ID: {did}</div>
    <span class="text-white text-xs px-3 py-1 rounded-full {status_color}">
      {status.capitalize()}
    </span>
  </div>

  <div class="mt-3">
    <span class="inline-block px-2 py-1 text-xs bg-gray-800 text-white rounded">
      {ai_class}
    </span>
  </div>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">AI Summary</h3>
  <p class="text-gray-800">{ai_summary}</p>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">Draft Reply</h3>
  <pre class="bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap">{ai_reply}</pre>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">Raw Message</h3>
  <pre class="bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap">{raw_text}</pre>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">Original Payload</h3>
  <pre class="bg-gray-100 p-4 rounded text-xs whitespace-pre-wrap">{payload_pretty}</pre>
""")

    if status == "pending":
        html.append(f"""
        <div class="flex space-x-3 mt-4">

          <a href="/edit/{did}"
             class="bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-800">
             Edit
          </a>

          <form action="/approve/{did}" method="post">
            <button class="text-white px-4 py-2 rounded-lg transition"
                    style="background-color: rgb(29, 93, 169);"
                    onmouseover="this.style.backgroundColor='rgb(22, 71, 130)'"
                    onmouseout="this.style.backgroundColor='rgb(29, 93, 169)'">
              Approve
            </button>
          </form>

        </div>
        """)
    elif status == "approved":
        _root = (root_path or "").rstrip("/")
        action = f"{_root}/execute/{did}" if _root else f"/execute/{did}"

        html.append(f"""
  <form action="{action}" method="post" enctype="application/x-www-form-urlencoded"
        onsubmit="const b=this.querySelector('button[type=submit]'); b.disabled=true; b.innerText='Sending…';">
    <input type="hidden" name="_" value="1"/>
    <button type="submit"
            class="mt-4 bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700">
      Send Email
    </button>
  </form>
""")

    html.append("</div>")
    return "".join(html)


# ==============================================================================
# Routes
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = """
    <html>
    <head>
        <title>DEAPInsights.ai</title>
        <style>
            .bento-button {
                display: inline-block;
                padding: 20px 32px;
                background-color: rgb(29, 93, 169);
                color: white;
                font-size: 20px;
                font-weight: 600;
                text-decoration: none;
                border-radius: 12px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .bento-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 14px rgba(0,0,0,0.25);
            }
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: #f5f7fa;
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        <a class="bento-button" href="/drafts">Emails</a>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/webhook", response_class=HTMLResponse)
def webhook_info() -> HTMLResponse:
    html = html_page_start("/webhook")
    html += """
    <h3>/webhook expects a <code>POST</code> with JSON</h3>
    <pre>
POST /webhook
Content-Type: application/json

{
  "event": "message.created",
  "text": "Hello"
}
    </pre>
    """
    html += html_page_end()
    return HTMLResponse(html)


@app.post("/webhook")
async def webhook(request: Request) -> dict:
    payload = await request.json()
    print(">>> WEBHOOK HIT with payload:", payload, flush=True)
    draft_id = save_draft(payload)
    return {"ok": True, "draft_id": draft_id}


@app.get("/drafts", response_class=HTMLResponse)
def list_drafts(request: Request) -> HTMLResponse:
    qs = parse_qs(urlparse(str(request.url)).query)
    ok_msg = (qs.get("ok") or [""])[0]

    try:
        rows = fetch_all_drafts()
    except sqlite3.OperationalError as e:
        return HTMLResponse(f"<h1>DB error</h1><pre>{e}</pre>", status_code=500)

    html = html_page_start("Drafts")
    html += "<h1>Drafts</h1>"

    if ok_msg:
        html += f"<div class='ok'>Done: {ok_msg}</div>"

    if not rows:
        html += "<div>No drafts yet. POST JSON to <code>/webhook</code> and refresh.</div>"
    else:
        root_path = request.scope.get("root_path", "")
        for did, content, status in rows:
            html += build_draft_card_html(did, content, status, root_path)

    html += html_page_end()
    return HTMLResponse(html)


@app.get("/edit/{draft_id}", response_class=HTMLResponse)
def edit_draft_form(draft_id: str):
    conn = get_db_connection()
    row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
    conn.close()

    if not row:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    data = json.loads(row[0])
    current_text = data.get("draft_reply", "")

    html = html_page_start("Edit Draft")
    html += f"""
    <h2 class="text-xl font-bold mb-4">Edit Draft Reply</h2>

    <form action="/edit/{draft_id}" method="post" class="space-y-4">
      <textarea name="draft_reply"
                class="w-full h-64 p-4 border rounded-lg">{current_text}</textarea>

      <button class="bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700">
        Save Changes
      </button>

      <a href="/drafts" class="ml-4 text-gray-600 hover:underline">Cancel</a>
    </form>
    """
    html += html_page_end()
    return HTMLResponse(html)


@app.post("/edit/{draft_id}")
def edit_draft_submit(draft_id: str, draft_reply: str = Form(...)):
    conn = get_db_connection()
    row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()

    if not row:
        conn.close()
        return RedirectResponse("/drafts?ok=missing", status_code=303)

    data = json.loads(row[0])
    data["draft_reply"] = draft_reply

    conn.execute(
        "UPDATE drafts SET content=? WHERE id=?",
        (json.dumps(data, ensure_ascii=False, indent=2), draft_id)
    )
    conn.commit()
    conn.close()

    return RedirectResponse("/drafts?ok=edited", status_code=303)


@app.post("/approve/{draft_id}")
def approve_draft(draft_id: str) -> RedirectResponse:
    update_draft_status(draft_id, "approved")
    return RedirectResponse(url="/drafts?ok=approved", status_code=303)


# ==============================================================================
# Execute (send) an approved draft
# ==============================================================================

@app.post("/execute/{draft_id}")
async def execute_draft(draft_id: str, background_tasks: BackgroundTasks):
    print(f">>> BUTTON CLICKED: Execute for {draft_id}", flush=True)
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT id FROM drafts WHERE id=?", (draft_id,)).fetchone()

        if not row:
            print(f">>> ERROR: Draft {draft_id} not found in DB", flush=True)
            return RedirectResponse("/drafts?ok=missing", status_code=303)

        background_tasks.add_task(send_and_mark_task, draft_id)
        print(f">>> SUCCESS: Task queued for {draft_id}. Redirecting...", flush=True)

        resp = RedirectResponse(url="/drafts?ok=sending", status_code=303)
        resp.background = background_tasks
        return resp

    except Exception as e:
        print(f">>> /execute handler FAILED for {draft_id}: {repr(e)}", flush=True)
        return RedirectResponse("/drafts?ok=error", status_code=303)


def send_and_mark_task(draft_id: str):
    print(f">>> WORKER START: Processing {draft_id}", flush=True)
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if not row:
            print(">>> WORKER ERROR: Row not found", flush=True)
            return

        data = json.loads(row["content"])
        payload = data.get("payload", {})

        to_addr = (data.get("sender_email") or extract_sender_address(payload) or "").strip()
        if not to_addr:
            raise RuntimeError("Cannot determine recipient email address (no Reply-To / From)")

        subject = f"Re: {payload.get('subject', 'Inquiry')}".strip()
        body = data.get("draft_reply", "")

        send_email_smtp(
            to_addr=to_addr,
            subject=subject,
            body=body,
            in_reply_to=payload.get("message_id") or payload.get("Message-ID"),
            refs=payload.get("references") or payload.get("References"),
        )

        conn.execute("UPDATE drafts SET status='executed' WHERE id=?", (draft_id,))
        print(f">>> WORKER COMPLETE: {draft_id} sent successfully", flush=True)

    except Exception as e:
        print(f">>> WORKER FAILED: {str(e)}", flush=True)
        try:
            conn.execute("UPDATE drafts SET status='failed' WHERE id=?", (draft_id,))
        except:
            pass
        set_draft_error(draft_id, str(e))
    finally:
        conn.close()


# ==============================================================================
# IMAP POLLER THREAD (with diagnostics)
# ==============================================================================

IMAP_EMAIL = os.getenv("IMAP_EMAIL")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:8000/webhook")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 30))


def _html_to_text(html: str) -> str:
    txt = unescape(html)
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", "", txt)
    txt = re.sub(r"(?is)<br\\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</p>", "\n\n", txt)
    txt = re.sub(r"(?is)<[^>]+>", "", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()


def imap_connect():
    print(">>> IMAP: connecting", {"host": IMAP_HOST, "port": IMAP_PORT, "email": IMAP_EMAIL})
    M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        M.login(IMAP_EMAIL, IMAP_PASSWORD)
    except imaplib.IMAP4.error as e:
        print(">>> IMAP LOGIN FAILED:", repr(e))
        raise
    print(">>> IMAP: login OK")
    return M


def imap_fetch_unread(include_html_fallback=True):
    M = imap_connect()

    status, _ = M.select("INBOX")
    print(">>> IMAP: SELECT INBOX =", status)
    if status != "OK":
        M.logout()
        return []

    try:
        status, data = M.uid("SEARCH", None, '(X-GM-RAW "is:unread")')
        print(">>> IMAP: UID SEARCH X-GM-RAW is:unread =", status, "count:", len((data[0] or b"").split()))
    except Exception as e:
        print(">>> IMAP: UID SEARCH error; falling back to UNSEEN:", e)
        status, data = M.search(None, "UNSEEN")
        print(">>> IMAP: SEARCH UNSEEN =", status, "count:", len((data[0] or b"").split()))

    if status != "OK":
        M.close()
        M.logout()
        return []

    ids = (data[0] or b"").split()
    messages = []

    for uid in ids:
        status, msg_data = M.uid("FETCH", uid, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            print(">>> IMAP: FETCH failed for", uid)
            continue

        msg = email.message_from_bytes(msg_data[0][1])

        subject = msg.get("Subject", "")
        sender = msg.get("From", "")

        # Extract body
        body = ""
        if msg.is_multipart():
            # Prefer text/plain
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = part.get("Content-Disposition", "") or ""
                if ctype == "text/plain" and "attachment" not in disp.lower():
                    try:
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                        break
                    except Exception:
                        pass
            if not body and include_html_fallback:
                for part in msg.walk():
                    ctype = part.get_content_type()
                    disp = part.get("Content-Disposition", "") or ""
                    if ctype == "text/html" and "attachment" not in disp.lower():
                        try:
                            html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore")
                            body = _html_to_text(html)
                            break
                        except Exception:
                            pass
        else:
            raw = msg.get_payload(decode=True) or b""
            try:
                text = raw.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception:
                text = raw.decode("utf-8", errors="ignore")
            body = _html_to_text(text) if msg.get_content_type() == "text/html" else text

        payload = {
            "source": "gmail_imap",
            "from": sender,
            "subject": subject,
            "body": body,
            "message_id": msg.get("Message-ID"),
            "references": msg.get("References"),
            "reply_to": msg.get("Reply-To"),   # include Reply-To for more accurate responses
        }
        print(">>> IMAP: fetched message:", {k: (v[:140] + "…") if isinstance(v, str) and len(v) > 140 else v for k, v in payload.items()})

        # Mark as read
        try:
            M.uid("STORE", uid, "+FLAGS", "\\Seen")
            print(">>> IMAP: marked seen", uid)
        except Exception as e:
            print(">>> IMAP: mark seen failed:", e)

        messages.append(payload)

    try:
        M.close()
    finally:
        M.logout()
    return messages


def imap_poller():
    print(">>> IMAP poller thread started with interval", POLL_INTERVAL)
    print(">>> WEBHOOK_URL:", WEBHOOK_URL)
    while True:
        try:
            msgs = imap_fetch_unread()
            for m in msgs:
                try:
                    r = requests.post(WEBHOOK_URL, json=m, timeout=15)
                    print(">>> Webhook POST:", r.status_code, (r.text[:200] + "…") if len(r.text) > 200 else r.text)
                except Exception as e:
                    print(">>> Webhook send error:", e)
        except Exception as e:
            print(">>> Poller cycle error:", e)

        time.sleep(POLL_INTERVAL)


# Start the thread only once
try:
    _DEAP_POLL_THREAD
except NameError:
    _DEAP_POLL_THREAD = threading.Thread(target=imap_poller, daemon=True)
    _DEAP_POLL_THREAD.start()


@app.post("/admin/poll-now")
def admin_poll_now() -> dict:
    try:
        msgs = imap_fetch_unread()
        posted = []
        for m in msgs:
            r = requests.post(WEBHOOK_URL, json=m, timeout=15)
            posted.append({"status": r.status_code, "text": r.text[:200]})
        return {"ok": True, "fetched": len(msgs), "posted": posted}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


# ==============================================================================
# SMTP SENDING (no Resend)
# ==============================================================================

SMTP_EMAIL = os.getenv("SMTP_EMAIL")                 # e.g., your Gmail address
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")   # e.g., Gmail App Password
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))         # 587 for STARTTLS, 465 for SSL
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "").lower() in ("1", "true", "yes")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "DEAP")  # display name for outgoing mail


def send_email_smtp(to_addr: str, subject: str, body: str,
                    in_reply_to=None, refs=None):
    """
    Sends email via SMTP (Gmail or other).
    - Auth using SMTP_EMAIL / SMTP_APP_PASSWORD
    - Uses STARTTLS by default on port 587 (or SSL if configured)
    - Preserves threading with In-Reply-To / References
    """
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        raise RuntimeError("Missing SMTP_EMAIL or SMTP_APP_PASSWORD environment variables")

    if not to_addr:
        raise ValueError("Recipient address is empty")

    # Build message
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_EMAIL}>"
    msg["To"] = to_addr

    # Threading headers for proper reply chains
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if refs:
        msg["References"] = refs

    msg.set_content(body)

    print(f">>> SMTP: sending to={to_addr} subj={subject}", flush=True)

    if SMTP_USE_SSL or SMTP_PORT == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context, timeout=30) as server:
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            if SMTP_STARTTLS:
                context = ssl.create_default_context()
                server.starttls(context=context)
                server.ehlo()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.send_message(msg)

    print(">>> SMTP: sent OK", flush=True)
    return True


# ==============================================================================
# Debug: print loaded routes on startup (helps verify you’re running the right file)
# ==============================================================================

print(">>> ROUTES LOADED:")
for r in app.routes:
    print("   ", r.path, getattr(r, "methods", None))