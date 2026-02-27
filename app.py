import sqlite3, uuid, os, json, re, threading, requests, time, imaplib, email, smtplib, ssl
from typing import List, Tuple
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse, parse_qs
from html import unescape
from email.utils import parseaddr
from email.message import EmailMessage

# ==============================================================================
# AI (Mistral) setup (unchanged)
# ==============================================================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}" if MISTRAL_API_KEY else "", "Content-Type": "application/json"}

def run_local_ai_pipeline(message_text: str) -> dict:
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
            ),
        },
        {"role": "user", "content": f"Customer message:\n{message_text}"},
    ]
    if not MISTRAL_API_KEY:
        trimmed = message_text[:140] + "..." if len(message_text) > 140 else message_text
        return {"classification": "general_query", "summary": trimmed, "draft_reply": "Thanks — I'll get back to you shortly."}
    try:
        payload = {"model": "mistral-small-latest", "messages": messages, "response_format": {"type": "json_object"}, "max_tokens": 300, "temperature": 0.2}
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        return {
            "classification": data.get("classification", "general_query"),
            "summary": data.get("summary", message_text[:120]),
            "draft_reply": data.get("draft_reply", "Thanks — I'll get back to you shortly."),
        }
    except Exception as e:
        print(">>> Mistral JSON parse or request failed:", e, flush=True)
        trimmed = message_text[:140] + "..." if len(message_text) > 140 else message_text
        return {"classification": "general_query", "summary": trimmed, "draft_reply": "Thanks — I'll get back to you shortly."}

# ==============================================================================
# App + DB setup
# ==============================================================================
app = FastAPI(title="DEAP DEMO")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f">>> {request.method} {request.url.path}", flush=True)
    return await call_next(request)

@app.on_event("startup")
async def _startup():
    init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

DB_PATH = os.path.join(os.path.dirname(__file__), "drafts.db")
print(">>> RUNNING FILE:", __file__)
print(">>> DB PATH:", os.path.abspath(DB_PATH))

# ==============================================================================
# DB
# ==============================================================================
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""CREATE TABLE IF NOT EXISTS drafts (id TEXT PRIMARY KEY, content TEXT)""")
        cols = [row[1] for row in conn.execute("PRAGMA table_info(drafts)").fetchall()]
        if "status" not in cols:
            print(">>> Adding 'status' column to drafts table")
            conn.execute("ALTER TABLE drafts ADD COLUMN status TEXT DEFAULT 'pending'")

def set_draft_error(draft_id: str, error: str) -> None:
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if not row: return
            data = json.loads(row[0] or "{}")
            data["last_error"] = (error or "")[:2000]
            conn.execute("UPDATE drafts SET content=?, status=? WHERE id=?", (json.dumps(data, ensure_ascii=False, indent=2), "failed", draft_id))
            conn.commit()
    except Exception as e:
        print(">>> set_draft_error failed:", repr(e), flush=True)

# ==============================================================================
# Helpers
# ==============================================================================
def extract_sender_address(payload: dict) -> str:
    reply_to = payload.get("reply_to") or payload.get("Reply-To")
    frm = payload.get("from") or payload.get("From")
    _, addr = parseaddr((reply_to or frm or "").strip())
    return (addr or "").strip()

def save_draft(payload: dict) -> str:
    raw_text = payload.get("text") or payload.get("body") or payload.get("message") or json.dumps(payload, ensure_ascii=False)
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
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("INSERT INTO drafts (id, content, status) VALUES (?, ?, ?)", (draft_id, json.dumps(enriched, ensure_ascii=False, indent=2), "pending"))
    conn.commit(); conn.close()
    return draft_id

def update_draft_status(draft_id: str, new_status: str) -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE drafts SET status = ? WHERE id = ?", (new_status, draft_id))
        conn.commit()

def fetch_all_drafts() -> List[Tuple[str, str, str]]:
    conn = get_db_connection()
    rows = conn.cursor().execute("SELECT id, content, status FROM drafts ORDER BY ROWID DESC").fetchall()
    conn.close()
    return rows

# ==============================================================================
# HTML
# ==============================================================================
def html_page_start(title: str = "Emails") -> str:
    return f"""
<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{ theme: {{ extend: {{ fontFamily: {{ sans: ['Inter','system-ui','sans-serif'] }} }} }} }};
</script>
</head>
<body class="bg-gray-50 font-sans">
<header class="text-white py-4 px-8 shadow flex items-center space-x-4" style="background-color: rgb(29, 93, 169);">
  <img src="/static/logo.png" class="h-20 w-auto rounded-lg border-2 border-white shadow" alt="Company Logo">
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
    payload = data.get("payload", {})
    from_addr = payload.get("from") or payload.get("From") or data.get("sender_email") or ""
    subject = payload.get("subject") or "Inquiry"

    status_color = {
        "pending": "bg-amber-500",
        "approved": "bg-blue-600",
        "executed": "bg-emerald-600",
        "failed": "bg-red-600",
    }.get(status, "bg-gray-600")

    last_error = data.get("last_error")

    html = []
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
    <span class="text-white text-xs px-3 py-1 rounded-full {status_color}">{status.capitalize()}</span>
  </div>

  <div class="mt-3">
    <span class="inline-block px-2 py-1 text-xs bg-gray-800 text-white rounded">{ai_class}</span>
  </div>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">AI Summary</h3>
  <p class="text-gray-800">{ai_summary}</p>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">Draft Reply</h3>
  <pre class="bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap">{ai_reply}</pre>

  <h3 class="mt-4 text-sm font-semibold text-gray-700">Original Email</h3>
  <div class="bg-gray-50 p-4 rounded border text-sm">
    <div><span class="font-semibold text-gray-700">From:</span> <span class="text-gray-900">{from_addr}</span></div>
    <div class="mt-1"><span class="font-semibold text-gray-700">Subject:</span> <span class="text-gray-900">{subject}</span></div>
    <div class="mt-3">
      <div class="font-semibold text-gray-700 mb-1">Message</div>
      <div class="whitespace-pre-wrap text-gray-900">{raw_text}</div>
    </div>
  </div>
""")

    if status == "pending":
        html.append(f"""
  <div class="flex space-x-3 mt-4">
    <a href="/edit/{did}" class="bg-gray-700 text-white px-4 py-2 rounded-lg hover:bg-gray-800">Edit</a>
    <form action="/approve/{did}" method="post">
      <button class="text-white px-4 py-2 rounded-lg transition" style="background-color: rgb(29, 93, 169);" onmouseover="this.style.backgroundColor='rgb(22, 71, 130)'" onmouseout="this.style.backgroundColor='rgb(29, 93, 169)'">
        Approve
      </button>
    </form>
  </div>
""")
    elif status == "approved":
        _root = (root_path or "").rstrip("/")
        action = f"{_root}/execute/{did}" if _root else f"/execute/{did}"
        # Hide the button immediately after submit so it doesn't linger
        html.append(f"""
  <form action="{action}" method="post" enctype="application/x-www-form-urlencoded"
        onsubmit="const b=this.querySelector('button[type=submit]'); b.style.display='none'; const s=document.getElementById('s-{did}'); if(s) s.classList.remove('hidden');">
    <input type="hidden" name="_" value="1"/>
    <button type="submit" class="mt-4 bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700">Send Email</button>
    <span id="s-{did}" class="ml-2 text-sm text-emerald-700 hidden">Sending…</span>
  </form>
""")

    html.append("</div>")
    return "".join(html)

# ==============================================================================
# Routes
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    # Splash page with a bento-style rounded rectangle linking to Emails
    html = html_page_start("DEAPInsights.ai")
    html += """
<div class="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-6">
  <a href="/emails" class="block rounded-2xl p-6 bg-white border border-gray-200 shadow-sm hover:shadow transition">
    <div class="flex items-center">
      <div class="h-10 w-10 flex items-center justify-center rounded-xl" style="background-color: rgba(29,93,169,0.1);">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="rgb(29,93,169)">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 8l9 6 9-6M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      </div>
      <div class="ml-4">
        <div class="text-lg font-semibold text-gray-900">Emails</div>
        <div class="text-sm text-gray-600">Review, edit, and send replies</div>
      </div>
    </div>
  </a>
</div>
"""
    html += html_page_end()
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

# -------- Drafts list -> Emails list (renamed path/UI only) --------
@app.get("/emails", response_class=HTMLResponse)
def list_emails(request: Request) -> HTMLResponse:
    qs = parse_qs(urlparse(str(request.url)).query)
    ok_msg = (qs.get("ok") or [""])[0]
    try:
        rows = fetch_all_drafts()
    except sqlite3.OperationalError as e:
        return HTMLResponse(f"<h1>DB error</h1><pre>{e}</pre>", status_code=500)

    html = html_page_start("Emails")
    html += "<h1 class='text-2xl font-semibold mb-4'>Emails</h1>"
    if ok_msg:
        html += f"<div class='mb-4 text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-3 py-2'>Done: {ok_msg}</div>"
    if not rows:
        html += "<div>No emails yet. POST JSON to <code>/webhook</code> and refresh.</div>"
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
    html = html_page_start("Edit Email")
    html += f"""
<h2 class="text-xl font-bold mb-4">Edit Draft Reply</h2>
<form action="/edit/{draft_id}" method="post" class="space-y-4">
  <textarea name="draft_reply" class="w-full h-64 p-4 border rounded-lg">{current_text}</textarea>
  <button class="bg-emerald-600 text-white px-4 py-2 rounded hover:bg-emerald-700">Save Changes</button>
  <a href="/emails" class="ml-4 text-gray-600 hover:underline">Cancel</a>
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
        return RedirectResponse("/emails?ok=missing", status_code=303)
    data = json.loads(row[0]); data["draft_reply"] = draft_reply
    conn.execute("UPDATE drafts SET content=? WHERE id=?", (json.dumps(data, ensure_ascii=False, indent=2), draft_id))
    conn.commit(); conn.close()
    return RedirectResponse("/emails?ok=edited", status_code=303)

@app.post("/approve/{draft_id}")
def approve_draft(draft_id: str) -> RedirectResponse:
    update_draft_status(draft_id, "approved")
    return RedirectResponse(url="/emails?ok=approved", status_code=303)

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
                return RedirectResponse("/emails?ok=missing", status_code=303)
        background_tasks.add_task(send_and_mark_task, draft_id)
        print(f">>> SUCCESS: Task queued for {draft_id}. Redirecting...", flush=True)
        resp = RedirectResponse(url="/emails?ok=sending", status_code=303)
        resp.background = background_tasks
        return resp
    except Exception as e:
        print(f">>> /execute handler FAILED for {draft_id}: {repr(e)}", flush=True)
        return RedirectResponse("/emails?ok=error", status_code=303)

def send_and_mark_task(draft_id: str):
    print(f">>> WORKER START: Processing {draft_id}", flush=True)
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if not row:
            print(">>> WORKER ERROR: Row not found", flush=True); return
        data = json.loads(row["content"])
        payload = data.get("payload", {})
        to_addr = (data.get("sender_email") or extract_sender_address(payload) or "").strip()
        if not to_addr: raise RuntimeError("Cannot determine recipient email address (no Reply-To / From)")
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
        try: conn.execute("UPDATE drafts SET status='failed' WHERE id=?", (draft_id,))
        except: pass
        set_draft_error(draft_id, str(e))
    finally:
        conn.close()

# ==============================================================================
# IMAP Poller (unchanged)
# ==============================================================================
IMAP_EMAIL = os.getenv("IMAP_EMAIL")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:8000/webhook")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 30))

def _html_to_text(html: str) -> str:
    txt = unescape(html)
    txt = re.sub(r"(?is)\<(script|style).*?\>.*?\</\\1\>", "", txt)
    txt = re.sub(r"(?is)\<br\s*/?\>", "\n", txt)
    txt = re.sub(r"(?is)\</p\>", "\n\n", txt)
    txt = re.sub(r"(?is)\<[^>]+\>", "", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

def imap_connect():
    print(">>> IMAP: connecting", {"host": IMAP_HOST, "port": IMAP_PORT, "email": IMAP_EMAIL})
    M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try: M.login(IMAP_EMAIL, IMAP_PASSWORD)
    except imaplib.IMAP4.error as e:
        print(">>> IMAP LOGIN FAILED:", repr(e)); raise
    print(">>> IMAP: login OK"); return M

def imap_fetch_unread(include_html_fallback=True):
    M = imap_connect()
    status, _ = M.select("INBOX"); print(">>> IMAP: SELECT INBOX =", status)
    if status != "OK": M.logout(); return []
    try:
        status, data = M.uid("SEARCH", None, '(X-GM-RAW "is:unread")'); print(">>> IMAP: UID SEARCH =", status, "count:", len((data[0] or b"").split()))
    except Exception as e:
        print(">>> IMAP: UID SEARCH error; fallback to UNSEEN:", e); status, data = M.search(None, "UNSEEN"); print(">>> IMAP: SEARCH UNSEEN =", status, "count:", len((data[0] or b"").split()))
    if status != "OK": M.close(); M.logout(); return []
    ids = (data[0] or b"").split(); messages = []
    for uid in ids:
        status, msg_data = M.uid("FETCH", uid, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]: print(">>> IMAP: FETCH failed for", uid); continue
        msg = email.message_from_bytes(msg_data[0][1])
        subject = msg.get("Subject", ""); sender = msg.get("From", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type(); disp = (part.get("Content-Disposition", "") or "")
                if ctype == "text/plain" and "attachment" not in disp.lower():
                    try: body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore"); break
                    except Exception: pass
            if not body and include_html_fallback:
                for part in msg.walk():
                    ctype = part.get_content_type(); disp = (part.get("Content-Disposition", "") or "")
                    if ctype == "text/html" and "attachment" not in disp.lower():
                        try: html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore"); body = _html_to_text(html); break
                        except Exception: pass
        else:
            raw = msg.get_payload(decode=True) or b""
            try: text = raw.decode(msg.get_content_charset() or "utf-8", errors="ignore")
            except Exception: text = raw.decode("utf-8", errors="ignore")
            body = _html_to_text(text) if msg.get_content_type() == "text/html" else text

        payload = {
            "source": "gmail_imap",
            "from": sender,
            "subject": subject,
            "body": body,
            "message_id": msg.get("Message-ID"),
            "references": msg.get("References"),
            "reply_to": msg.get("Reply-To"),
        }
        print(">>> IMAP: fetched message:", {k: (v[:140] + "…") if isinstance(v, str) and len(v) > 140 else v for k, v in payload.items()})
        try: M.uid("STORE", uid, "+FLAGS", "\\Seen"); print(">>> IMAP: marked seen", uid)
        except Exception as e: print(">>> IMAP: mark seen failed:", e)
        messages.append(payload)
    try: M.close()
    finally: M.logout()
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
# SMTP sending (unchanged)
# ==============================================================================
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "").lower() in ("1", "true", "yes")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "DEAP")

def send_email_smtp(to_addr: str, subject: str, body: str, in_reply_to=None, refs=None):
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD: raise RuntimeError("Missing SMTP_EMAIL or SMTP_APP_PASSWORD environment variables")
    if not to_addr: raise ValueError("Recipient address is empty")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_EMAIL}>"
    msg["To"] = to_addr
    if in_reply_to: msg["In-Reply-To"] = in_reply_to
    if refs: msg["References"] = refs
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
                server.starttls(context=context); server.ehlo()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.send_message(msg)
    print(">>> SMTP: sent OK", flush=True)
    return True

# ==============================================================================
# Debug: routes list
# ==============================================================================
print(">>> ROUTES LOADED:")
for r in app.routes:
    print(" ", r.path, getattr(r, "methods", None))