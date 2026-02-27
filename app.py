#!/usr/bin/env python3
import os, sqlite3, uuid, json, threading, time, re, ssl, smtplib, email, imaplib, requests
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from email.message import EmailMessage
from email.utils import parseaddr
from html import escape, unescape

# =========================================================
#  CONFIG
# =========================================================
DB_PATH = "drafts.db"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "DEAP")

IMAP_EMAIL = os.getenv("IMAP_EMAIL")
IMAP_PASS = os.getenv("IMAP_PASSWORD")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:8000/webhook")

# =========================================================
#  APP INIT
# =========================================================
app = FastAPI(title="Email Automation Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
#  DATABASE
# =========================================================
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

# =========================================================
#  AI PIPELINE (MISTRAL)
# =========================================================
def run_ai(text: str) -> dict:
    """Return structured JSON from Mistral, fallback to defaults."""
    if not MISTRAL_API_KEY:
        return {
            "classification": "general_query",
            "summary": text[:150],
            "draft_reply": "Thanks — we'll get back to you shortly."
        }

    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": (
                        "You are a business assistant.\n"
                        "Return JSON only with keys: "
                        "classification, summary, draft_reply."
                    )},
                    {"role": "user", "content": text}
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=20
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except:
        return {
            "classification": "general_query",
            "summary": text[:150],
            "draft_reply": "Thanks — we'll respond soon."
        }

# =========================================================
#  HELPERS
# =========================================================
def extract_sender(payload):
    _, addr = parseaddr(payload.get("from") or payload.get("reply_to") or "")
    return addr

def save_draft(payload):
    text = payload.get("body") or payload.get("text") or json.dumps(payload)
    ai = run_ai(text)
    sender = extract_sender(payload)

    enriched = {
        "raw_text": text,
        "classification": ai["classification"],
        "summary": ai["summary"],
        "draft_reply": ai["draft_reply"],
        "payload": payload,
        "sender": sender
    }

    draft_id = str(uuid.uuid4())
    conn = db()
    conn.execute(
        "INSERT INTO drafts (id, content) VALUES (?, ?)",
        (draft_id, json.dumps(enriched))
    )
    conn.commit()
    conn.close()
    return draft_id

# =========================================================
#  HTML TEMPLATES (CLEAN)
# =========================================================
def page(title, body):
    return HTMLResponse(f"""
    <!doctype html>
    <html><head>
    <title>{escape(title)}</title>
    <meta charset="utf-8" />
    <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50">
    <div class="max-w-3xl mx-auto p-6">
      {body}
    </div>
    </body></html>
    """)

# =========================================================
#  ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return page("Home", """
    <h1 class="text-2xl font-bold">Dashboard</h1>
    <a href="/emails" class="mt-4 inline-block bg-blue-600 text-white px-4 py-2 rounded">
      Emails
    </a>
    """)

@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()
    draft_id = save_draft(payload)
    return {"ok": True, "draft_id": draft_id}

@app.get("/emails", response_class=HTMLResponse)
def list_emails():
    conn = db()
    rows = conn.execute("SELECT id, content, status FROM drafts ORDER BY rowid DESC").fetchall()
    conn.close()

    items = []
    for r in rows:
        data = json.loads(r["content"])
        items.append(f"""
        <div class="bg-white shadow p-4 rounded mb-4">
            <div class="flex justify-between">
                <div><b>{escape(data['summary'])}</b></div>
                <span class="text-sm">{r['status']}</span>
            </div>
            <pre class="text-sm bg-gray-100 p-2 rounded mt-2">
{escape(data['draft_reply'])}
            </pre>
            <a href="/edit/{r['id']}" class="text-blue-600 text-sm">Edit</a>
            |
            <form action="/approve/{r['id']}" method="post" style="display:inline">
                <button class="text-green-700 text-sm">Approve</button>
            </form>
        </div>
        """)

    return page("Emails", "<h1>Emails</h1>" + "".join(items))

@app.get("/edit/{draft_id}", response_class=HTMLResponse)
def edit_form(draft_id: str):
    conn = db()
    row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
    conn.close()

    if not row:
        return page("Not found", "<h1>Not found</h1>")

    data = json.loads(row["content"])
    reply = escape(data["draft_reply"])

    return page("Edit", f"""
    <h1>Edit Draft</h1>
    <form action="/edit/{draft_id}" method="post">
      <textarea name="draft_reply" class="w-full h-64 border p-2">{reply}</textarea>
      <button class="bg-blue-600 text-white px-4 py-2 mt-4 rounded">Save</button>
    </form>
    """)

@app.post("/edit/{draft_id}")
def edit_submit(draft_id: str, draft_reply: str = Form(...)):
    conn = db()
    row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
    if not row:
        return RedirectResponse("/emails", 303)

    data = json.loads(row["content"])
    data["draft_reply"] = draft_reply

    conn.execute("UPDATE drafts SET content=? WHERE id=?", (json.dumps(data), draft_id))
    conn.commit()
    conn.close()

    return RedirectResponse("/emails", 303)

@app.post("/approve/{draft_id}")
def approve(draft_id):
    conn = db()
    conn.execute("UPDATE drafts SET status='approved' WHERE id=?", (draft_id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/emails", 303)

# =========================================================
#  SENDING EMAILS
# =========================================================
def send_email(to_addr, subject, body):
    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_EMAIL}>"
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(SMTP_EMAIL, SMTP_PASS)
        s.send_message(msg)

@app.post("/execute/{draft_id}")
def execute_email(draft_id: str, tasks: BackgroundTasks):
    tasks.add_task(_send_worker, draft_id)
    return RedirectResponse("/emails", 303)

def _send_worker(draft_id: str):
    conn = db()
    row = conn.execute("SELECT content FROM drafts WHERE id=?", (draft_id,)).fetchone()
    if not row:
        return
    data = json.loads(row["content"])

    to = data.get("sender")
    subject = "Re: " + (data["payload"].get("subject") or "Your inquiry")
    body = data["draft_reply"]

    try:
        send_email(to, subject, body)
        conn.execute("UPDATE drafts SET status='executed' WHERE id=?", (draft_id,))
    except Exception as e:
        conn.execute("UPDATE drafts SET status='failed' WHERE id=?", (draft_id,))
    conn.commit()
    conn.close()

# =========================================================
#  OPTIONAL IMAP POLLER
# =========================================================
def start_imap_poller():
    if not IMAP_EMAIL or not IMAP_PASS:
        return

    def loop():
        while True:
            try:
                M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
                M.login(IMAP_EMAIL, IMAP_PASS)
                M.select("INBOX")
                typ, data = M.search(None, "UNSEEN")
                for uid in data[0].split():
                    typ, msgdata = M.fetch(uid, "(RFC822)")
                    msg = email.message_from_bytes(msgdata[0][1])
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                    payload = {
                        "from": msg.get("From"),
                        "subject": msg.get("Subject"),
                        "body": body
                    }
                    requests.post(WEBHOOK_URL, json=payload, timeout=5)
                    M.store(uid, "+FLAGS", "\\Seen")
                M.logout()
            except:
                pass
            time.sleep(30)

    threading.Thread(target=loop, daemon=True).start()

start_imap_poller()