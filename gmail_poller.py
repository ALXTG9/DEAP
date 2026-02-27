import os
import time
import email
import imaplib
import requests

IMAP_EMAIL = os.getenv("IMAP_EMAIL")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 30))


def connect():
    M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    M.login(IMAP_EMAIL, IMAP_PASSWORD)
    return M


def fetch_unread_messages():
    M = connect()
    M.select("INBOX")

    status, data = M.search(None, "UNSEEN")

    if status != "OK":
        M.close()
        M.logout()
        return []

    message_ids = data[0].split()
    messages = []

    for msg_id in message_ids:
        status, msg_data = M.fetch(msg_id, "(RFC822)")
        if status != "OK":
            continue

        msg = email.message_from_bytes(msg_data[0][1])

        subject = msg.get("Subject", "")
        sender = msg.get("From", "")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        messages.append({
            "source": "gmail_imap",
            "from": sender,
            "subject": subject,
            "body": body,
        })

        # Mark as read
        M.store(msg_id, "+FLAGS", "\\Seen")

    M.close()
    M.logout()

    return messages


def main():
    print("Starting IMAP poller...")
    while True:
        try:
            messages = fetch_unread_messages()
            for msg in messages:
                print("Forwarding email to webhook:", msg)
                try:
                    r = requests.post(WEBHOOK_URL, json=msg, timeout=10)
                    print("Webhook response:", r.status_code, r.text)
                except Exception as e:
                    print("Webhook error:", e)

        except Exception as e:
            print("Poller error:", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()