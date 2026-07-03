import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ChatStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
            """)

    def list_chats(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, model_key, created_at, updated_at "
                "FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_chat(self, model_key):
        now = _now()
        chat = {
            "id": str(uuid.uuid4()),
            "title": "New chat",
            "model_key": model_key,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats (id, title, model_key, created_at, updated_at) "
                "VALUES (:id, :title, :model_key, :created_at, :updated_at)",
                chat,
            )
        return chat

    def get_chat(self, chat_id):
        with self._connect() as conn:
            chat = conn.execute(
                "SELECT id, title, model_key, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
            if chat is None:
                raise KeyError(chat_id)
            messages = conn.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        out = dict(chat)
        out["messages"] = [dict(row) for row in messages]
        return out

    def update_chat(self, chat_id, model_key=None, title=None):
        chat = self.get_chat(chat_id)
        model_key = model_key or chat["model_key"]
        title = title or chat["title"]
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET model_key = ?, title = ?, updated_at = ? WHERE id = ?",
                (model_key, title, now, chat_id),
            )
        return self.get_chat(chat_id)

    def delete_chat(self, chat_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def add_message(self, chat_id, role, content):
        content = (content or "").strip()
        if not content:
            raise ValueError("message content is required")
        chat = self.get_chat(chat_id)
        now = _now()
        title = chat["title"]
        if role == "user" and title == "New chat":
            title = content.replace("\n", " ")[:48]
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, chat_id),
            )
        return {
            "id": cur.lastrowid,
            "role": role,
            "content": content,
            "created_at": now,
        }


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
