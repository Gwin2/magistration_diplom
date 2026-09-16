from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

SESSION_COOKIE = "nn_trainer_session"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
PASSWORD_MIN_LENGTH = 8
_USERNAME_PATTERN = re.compile(r"^[a-z0-9_.-]{3,64}$")
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


def normalize_username(value: str) -> str:
    username = str(value or "").strip().lower()
    if not _USERNAME_PATTERN.fullmatch(username):
        raise ValueError("Логин: 3–64 символа a-z, 0-9, '.', '_' или '-'.")
    return username


def validate_password(value: str) -> str:
    password = str(value or "")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Пароль должен содержать минимум {PASSWORD_MIN_LENGTH} символов.")
    if len(password) > 256:
        raise ValueError("Пароль не должен быть длиннее 256 символов.")
    return password


def hash_password(password: str) -> str:
    password = validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    )
    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii")

    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${encode(salt)}${encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, raw_n, raw_r, raw_p, raw_salt, raw_digest = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(raw_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(raw_digest.encode("ascii"))
        actual = hashlib.scrypt(
            str(password).encode("utf-8"),
            salt=salt,
            n=int(raw_n),
            r=int(raw_r),
            p=int(raw_p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError, base64.binascii.Error):
        return False


class AuthStore:
    def __init__(self, state_dir: Path) -> None:
        self.users_file = state_dir / "users.json"
        self.session_ttl = max(300, int(os.environ.get("UAV_CONTROL_SESSION_TTL", "28800")))
        self._lock = threading.RLock()
        self._sessions: dict[str, tuple[str, float]] = {}
        self._users = self._load_users()
        if not self._users:
            self._users = {
                normalize_username(
                    os.environ.get("UAV_CONTROL_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)
                ): {
                    "role": "admin",
                    "password_hash": hash_password(
                        os.environ.get("UAV_CONTROL_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
                    ),
                    "created_at": int(time.time()),
                }
            }
            self._save_users()

    def _load_users(self) -> dict[str, dict[str, Any]]:
        if not self.users_file.exists():
            return {}
        try:
            payload = json.loads(self.users_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        rows = payload.get("users", {}) if isinstance(payload, dict) else {}
        if not isinstance(rows, dict):
            return {}
        users: dict[str, dict[str, Any]] = {}
        for raw_username, raw_user in rows.items():
            if not isinstance(raw_user, dict):
                continue
            try:
                username = normalize_username(raw_username)
            except ValueError:
                continue
            role = raw_user.get("role")
            password_hash = raw_user.get("password_hash")
            if role not in {"admin", "user"} or not isinstance(password_hash, str):
                continue
            users[username] = {
                "role": role,
                "password_hash": password_hash,
                "created_at": raw_user.get("created_at", int(time.time())),
            }
        return users

    def _save_users(self) -> None:
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.users_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "users": self._users}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.users_file)

    @staticmethod
    def _public_user(username: str, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": username,
            "role": record["role"],
            "created_at": record.get("created_at"),
        }

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        try:
            username = normalize_username(username)
        except ValueError:
            return None
        with self._lock:
            record = self._users.get(username)
            if not record or not verify_password(password, record["password_hash"]):
                return None
            return self._public_user(username, record)

    def create_session(self, username: str) -> str:
        with self._lock:
            if username not in self._users:
                raise ValueError("Пользователь не найден.")
            token = secrets.token_urlsafe(32)
            self._sessions[token] = (username, time.time() + self.session_ttl)
            return token

    def current_user(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            username, expires_at = session
            if expires_at <= time.time() or username not in self._users:
                self._sessions.pop(token, None)
                return None
            return self._public_user(username, self._users[username])

    def revoke_session(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._sessions.pop(token, None)

    def revoke_user_sessions(self, username: str) -> None:
        with self._lock:
            for token, (session_username, _expires_at) in list(self._sessions.items()):
                if session_username == username:
                    self._sessions.pop(token, None)

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._public_user(username, self._users[username])
                for username in sorted(self._users)
            ]

    def create_user(self, username: str, password: str, role: str) -> dict[str, Any]:
        username = normalize_username(username)
        password = validate_password(password)
        if role not in {"admin", "user"}:
            raise ValueError("Роль должна быть admin или user.")
        with self._lock:
            if username in self._users:
                raise ValueError("Пользователь уже существует.")
            self._users[username] = {
                "role": role,
                "password_hash": hash_password(password),
                "created_at": int(time.time()),
            }
            self._save_users()
            return self._public_user(username, self._users[username])

    def update_user(
        self,
        username: str,
        new_username: str | None = None,
        password: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        username = normalize_username(username)
        new_username = normalize_username(new_username) if new_username else username
        if password is not None:
            password = validate_password(password)
        if role is not None and role not in {"admin", "user"}:
            raise ValueError("Роль должна быть admin или user.")
        with self._lock:
            record = self._users.get(username)
            if not record:
                raise ValueError("Пользователь не найден.")
            if new_username != username and new_username in self._users:
                raise ValueError("Новый логин уже занят.")
            next_role = role or record["role"]
            if record["role"] == "admin" and next_role == "user" and self._admin_count() <= 1:
                raise ValueError("Нельзя убрать роль последнего администратора.")
            updated = {
                **record,
                "role": next_role,
                "password_hash": (
                    hash_password(password) if password is not None else record["password_hash"]
                ),
            }
            if new_username != username:
                self._users.pop(username)
                self._users[new_username] = updated
                for token, (session_username, expires_at) in list(self._sessions.items()):
                    if session_username == username:
                        self._sessions[token] = (new_username, expires_at)
            else:
                self._users[username] = updated
            self._save_users()
            return self._public_user(new_username, updated)

    def delete_user(self, username: str) -> None:
        username = normalize_username(username)
        with self._lock:
            record = self._users.get(username)
            if not record:
                raise ValueError("Пользователь не найден.")
            if record["role"] == "admin" and self._admin_count() <= 1:
                raise ValueError("Нельзя удалить последнего администратора.")
            self._users.pop(username)
            self.revoke_user_sessions(username)
            self._save_users()

    def change_password(self, username: str, current_password: str, new_password: str) -> None:
        username = normalize_username(username)
        new_password = validate_password(new_password)
        with self._lock:
            record = self._users.get(username)
            if not record or not verify_password(current_password, record["password_hash"]):
                raise ValueError("Текущий пароль неверен.")
            record["password_hash"] = hash_password(new_password)
            self._save_users()
            self.revoke_user_sessions(username)

    def _admin_count(self) -> int:
        return sum(record["role"] == "admin" for record in self._users.values())
