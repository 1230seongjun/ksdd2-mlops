from hashlib import sha256
from secrets import token_urlsafe
from pwdlib import PasswordHash


_password_hasher = PasswordHash.recommended()
_dummy_password_hash = _password_hasher.hash(token_urlsafe(32))

def register_worker(conn, invite_token: str, password_hash: str) -> int | None:
    token_hash = sha256(invite_token.encode("utf-8")).hexdigest()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE invitations
                SET used_at = now()
                WHERE token_hash = %s
                  AND used_at IS NULL
                  AND revoked_at IS NULL
                  AND expires_at > now()
                RETURNING email
                """,
                (token_hash,),
            )
            invitation = cur.fetchone()
            if invitation is None:
                return None

            email = invitation[0]
            cur.execute(
                """
                INSERT INTO users (email, password_hash, role)
                VALUES (%s, %s, 'worker')
                RETURNING id
                """,
                (email, password_hash),
            )
            user_id = cur.fetchone()[0]

    return user_id

def create_invitation(conn, admin_id: int, email: str) -> str | None:
    token = token_urlsafe(32)
    token_hash = sha256(token.encode("utf-8")).hexdigest()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO invitations (token_hash, email, created_by, expires_at)
                SELECT %s, %s, id, now() + interval '24 hours'
                FROM users
                WHERE id = %s AND role = 'admin'
                RETURNING token_hash
                """,
                (token_hash, email, admin_id),
            )
            if cur.fetchone() is None:
                return None

    return token

def create_session(conn, user_id: int) -> str:
    token = token_urlsafe(32)
    token_hash = sha256(token.encode("utf-8")).hexdigest()

    with conn.transaction():
        conn.execute(
            """
            INSERT INTO sessions (token_hash, user_id, expires_at)
            VALUES (%s, %s, now() + interval '8 hours')
            """,
            (token_hash, user_id),
        )

    return token


def get_session_user(conn, session_token: str) -> tuple[int, str] | None:
    token_hash = sha256(session_token.encode("utf-8")).hexdigest()

    with conn.transaction():
        return conn.execute(
            """
            SELECT users.id, users.role
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = %s
              AND sessions.expires_at > now()
            """,
            (token_hash,),
        ).fetchone()

def revoke_session(conn, session_token: str) -> None:
    token_hash = sha256(session_token.encode("utf-8")).hexdigest()

    with conn.transaction():
        conn.execute(
            "DELETE FROM sessions WHERE token_hash = %s",
            (token_hash,),
        )

def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def authenticate_user(conn, email: str, password: str) -> int | None:
    with conn.transaction():
        account = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = %s",
            (email,),
        ).fetchone()

    if account is None:
        _password_hasher.verify(password, _dummy_password_hash)
        return None

    user_id, stored_hash = account
    if not _password_hasher.verify(password, stored_hash):
        return None

    return user_id
