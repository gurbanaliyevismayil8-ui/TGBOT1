# db.py
import os
import psycopg2
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

# DATABASE_URL = "postgres://ucrafht26k0hq1:p0143fd9285aa8ad7401e2055090349af87a237301813828d8d9c6b05ea5658fc@c2ath2egdsh9dm.cluster-czz5s0kz4scl.eu-west-1.rds.amazonaws.com:5432/dvap90udaqrcv"
DATABASE_URL = os.environ["DATABASE_URL"]

@contextmanager
def get_conn():
	conn = psycopg2.connect(DATABASE_URL, sslmode="require")
	try:
		yield conn
		conn.commit()
	except Exception:
		conn.rollback()
		raise
	finally:
		conn.close()


# ======================
# USERS
# ======================
def ensure_user(chat_id: int) -> None:
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO users (chat_id, first_seen_at)
				VALUES (%s, %s)
				ON CONFLICT (chat_id) DO NOTHING
				""",
				(chat_id, datetime.now(timezone.utc)),
			)


def get_all_users():
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT chat_id FROM users ORDER BY first_seen_at DESC")
			return [row[0] for row in cur.fetchall()]


def get_stats() -> dict:
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT COUNT(*) FROM users")
			users = int(cur.fetchone()[0])

			cur.execute("SELECT COUNT(*) FROM subscriptions")
			subs = int(cur.fetchone()[0])

			cur.execute("SELECT COUNT(*) FROM user_access WHERE expires_at > now()")
			paid_active = int(cur.fetchone()[0])

	return {"users": users, "subs": subs, "paid_active": paid_active}


# ======================
# SUBSCRIPTIONS (таблица: subscriptions(chat_id, query, created_at), PK(chat_id,query))
# ======================
def add_subscription(chat_id: int, query: str) -> bool:
	query = query.strip().lower()

	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT 1 FROM subscriptions WHERE chat_id=%s AND query=%s",
				(chat_id, query),
			)
			if cur.fetchone():
				return False

			cur.execute(
				"""
				INSERT INTO subscriptions (chat_id, query, created_at)
				VALUES (%s, %s, %s)
				""",
				(chat_id, query, datetime.now(timezone.utc)),
			)
			return True


def remove_subscription(chat_id: int, query: str) -> int:
	query = query.strip().lower()

	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT 1 FROM subscriptions WHERE chat_id=%s AND query=%s",
				(chat_id, query),
			)
			if not cur.fetchone():
				return 0

			cur.execute(
				"DELETE FROM subscription_seen WHERE chat_id=%s AND query=%s",
				(chat_id, query),
			)
			cur.execute(
				"DELETE FROM subscriptions WHERE chat_id=%s AND query=%s",
				(chat_id, query),
			)
			return 1

def list_company_codes_filtered(status: str = "all", limit: int = 50):
    """
    status: all | used | unused | expired | active | inactive
    expired = code has users, but none of them are currently active (expires_at <= now())
    """
    status = (status or "all").strip().lower()
    limit = max(1, min(int(limit or 50), 200))

    base_sql = """
        SELECT
            cc.code,
            cc.company_name,
            cc.max_users,
            cc.duration_days,
            cc.is_active,
            cc.created_at,
            COUNT(ua.chat_id) AS total_users,
            COUNT(*) FILTER (WHERE ua.expires_at > now()) AS active_users,
            MAX(ua.expires_at) AS latest_expires
        FROM company_codes cc
        LEFT JOIN user_access ua ON ua.code = cc.code
        GROUP BY cc.code, cc.company_name, cc.max_users, cc.duration_days, cc.is_active, cc.created_at
    """

    where_having = ""
    if status == "used":
        where_having = " HAVING COUNT(ua.chat_id) > 0 "
    elif status == "unused":
        where_having = " HAVING COUNT(ua.chat_id) = 0 "
    elif status == "expired":
        where_having = " HAVING COUNT(ua.chat_id) > 0 AND COUNT(*) FILTER (WHERE ua.expires_at > now()) = 0 "
    elif status == "active":
        where_having = " HAVING cc.is_active = TRUE "
    elif status == "inactive":
        where_having = " HAVING cc.is_active = FALSE "

    tail_sql = " ORDER BY cc.created_at DESC LIMIT %s"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(base_sql + where_having + tail_sql, (limit,))
            return cur.fetchall()


def list_subscriptions(chat_id: int):
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT query
				FROM subscriptions
				WHERE chat_id=%s
				ORDER BY created_at DESC
				""",
				(chat_id,),
			)
			return [row[0] for row in cur.fetchall()]


def get_all_subscriptions():
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT chat_id, query FROM subscriptions")
			return cur.fetchall()  # [(chat_id, query), ...]


# ======================
# SEEN EVENTS (таблица: subscription_seen(chat_id, query, event_id))
# ======================
def sub_has_seen(chat_id: int, query: str, event_id: str) -> bool:
	query = query.strip().lower()
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				SELECT 1
				FROM subscription_seen
				WHERE chat_id=%s AND query=%s AND event_id=%s
				""",
				(chat_id, query, event_id),
			)
			return cur.fetchone() is not None


def mark_sub_seen(chat_id: int, query: str, event_id: str) -> None:
    query = query.strip().lower()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO subscription_seen (chat_id, query, event_id, seen_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chat_id, query, event_id) DO NOTHING
                """,
                (chat_id, query, event_id, datetime.now(timezone.utc)),
            )

# def mark_sub_seen(chat_id: int, query: str, event_id: str) -> None:
# 	query = query.strip().lower()
# 	with get_conn() as conn:
# 		with conn.cursor() as cur:
# 			cur.execute(
# 				"""
# 				INSERT INTO subscription_seen (chat_id, query, event_id)
# 				VALUES (%s, %s, %s)
# 				ON CONFLICT DO NOTHING
# 				""",
# 				(chat_id, query, event_id),
# 			)


# ======================
# PAID ACCESS (company codes)
# ======================

def create_company_code(code: str, company_name: str, max_users: int = 3, duration_days: int = 30) -> None:
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO company_codes(code, company_name, max_users, duration_days, created_at, is_active)
				VALUES (%s, %s, %s, %s, %s, TRUE)
				ON CONFLICT (code) DO UPDATE SET
				  company_name=EXCLUDED.company_name,
				  max_users=EXCLUDED.max_users,
				  duration_days=EXCLUDED.duration_days,
				  is_active=TRUE
				""",
				(code, company_name, max_users, duration_days, datetime.now(timezone.utc)),
			)

# def create_company_code(code: str, company_name: str, max_users: int = 3) -> None:
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 """
#                 INSERT INTO company_codes(code, company_name, max_users, created_at, is_active)
#                 VALUES (%s, %s, %s, %s, TRUE)
#                 ON CONFLICT (code) DO UPDATE SET
#                   company_name=EXCLUDED.company_name,
#                   max_users=EXCLUDED.max_users,
#                   is_active=TRUE
#                 """,
#                 (code, company_name, max_users, datetime.now(timezone.utc)),
#             )


def deactivate_code(code: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            # 1) disable code
            cur.execute("UPDATE company_codes SET is_active=FALSE WHERE code=%s", (code,))
            # 2) kick everyone who used it (expire access immediately)
            cur.execute(
                "UPDATE user_access SET expires_at = now() WHERE code=%s AND expires_at > now()",
                (code,)
            )
def code_info(code: str):
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT code, company_name, max_users, duration_days, is_active FROM company_codes WHERE code=%s",
				(code,),
			)
			return cur.fetchone()


# def code_info(code: str):
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 "SELECT code, company_name, max_users, is_active FROM company_codes WHERE code=%s",
#                 (code,),
#             )
#             return cur.fetchone()


def code_usage_count(code: str) -> int:
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute("SELECT COUNT(*) FROM user_access WHERE code=%s", (code,))
			return int(cur.fetchone()[0])


def get_user_access(chat_id: int):
	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"SELECT code, activated_at, expires_at FROM user_access WHERE chat_id=%s",
				(chat_id,),
			)
			return cur.fetchone()


def is_paid_active(chat_id: int) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ua.expires_at, cc.is_active
                FROM user_access ua
                JOIN company_codes cc ON cc.code = ua.code
                WHERE ua.chat_id=%s
                """,
                (chat_id,),
            )
            row = cur.fetchone()

    if not row:
        return False

    expires_at, code_active = row

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    return bool(code_active) and (expires_at > datetime.now(timezone.utc))



def redeem_code(chat_id: int, code: str, duration_days: int = None) -> tuple[bool, str, datetime | None]:
	code = (code or "").strip()
	if not code:
		return False, "Kod boşdur.", None

	info = code_info(code)
	if not info:
		return False, "Bu kod tapılmadı.", None

	if len(info) == 4:  # Старый формат без duration_days
		_, company_name, max_users, is_active = info
		duration_days = duration_days or 30  # Значение по умолчанию
	else:
		_, company_name, max_users, duration_days_db, is_active = info
		duration_days = duration_days or duration_days_db or 30

	if not is_active:
		return False, "Bu kod deaktiv edilib.", None

	used = code_usage_count(code)

	current = get_user_access(chat_id)
	if not (current and current[0] == code):
		if used >= int(max_users):
			return False, f"Bu kodun limitinə çatılıb ({used}/{max_users}).", None

	now = datetime.now(timezone.utc)

	if current:
		_, _, cur_exp = current
		if cur_exp.tzinfo is None:
			cur_exp = cur_exp.replace(tzinfo=timezone.utc)
		base = cur_exp if cur_exp > now else now
	else:
		base = now

	new_expires = base + timedelta(days=duration_days)

	with get_conn() as conn:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO user_access(chat_id, code, activated_at, expires_at)
				VALUES (%s, %s, %s, %s)
				ON CONFLICT (chat_id) DO UPDATE SET
				  code=EXCLUDED.code,
				  activated_at=EXCLUDED.activated_at,
				  expires_at=EXCLUDED.expires_at
				""",
				(chat_id, code, now, new_expires),
			)

	return True, f"✅ Aktiv edildi: {company_name}", new_expires

# def redeem_code(chat_id: int, code: str, duration_days: int = 30) -> tuple[bool, str, datetime | None]:
#     code = (code or "").strip()
#     if not code:
#         return False, "Kod boşdur.", None

#     info = code_info(code)
#     if not info:
#         return False, "Bu kod tapılmadı.", None

#     _, company_name, max_users, is_active = info
#     if not is_active:
#         return False, "Bu kod deaktiv edilib.", None

#     used = code_usage_count(code)

#     current = get_user_access(chat_id)
#     # Если юзер уже на этом коде — не считаем как новый слот
#     if not (current and current[0] == code):
#         if used >= int(max_users):
#             return False, f"Bu kodun limitinə çatılıb ({used}/{max_users}).", None

#     now = datetime.now(timezone.utc)

#     # продление: 30 дней от max(now, expires_at)
#     if current:
#         _, _, cur_exp = current
#         if cur_exp.tzinfo is None:
#             cur_exp = cur_exp.replace(tzinfo=timezone.utc)
#         base = cur_exp if cur_exp > now else now
#     else:
#         base = now

#     new_expires = base + timedelta(days=duration_days)

#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 """
#                 INSERT INTO user_access(chat_id, code, activated_at, expires_at)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (chat_id) DO UPDATE SET
#                   code=EXCLUDED.code,
#                   activated_at=EXCLUDED.activated_at,
#                   expires_at=EXCLUDED.expires_at
#                 """,
#                 (chat_id, code, now, new_expires),
#             )

#     return True, f"✅ Aktiv edildi: {company_name}", new_expires