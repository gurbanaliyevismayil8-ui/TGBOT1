# main.py
import os
import re
import unicodedata
import asyncio
import secrets
import string
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple, Set
from telegram.ext import MessageHandler, filters

import requests
from dotenv import load_dotenv

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from db import (
    ensure_user,
    add_subscription,
    remove_subscription,
    list_subscriptions,
    get_all_subscriptions,
    sub_has_seen,
    mark_sub_seen,
    get_all_users,
    get_stats,
    # paid
    is_paid_active,
    redeem_code,
    get_user_access,
    create_company_code,
    deactivate_code,
    code_info,
    code_usage_count,
)

# ======================
# CONFIG
# ======================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise SystemExit("❌ BOT_TOKEN boşdur. .env faylında BOT_TOKEN yaz.")

ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0") or "0")
DEFAULT_MAX_USERS = int(os.getenv("DEFAULT_MAX_USERS", "3") or "3")
ACCESS_DAYS = int(os.getenv("ACCESS_DAYS", "30") or "30")

def is_admin(chat_id: int) -> bool:
    return ADMIN_CHAT_ID != 0 and chat_id == ADMIN_CHAT_ID

ACTIVE_ONLY = os.getenv("ACTIVE_ONLY", "1").strip() == "1"
DAYS_BACK = int(os.getenv("DAYS_BACK", "60"))

SMART_PAGES = int(os.getenv("SMART_PAGES", "9"))
SMART_PAGE_SIZE = int(os.getenv("SMART_PAGE_SIZE", "25"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
LIST_TTL_SEC = int(os.getenv("LIST_TTL_SEC", "180"))  # cache list pages (seconds)
CAND_TTL_SEC = int(os.getenv("CAND_TTL_SEC", "60"))   # cache merged candidates (seconds)
MAX_DETAIL_CHECK = int(os.getenv("MAX_DETAIL_CHECK", "250"))  # safety cap when query needs detail

DEBUG_MATCH = os.getenv("DEBUG_MATCH", "1").strip() == "1"  # можно 0 чтобы выключить

RESULT_LIMIT = int(os.getenv("RESULT_LIMIT", "6"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "20000"))

DEFAULT_EVENT_TYPE = int(os.getenv("DEFAULT_EVENT_TYPE", "2"))
DEFAULT_EVENT_STATUS = int(os.getenv("DEFAULT_EVENT_STATUS", "1"))

BASE_API = "https://etender.gov.az/api/events"
DETAIL_BASE = "https://etender.gov.az/main/competition/detail/"
BOT_STARTED_AT = datetime.now(timezone.utc)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Accept-Language": "az,en;q=0.8,ru;q=0.7",
}

# --- HTTP session (keep-alive) ---
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

async def async_get(url: str, *, params=None, headers=None, timeout: float = REQUEST_TIMEOUT):
    """Run blocking requests.get in a thread to avoid blocking the asyncio event loop."""
    def _do():
        h = headers or HEADERS
        return SESSION.get(url, params=params, headers=h, timeout=timeout)
    return await asyncio.to_thread(_do)


PAYWALL_AZ = (
    "🔒 *Bu bot yalnız ödənişli girişlə işləyir.*\n\n"
    "Şirkətinizdən aldığınız kodu aktiv edin:\n"
    "`/redeem KOD`\n\n"
    "Məsələn:\n"
    "`/redeem ABCD-1234-EF`\n"
    "Suallar - @IsmayilGurbanaliyev"
)

def user_commands_text(is_paid: bool) -> str:
    if is_paid:
        return (
            "📌 *Komandalar:*\n"
            "• `/search söz (tender saytinda axtariw(agilli axtariw))`\n"
            "• `/subscribe söz ( her 6 saat tender axtarir ve tapsa yolluyur`\n"
            "• `/unsubscribe söz`\n"
            "• `/subs (abune olan tenderleri gorsedir)`\n"

        )
    return (
        "🔒 *Giriş yoxdur.*\n"
        "Kodu aktiv et:\n"
        "• `/redeem KOD`\n"
        "\n"
        "Məsələn:\n"
        "• `/redeem ABCD-1234-EF`\n"
    )

def require_paid(chat_id: int) -> bool:
    if is_admin(chat_id):
        return True
    return is_paid_active(chat_id)

async def send_paywall(update: Update):
    await update.message.reply_text(PAYWALL_AZ, parse_mode=ParseMode.MARKDOWN)

# ======================
# SMART MATCH
# ======================
AZ_DIACRITICS_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o",
    "ə": "e", "Ə": "e",
    "ı": "i", "İ": "i",
})

PLAIN_TO_AZ = {
    "c": ["ç"],
    "s": ["ş"],
    "g": ["ğ"],
    "u": ["ü"],
    "o": ["ö"],
    "e": ["ə"],
    "i": ["ı", "i"],
}

SYNONYMS = {
    # 🚚 LOGISTIKA / NƏQLİYYAT
    "logistika": [
        "logistika",
        "nəqliyyat",
        "daşınma",
        "yükdaşıma",
        "yüklərin daşınması",
        "yük logistikası",
        "nəqliyyat xidmətləri",
        "daşıma xidmətləri",
        "ekspeditor xidmətləri",
        "poçt daşımaları",
        "kuryer xidmətləri"
    ],

    # 🏗 TIKINTI / INŞAAT
    "tikinti": [
        "tikinti",
        "tikinti işləri",
        "inşaat",
        "inşaat işləri",
        "təmir-tikinti",
        "əsaslı təmir",
        "kapital təmir",
        "bərpa",
        "rekonstruksiya",
        "obyektlərin tikintisi",
        "mühəndis tikinti işləri"
    ],

    # 🧱 TIKINTI MATERIALLARI
    "tikinti materialları": [
        "tikinti materialları",
        "inşaat materialları",
        "tikinti malları",
        "dam örtükləri",
        "izolyasiya materialları",
        "beton məmulatları",
        "metal konstruksiyalar",
        "polimer materiallar",
        "pvc materialları",
        "borular və fitinqlər"
    ],

    # ⚡ ÖLÇÜ / MULTIMETRLƏR
    "ölçü cihazları": [
        "ölçü cihazları",
        "ölçmə cihazları",
        "ölçü avadanlıqları",
        "ölçmə və nəzarət avadanlıqları",
        "elektron ölçü cihazları",
        "laboratoriya ölçü cihazları",
        "multimetrlər",
        "rəqəmsal multimetrlər",
        "voltmetr",
        "ampermetr",
        "ohmmetr"
    ],

    # 🛠 AVADANLIQ / TEXNIKA
    "avadanlıq": [
        "avadanlıq",
        "texniki avadanlıq",
        "istehsalat avadanlığı",
        "mexaniki avadanlıq",
        "elektrik avadanlığı",
        "sənaye avadanlığı",
        "maşın və mexanizmlər",
        "ehtiyat hissələri",
        "texniki vasitələr"
    ],

    # 🏭 SƏNAYE
    "sənaye": [
        "sənaye",
        "sənaye avadanlığı",
        "istehsalat",
        "istehsal sahəsi",
        "zavod avadanlığı",
        "fabrik avadanlığı",
        "texnoloji proseslər",
        "sənaye infrastrukturu"
    ],

    # 🧰 TƏMIR / SERVIS
    "təmir": [
        "təmir",
        "texniki xidmət",
        "servis",
        "profilaktik təmir",
        "bərpa işləri",
        "avadanlığın təmiri",
        "texniki baxış",
        "istismar xidməti"
    ],

    # 🛡 TƏHLÜKƏSIZLIK / MÜHAFIZƏ
    "təhlükəsizlik": [
        "təhlükəsizlik",
        "mühafizə",
        "mühafizə xidmətləri",
        "təhlükəsizlik sistemləri",
        "video müşahidə",
        "nəzarət sistemləri",
        "siqnalizasiya",
        "keçid nəzarət sistemləri"
    ],

    # 📦 POÇT / KURYER
    "poçt": [
        "poçt",
        "poçt xidmətləri",
        "kuryer",
        "kuryer xidmətləri",
        "çatdırılma",
        "sənədlərin daşınması",
        "məktub daşınması"
    ]
}

PRESET_COMMANDS = {
    "logistika": [
        "logistika",
        "nəqliyyat",
        "daşınma",
        "yükdaşıma",
        "yük logistikası",
        "kuryer",
        "poçt"
    ],

    "tikinti": [
        "tikinti",
        "inşaat",
        "təmir",
        "rekonstruksiya",
        "tikinti materialları",
        "dam örtükləri",
        "beton"
    ],

    "avadanliq": [
        "avadanlıq",
        "texniki avadanlıq",
        "sənaye avadanlığı",
        "mexaniki avadanlıq",
        "elektrik avadanlığı"
    ],

    "olcu": [
        "ölçü cihazları",
        "ölçmə cihazları",
        "multimetrlər",
        "voltmetr",
        "ampermetr",
        "ohmmetr"
    ],

    "temir": [
        "təmir",
        "texniki xidmət",
        "servis",
        "profilaktik təmir",
        "bərpa işləri"
    ],

    "tehlukesizlik": [
        "təhlükəsizlik",
        "mühafizə",
        "video müşahidə",
        "siqnalizasiya",
        "nəzarət sistemləri"
    ],

    "poct": [
        "poçt",
        "kuryer",
        "çatdırılma",
        "poçt xidmətləri"
    ]
}

def _tokenize_haystack_words(haystack: str) -> List[str]:
    h_fold = normalize_text(fold_diacritics(haystack))
    return [w for w in re.split(r"[^a-z0-9]+", h_fold) if w]

def token_match_debug(token: str, haystack: str) -> Dict:
    """
    Возвращает подробную информацию, почему токен совпал/не совпал.
    """
    token_norm = normalize_text(token)
    variants = generate_variants(token_norm)

    h = normalize_text(haystack)
    h_fold = normalize_text(fold_diacritics(haystack))
    words = _tokenize_haystack_words(haystack)

    checks = []

    for v in variants:
        v_norm = normalize_text(v)
        v_fold = normalize_text(fold_diacritics(v_norm))

        # 1) substring по оригиналу
        if v_norm and v_norm in h:
            return {
                "matched": True,
                "method": "substring",
                "token": token_norm,
                "variant": v_norm,
                "evidence": f"'{v_norm}' in text",
                "checks": checks,
            }

        # 2) substring по folded (без диакритики)
        if v_fold and v_fold in h_fold:
            return {
                "matched": True,
                "method": "substring_folded",
                "token": token_norm,
                "variant": v_norm,
                "evidence": f"fold('{v_norm}')='{v_fold}' in fold(text)",
                "checks": checks,
            }

        # 3) fuzzy (как у тебя сейчас)
        if 5 <= len(v_norm) <= 10:
            for w in words:
                if len(w) < 5:
                    continue
                ratio = fuzzy_ratio(v_norm, w)
                # сохраним самые близкие сравнения для отчета
                if ratio >= 0.70:
                    checks.append({"variant": v_norm, "word": w, "ratio": round(ratio, 3)})

                if ratio >= 0.85:
                    return {
                        "matched": True,
                        "method": "fuzzy",
                        "token": token_norm,
                        "variant": v_norm,
                        "evidence": f"fuzzy('{v_norm}', '{w}')={round(ratio,3)} >= 0.85",
                        "checks": sorted(checks, key=lambda x: x["ratio"], reverse=True)[:8],
                    }

    return {
        "matched": False,
        "method": None,
        "token": token_norm,
        "variant": None,
        "evidence": "no match",
        "checks": sorted(checks, key=lambda x: x["ratio"], reverse=True)[:8],
    }

def full_query_match_debug(query: str, haystack: str) -> Dict:
    """
    Объясняет match целиком: по каждому токену (>=3) дает результат.
    """
    tokens = tokenize_query(query)
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return {"matched": False, "reason": "no tokens >=3", "tokens": []}

    results = []
    for t in tokens:
        res = token_match_debug(t, haystack)
        results.append(res)
        if not res["matched"]:
            return {"matched": False, "reason": f"token '{t}' failed", "tokens": results}

    return {"matched": True, "reason": "all tokens matched", "tokens": results}

def fold_diacritics(s: str) -> str:
    return s.translate(AZ_DIACRITICS_MAP)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_query(q: str) -> List[str]:
    q = normalize_text(q)
    parts = re.split(r"[,\s]+", q)
    return [p for p in parts if p]

def generate_variants(token: str) -> List[str]:
    t0 = normalize_text(token)
    variants: Set[str] = set()

    variants.add(t0)
    variants.add(normalize_text(fold_diacritics(t0)))

    if t0 in SYNONYMS:
        for v in SYNONYMS[t0]:
            variants.add(normalize_text(v))
            variants.add(normalize_text(fold_diacritics(v)))

    def try_diacriticize(s: str) -> str:
        out = []
        for ch in s:
            if ch in PLAIN_TO_AZ and len(PLAIN_TO_AZ[ch]) == 1:
                out.append(PLAIN_TO_AZ[ch][0])
            else:
                out.append(ch)
        return "".join(out)

    variants.add(normalize_text(try_diacriticize(t0)))

    if len(t0) >= 4:
        variants.update({t0 + "ı", t0 + "i", t0 + "ın", t0 + "in", t0 + "ının", t0 + "inin"})

    final = [v for v in variants if v]
    final.sort(key=lambda x: (x != t0, len(x)))
    return final

def fuzzy_ratio(a: str, b: str) -> float:
    a = normalize_text(fold_diacritics(a))
    b = normalize_text(fold_diacritics(b))
    return SequenceMatcher(None, a, b).ratio()

def token_match(token_variants: List[str], haystack: str) -> bool:
    h = normalize_text(haystack)
    h_fold = normalize_text(fold_diacritics(h))

    for v in token_variants:
        v_norm = normalize_text(v)
        v_fold = normalize_text(fold_diacritics(v_norm))

        if v_norm in h or v_fold in h_fold:
            return True

        if 5 <= len(v_norm) <= 14:
            words = re.split(r"[^a-z0-9]+", h_fold)
            for w in words:
                if len(w) < 5:
                    continue
                if fuzzy_ratio(v_norm, w) >= 0.85:
                    return True

    return False

def full_query_match(query: str, haystack: str) -> bool:
    tokens = tokenize_query(query)
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return False
    for t in tokens:
        if not token_match(generate_variants(t), haystack):
            return False
    return True

# ======================
# DATE FILTERS
# ======================
def parse_dt(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except Exception:
        return None

def is_active_item(item: Dict) -> bool:
    if not ACTIVE_ONLY:
        return True
    end_dt = parse_dt(item.get("endDate"))
    if end_dt is None:
        return True
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    return end_dt >= datetime.now(timezone.utc)

def is_fresh_item(item: Dict) -> bool:
    if DAYS_BACK <= 0:
        return True
    pub_dt = parse_dt(item.get("publishDate"))
    if pub_dt is None:
        return True
    if pub_dt.tzinfo is None:
        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
    return pub_dt >= datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

def filter_item(item: Dict) -> bool:
    return is_active_item(item) and is_fresh_item(item)

# ======================
# API FETCH
# ======================

# ======================
# API FETCH
# ======================

# --- DETAIL CACHE ---
DETAIL_CACHE: Dict[str, Tuple[float, Dict]] = {}
DETAIL_TTL_SEC = 60 * 30 * 3  # 3 hour
CANDIDATES_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
CANDIDATES_TTL = 60 * 10  # 10 минут

# --- LIST PAGE CACHE ---
LIST_CACHE: Dict[Tuple[int, int, int, int], Tuple[float, List[Dict], Optional[str]]] = {}

EVENT_DETAIL_URL = "https://etender.gov.az/api/events/"

def fetch_event_detail(event_id: str) -> Optional[Dict]:
    now = datetime.now(timezone.utc).timestamp()
    cached = DETAIL_CACHE.get(event_id)
    if cached and (now - cached[0] < DETAIL_TTL_SEC):
        return cached[1]

    try:
        r = SESSION.get(
            f"{EVENT_DETAIL_URL}{event_id}",
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code >= 400:
            return None


        data = r.json()
        if isinstance(data, dict) and data.get("id"):
            DETAIL_CACHE[event_id] = (now, data)
            return data
    except Exception:
        return None

    return None

def make_params(page_number: int, page_size: int) -> Dict:
    return {
        "EventType": DEFAULT_EVENT_TYPE,
        "EventStatus": DEFAULT_EVENT_STATUS,
        "PageNumber": page_number,
        "PageSize": page_size,
        "buyerOrganizationName": "",
        "documentNumber": "",
        "publishDateFrom": "",
        "publishDateTo": "",
        "AwardedparticipantName": "",
        "AwardedparticipantVoen": "",
        "DocumentViewType": "",
    }

def fetch_events_page(page_number: int, page_size: int) -> Tuple[List[Dict], Optional[str]]:
    key = (DEFAULT_EVENT_TYPE, DEFAULT_EVENT_STATUS, page_number, page_size)
    now = datetime.now(timezone.utc).timestamp()

    cached = LIST_CACHE.get(key)
    if cached and (now - cached[0] < LIST_TTL_SEC):
        return cached[1], cached[2]

    try:
        r = SESSION.get(
            BASE_API,
            params=make_params(page_number, page_size),
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        return [], f"HTTP error: {e}"

    if r.status_code >= 400:
        return [], f"HTTP {r.status_code}: {r.text[:200]}"

    try:
        data = r.json()
    except Exception:
        return [], "JSON oxunmadı."

    items: List[Dict] = []

    for k in ["data", "Data", "items", "Items", "results", "Results"]:
        v = data.get(k)
        if isinstance(v, list):
            items = v
            break

    if not items:
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                items = v
                break

    LIST_CACHE[key] = (now, items, None)
    return items, None

def extract_id(item: Dict) -> str:
    for k in ["eventId", "EventId", "id", "Id"]:
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""

def extract_text_list_only(item: Dict) -> str:
    parts: List[str] = []
    for k in ["eventName", "EventName", "buyerOrganizationName", "BuyerOrganizationName"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)

def extract_text(item: Dict) -> str:
    parts = []

    # базовые поля из списка
    for k in ["eventName", "EventName", "buyerOrganizationName", "BuyerOrganizationName"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # подтягиваем detail (categoryCodes, tenderName, organizationName, etc.)
    tid = extract_id(item)
    if tid:
        detail = fetch_event_detail(tid)
        if isinstance(detail, dict):
            # иногда в detail другие названия ключей
            for k in ["tenderName", "eventName", "organizationName", "buyerOrganizationName"]:
                v = detail.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

            cats = detail.get("categoryCodes")
            if isinstance(cats, list):
                # "43211500 Kompüterlər" и т.п.
                parts.extend([str(x).strip() for x in cats if str(x).strip()])

    return " ".join(parts)

def extract_display_text(item: Dict) -> str:
    parts = []
    for k in ["eventName", "EventName", "buyerOrganizationName", "BuyerOrganizationName"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    # если в list API нет eventName/buyerOrganizationName, пробуем detail
    tid = extract_id(item)
    if tid:
        detail = fetch_event_detail(tid)
        if isinstance(detail, dict):
            for k in ["tenderName", "organizationName"]:
                v = detail.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

    return " ".join(parts).strip()

def get_candidate_items() -> List[Dict]:
    now = datetime.now(timezone.utc).timestamp()
    cached = CANDIDATES_CACHE.get("items")

    if cached and (now - cached[0] < CANDIDATES_TTL):
        return cached[1]

    collected: List[Dict] = []
    seen_ids: Set[str] = set()

    for page in range(1, SMART_PAGES + 1):
        items, err = fetch_events_page(page, SMART_PAGE_SIZE)
        if err:
            continue

        for it in items:
            tid = extract_id(it)
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)

            if filter_item(it):
                collected.append(it)

    CANDIDATES_CACHE["items"] = (now, collected)
    return collected


def _extract_text_list_only(item: Dict) -> str:
    """Text from list API only (no detail HTTP)."""
    parts = []
    for k in ["eventName", "EventName", "buyerOrganizationName", "BuyerOrganizationName"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)


def smart_search(query: str) -> List[Dict]:
    q = normalize_text(query)
    candidates = get_candidate_items()

    # If query has digits (often category codes), we may need detail,
    # but we still try list-only match first to keep it fast.
    tokens = tokenize_query(q)
    needs_detail = any(any(ch.isdigit() for ch in t) for t in tokens)

    matched: List[Dict] = []

    # 1) Fast pass: list-only match (no detail HTTP)
    for it in candidates:
        base_text = _extract_text_list_only(it)
        if base_text and full_query_match(q, base_text):
            matched.append(it)

    if matched or not needs_detail:
        return matched

    # 2) Slower pass for digit/category queries: check detail, but cap work
    MAX_DETAIL_CHECK = int(os.getenv("MAX_DETAIL_CHECK", "250"))
    checked = 0
    for it in candidates:
        if checked >= MAX_DETAIL_CHECK:
            break
        checked += 1
        text = extract_text(it)  # may fetch detail
        if text and full_query_match(q, text):
            matched.append(it)

    return matched


# ======================
# TELEGRAM FORMAT
# ======================
def format_search_results(items: List[Dict], query: str, limit: int) -> str:
    if not items:
        return "Bu sorğu üzrə heç nə tapılmadı (aktiv/son günlər filtri ilə)."

    out = [f"🔎 *Axtarış:* *{query}*"]
    filt = []
    filt.append("yalnız aktiv" if ACTIVE_ONLY else "hamısı")
    if DAYS_BACK > 0:
        filt.append(f"son {DAYS_BACK} gün")
    out.append(f"_(filtr: {', '.join(filt)})_\n")

    count = 0
    for it in items:
        tid = extract_id(it)
        text = extract_display_text(it)
        if not tid or not text:
            continue
        url = f"{DETAIL_BASE}{tid}"
        out.append(f"• {text}\n{url}")
        count += 1
        if count >= limit:
            break

    return "\n\n".join(out)

def format_new_notification(query: str, it: Dict) -> str:
    tid = extract_id(it)
    text = extract_display_text(it)  # ✅ БЕЗ categoryCodes
    url = f"{DETAIL_BASE}{tid}" if tid else ""

    return (
        f"🆕 *Yeni tender (subscribe):* *{query}*\n\n"
        f"{text}\n"
        f"{url}"
    )
# ======================
# COMMANDS
# ======================
def _fmt_timedelta(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    sec = int(delta.total_seconds())
    if sec < 60:
        return f"{sec} san"
    if sec < 3600:
        return f"{sec // 60} dəq"
    if sec < 86400:
        return f"{sec // 3600} saat"
    return f"{sec // 86400} gün"

def _get_query(context: ContextTypes.DEFAULT_TYPE) -> str:
    return " ".join(context.args).strip()

def _extract_id_from_arg(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    m = re.search(r"/detail/(\d+)", s)
    if m:
        return m.group(1)
    return s if s.isdigit() else ""

def find_candidate_by_id(event_id: str) -> Optional[Dict]:
    # ищем в текущих candidates (те же страницы и фильтры)
    candidates = get_candidate_items()
    for it in candidates:
        if extract_id(it) == str(event_id):
            return it
    return None

async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Debug yalnız admin üçün.")
        return

    # /debug <query> <id_or_url>
    text = update.message.text or ""
    parts = text.split(" ", 2)
    if len(parts) < 3:
        await update.message.reply_text("İstifadə: /debug <query> <tender_id və ya link>\nMəs: /debug məktüb 345200")
        return

    query = parts[1].strip()
    tid = _extract_id_from_arg(parts[2])
    if not tid:
        await update.message.reply_text("Tender ID tapılmadı. ID və ya detail link göndər.")
        return

    it = await asyncio.to_thread(find_candidate_by_id, tid)
    if not it:
        await update.message.reply_text(
            "Bu tender hal-hazırda candidates-də tapılmadı.\n"
            "Səbəb ola bilər: SMART_PAGES azdır, ACTIVE_ONLY/DAYS_BACK filtri çıxarıb, və ya EventType/EventStatus fərqlidir."
        )
        return

    text_used = await asyncio.to_thread(extract_text, it)  # eventName + buyerOrganizationName
    dbg = full_query_match_debug(query, text_used)

    lines = []
    lines.append("🧪 *DEBUG MATCH*")
    lines.append(f"🔎 Query: *{normalize_text(query)}*")
    lines.append(f"🆔 Tender: `{tid}`")
    lines.append(f"🔗 {DETAIL_BASE}{tid}")
    lines.append("")
    lines.append(f"📝 Text used: _{text_used}_")
    lines.append("")
    lines.append(f"✅ Matched: *{dbg['matched']}*")
    lines.append(f"ℹ️ Reason: _{dbg['reason']}_")
    lines.append("")

    for r in dbg["tokens"]:
        lines.append(f"— Token: *{r['token']}*")
        lines.append(f"   matched: *{r['matched']}*")
        if r["matched"]:
            lines.append(f"   method: `{r['method']}`")
            lines.append(f"   variant: `{r['variant']}`")
            lines.append(f"   evidence: _{r['evidence']}_")
        else:
            lines.append(f"   evidence: _{r['evidence']}_")

        if r.get("closest"):
            lines.append("   closest:")
            for v, w, ratio in r["closest"][:5]:
                lines.append(f"     • {v} ~ {w} = {ratio}")
        lines.append("")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    if require_paid(chat_id):
        msg = (
            "Salam! ✅ Giriş aktivdir.\n\n"
            "Komandalar:\n"
            "• /search söz (etender axtariwi(agilli axtariw))\n"
            "• /subscribe söz (her 6 saat axtariw uzre yeni tender verir)\n"
            "• /unsubscribe söz\n"
            "• /subs (aktiv abuneler)\n"
            "Suallar - @IsmayilGurbanaliyev\n"
        )
        await update.message.reply_text(msg)
    else:
        await send_paywall(update)

async def cmd_redeem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    code = _get_query(context).strip()
    if not code:
        await update.message.reply_text("İstifadə: /redeem KOD")
        return

    ok, msg, expires = redeem_code(chat_id, code, duration_days=ACCESS_DAYS)
    if not ok:
        await update.message.reply_text("❌ " + msg)
        return

    exp_txt = expires.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if expires else ""
    await update.message.reply_text(f"{msg}\n⏳ Bitmə tarixi: {exp_txt}")

TG_MAX = 3900  # запас, чтобы точно не упереться

async def send_long(update: Update, text: str, *, parse_mode=None, preview=False):
    parts = text.split("\n\n")
    chunk = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue

        candidate = (chunk + "\n\n" + p) if chunk else p
        if len(candidate) > TG_MAX:
            if chunk:
                await update.message.reply_text(chunk, parse_mode=parse_mode, disable_web_page_preview=not preview)
            chunk = p
        else:
            chunk = candidate

    if chunk:
        await update.message.reply_text(chunk, parse_mode=parse_mode, disable_web_page_preview=not preview)

async def cmd_preview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    if not require_paid(chat_id):
        await send_paywall(update)
        return

    q_raw = _get_query(context)
    if not q_raw:
        await update.message.reply_text("İstifadə: /preview <söz/ifadə>\nMəsələn: /preview server avadanlıq")
        return

    q_norm = normalize_text(q_raw)
    items = await asyncio.to_thread(smart_search, q_norm)

    total = len(items)
    top3 = items[:3]

    if total == 0:
        await update.message.reply_text(
            f"🔎 *Preview:* *{q_norm}*\n\nHeç nə tapılmadı (filtrlə).",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    lines = [f"🔎 *Preview:* *{q_norm}*", f"✅ Tapıldı: *{total}* nəticə", ""]
    for it in top3:
        tid = extract_id(it)
        text = await asyncio.to_thread(extract_text, it)
        if tid and text:
            lines.append(f"• {text}\n{DETAIL_BASE}{tid}\n")

    lines.append("ℹ️ Abunə olmaq üçün:\n" f"`/subscribe {q_norm}`")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    st = get_stats()
    uptime = _fmt_timedelta(BOT_STARTED_AT)

    access = get_user_access(chat_id)
    if access:
        _, _, expires_at = access
        exp_txt = expires_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        access_line = f"🔑 Access: aktiv (bitir: {exp_txt})\n"
    else:
        access_line = "🔑 Access: yoxdur\n"

    msg = (
        "🟢 Bot aktivdir\n"
        f"⏱ Uptime: {uptime}\n"
        f"👥 İstifadəçilər: {st.get('users', 0)}\n"
        f"💳 Aktiv ödənişli: {st.get('paid_active', 0)}\n"
        f"📌 Abunəliklər: {st.get('subs', 0)}\n"
        f"🔁 Yoxlama intervalı: {CHECK_INTERVAL} san\n\n"
        + access_line
    )
    await update.message.reply_text(msg)

async def cmd_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    if not require_paid(chat_id):
        await send_paywall(update)
        return

    cmd = update.message.text.lstrip("/").lower()

    if cmd not in PRESET_COMMANDS:
        await update.message.reply_text("❌ Bu komanda üçün preset tapılmadı.")
        return

    await update.message.reply_text(
        f"🔎 Axtarış edilir: {cmd}\n(avtomatik sektor axtarışı)",
    )

    # ✅ ищем по всем ключевым словам из группы и собираем уникальные tender-id
    items_all = []
    seen = set()

    for kw in PRESET_COMMANDS[cmd]:
        for it in await asyncio.to_thread(smart_search, kw):
            tid = extract_id(it)
            if tid and tid not in seen:
                seen.add(tid)
                items_all.append(it)

    text = format_search_results(items_all, cmd, RESULT_LIMIT)
    await send_long(update, text, parse_mode=None, preview=False)

async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    paid = require_paid(chat_id)
    txt = "❌ Bu komanda tanınmadı.\n\n" + user_commands_text(paid)

    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def on_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)

    paid = require_paid(chat_id)
    txt = "ℹ️ Mən komandalarla işləyirəm.\n\n" + user_commands_text(paid)

    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not require_paid(chat_id):
        await send_paywall(update)
        return

    q = _get_query(context)
    if not q:
        await update.message.reply_text("Belə yaz: /search Poçt")
        return

    await update.message.reply_text("🔎 Axtarış edilir… zəhmət olmasa gözlə.")

    items = await asyncio.to_thread(smart_search, q)
    text = format_search_results(items, q, RESULT_LIMIT)
    await send_long(update, text, parse_mode=None, preview=False)

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not require_paid(chat_id):
        await send_paywall(update)
        return

    q = _get_query(context)
    if not q:
        await update.message.reply_text("Belə yaz: /subscribe poct")
        return

    ensure_user(chat_id)

    MAX_SUBS_PER_USER = 7

    # ✅ LIMIT CHECK
    current_subs = list_subscriptions(chat_id)
    if len(current_subs) >= MAX_SUBS_PER_USER:
        await update.message.reply_text(
            f"🚫 Limitə çatdınız.\n\n"
            f"📌 Maksimum: *{MAX_SUBS_PER_USER} abunə*\n"
            f"Silmək üçün: `/unsubscribe söz`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    ok = add_subscription(chat_id, q)
    if not ok:
        await update.message.reply_text("Sən artıq bu sorğuya abunə olmusan.")
        return

    # помечаем текущие совпадения как seen (чтобы не слать старое)
    current = await asyncio.to_thread(smart_search, q)
    for it in current:
        tid = extract_id(it)
        if tid:
            mark_sub_seen(chat_id, q, tid)

    await update.message.reply_text(
        f"✅ Abunəlik əlavə olundu: *{q}*\n"
        f"📊 İstifadə: {len(current_subs)+1}/{MAX_SUBS_PER_USER}\n"
        f"Yalnız *yeni tender* çıxanda bildiriş göndərəcəyəm.",
        parse_mode=ParseMode.MARKDOWN,
    )

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not require_paid(chat_id):
        await send_paywall(update)
        return

    q = _get_query(context)
    if not q:
        await update.message.reply_text("Belə yaz: /unsubscribe Poçt")
        return

    removed = remove_subscription(chat_id, q)
    if removed:
        await update.message.reply_text(f"🗑 Unsubscribe edildi: *{q}*", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("Bu sorğu üzrə abunəlik tapılmadı.")

async def cmd_subs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not require_paid(chat_id):
        await send_paywall(update)
        return

    subs = list_subscriptions(chat_id)
    if not subs:
        await update.message.reply_text("Abunəliyin yoxdur. /subscribe söz")
        return

    lines = ["📌 *Sənin abunəliklərin:*"]
    for q in subs:
        lines.append(f"• {q}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Bu komanda yalnız admin üçündür.")
        return

    text = update.message.text or ""
    payload = text.split(" ", 1)
    msg = payload[1].strip() if len(payload) > 1 else ""

    if not msg:
        await update.message.reply_text("İstifadə: /broadcast <mətn>")
        return

    users = get_all_users()
    await update.message.reply_text(f"📣 Göndərilir… istifadəçi sayı: {len(users)}")

    ok = 0
    failed = 0
    for uid in users:
        try:
            await context.bot.send_message(chat_id=uid, text=msg, disable_web_page_preview=True)
            ok += 1
            await asyncio.sleep(0.05)
        except Exception:
            failed += 1

    await update.message.reply_text(f"✅ Hazırdır.\nUğurlu: {ok}\nXəta: {failed}")

# ----------------------
# ADMIN: codes
# ----------------------
def _gen_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    a = "".join(secrets.choice(alphabet) for _ in range(4))
    b = "".join(secrets.choice(alphabet) for _ in range(4))
    c = "".join(secrets.choice(alphabet) for _ in range(2))
    return f"{a}-{b}-{c}"

async def cmd_createcode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Admin only.")
        return

    # /createcode CompanyName 3
    text = update.message.text or ""
    rest = text.split(" ", 1)
    if len(rest) < 2 or not rest[1].strip():
        await update.message.reply_text("İstifadə: /createcode <company_name> [max_users]")
        return

    parts = rest[1].strip().rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        company_name = parts[0].strip()
        max_users = int(parts[1])
    else:
        company_name = rest[1].strip()
        max_users = DEFAULT_MAX_USERS

    code = _gen_code()
    create_company_code(code, company_name, max_users=max_users)

    await update.message.reply_text(
        f"✅ Kod yaradıldı\n🏢 {company_name}\n👥 Limit: {max_users}\n🔑 `{code}`\n\n"
        f"İstifadə: `/redeem {code}`",
        parse_mode=ParseMode.MARKDOWN,
    )

async def cmd_codeinfo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Admin only.")
        return

    code = _get_query(context).strip()
    if not code:
        await update.message.reply_text("İstifadə: /codeinfo <code>")
        return

    info = code_info(code)
    if not info:
        await update.message.reply_text("Kod tapılmadı.")
        return

    _, company_name, max_users, is_active = info
    used = code_usage_count(code)

    await update.message.reply_text(
        f"🔑 `{code}`\n🏢 {company_name}\n👥 {used}/{max_users}\n✅ Aktiv: {is_active}",
        parse_mode=ParseMode.MARKDOWN,
    )

async def cmd_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Admin only.")
        return

    code = _get_query(context).strip()
    if not code:
        await update.message.reply_text("İstifadə: /revoke <code>")
        return

    deactivate_code(code)
    await update.message.reply_text("✅ Kod deaktiv edildi.\n")

# ======================
# BACKGROUND JOB
# ======================
async def job_check_subscriptions(context: ContextTypes.DEFAULT_TYPE):
    try:
        subs = get_all_subscriptions()
        if not subs:
            return

        # heavy: fetch candidates in a worker thread (avoids blocking asyncio loop)
        candidates = await asyncio.to_thread(get_candidate_items)

        for chat_id, query in subs:
            # если не платный — не проверяем
            if not is_paid_active(chat_id) and not is_admin(chat_id):
                continue

            new_items = []
            for it in candidates:
                tid = extract_id(it)
                if not tid:
                    continue
                if sub_has_seen(chat_id, query, tid):
                    continue

                # heavy text building: run in thread to avoid blocking
                text = await asyncio.to_thread(extract_text, it)
                if text and full_query_match(query, text):
                    new_items.append(it)

            for it in new_items:
                tid = extract_id(it)
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=format_new_notification(query, it),
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=False,
                    )
                    if tid:
                        mark_sub_seen(chat_id, query, tid)
                except Exception:
                    pass
    except asyncio.CancelledError:
        # graceful shutdown
        return

async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.message.reply_text("⛔ Bu komanda yalnız admin üçündür.")
        return

    msg = (
        "🛠 *Admin Panel*\n\n"
        "🔑 Kodlar:\n"
        "• `/createcode <company> [max_users]` — kod yarat\n"
        "• `/codeinfo <code>` — kod məlumatı\n"
        "• `/revoke <code>` — kodu deaktiv et\n\n"
        "📣 İdarəetmə:\n"
        "• `/broadcast <text>` — hamıya mesaj\n"
        "• `/status` — bot statusu\n"
    )

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

# ======================
# ERROR HANDLER
# ======================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("❌ ERROR:", context.error)
    except Exception:
        pass

# ======================
# MAIN
# ======================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("redeem", cmd_redeem))

    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("preview", cmd_preview))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("subs", cmd_subs))
    app.add_handler(CommandHandler("status", cmd_status))

    app.add_handler(CommandHandler("broadcast", cmd_broadcast))
    app.add_handler(CommandHandler("debug", cmd_debug))

    # admin codes
    app.add_handler(CommandHandler("createcode", cmd_createcode))
    app.add_handler(CommandHandler("codeinfo", cmd_codeinfo))
    app.add_handler(CommandHandler("revoke", cmd_revoke))
    app.add_handler(CommandHandler("admin", cmd_admin))
    for preset in PRESET_COMMANDS.keys():
        app.add_handler(CommandHandler(preset, cmd_preset))
    # Unknown commands (must be after all other command handlers)
    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))

    # Optional: reply to any plain text (non-command)
    # app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_plain_text))

    app.add_error_handler(on_error)

    if app.job_queue:
        app.job_queue.run_repeating(job_check_subscriptions, interval=CHECK_INTERVAL, first=10)
    else:
        print("⚠️ JobQueue yoxdur. requirements.txt-da python-telegram-bot[job-queue] olmalıdır.")

    print("✅ Telegram bot işə düşdü. (dayandırmaq üçün Ctrl+C)")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()