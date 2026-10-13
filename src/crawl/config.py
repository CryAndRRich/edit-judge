"""Paths and settings shared by all crawlers.

Secrets and per-user settings are read from the `.env` file at the repository root
(never committed; see .env.example). Supported keys:

    CRAWLER_CONTACT   contact (e.g. an email) appended to the User-Agent
    CRAWLER_SALT      salt for hashing usernames; generated into data/raw/.salt if absent
    PSG_COOKIE        PhotoshopGurus session cookie, "xf_user=...; xf_session=..."
"""

import secrets

from ..common.env import ENV, load_env  # noqa: F401  (load_env re-exported)
from ..paths import DATA_DIR, ENV_FILE, RAW_DIR  # noqa: F401  (re-exported for the source modules)


def user_agent() -> str:
    contact = ENV.get("CRAWLER_CONTACT", "").strip()
    suffix = f"; contact: {contact}" if contact else ""
    return f"EditJudge-crawler/0.1 (non-commercial student project{suffix})"


def username_salt() -> str:
    """Salt for hashing usernames. Stable across runs so hashes can be joined."""
    if ENV.get("CRAWLER_SALT"):
        return ENV["CRAWLER_SALT"]
    salt_file = RAW_DIR / ".salt"
    if not salt_file.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        salt_file.write_text(secrets.token_hex(16), encoding="utf-8")
    return salt_file.read_text(encoding="utf-8").strip()
