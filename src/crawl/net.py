"""Polite HTTP access: per-host pacing, robots.txt, bounded retries. Safe to use from several threads.

Pacing is per host: requests to the same host are at least `delay` seconds apart (they wait for
each other), while requests to different hosts may run in parallel. So parallel downloads never
send one host more than one request per `delay`.

A refusal (HTTP 401/403 or a bot-challenge page) raises `Blocked` and is never retried:
the project rule is to stop, not to work around a block.
"""

import logging
import threading
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class Blocked(Exception):
    """The site refused access. Do not retry or bypass."""


class Disallowed(Exception):
    """robots.txt disallows this URL for our user agent."""


class GaveUp(LookupError):
    """Still failing after the allowed retries; the caller records it and moves on."""


class PoliteSession:
    def __init__(self, user_agent: str, delay: float = 3.0, cookies: dict[str, str] | None = None,
                 cookie_domain: str | None = None, max_retries: int = 3, timeout: float = 30.0):
        # Main-thread session. Cookies live in its jar (not a fixed header) so that session cookies
        # the site rotates via Set-Cookie are picked up; a stale fixed header gets logged out.
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        for name, value in (cookies or {}).items():
            self.session.cookies.set(name, value, domain=cookie_domain, path="/")
        self.user_agent = user_agent
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self._main = threading.get_ident()
        self._local = threading.local()
        self._lock = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}

    # --- per-thread sessions and per-host pacing ----------------------------------------------
    def _session(self) -> requests.Session:
        """The main thread uses self.session; worker threads get their own copy (headers and cookies)."""
        if threading.get_ident() == self._main:
            return self.session
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update(self.session.headers)
            s.cookies.update(self.session.cookies)
            self._local.session = s
        return self._local.session

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _send(self, method: str, url: str, **kwargs) -> requests.Response:
        """One paced request: request *starts* to a host are at least `delay` seconds apart.

        The host lock is held only while reserving the start slot, not during the transfer, so
        one slow, large download does not stall other requests to the same host (each still
        starts at most once per `delay`)."""
        host = urlsplit(url).netloc
        with self._host_lock(host):
            wait = self._last.get(host, 0.0) + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
        return self._session().request(method, url, timeout=self.timeout, **kwargs)

    # --- robots.txt ----------------------------------------------------------------------------
    def _robots_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._host_lock("robots:" + origin):
            if origin not in self._robots:
                parser = RobotFileParser()
                try:
                    resp = self._send("GET", origin + "/robots.txt")
                except requests.RequestException as exc:
                    # Dead hosts are common among old image hosts; record and move on.
                    raise GaveUp("host_unreachable") from exc
                # A missing robots.txt means everything is allowed; any other failure is treated as a refusal.
                if resp.status_code == 404:
                    parser.parse([])
                elif resp.ok:
                    parser.parse(resp.text.splitlines())
                else:
                    raise Blocked(f"robots.txt for {origin} returned HTTP {resp.status_code}")
                self._robots[origin] = parser
        return self._robots[origin]

    # --- public API ------------------------------------------------------------------------------
    def get(self, url: str, *, robots: bool = True, **kwargs) -> requests.Response:
        """GET with pacing, bounded retries and refusal detection.

        robots=False is only for official APIs used with credentials (e.g. Reddit's OAuth API),
        which are governed by the API terms rather than by robots.txt."""
        if robots and not self._robots_for(url).can_fetch(self.user_agent, url):
            raise Disallowed(url)
        return self.request("GET", url, **kwargs)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._send(method, url, **kwargs)
            except requests.RequestException as exc:
                log.warning("request error (%s/%s) %s: %s", attempt, self.max_retries, url, exc)
                time.sleep(self.delay * 2 ** attempt)
                continue
            if resp.status_code in (401, 403) or "<title>Just a moment...</title>" in resp.text[:2000]:
                raise Blocked(f"HTTP {resp.status_code} for {url}")
            if resp.status_code in RETRY_STATUSES:
                retry_after = resp.headers.get("Retry-After", "")
                pause = float(retry_after) if retry_after.isdigit() else self.delay * 2 ** attempt
                log.warning("HTTP %s (%s/%s) %s; waiting %.0fs", resp.status_code, attempt, self.max_retries, url, pause)
                time.sleep(pause)
                continue
            return resp
        raise GaveUp(f"gave_up_after_{self.max_retries}_attempts")
