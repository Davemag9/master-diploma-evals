"""Rate-limited, multi-key Gemini generation client, tuned for the free tier.

gemini-flash-latest (currently resolving to gemini-3.6-flash) has a VERY
tight free-tier quota: RPM~5, RPD~20. To run the generation comparison over
the full 107-question benchmark (214 calls) we round-robin across several
API keys (separate Google Cloud projects, each with its own ~20/day quota),
the same way embed_client.py does for the embedding model.

This client:
  - spaces requests at least 13s apart per key (RPM~5 + margin),
  - retries on 429, honoring the server's retryDelay when present,
  - rotates to the next key on a daily-quota 429, tracking usage per key
    on disk so a safety cap (not the exact server limit) is enforced,
  - raises AllKeysExhausted when every key is spent for the day so a
    caller can stop cleanly and resume tomorrow,
  - exposes on_wait(seconds) so a progress bar can show rate-limit pauses.
"""
import http.client
import json
import re
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

API_KEYS = [ "Хто лишив ключі тут той лох"]
GEN_MODEL = "gemini-flash-latest"

MIN_INTERVAL = 13.0  # seconds; RPM~5 free tier -> 12s min, +1s margin
DAILY_SAFETY_CAP = 18  # stay under the real RPD~20 free-tier ceiling, per key
# per-key override for keys known to have a much higher (or no) daily cap
DAILY_CAP_OVERRIDES = {}


def _cap_for(key_idx):
    return DAILY_CAP_OVERRIDES.get(key_idx, DAILY_SAFETY_CAP)

QUOTA_DIR = Path(__file__).parent / "data" / "cache"


class DailyQuotaExceeded(RuntimeError):
    pass


class AllKeysExhausted(RuntimeError):
    pass


class GenerationClient:
    def __init__(self):
        QUOTA_DIR.mkdir(parents=True, exist_ok=True)
        self._quota_path = QUOTA_DIR / f"gen_quota_multi_{date.today().isoformat()}.json"
        self._usage = self._load_usage()
        self._key_idx = 0
        self._exhausted_keys = set(i for i in range(len(API_KEYS)) if self._usage.get(str(i), 0) >= _cap_for(i))
        self._last_call = {i: 0.0 for i in range(len(API_KEYS))}
        self._advance_to_usable_key()

    def _load_usage(self):
        if self._quota_path.exists():
            return json.loads(self._quota_path.read_text(encoding="utf-8"))
        return {str(i): 0 for i in range(len(API_KEYS))}

    def _save_usage(self):
        self._quota_path.write_text(json.dumps(self._usage), encoding="utf-8")

    def _advance_to_usable_key(self):
        while self._key_idx in self._exhausted_keys and len(self._exhausted_keys) < len(API_KEYS):
            self._key_idx = (self._key_idx + 1) % len(API_KEYS)

    def remaining_today(self):
        return sum(max(0, _cap_for(i) - self._usage.get(str(i), 0)) for i in range(len(API_KEYS)))

    def _rotate_key(self):
        self._exhausted_keys.add(self._key_idx)
        if len(self._exhausted_keys) >= len(API_KEYS):
            raise AllKeysExhausted(
                f"All {len(API_KEYS)} generation API keys hit their daily quota. "
                f"Usage today: {self._usage}. Free-tier quota resets ~00:00 Pacific Time."
            )
        self._key_idx = (self._key_idx + 1) % len(API_KEYS)
        self._advance_to_usable_key()

    def _wait_for_rate_limit(self, on_wait=None):
        elapsed = time.time() - self._last_call[self._key_idx]
        wait = MIN_INTERVAL - elapsed
        if wait > 0:
            if on_wait:
                on_wait(wait)
            time.sleep(wait)

    def generate(self, prompt, on_wait=None):
        if self._usage.get(str(self._key_idx), 0) >= _cap_for(self._key_idx):
            self._rotate_key()

        self._wait_for_rate_limit(on_wait)

        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")

        backoff = 15
        for attempt in range(6):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEN_MODEL}:generateContent?key={API_KEYS[self._key_idx]}"
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                self._last_call[self._key_idx] = time.time()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    self._usage[str(self._key_idx)] = self._usage.get(str(self._key_idx), 0) + 1
                    self._save_usage()
                    return _extract_text(body)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code == 429:
                    if "PerDay" in err_body or "RequestsPerDay" in err_body:
                        self._usage[str(self._key_idx)] = _cap_for(self._key_idx)  # mark spent even if we guessed wrong
                        self._save_usage()
                        self._rotate_key()
                        continue
                    retry_s = _parse_retry_delay(err_body) or backoff
                    if on_wait:
                        on_wait(retry_s)
                    time.sleep(retry_s)
                    backoff *= 2
                    continue
                if e.code in (500, 502, 503, 504) and attempt < 5:
                    # transient server-side overload -- retry with backoff, don't burn a key
                    if on_wait:
                        on_wait(backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Generation API error {e.code}: {err_body[:400]}")
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as e:
                # transient network hiccup (dropped connection, DNS blip, etc.) -- retry, don't burn a key
                if attempt < 5:
                    if on_wait:
                        on_wait(backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Generation API network error after retries: {e}")
        raise RuntimeError("Generation API: exceeded retries (rate limit)")


def _extract_text(body):
    try:
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        return ""


def _parse_retry_delay(err_body):
    m = re.search(r'"retryDelay":\s*"(\d+)s"', err_body)
    return int(m.group(1)) if m else None
