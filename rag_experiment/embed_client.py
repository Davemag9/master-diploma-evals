"""Rate-limited Gemini embedding client with a persistent on-disk cache.

Free tier for gemini-embedding-001 turned out to cap at RPD=1000 (a DAILY
request limit, not just per-minute) per API key/project. To get more
headroom we round-robin across multiple API keys (each is a separate
Google Cloud project with its own independent 1000/day quota) and switch
to the next key once one reports its daily quota exhausted.

Every new embedding is cached by content hash, and the cache is flushed to
disk periodically (not just at the very end) so a crash or a 429 partway
through a big batch never loses already-paid-for embeddings.
"""
import hashlib
import http.client
import json
import time
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

API_KEYS = [ "Хто лишив ключі тут той лох"]
EMBED_MODEL = "gemini-embedding-001"

CACHE_PATH = Path(__file__).parent / "data" / "cache" / "embeddings.json"
QUOTA_DIR = Path(__file__).parent / "data" / "cache"
MIN_INTERVAL = 0.8  # seconds between calls per key
SAVE_EVERY = 10  # flush cache to disk after this many NEW entries


class AllKeysExhausted(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self):
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CACHE_PATH.exists():
            self._cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        else:
            self._cache = {}
        self._dirty_count = 0
        self._last_call = 0.0
        self._calls_made = 0
        self._key_idx = 0
        self._exhausted_keys = set()
        self._quota_path = QUOTA_DIR / f"embed_quota_{date.today().isoformat()}.json"
        self._usage = self._load_usage()

    def _load_usage(self):
        if self._quota_path.exists():
            return json.loads(self._quota_path.read_text(encoding="utf-8"))
        return {str(i): 0 for i in range(len(API_KEYS))}

    def _save_usage(self):
        self._quota_path.write_text(json.dumps(self._usage), encoding="utf-8")

    def _save_cache(self):
        CACHE_PATH.write_text(json.dumps(self._cache), encoding="utf-8")
        self._dirty_count = 0

    def _maybe_save_cache(self):
        if self._dirty_count >= SAVE_EVERY:
            self._save_cache()

    @staticmethod
    def _key(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _pace(self):
        elapsed = time.time() - self._last_call
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

    def _current_key(self):
        return API_KEYS[self._key_idx]

    def _rotate_key(self):
        self._exhausted_keys.add(self._key_idx)
        if len(self._exhausted_keys) >= len(API_KEYS):
            raise AllKeysExhausted(
                f"All {len(API_KEYS)} embedding API keys hit their daily quota. "
                f"Usage today: {self._usage}. Free-tier quota resets ~00:00 Pacific Time."
            )
        self._key_idx = (self._key_idx + 1) % len(API_KEYS)
        while self._key_idx in self._exhausted_keys:
            self._key_idx = (self._key_idx + 1) % len(API_KEYS)

    def _call_api(self, text):
        payload = json.dumps({"content": {"parts": [{"text": text}]}}).encode("utf-8")
        backoff = 5
        for attempt in range(6):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={self._current_key()}"
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            self._pace()
            try:
                self._last_call = time.time()
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self._calls_made += 1
                    self._usage[str(self._key_idx)] = self._usage.get(str(self._key_idx), 0) + 1
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["embedding"]["values"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code == 429:
                    if "PerDay" in err_body or "RequestsPerDay" in err_body:
                        self._rotate_key()
                        continue  # retry immediately on the new key, no backoff needed
                    if attempt < 5:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                raise RuntimeError(f"Embedding API error {e.code}: {err_body[:400]}")
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError, OSError) as e:
                if attempt < 5:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"Embedding API network error after retries: {e}")
        raise RuntimeError("Embedding API: exceeded retries")

    def embed(self, text):
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]
        try:
            vec = self._call_api(text)
        finally:
            self._save_usage()
        self._cache[key] = vec
        self._dirty_count += 1
        self._maybe_save_cache()
        return vec

    def embed_many(self, texts, on_progress=None):
        vectors = []
        try:
            for i, t in enumerate(texts):
                vectors.append(self.embed(t))
                if on_progress:
                    on_progress(i + 1, len(texts))
        finally:
            self._save_cache()  # always persist whatever we got, even on crash/exception
        return vectors
