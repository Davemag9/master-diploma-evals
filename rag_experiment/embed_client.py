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
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # MIN_INTERVAL is a per-key requests-per-minute guard, so each key gets
        # its own clock and calls cycle across keys: len(API_KEYS)x the
        # throughput without exceeding any single key's rate limit.
        self._last_call = {i: 0.0 for i in range(len(API_KEYS))}
        self._calls_made = 0
        self._key_idx = 0
        self._exhausted_keys = set()
        self._lock = threading.Lock()
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

    def _pace_key(self, key_idx):
        elapsed = time.time() - self._last_call[key_idx]
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

    def _next_key(self):
        """Move to the next key that still has quota, for the next request."""
        if len(self._exhausted_keys) >= len(API_KEYS):
            return
        self._key_idx = (self._key_idx + 1) % len(API_KEYS)
        while self._key_idx in self._exhausted_keys:
            self._key_idx = (self._key_idx + 1) % len(API_KEYS)

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

    def _call_api(self, text, key_idx=None):
        """Embed one text. `key_idx` pins the call to one key, which is what
        lets several calls run at once: each key keeps its own clock, so N
        workers on N keys stay inside every key's rate limit."""
        payload = json.dumps({"content": {"parts": [{"text": text}]}}).encode("utf-8")
        backoff = 5
        pinned = key_idx is not None
        if not pinned:
            self._next_key()
            key_idx = self._key_idx
        for attempt in range(6):
            if key_idx in self._exhausted_keys:
                if len(self._exhausted_keys) >= len(API_KEYS):
                    raise AllKeysExhausted(
                        f"All {len(API_KEYS)} embedding API keys hit their daily quota. "
                        f"Usage today: {self._usage}. Free-tier quota resets ~00:00 Pacific Time."
                    )
                key_idx = (key_idx + 1) % len(API_KEYS)
                while key_idx in self._exhausted_keys:
                    key_idx = (key_idx + 1) % len(API_KEYS)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={API_KEYS[key_idx]}"
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            self._pace_key(key_idx)
            try:
                self._last_call[key_idx] = time.time()
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                with self._lock:
                    self._calls_made += 1
                    self._usage[str(key_idx)] = self._usage.get(str(key_idx), 0) + 1
                return body["embedding"]["values"]
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code == 429:
                    if "PerDay" in err_body or "RequestsPerDay" in err_body:
                        with self._lock:
                            self._exhausted_keys.add(key_idx)
                            spent = len(self._exhausted_keys) >= len(API_KEYS)
                        if spent:
                            raise AllKeysExhausted(
                                f"All {len(API_KEYS)} embedding API keys hit their daily quota. "
                                f"Usage today: {self._usage}. "
                                "Free-tier quota resets ~00:00 Pacific Time."
                            )
                        continue  # retry on another key, no backoff needed
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
        """Embed a list of texts, cache-first, with one worker per API key.

        Cache hits are free and stay on this thread; only the misses are
        farmed out, each pinned to its own key so no key exceeds its rate
        limit. Results keep the input order.
        """
        vectors = [None] * len(texts)
        misses = []
        for i, t in enumerate(texts):
            key = self._key(t)
            cached = self._cache.get(key)
            if cached is not None:
                vectors[i] = cached
            else:
                misses.append((i, t))

        done = len(texts) - len(misses)
        if on_progress and done:
            on_progress(done, len(texts))

        if not misses:
            return vectors

        error = None
        try:
            with ThreadPoolExecutor(max_workers=len(API_KEYS)) as pool:
                futures = {}
                for n, (slot, text) in enumerate(misses):
                    fut = pool.submit(self._call_api, text, n % len(API_KEYS))
                    futures[fut] = (slot, text)
                for fut in as_completed(futures):
                    slot, text = futures[fut]
                    try:
                        vec = fut.result()
                    except Exception as e:  # remember the first failure
                        error = error or e
                        continue
                    vectors[slot] = vec
                    with self._lock:
                        self._cache[self._key(text)] = vec
                        self._dirty_count += 1
                    done += 1
                    if on_progress:
                        on_progress(done, len(texts))
        finally:
            self._save_cache()  # always persist whatever we got
            self._save_usage()

        if error is not None:
            raise error
        return vectors
