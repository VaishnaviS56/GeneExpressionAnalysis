from __future__ import annotations

from functools import lru_cache

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gea_agent.config import SETTINGS


@lru_cache(maxsize=1)
def get_retrying_session() -> requests.Session:
    retry = Retry(
        total=max(0, int(SETTINGS.http_max_retries)),
        connect=max(0, int(SETTINGS.http_max_retries)),
        read=max(0, int(SETTINGS.http_max_retries)),
        backoff_factor=float(SETTINGS.http_backoff_factor),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "GEA-Agent/1.0"})
    return session
