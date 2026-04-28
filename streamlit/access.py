"""Access-control gates for resource-intensive Streamlit pages.

Two-gate policy for the fleet tuner:
  1. The deployment must have set `TUNER_ENABLED=1` (the actual lock).
  2. The request must originate from a private/local IP, derived from
     `X-Forwarded-For` (best-effort; missing header is treated as local
     because direct host access has no XFF).
"""

from __future__ import annotations

import ipaddress
import os
from typing import Tuple

import streamlit as st


_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(ip in net for net in _PRIVATE_NETWORKS)


def tuner_access_state() -> Tuple[bool, str]:
    """Return (enabled, reason). When enabled is False, reason is a
    user-facing string explaining why."""
    if os.environ.get("TUNER_ENABLED", "0") != "1":
        return False, (
            "Fleet tuner is disabled in this deployment. "
            "It must be explicitly enabled by the operator (TUNER_ENABLED=1) "
            "and is intended for local-access use only."
        )

    headers = getattr(st.context, "headers", {}) or {}
    xff = headers.get("X-Forwarded-For", "") or headers.get("x-forwarded-for", "")
    client_ip = xff.split(",")[0].strip() if xff else ""

    if not client_ip:
        # Direct host access has no XFF; assume local.
        return True, ""

    if _is_private_ip(client_ip):
        return True, ""

    return False, (
        f"Fleet tuner is local-access only. Connection from {client_ip} is blocked. "
        "Run from the docker host's local network to enable."
    )
