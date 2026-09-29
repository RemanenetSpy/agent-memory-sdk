#!/usr/bin/env python
"""Seed a demo memory store with realistic multi-domain data.

Usage:
    python scripts/seed_demo.py [--data-dir PATH]

Creates memories across all types, scopes, and confidence levels so the
Streamlit dashboard shows populated charts, a full browse list, and
working resolve results.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory.manager import Memory
from agent_memory.models import MemoryScope, MemoryType


def seed(data_dir: str = "/tmp/agent_memory_demo") -> Memory:
    m = Memory(persist_dir=data_dir, collection_name="agent_memories")

    # ── Authentication & Account ──────────────────────────────────────────────
    m.remember(
        "How do I reset my password?",
        "Go to Settings → Security → Reset Password. A reset link is emailed to you. "
        "Links expire after 30 minutes.",
        type=MemoryType.CONVERSATION, scope=MemoryScope.GLOBAL,
        tags=["auth", "password", "account"], confidence=1.0,
    )
    m.remember(
        "How do I enable two-factor authentication?",
        "Settings → Security → Two-Factor Auth → Enable. "
        "Supports TOTP (Google Authenticator, Authy) and SMS.",
        type=MemoryType.WORKFLOW, scope=MemoryScope.GLOBAL,
        tags=["auth", "security", "2fa"], confidence=1.0,
    )
    m.remember(
        "What are the SSO options?",
        "We support SAML 2.0, Google Workspace, Okta, and Azure AD. "
        "Configure under Settings → Security → SSO.",
        type=MemoryType.FACT, scope=MemoryScope.TEAM,
        tags=["auth", "sso", "enterprise"], confidence=0.97,
    )
    m.remember(
        "How do I revoke an API token?",
        "Settings → API Keys → select the key → Revoke. "
        "Revocation is immediate; existing requests in-flight are not affected.",
        type=MemoryType.WORKFLOW, scope=MemoryScope.USER,
        tags=["auth", "api", "security"], confidence=0.99,
    )

    # ── Billing & Subscriptions ────────────────────────────────────────────────
    m.remember(
        "What payment methods do you accept?",
        "Visa, Mastercard, American Express, PayPal, and wire transfer for annual plans.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["billing", "payments"], confidence=1.0,
    )
    m.remember(
        "How do I cancel my subscription?",
        "Account → Billing → Cancel Plan. Your access continues until the current period ends. "
        "No partial refunds for monthly plans.",
        type=MemoryType.CONVERSATION, scope=MemoryScope.USER,
        tags=["billing", "subscription", "cancel"], confidence=0.95,
    )
    m.remember(
        "How do I upgrade to the Pro plan?",
        "Account → Billing → Change Plan → Pro. Prorated credit is applied immediately.",
        type=MemoryType.WORKFLOW, scope=MemoryScope.USER,
        tags=["billing", "upgrade", "pro"], confidence=0.98,
    )
    m.remember(
        "Is there a free trial?",
        "Yes — 14-day free trial on all paid plans. No credit card required. "
        "Automatically moves to free tier at end of trial.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["billing", "trial", "pricing"], confidence=1.0,
    )

    # ── API & Developer ────────────────────────────────────────────────────────
    m.remember(
        "What are the API rate limits?",
        "Free: 100 req/min. Pro: 1 000 req/min. Enterprise: unlimited with burst allowance.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["api", "limits", "developer"], confidence=1.0,
        requires_verification=True,
    )
    m.remember(
        "How do I authenticate API requests?",
        "Include Authorization: Bearer <YOUR_TOKEN> in every request header. "
        "Tokens are generated in Settings → API Keys.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["api", "auth", "developer"], confidence=1.0,
    )
    m.remember(
        "Where is the API documentation?",
        "Full reference at docs.example.com/api. "
        "OpenAPI spec at docs.example.com/api/openapi.json. Postman collection available.",
        type=MemoryType.DOCUMENT, scope=MemoryScope.GLOBAL,
        tags=["api", "docs", "developer"], confidence=0.98,
    )
    m.remember(
        "Which SDKs are officially supported?",
        "Python (pip install example-sdk), Node.js (npm install example-sdk), "
        "Go (go get example.com/sdk), and Ruby (gem install example-sdk).",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["sdk", "api", "developer"], confidence=0.97,
    )
    m.remember(
        "How do I handle API pagination?",
        "Use cursor-based pagination: pass cursor=<next_cursor> returned in each response. "
        "Default page size is 20; max is 100 via limit=100.",
        type=MemoryType.FACT, scope=MemoryScope.PROJECT,
        tags=["api", "pagination", "developer"], confidence=0.96,
    )

    # ── Data & Storage ─────────────────────────────────────────────────────────
    m.remember(
        "Can I export my data?",
        "Yes — Settings → Data → Export. Formats: CSV, JSON, or Parquet. "
        "Exports are available for download for 7 days.",
        type=MemoryType.CONVERSATION, scope=MemoryScope.USER,
        tags=["data", "export", "storage"], confidence=0.90,
    )
    m.remember(
        "How long is data retained?",
        "Active accounts: indefinitely. "
        "Deleted accounts: data purged after 30 days. "
        "Audit logs retained for 1 year on Pro+.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["data", "retention", "compliance"], confidence=0.93,
    )
    m.remember(
        "What is the storage limit per plan?",
        "Free: 1 GB. Pro: 50 GB. Business: 500 GB. Enterprise: custom.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["storage", "limits", "pricing"], confidence=1.0,
    )

    # ── Team & Collaboration ───────────────────────────────────────────────────
    m.remember(
        "How do I invite team members?",
        "Settings → Team → Invite Members. Enter email addresses, select role, send. "
        "Invitations expire after 7 days.",
        type=MemoryType.WORKFLOW, scope=MemoryScope.TEAM,
        tags=["team", "invite", "collaboration"], confidence=1.0,
    )
    m.remember(
        "What roles are available?",
        "Owner, Admin, Editor, Viewer, and Guest. "
        "Roles control access to billing, settings, data, and API keys.",
        type=MemoryType.FACT, scope=MemoryScope.TEAM,
        tags=["team", "roles", "permissions"], confidence=0.99,
    )
    m.remember(
        "Can I set up audit logging?",
        "Yes — Settings → Security → Audit Log. Available on Business and Enterprise plans. "
        "Logs all user actions with timestamps and IPs.",
        type=MemoryType.FACT, scope=MemoryScope.WORKSPACE,
        tags=["audit", "security", "compliance", "enterprise"], confidence=0.94,
    )

    # ── SLA & Support ──────────────────────────────────────────────────────────
    m.remember(
        "What is the enterprise SLA?",
        "99.99% uptime SLA. P1 incident response: 15 minutes. "
        "P2: 2 hours. Dedicated support engineer included.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["sla", "enterprise", "support"], confidence=0.92,
        requires_verification=True,
    )
    m.remember(
        "How do I contact support?",
        "In-app chat (24/7 for Pro+), email support@example.com, "
        "or open a ticket at help.example.com. Phone support for Enterprise.",
        type=MemoryType.CONVERSATION, scope=MemoryScope.GLOBAL,
        tags=["support", "contact"], confidence=1.0,
    )

    # ── Integrations ──────────────────────────────────────────────────────────
    m.remember(
        "Which integrations are available?",
        "Slack, Jira, GitHub, GitLab, Zapier, Make, Salesforce, HubSpot, "
        "and all major cloud providers (AWS, GCP, Azure).",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["integrations", "third-party"], confidence=0.96,
    )
    m.remember(
        "How do I set up the Slack integration?",
        "Settings → Integrations → Slack → Connect. "
        "Authorize the app, pick your workspace channel, then choose which events to post.",
        type=MemoryType.WORKFLOW, scope=MemoryScope.TEAM,
        tags=["integrations", "slack", "notifications"], confidence=0.98,
    )
    m.remember(
        "Does the API support webhooks?",
        "Yes — Settings → Webhooks → Add Endpoint. "
        "HMAC-SHA256 signature verification included. Retry logic with exponential back-off.",
        type=MemoryType.FACT, scope=MemoryScope.PROJECT,
        tags=["api", "webhooks", "developer"], confidence=0.97,
    )

    # ── Code snippets ─────────────────────────────────────────────────────────
    m.remember(
        "Python SDK — how to initialise the client?",
        "```python\nfrom example_sdk import Client\nclient = Client(api_key='YOUR_KEY')\nresult = client.query('What is the status?')\n```",
        type=MemoryType.CODE, scope=MemoryScope.PROJECT,
        tags=["python", "sdk", "code"], confidence=1.0,
    )
    m.remember(
        "Node.js SDK — how to make a request?",
        "```js\nimport { Client } from 'example-sdk';\nconst client = new Client({ apiKey: process.env.API_KEY });\nconst res = await client.query('Status check');\n```",
        type=MemoryType.CODE, scope=MemoryScope.PROJECT,
        tags=["nodejs", "sdk", "code"], confidence=1.0,
    )

    # ── User preferences ──────────────────────────────────────────────────────
    m.remember(
        "User prefers dark mode",
        "User has dark mode enabled across all interfaces. "
        "Apply dark theme by default; do not suggest switching.",
        type=MemoryType.PREFERENCE, scope=MemoryScope.USER,
        tags=["preference", "ui", "dark-mode"], confidence=0.99,
    )
    m.remember(
        "User's timezone is Europe/London",
        "Always display times in Europe/London (GMT/BST). "
        "User is based in London, UK.",
        type=MemoryType.PREFERENCE, scope=MemoryScope.USER,
        tags=["preference", "timezone", "locale"], confidence=1.0,
    )
    m.remember(
        "User prefers concise responses",
        "Keep answers under 3 sentences where possible. "
        "Avoid bullet lists unless explicitly comparing 3+ items.",
        type=MemoryType.PREFERENCE, scope=MemoryScope.USER,
        tags=["preference", "communication"], confidence=0.95,
    )

    # ── Archived / expired entries (to show non-active states) ────────────────
    past = datetime.now(timezone.utc) - timedelta(days=2)
    m.store.store(__import__("agent_memory.models", fromlist=["MemoryEntry"]).MemoryEntry(
        query="Old API endpoint — /v1/users (deprecated)",
        response="The /v1/users endpoint was removed in v2. Use /v2/accounts instead.",
        type=MemoryType.FACT, scope=MemoryScope.PROJECT,
        tags=["api", "deprecated"], confidence=0.5, archived=True,
        created_at=past, updated_at=past,
    ))

    future = datetime.now(timezone.utc) - timedelta(hours=1)
    m.store.store(__import__("agent_memory.models", fromlist=["MemoryEntry"]).MemoryEntry(
        query="Temporary maintenance window tonight",
        response="Scheduled maintenance 02:00–04:00 UTC on 2024-01-15. API will be unavailable.",
        type=MemoryType.FACT, scope=MemoryScope.GLOBAL,
        tags=["maintenance", "status"], confidence=1.0,
        expires_at=future,
    ))

    # ── Demo: from_conversation ───────────────────────────────────────────────
    m.from_conversation(
        human="I'm a senior engineer and I prefer dark mode and Python.",
        assistant="Got it — noted your preference for dark mode and Python.",
        scope=MemoryScope.USER,
    )

    # ── Demo: mark_correct / mark_wrong feedback loop ─────────────────────────
    d_correct = m.resolve("How do I reset my password?")
    if d_correct.action.value != "none":
        m.mark_correct(d_correct)   # boost confidence on a verified good answer

    d_wrong = m.resolve("What is the enterprise SLA?")
    if d_wrong.action.value != "none":
        m.mark_wrong(d_wrong)       # penalise the SLA memory (marked requires_verification)

    # ── Demo: verify() — simulates a tool-check that confirms the answer ──────
    d_verify = m.resolve("What are the API rate limits?")
    if d_verify.action.value == "verify":
        m.verify(d_verify, verifier=lambda d: True)   # simulated check passes

    # ── Demo: graph scores ────────────────────────────────────────────────────
    m.refresh_graph_scores()   # wires PageRank into scoring (graph_weight=0 by default)

    # ── Bump access counts on frequently-used memories ────────────────────────
    entries = m.store.list_all(limit=5)
    for i, entry in enumerate(entries):
        if entry:
            entry.access_count = (i + 1) * 3
            entry.last_accessed_at = datetime.now(timezone.utc)
            m.store.update(entry)

    total = m.store.count
    print(f"✓ Seeded {total} memories into {data_dir}")
    print(f"  Types:  {', '.join(f'{t.value}' for t in MemoryType)}")
    print(f"  Scopes: {', '.join(f'{s.value}' for s in MemoryScope)}")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed demo data")
    parser.add_argument("--data-dir", default="/tmp/agent_memory_demo", help="Store path")
    args = parser.parse_args()
    seed(args.data_dir)
