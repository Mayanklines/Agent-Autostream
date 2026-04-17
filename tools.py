"""
Mock Lead Capture Tool.
Simulates the backend API call that stores a qualified lead.
"""

import json
import os
from datetime import datetime


LEADS_FILE = os.path.join(os.path.dirname(__file__), "..", "captured_leads.json")


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.

    Args:
        name: The lead's full name.
        email: The lead's email address.
        platform: The creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        A dict with capture status and lead_id.
    """
    print(f"\n{'='*50}")
    print(f"✅  Lead captured successfully!")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")

    # Persist to a local JSON file (simulates DB write)
    lead = {
        "lead_id": f"LEAD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "source": "Inflx Social Agent",
        "status": "new"
    }

    # Load existing leads
    existing = []
    if os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(lead)

    # Save back
    with open(LEADS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return {
        "success": True,
        "lead_id": lead["lead_id"],
        "message": f"Lead for {name} captured and queued for follow-up."
    }
