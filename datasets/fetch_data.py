from datetime import datetime, timezone
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise ValueError("API_TOKEN not found in .env file.")

BASE_URL = "https://ev.caltech.edu/api/v1/sessions"

sites = ["caltech", "jpl", "office001"]

start_time = datetime(2016, 1, 1, tzinfo=timezone.utc)
end_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Needs to be like this
start_str = start_time.strftime("%a, %-d %b %Y %H:%M:%S GMT")
end_str = end_time.strftime("%a, %-d %b %Y %H:%M:%S GMT")

where = (
    f'connectionTime >= "{start_str}" '
    f'and connectionTime <= "{end_str}"'
)

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

#for site_name, site_id in sites.items():
for site in sites:
    print(f"\nDownloading {site}...")

    all_sessions = []

    url = f"{BASE_URL}/{site}"
    params = {"where": where}

    while True:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        payload = response.json()

        if "_items" not in payload:
            print(json.dumps(payload, indent=2))
            raise RuntimeError("Unexpected API response.")

        sessions = payload["_items"]
        all_sessions.extend(sessions)

        print(f"Retrieved {len(sessions)} sessions ({len(all_sessions)} total)")

        next_link = payload.get("_links", {}).get("next")

        if not next_link:
            break

        url = next_link["href"]

        # Handle relative URLs returned by ACN API
        if not url.startswith("http"):
            url = "https://ev.caltech.edu/api/v1/" + url.lstrip("/")

        params = None

    filename = f"{site}_2016_2025.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, indent=2)

    print(f"Saved {len(all_sessions)} sessions to {filename}")

print("\nDone!")