"""Submit a forged-image .zip to the watermark-forgery leaderboard.

Adapted from the provided ``submission.py``. The API key is read from the
``TML_API_KEY`` environment variable instead of being hard-coded, so no secret
lands in git.

    export TML_API_KEY=<your key>
    python -m scripts.submit --file submissions/best.zip

Reminder: one submission per group every 60 minutes; a failed/malformed
submission only costs a 2-minute cooldown.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

BASE_URL = "http://34.63.153.158"  # do not change
TASK_ID = "22-forging-task"  # do not change


def parse_args():
    p = argparse.ArgumentParser(description="Submit a forged-image zip to the leaderboard")
    p.add_argument("--file", required=True, help="path to the submission .zip")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("TML_API_KEY")
    if not api_key:
        sys.exit("TML_API_KEY environment variable is not set")
    if not os.path.isfile(args.file):
        sys.exit(f"File not found: {args.file}")

    try:
        with open(args.file, "rb") as f:
            files = {"file": (os.path.basename(args.file), f, "zip")}
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": api_key},
                files=files,
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            sys.exit("Upload rejected: file too large (HTTP 413).")
        resp.raise_for_status()
        print("Successfully submitted.")
        print("Server response:", body)
        if body.get("submission_id"):
            print("Submission ID:", body["submission_id"])
    except requests.exceptions.RequestException as e:
        print(f"Submission error: {e}")
        detail = getattr(e, "response", None)
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
