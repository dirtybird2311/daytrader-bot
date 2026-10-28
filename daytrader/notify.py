"""Optional delivery of a finished report by email and as a GitHub job summary.

Email needs three environment variables (GitHub repo secrets):
  SMTP_USER       e.g. you@gmail.com
  SMTP_PASSWORD   a Gmail app password (not your normal password)
  REPORT_EMAIL_TO where to send it
Without them, email is skipped quietly.
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path


def summary_markdown(summary: dict) -> str:
    verdict = "✅ PASS" if summary["passed"] else "❌ FAIL"
    lines = [
        f"## Backtest {summary['period']}: {verdict}" + (" (SYNTHETIC DATA)" if summary.get("synthetic") else ""),
        "",
        f"Sharpe **{summary['sharpe']}** · annual return **{summary['annual_return']:.1%}** · "
        f"max drawdown **{summary['max_drawdown']:.1%}** · trades **{summary['trades']:,}**",
        "",
        "| Check | Result |", "|---|---|",
    ]
    for name, ok in summary["checks"].items():
        lines.append(f"| {name} | {'✓ Pass' if ok else ('— Not run' if ok is None else '✕ Fail')} |")
    lines += ["", "Full report: download the **report** artifact below, or check your email."]
    return "\n".join(lines)


def send_email(summary: dict, report: Path) -> bool:
    user, pw, to = (os.environ.get(k) for k in ("SMTP_USER", "SMTP_PASSWORD", "REPORT_EMAIL_TO"))
    if not (user and pw and to):
        print("Email not configured; skipping.")
        return False
    msg = EmailMessage()
    verdict = "PASS" if summary["passed"] else "FAIL"
    msg["Subject"] = f"Backtest {summary['period']}: {verdict} (Sharpe {summary['sharpe']})"
    msg["From"], msg["To"] = user, to
    msg.set_content(summary_markdown(summary) + "\n\nOpen the attached HTML file in your browser.")
    msg.add_attachment(report.read_bytes(), maintype="text", subtype="html", filename=report.name)
    with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"), 465) as s:
        s.login(user, pw)
        s.send_message(msg)
    print(f"Emailed report to {to}")
    return True


def main(summary_path: str = "reports/latest_summary.json") -> None:
    summary = json.loads(Path(summary_path).read_text())
    md = summary_markdown(summary)
    step = os.environ.get("GITHUB_STEP_SUMMARY")
    if step:
        with open(step, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    else:
        print(md)
    send_email(summary, Path(summary["report"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
