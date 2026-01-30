from __future__ import annotations

import matplotlib.pyplot as plt


def add_safety_clip_note(fig: plt.Figure, safety_clip: dict | None, where: str = "br"):
    """
    Adds a small note to the figure: "Safety clip: Used/Unused/Off".

    safety_clip dict (expected):
      {"enabled": bool, "any_used": bool, ...}
    """
    status = "Off"
    if safety_clip is not None:
        enabled = bool(safety_clip.get("enabled", False))
        any_used = bool(safety_clip.get("any_used", False))
        if enabled:
            status = "Used" if any_used else "Unused"
        else:
            status = "Off"

    txt = f"Safety clip: {status}"
    if where == "tr":
        x, y, va = 0.99, 0.99, "top"
    else:
        x, y, va = 0.99, 0.01, "bottom"

    fig.text(x, y, txt, ha="right", va=va, fontsize=9, alpha=0.85)
