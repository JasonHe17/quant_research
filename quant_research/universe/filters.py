"""Universe filter placeholders."""

from __future__ import annotations

import pandas as pd

from quant_research.universe.models import Universe


def identity_filter(universe: Universe) -> Universe:
    """Return a universe unchanged."""

    return universe


def active_on(universe: Universe, date: str) -> Universe:
    """Filter members whose effective interval contains ``date``."""

    members = universe.members.copy()
    start_ok = members["effective_from"].isna() | (members["effective_from"] <= date)
    end_ok = members["effective_to"].isna() | (members["effective_to"] >= date)
    filtered = members.loc[start_ok & end_ok].reset_index(drop=True)
    diagnostics = pd.concat(
        [
            universe.diagnostics,
            pd.DataFrame(
                [
                    {
                        "universe_name": universe.spec.name,
                        "filter": "active_on",
                        "date": date,
                        "member_count": len(filtered),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return Universe(
        spec=universe.spec,
        members=filtered,
        diagnostics=diagnostics,
        artifacts=dict(universe.artifacts),
    )
