"""Canonical team-name matching across data sources (league APIs, ESPN,
The Odds API), which disagree on accents, punctuation, and city spellings."""

import unicodedata

# Map known alternate spellings to one canonical form (lowercase, no periods).
TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "oakland athletics": "athletics",
    "los angeles angels of anaheim": "los angeles angels",
    "st louis cardinals": "st. louis cardinals",
    "st louis blues": "st. louis blues",
    "washington football team": "washington commanders",
}


def team_key(name: str) -> str:
    """Canonical key for a team name: accent-stripped, lowercase, no
    periods, aliases applied. Use for any cross-source lookup."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().replace(".", "").split())
    s = TEAM_ALIASES.get(s, s)
    return s.replace(".", "")
