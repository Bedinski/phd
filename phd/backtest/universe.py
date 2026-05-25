"""Universe selection. Returns a list of tickers for the requested universe."""

from __future__ import annotations

# A handful of canonical tickers. The full S&P 500 list belongs in a data file
# and is fetched lazily — but for the smoke path we keep a small static list.

SP500_TOP10 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "BRK-B", "LLY", "AVGO", "TSLA",
]

NASDAQ100_TOP10 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AVGO", "ADBE", "COST",
]


def resolve(base: str, custom: list[str] | None = None) -> list[str]:
    if base == "sp500_top10":
        return list(SP500_TOP10)
    if base == "nasdaq100":
        # Use the curated top-10 as a stand-in until we wire the full list.
        return list(NASDAQ100_TOP10)
    if base == "sp500":
        # Stand-in: top 10 — full universe download will be wired in later via
        # Alpaca / a static CSV checked into the repo.
        return list(SP500_TOP10)
    if base == "custom":
        if not custom:
            raise ValueError("universe.base=custom requires custom_tickers")
        return list(custom)
    raise ValueError(f"unknown universe base: {base!r}")
