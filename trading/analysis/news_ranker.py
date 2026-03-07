"""
Rank news articles by relevance to a market event (symbol, company, direction).
Adds relevance_score and relevance_reason to each article for UI display.
"""
import re
from typing import List


def rank_news_by_relevance(
    articles: List[dict],
    symbol: str,
    company: str,
    direction: str,
) -> List[dict]:
    """
    Rank articles by relevance to the event. Mutates and returns the same list
    with "relevance_score" (0-5) and "relevance_reason" added to each article.
    Results sorted by relevance_score descending.
    """
    symbol_upper = (symbol or "").upper()
    company_lower = (company or "").lower()
    keywords = set()
    if symbol_upper:
        keywords.add(symbol_upper)
    if company_lower:
        keywords.add(company_lower)
        # Add single-word company name if multi-word (e.g. "Apple" from "Apple Inc")
        for part in company_lower.split():
            if len(part) > 2:
                keywords.add(part)

    for a in articles:
        title = (a.get("title") or "").lower()
        desc = (a.get("description") or "").lower()
        text = f"{title} {desc}"
        score = 0
        reasons = []
        # Direct symbol/company mention
        if symbol_upper and symbol_upper in (a.get("title") or "").upper():
            score += 2
            reasons.append("title mentions symbol")
        if company_lower and company_lower in title:
            score += 2
            reasons.append("title mentions company")
        # Keyword in title
        for kw in keywords:
            if len(kw) > 2 and kw.lower() in title:
                score += 1
                reasons.append(f"keyword '{kw}' in title")
                break
        # Direction-related words (earnings, beat, miss, surge, drop, etc.)
        if direction == "up":
            if re.search(r"\b(surge|rally|gain|beat|earnings|raise|upgrade)\b", text):
                score += 1
                reasons.append("positive/up language")
        else:
            if re.search(r"\b(drop|fall|miss|cut|downgrade|decline)\b", text):
                score += 1
                reasons.append("negative/down language")
        score = min(5, score)
        a["relevance_score"] = score
        a["relevance_reason"] = "; ".join(reasons) if reasons else "generic"

    return sorted(articles, key=lambda x: x.get("relevance_score", 0), reverse=True)
