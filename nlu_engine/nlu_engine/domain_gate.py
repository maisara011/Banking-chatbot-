BANK_KEYWORDS = {
    "account", "balance", "transfer", "money", "send", "receive",
    "card", "debit", "credit", "atm", "bank", "withdraw", "deposit",
    "pin", "password"
}

BANK_ENTITIES = {
    "ACCOUNT_NUMBER",
    "AMOUNT",
    "CARD_TYPE",
    "ACCOUNT_TYPE"
}


def is_banking_query(text: str, entities: list) -> bool:
    """
    Hybrid domain gate:
    - Keyword-based
    - Entity-based
    """

    text_l = text.lower()

    # Signal 1: Keyword presence
    keyword_hit = any(k in text_l for k in BANK_KEYWORDS)

    # Signal 2: Entity presence
    entity_hit = any(
        e.get("entity") in BANK_ENTITIES
        for e in entities
    )

    return keyword_hit or entity_hit
