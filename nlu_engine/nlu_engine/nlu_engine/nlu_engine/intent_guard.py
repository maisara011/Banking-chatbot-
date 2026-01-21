# nlu_engine/intent_guard.py

BANKING_INTENTS = {
    "transfer_money",
    "check_balance",
    "card_block",
    "greet"
}

def is_banking_intent(intent: str) -> bool:
    return intent in BANKING_INTENTS
