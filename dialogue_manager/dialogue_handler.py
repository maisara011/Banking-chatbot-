
# dialogue_manager/dialogue_handler.py

from database.bank_crud import (
    get_account,
    transfer_money,
    list_accounts
)
from database.db import get_conn


def log_interaction(user_text, intent, confidence=0.85, entities=None, success=True):
    """Log every interaction to chat_history for admin panel"""
    conn = get_conn()
    real_intent = intent if intent else "unknown"
    
    # Convert entities list to string for storage
    entities_str = str(entities) if entities else ""
    
    conn.execute("""
        INSERT INTO chat_history (user_query, predicted_intent, confidence, entities, success)
        VALUES (?, ?, ?, ?, ?)
    """, (user_text, real_intent, confidence, entities_str, 1 if success else 0))
    conn.commit()
    conn.close()


class DialogueManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active_intent = None
        self.slots = {}
        self.awaiting = None
        self.in_flow = False

    def handle(self, intent, entities, user_text):
        user_text = user_text.strip()
        
        # Save original intent before any override
        original_intent = intent
        
        # -------- GLOBAL CANCEL --------
        if user_text.lower() in ["cancel", "stop", "exit"]:
            self.reset()
            return "❌ Operation cancelled. How else can I help you?"

        # -------- INTENT LOCK --------
        if self.in_flow:
            intent = self.active_intent
        else:
            self.reset()
            self.active_intent = intent

        # -------- OUT OF SCOPE DETECTION --------
        if original_intent == "out_of_scope":
            self.reset()
            return None  # Signal to use LLM fallback

        # -------- GREET --------
        if intent == "greet":
            log_interaction(user_text, intent, 1.0, entities)
            return "Hello 👋 How can I help you today?"

        # -------- TRANSFER MONEY --------
        if intent == "transfer_money":
            self.in_flow = True
            return self._handle_transfer(entities, user_text)

        # -------- CHECK BALANCE --------
        if intent == "check_balance":
            self.in_flow = True
            return self._handle_check_balance(entities, user_text)

        # -------- CARD BLOCK --------
        if intent == "card_block":
            self.in_flow = True
            return self._handle_card_block(entities, user_text)

        # Log unknown queries
        log_interaction(user_text, intent, 0.5, entities, success=False)
        return "Sorry, I didn't understand. Please try again."

    # ================= TRANSFER FLOW =================
    def _handle_transfer(self, entities, user_text):
        # 1️⃣ Handle awaiting input first
        if self.awaiting == "amount":
            amount = self._parse_amount(user_text)
            if amount is None:
                return "__ERROR__:Please enter a valid numeric amount."
            self.slots["amount"] = amount
            self.awaiting = None

        elif self.awaiting == "password":
            self.slots["password"] = user_text
            self.awaiting = None

            accounts = list_accounts()
            receivers = [
                f"{u} ({a})" for a, u in accounts
                if a != self.slots["from_account"]
            ]
            self.awaiting = "receiver"
            return "Select receiver account:\n" + "\n".join(receivers)

        elif self.awaiting == "receiver":
            to_acc = user_text.split("(")[-1].replace(")", "")
            result = transfer_money(
                self.slots["from_account"],
                to_acc,
                self.slots["amount"],
                self.slots["password"]
            )
            
            # Log the transfer attempt
            success = result.startswith("✅")
            log_interaction(user_text, "transfer_money", 0.95, entities, success)
            
            self.reset()
            if result.startswith("❌"):
                return "__ERROR__:" + result.replace("❌ ", "")
            return "✅ Transfer Successful"

        # 2️⃣ Entity-based slot filling
        for e in entities:
            if e["entity"] == "ACCOUNT_NUMBER":
                self.slots["from_account"] = str(e["value"]).strip()
            elif e["entity"] == "AMOUNT":
                self.slots["amount"] = int(e["value"])

        # 3️⃣ Account number
        if "from_account" not in self.slots:
            if user_text.isdigit():
                self.slots["from_account"] = user_text
            else:
                return "Please enter your account number."

        acc = get_account(self.slots["from_account"])
        if not acc:
            self.slots.pop("from_account", None)
            return "__ERROR__:Invalid account number. Please re-enter your account number."

        # 4️⃣ Amount
        if "amount" not in self.slots:
            self.awaiting = "amount"
            return "How much amount do you want to transfer?"

        # 5️⃣ Password
        if "password" not in self.slots:
            self.awaiting = "password"
            return "Please enter your password to proceed."

        return "Processing transfer..."

    # ================= CHECK BALANCE FLOW =================
    def _handle_check_balance(self, entities, user_text):
        # Awaiting account number
        if self.awaiting == "account":
            acc_no = user_text.strip()
            acc = get_account(acc_no)
            if not acc:
                log_interaction(user_text, "check_balance", 0.9, entities, success=False)
                return "__ERROR__:Invalid account number. Please re-enter."
            balance = acc[3]
            log_interaction(user_text, "check_balance", 0.95, entities, success=True)
            self.reset()
            return f"💰 Your account balance is ₹{balance}"

        # Entity-based
        for e in entities:
            if e["entity"] == "ACCOUNT_NUMBER":
                acc = get_account(e["value"])
                if not acc:
                    log_interaction(user_text, "check_balance", 0.9, entities, success=False)
                    return "__ERROR__:Invalid account number."
                log_interaction(user_text, "check_balance", 0.95, entities, success=True)
                self.reset()
                return f"💰 Your account balance is ₹{acc[3]}"

        # Ask for account
        self.awaiting = "account"
        return "Please provide your account number to check balance."

    # ================= CARD BLOCK FLOW =================
    def _handle_card_block(self, entities, user_text):
        # Awaiting card type
        if self.awaiting == "card_type":
            card_type = user_text.lower()
            if card_type not in ["debit", "credit"]:
                return "__ERROR__:Please enter debit or credit."
            log_interaction(user_text, "card_block", 0.95, entities, success=True)
            self.reset()
            return f"🔒 Your {card_type} card has been blocked successfully."

        # Entity-based
        for e in entities:
            if e["entity"] == "CARD_TYPE":
                log_interaction(user_text, "card_block", 0.95, entities, success=True)
                self.reset()
                return f"🔒 Your {e['value']} card has been blocked successfully."

        self.awaiting = "card_type"
        return "Please provide your card type to block (debit / credit)."

    # ================= HELPERS =================
    def _parse_amount(self, text):
        import re
        match = re.search(r"\d+", text.replace(",", ""))
        return int(match.group()) if match else None
