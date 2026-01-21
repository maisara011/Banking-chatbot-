
# nlu_engine/entity_extractor.py
"""
Robust rule-based entity extractor (currency/amount, account numbers, txn ids).
Rules:
 - TXN IDs (txn, txn id, transaction id, utr, ref no, etc.) — high priority
 - ACCOUNT NUMBER when context words appear (account, acct, a/c, to account, account no)
 - AMOUNT only when currency symbol or currency words present ($, ₹, Rs, INR, dollars, rupees)
 - Avoid overlapping labels by reserving matched spans
"""

import re

class EntityExtractor:
    def __init__(self):
        # regexes (case-insensitive)
        # TXN / transaction id patterns (examples: txn 12345, txn id: 12-34-ABC, UTR: ABC123456)
        self.txn_patterns = [
            re.compile(r'\b(?:txn(?:id| _id| id)?|transaction(?:\s*id)?|txnid|utr|ref(?:erence)?\s*no\.?|ref)\b[:\s\-]*([A-Za-z0-9\-_/]{4,40})', re.I),
            # Some banks use UTR/REF followed by alphanumeric code
            re.compile(r'\b(UTR|REF|TXN)\b[:\s\-]*([A-Za-z0-9\-_/]{4,40})', re.I),
        ]

        # Account context patterns: capture "account 1234", "to account 12345", "acct no 12345"
        self.account_patterns = [
            re.compile(r'\b(?:account|acct|a\/c|account\s*no|account\s*number|acct\.?)\b[:\s\-]*([0-9]{4,24})', re.I),
            re.compile(r'\bto\s+(?:account|acct|a\/c)\b[:\s\-]*([0-9]{4,24})', re.I),
            # sometimes "account ending 1234"
            re.compile(r'\baccount(?:\s+ending)?\s*(?:no\.?|number)?\s*([0-9]{4,24})\b', re.I),
        ]

        # AMOUNT patterns: require currency symbol or currency word
        self.amount_patterns = [
            # $1000, ₹1,000.00, Rs. 1000, INR 1000
            re.compile(r'(?P<amt>\b(?:₹|\$|Rs\.?|INR|USD)\s*[0-9][0-9,]*(?:\.\d+)?\b)', re.I),
            re.compile(r'(?P<amt>\b[0-9][0-9,]*(?:\.\d+)?\s*(?:rupees|rs|inr|usd|dollars)\b)', re.I),
        ]

        # A generic number pattern (only used with context; does not label by itself)
        self.number_pattern = re.compile(r'\b[0-9]{2,20}\b')

    def _reserve_span(self, reserved, start, end):
        """Return True if span overlaps existing reserved spans; otherwise add and return False"""
        for (a, b) in reserved:
            if not (end <= a or start >= b):
                return True
        reserved.append((start, end))
        return False

    def _normalize_amount(self, raw):
        # remove currency symbol/words and commas, strip and return numeric string
        s = raw.strip()
        s = re.sub(r'[₹,$]', '', s)
        s = re.sub(r'\b(Rs\.?|rs|INR|USD|usd|dollars|rupees)\b', '', s, flags=re.I)
        s = s.replace(',', '')
        s = s.strip()
        return s

    def extract(self, text):
        """
        Extract entities from text and return list of dicts:
          [ {"entity":"AMOUNT", "value":"1000"}, {"entity":"ACCOUNT_NUMBER","value":"11223344"}, ... ]
        """

        if not text:
            return []

        reserved_spans = []  # list of (start, end) spans already labeled
        results = []

        # 1) TXN IDs (high priority)
        for pat in self.txn_patterns:
            for m in pat.finditer(text):
                # group with code may be either group 1 or 2 depending on pattern
                groups = [g for g in m.groups() if g]
                if not groups:
                    continue
                code = groups[-1].strip()
                start, end = m.span()
                if self._reserve_span(reserved_spans, start, end):
                    continue
                results.append({"entity": "TXN_ID", "value": code})

        # 2) ACCOUNT NUMBERS - require context words
        for pat in self.account_patterns:
            for m in pat.finditer(text):
                # group(1) should be the numeric part
                try:
                    num = m.group(1)
                except IndexError:
                    continue
                if not num:
                    continue
                start, end = m.span()
                if self._reserve_span(reserved_spans, start, end):
                    continue
                # normalize: remove spaces/commas
                num_norm = re.sub(r'\D', '', num)
                if num_norm:
                    results.append({"entity": "ACCOUNT_NUMBER", "value": num_norm})

        # 3) AMOUNTS - require currency symbol or currency word present
        for pat in self.amount_patterns:
            for m in pat.finditer(text):
                raw = m.group(0)
                start, end = m.span()
                if self._reserve_span(reserved_spans, start, end):
                    continue
                normalized = self._normalize_amount(raw)
                if normalized:
                    results.append({"entity": "AMOUNT", "value": normalized})

        # 4) Additional rule: numbers that follow keywords like "to account" (caught above),
        # or "account ending 1234" (caught above). Avoid marking plain numbers otherwise.
        # But still check if there are numbers with context "transfer of 500 to account 1234"
        # (account 1234 would be caught earlier). For safety, we do not label other numbers.

        # Optional: If you want to detect phone-like or reference numbers with context "txn" or "ref", already covered.

        return results


# Simple wrapper
def extract(text):
    ex = EntityExtractor()
    return ex.extract(text)

                 
