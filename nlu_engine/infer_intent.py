
# nlu_engine/infer_intent.py
# Safer inference wrapper — delays heavy imports and provides clear error messages

import os
import json

class IntentClassifier:
    def __init__(self, model_dir="models/intent_model"):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir {model_dir} not found. Train the model first.")
        self.model_dir = model_dir
        # Delayed imports
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import numpy as np
        except Exception as e:
            raise RuntimeError("Failed to import transformers/torch. Ensure your environment has them installed.") from e

        self._torch = __import__("torch")
        self.np = __import__("numpy")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        # load id2label
        with open(os.path.join(model_dir, "id2label.json"), "r", encoding="utf-8") as f:
            self.id2label = json.load(f)

    def predict(self, text, top_k=1):
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze().cpu().numpy()
            probs = self._torch.softmax(self._torch.tensor(logits), dim=0).numpy()
            top_idx = self.np.argsort(probs)[::-1][:top_k]
            results = []
            for idx in top_idx:
                results.append({"intent": self.id2label[str(int(idx))], "score": float(probs[int(idx)])})
            return results

if __name__ == "__main__":
    try:
        ic = IntentClassifier()
        print(ic.predict("Please transfer 5000 to my savings account", top_k=3))
    except Exception as e:
        print("Error:", e)
              
