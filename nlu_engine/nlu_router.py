
# nlu_engine/nlu_router.py
from nlu_engine.infer_intent import IntentClassifier
from nlu_engine.entity_extractor import EntityExtractor

MODEL_DIR = "models/intent_model"

class NLUProcessor:
    def __init__(self):
        self.intent_model = IntentClassifier(model_dir=MODEL_DIR)
        self.entity_extractor = EntityExtractor()

    def process(self, text):
        intent_res = self.intent_model.predict(text, top_k=1)[0]
        intent = intent_res["intent"]
        confidence = intent_res["score"]
        entities = self.entity_extractor.extract(text)
        return intent, confidence, entities
      
