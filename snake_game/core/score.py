# core/score.py
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCORE_FILE = os.path.join(DATA_DIR, "score.json")

class ScoreManager:
    def __init__(self):
        self.current_score = 0
        self.high_score = 0
        self._ensure_data_dir()
        self._load()

    def reset(self):
        self.current_score = 0

    def increment(self, amount=1):
        self.current_score += amount
        if self.current_score > self.high_score:
            self.high_score = self.current_score
            self._save()

    def _load(self):
        try:
            with open(SCORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.high_score = data.get("high_score", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.high_score = 0

    def _save(self):
        try:
            with open(SCORE_FILE, "w", encoding="utf-8") as f:
                json.dump({"high_score": self.high_score}, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving score: {e}")

    def _ensure_data_dir(self):
        os.makedirs(DATA_DIR, exist_ok=True)
