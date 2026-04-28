import config
import numpy as np

CONF_WEIGHT = 0.5

def compute_severity(predicted_class: str, confidence: float) -> dict:
  lookup_key = str(predicted_class).lower().strip()
  base = config.BASE_SEVERITY.get(lookup_key, 1)
  shift = (confidence - 0.50) * CONF_WEIGHT
  adjusted = base + shift
  severity_score = int(np.clip(round(adjusted), 1, 5))
  return {
      "severity_score": severity_score,
      "severity_label": config.SEVERITY_DESCRIPTIONS[severity_score],
      "base_score": base,
      "confidence": round(float(confidence), 4),
      "predicted_class": predicted_class
  }

def print_severity_report(result: dict) -> None:
  bar = "#" * result["severity_score"] + "-" * (5 - result["severity_score"])
  print(f"\nSeverity: {result['severity_score']}/5 [{bar}]")
  print(f"Class: {result['predicted_class']} | Assessment: {result['severity_label']}\n")
