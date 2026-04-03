"""Log Pattern Analyzer tool for the RadVision Pro support agent.

Regex-based matching of error messages against known error signatures
loaded from product_spec.yaml. This is pattern matching — NOT vector search.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from src.config import SEED_PATH

logger = logging.getLogger(__name__)


class LogAnalysisResult(BaseModel):
    matched: bool
    pattern_id: Optional[str] = None
    kb_ref: Optional[str] = None
    component: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    matched_text: Optional[str] = None


class LogAnalyzer:
    """Matches log/error strings against known RadVision Pro error patterns."""

    def __init__(self, seed_path: Path = SEED_PATH) -> None:
        with seed_path.open() as fh:
            spec = yaml.safe_load(fh)

        self._patterns: list[dict] = []
        for raw in spec.get("error_patterns", []):
            try:
                self._patterns.append({**raw, "_re": re.compile(raw["regex"], re.IGNORECASE)})
            except re.error as exc:
                logger.warning("Invalid regex in pattern %s: %s", raw.get("pattern_id"), exc)

        logger.info("LogAnalyzer loaded %d error patterns", len(self._patterns))

    def analyze_log(self, error_text: str) -> LogAnalysisResult:
        """Return the first pattern that matches error_text, or matched=False."""
        if not error_text or not error_text.strip():
            return LogAnalysisResult(matched=False)

        for pattern in self._patterns:
            m = pattern["_re"].search(error_text)
            if m:
                return LogAnalysisResult(
                    matched=True,
                    pattern_id=pattern["pattern_id"],
                    kb_ref=pattern.get("kb_ref"),
                    component=pattern["component"],
                    severity=pattern["severity"],
                    description=pattern["description"],
                    matched_text=m.group(0),
                )

        return LogAnalysisResult(matched=False)


def analyze_log(error_text: str) -> dict:
    """Module-level entry point used by the LangGraph tool node."""
    return LogAnalyzer().analyze_log(error_text).model_dump()
