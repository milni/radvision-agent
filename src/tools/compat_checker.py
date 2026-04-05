"""Compatibility Checker tool for the RadVision Pro support agent.

Structured matrix lookups — NOT vector search.

- check_compatibility(version, gpu, feature)  →  feature support status
- check_platform(version, os)                 →  OS platform support status
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.config import SEED_PATH

logger = logging.getLogger(__name__)


class CompatibilityResult(BaseModel):
    status: str        # "supported" | "limited" | "unsupported" | "experimental" | "unknown"
    reason: str        # Full text from the matrix, or explanation when unknown
    recommendation: str


def _parse_status(status_str: str) -> str:
    """Extract the canonical status from a YAML value like 'limited — requires ...'."""
    lower = status_str.lower()
    for token in ("supported", "limited", "unsupported", "experimental"):
        if lower.startswith(token):
            return token
    return "unknown"


class CompatibilityChecker:
    """Structured lookup against the RadVision Pro compatibility matrix."""

    def __init__(self, seed_path: Path = SEED_PATH) -> None:
        spec = yaml.safe_load(seed_path.read_text())
        platforms = spec.get("platforms", {})

        self._gpus: dict[str, dict] = {g["name"]: g for g in platforms.get("gpus", [])}
        self._oses: dict[str, dict] = {o["name"]: o for o in platforms.get("operating_systems", [])}

        # Index features with normalized keys (lowercase, spaces instead of underscores)
        # so lookups are case/format-insensitive: "Cardiac 4D", "Cardiac_4D", "cardiac 4d" all work.
        self._feature_index: dict[tuple[str, str], dict[str, str]] = {}
        for entry in platforms.get("feature_compatibility", []):
            normalized_features = {
                k.lower().replace("_", " "): v
                for k, v in entry.get("features", {}).items()
            }
            self._feature_index[(entry["version"], entry["gpu"])] = normalized_features

        logger.info(
            "CompatibilityChecker loaded %d GPUs, %d OSes, %d feature-compat rows",
            len(self._gpus), len(self._oses), len(self._feature_index),
        )

    def _resolve_gpu(self, gpu: str) -> str:
        """Normalize gpu name to match the matrix key.

        Tries exact match first, then common vendor prefix variants so that
        triage-extracted short names like 'T4' match 'NVIDIA T4' in the matrix.
        """
        candidates = [gpu, f"NVIDIA {gpu}", f"AMD {gpu}", f"Intel {gpu}"]
        all_gpus = {g for (_, g) in self._feature_index} | set(self._gpus)
        for c in candidates:
            if c in all_gpus:
                return c
        return gpu  # return as-is; caller handles not-found

    def versions_for_gpu(self, gpu: str) -> list[str]:
        """Return all RadVision Pro versions that have compatibility data for this GPU."""
        gpu = self._resolve_gpu(gpu)
        return sorted({v for (v, g) in self._feature_index if g == gpu})

    def check_compatibility(self, version: str, gpu: str, feature: str) -> CompatibilityResult:
        """Look up feature support for a given version and GPU.

        Args:
            version: e.g. "4.2"
            gpu: e.g. "NVIDIA T4"
            feature: e.g. "Cardiac 4D" or "Cardiac_4D" or "MPR"
        """
        gpu = self._resolve_gpu(gpu)
        feature_key = feature.strip().lower().replace("_", " ")
        features = self._feature_index.get((version, gpu))

        if features is None:
            return CompatibilityResult(
                status="unknown",
                reason=f"No compatibility data for version {version!r} with GPU {gpu!r}.",
                recommendation="Contact MedViz support or consult the release notes.",
            )

        if feature_key not in features:
            return CompatibilityResult(
                status="unknown",
                reason=f"Feature {feature!r} not found for {version}/{gpu}. Available: {', '.join(features)}.",
                recommendation="Verify the feature name and consult the product documentation.",
            )

        status_str = features[feature_key]
        status = _parse_status(status_str)

        if status == "supported":
            recommendation = f"{gpu} fully supports {feature} in version {version}."
        elif status == "limited":
            recommendation = (
                f"{gpu} has limited support for {feature}. "
                "Check configuration requirements. "
                "Consider upgrading to NVIDIA A5000 or A6000 for full support."
            )
        elif status == "unsupported":
            recommendation = (
                f"{gpu} does not support {feature}. "
                "Upgrade to NVIDIA A5000 (24 GB VRAM) or NVIDIA A6000 (48 GB VRAM)."
            )
        elif status == "experimental":
            recommendation = f"{gpu} has experimental support for {feature}. Not for production use."
        else:
            recommendation = "Verify GPU compatibility before deployment."

        return CompatibilityResult(status=status, reason=status_str, recommendation=recommendation)

    def check_platform(self, version: str, os: str) -> CompatibilityResult:
        """Look up OS platform support.

        Note: OS support is version-agnostic in the product spec;
        the version argument is accepted for API consistency.
        """
        os_info = self._oses.get(os)

        if os_info is None:
            return CompatibilityResult(
                status="unknown",
                reason=f"OS {os!r} is not in the compatibility matrix.",
                recommendation="RHEL 8 and RHEL 9 are the primary supported platforms.",
            )

        raw_status: str = os_info["status"]
        status = "supported" if "supported" in raw_status else raw_status
        notes: str = os_info.get("notes", "")

        if raw_status == "fully_supported":
            recommendation = "Platform is fully supported."
        else:
            recommendation = f"Platform is supported. Note: {notes}" if notes else "Platform is supported."

        return CompatibilityResult(status=status, reason=notes or raw_status, recommendation=recommendation)


def check_compatibility(version: str, gpu: str, feature: str) -> dict:
    """Module-level entry point used by the LangGraph tool node."""
    return CompatibilityChecker().check_compatibility(version, gpu, feature).model_dump()


def check_platform(version: str, os: str) -> dict:
    """Module-level entry point used by the LangGraph tool node."""
    return CompatibilityChecker().check_platform(version, os).model_dump()
