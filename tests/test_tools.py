"""Tests for Phase 1 tools: LogAnalyzer and CompatibilityChecker.

All tests use the real product_spec.yaml seed — no mocking.
This validates that the seed data is internally consistent and that
the tools correctly parse and query it.
"""

import pytest

from src.tools.log_analyzer import LogAnalyzer, LogAnalysisResult, analyze_log
from src.tools.compat_checker import CompatibilityChecker, CompatibilityResult, check_compatibility, check_platform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> LogAnalyzer:
    return LogAnalyzer()


@pytest.fixture(scope="module")
def checker() -> CompatibilityChecker:
    return CompatibilityChecker()


# ---------------------------------------------------------------------------
# LogAnalyzer — basic matching
# ---------------------------------------------------------------------------


class TestLogAnalyzerMatches:
    def test_tls_renegotiation_error(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("AssocReject from PACS-01 reason=0x0006 after TLS handshake")
        assert result.matched is True
        assert result.pattern_id == "ERR-001"
        assert result.kb_ref == "KB-4231"
        assert result.component == "DICOM Gateway"
        assert result.severity == "high"

    def test_vram_fallback(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("VRAM_FALLBACK_TRIGGERED: switching to CPU compositor")
        assert result.matched is True
        assert result.pattern_id == "ERR-002"
        assert result.kb_ref == "KB-4298"
        assert result.component == "3D Rendering Engine"
        assert result.severity == "medium"

    def test_fhir_connection_pool_exhausted(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("ERROR: ConnectionPool.exhausted — all 10 slots busy")
        assert result.matched is True
        assert result.pattern_id == "ERR-003"
        assert result.kb_ref == "KB-4315"
        assert result.component == "Integration Gateway"

    def test_ice_timeout(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("WebRTC candidate_gather_slow: exceeded 5000ms deadline")
        assert result.matched is True
        assert result.pattern_id == "ERR-004"
        assert result.kb_ref == "KB-4330"
        assert result.component == "Collaboration Hub"
        assert result.severity == "low"

    def test_opengl_context_share_failure(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("OGL_CTX_SHARE_FAIL on display :1.0 — second monitor init")
        assert result.matched is True
        assert result.pattern_id == "ERR-005"
        assert result.kb_ref == "KB-3102"

    def test_hl7_thread_pool_full(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("HL7 thread_pool_full: dropping message ADT-A08")
        assert result.matched is True
        assert result.pattern_id == "ERR-006"
        assert result.kb_ref == "KB-3156"

    def test_dicom_store_timeout(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("DICOM timeout during C-STORE to destination AE")
        assert result.matched is True
        assert result.pattern_id == "ERR-007"
        assert result.kb_ref is None  # No KB article for this one

    def test_gpu_init_failure(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("GPU_INIT_FAIL: CUDA device 0 not available")
        assert result.matched is True
        assert result.pattern_id == "ERR-008"
        assert result.severity == "critical"

    def test_ldap_auth_failure(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("AUTH_FAIL LDAP: ldap_bind error -1 for user jdoe")
        assert result.matched is True
        assert result.pattern_id == "ERR-009"
        assert result.component == "Authentication"

    def test_disk_space_low(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("DISK_SPACE_LOW: storage threshold exceeded on /data")
        assert result.matched is True
        assert result.pattern_id == "ERR-010"
        assert result.severity == "critical"

    def test_alternate_tls_pattern(self, analyzer: LogAnalyzer) -> None:
        """TLS_RENEGO_FAIL is the alternate regex branch for ERR-001."""
        result = analyzer.analyze_log("TLS_RENEGO_FAIL on connection 192.168.1.10:4242")
        assert result.matched is True
        assert result.pattern_id == "ERR-001"

    def test_cardiac_render_cpu_compositor(self, analyzer: LogAnalyzer) -> None:
        """CardiacRender.*compositor=CPU is the alternate regex branch for ERR-002."""
        result = analyzer.analyze_log("CardiacRender phase=reconstruct compositor=CPU frames=120")
        assert result.matched is True
        assert result.pattern_id == "ERR-002"


# ---------------------------------------------------------------------------
# LogAnalyzer — no-match and edge cases
# ---------------------------------------------------------------------------


class TestLogAnalyzerNoMatch:
    def test_no_match_returns_false(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("INFO: RadVision Pro started successfully on port 8080")
        assert result.matched is False
        assert result.pattern_id is None
        assert result.kb_ref is None
        assert result.severity is None

    def test_empty_string(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("")
        assert result.matched is False

    def test_whitespace_only(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("   ")
        assert result.matched is False

    def test_case_insensitive_match(self, analyzer: LogAnalyzer) -> None:
        """Patterns must match regardless of case."""
        result = analyzer.analyze_log("assocreject from pacs reason=0x0006")
        assert result.matched is True
        assert result.pattern_id == "ERR-001"

    def test_matched_text_is_captured(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("prefix VRAM_FALLBACK_TRIGGERED suffix text")
        assert result.matched is True
        assert result.matched_text is not None
        assert "VRAM_FALLBACK_TRIGGERED" in result.matched_text.upper()

    def test_returns_pydantic_model(self, analyzer: LogAnalyzer) -> None:
        result = analyzer.analyze_log("anything here")
        assert isinstance(result, LogAnalysisResult)

    def test_module_level_function_returns_dict(self) -> None:
        result = analyze_log("VRAM_FALLBACK_TRIGGERED")
        assert isinstance(result, dict)
        assert result["matched"] is True
        assert result["pattern_id"] == "ERR-002"


# ---------------------------------------------------------------------------
# CompatibilityChecker — feature compatibility
# ---------------------------------------------------------------------------


class TestCompatCheckerFeatures:
    def test_a6000_cardiac4d_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA A6000", "Cardiac 4D")
        assert result.status == "supported"

    def test_t4_cardiac4d_limited(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA T4", "Cardiac 4D")
        assert result.status == "limited"
        # Should tell the field engineer what to do
        assert "PROGRESSIVE_RENDER" in result.reason or "limited" in result.reason.lower()

    def test_t4_vessel_analysis_unsupported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA T4", "Vessel Analysis")
        assert result.status == "unsupported"
        assert "VRAM" in result.reason or "unsupported" in result.reason.lower()

    def test_amd_mi210_vrt_experimental(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "AMD MI210", "VRT")
        assert result.status == "experimental"

    def test_amd_mi210_cardiac4d_unsupported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "AMD MI210", "Cardiac 4D")
        assert result.status == "unsupported"

    def test_a5000_mpr_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA A5000", "MPR")
        assert result.status == "supported"

    def test_unknown_version_returns_unknown(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("3.0", "NVIDIA A6000", "MPR")
        assert result.status == "unknown"

    def test_unknown_gpu_returns_unknown(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA RTX 4090", "MPR")
        assert result.status == "unknown"

    def test_unknown_feature_returns_unknown(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA A6000", "HolographicMode")
        assert result.status == "unknown"

    def test_result_has_recommendation(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA T4", "Cardiac 4D")
        assert result.recommendation
        assert len(result.recommendation) > 10

    def test_returns_pydantic_model(self, checker: CompatibilityChecker) -> None:
        result = checker.check_compatibility("4.2", "NVIDIA A6000", "MPR")
        assert isinstance(result, CompatibilityResult)

    def test_feature_name_normalisation(self, checker: CompatibilityChecker) -> None:
        """Cardiac_4D and Cardiac 4D should resolve to the same matrix entry."""
        r1 = checker.check_compatibility("4.2", "NVIDIA A6000", "Cardiac 4D")
        r2 = checker.check_compatibility("4.2", "NVIDIA A6000", "Cardiac_4D")
        assert r1.status == r2.status


# ---------------------------------------------------------------------------
# CompatibilityChecker — platform / OS checks
# ---------------------------------------------------------------------------


class TestCompatCheckerPlatform:
    def test_rhel8_fully_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "RHEL 8")
        assert result.status == "supported"

    def test_rhel9_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "RHEL 9")
        assert result.status == "supported"
        # Notes should mention the known WebRTC issue
        assert "KB-4330" in result.reason or "WebRTC" in result.reason

    def test_ubuntu_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "Ubuntu 22.04")
        assert result.status == "supported"

    def test_windows_server_supported(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "Windows Server 2022")
        assert result.status == "supported"

    def test_unknown_os_returns_unknown(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "Debian 12")
        assert result.status == "unknown"

    def test_platform_recommendation_non_empty(self, checker: CompatibilityChecker) -> None:
        result = checker.check_platform("4.2", "RHEL 8")
        assert result.recommendation


# ---------------------------------------------------------------------------
# CompatibilityChecker — list helpers
# ---------------------------------------------------------------------------


class TestCompatCheckerHelpers:
    def test_module_level_check_compatibility_returns_dict(self) -> None:
        result = check_compatibility("4.2", "NVIDIA T4", "Cardiac 4D")
        assert isinstance(result, dict)
        assert result["status"] == "limited"

    def test_module_level_check_platform_returns_dict(self) -> None:
        result = check_platform("4.2", "RHEL 8")
        assert isinstance(result, dict)
        assert result["status"] == "supported"
