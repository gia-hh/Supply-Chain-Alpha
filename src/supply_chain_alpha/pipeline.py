"""Phase-ordered project orchestration and reproducibility manifests."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

from supply_chain_alpha.data.disclosures import sha256_file, verify_raw_manifest
from supply_chain_alpha.utils.config import load_config
from supply_chain_alpha.utils.hashing import (
    atomic_write_json,
    config_sha256,
    source_tree_manifest,
    source_tree_manifest_sha256,
    write_source_tree_manifest,
)
from supply_chain_alpha.utils.status import (
    PhaseState,
    build_phase_status,
    phase_status_path,
    read_phase_status,
    write_phase_status,
)


class ExitCode(IntEnum):
    SUCCESS = 0
    ENGINEERING_FAILURE = 1
    EXTERNAL_DATA_BLOCKER = 2
    RESEARCH_GATE_FAILED = 3
    HOLDOUT_INTEGRITY_VIOLATION = 4


PHASE_NAMES = {
    1: "repository_and_schemas",
    2: "real_data_and_entity_resolution",
    3: "point_in_time_graph_coverage",
    4: "point_in_time_market_panel",
    5: "residual_return_engine",
    6: "diffusion_signal_engine",
    7: "development_validation_research",
    8: "portfolio_execution_cost_model",
    9: "research_freeze",
    10: "final_holdout",
    11: "final_report_and_closure",
}

PHASE_SCRIPTS = {
    2: "scripts/run_phase2.py",
    3: "scripts/run_phase3_coverage.py",
    4: "scripts/build_market_panel.py",
    5: "scripts/build_residual_returns.py",
    6: "scripts/build_signals.py",
    7: "scripts/run_signal_research.py",
    8: "scripts/run_portfolio_research.py",
    9: "scripts/freeze_research.py",
    10: "scripts/run_holdout.py",
    11: "scripts/build_final_report.py",
}

# Section 19 is an executable registry, not merely a prose checklist. Explicit
# paths ensure that deleting a test family makes Phase 1 fail closed.
MANDATORY_TEST_REGISTRY: dict[str, tuple[str, ...]] = {
    "engineering": (
        "tests/unit/test_config.py",
        "tests/unit/test_hashing.py",
        "tests/unit/test_status.py",
        "tests/unit/test_pipeline.py",
    ),
    "data_and_schema": (
        "tests/unit/test_schemas.py",
        "tests/unit/test_raw_manifest.py",
        "tests/unit/test_acquisition.py",
    ),
    "entity_resolution": ("tests/unit/test_entity_resolution.py",),
    "graph": ("tests/unit/test_edges.py",),
    "market": ("tests/unit/test_market.py",),
    "residuals": ("tests/unit/test_residuals.py",),
    "signals": ("tests/unit/test_signals.py",),
    "statistics": (
        "tests/unit/test_statistics.py",
        "tests/unit/test_experiment_registry.py",
    ),
    "portfolio": ("tests/unit/test_portfolio.py",),
    "holdout": ("tests/unit/test_holdout.py",),
    "tiny_end_to_end": ("tests/integration/test_tiny_e2e_pipeline.py",),
}
MANDATORY_TEST_FILES = tuple(
    dict.fromkeys(
        path for family in MANDATORY_TEST_REGISTRY.values() for path in family
    )
)
TINY_E2E_FIXTURE_FILES = (
    "fixtures/tiny_e2e/scenario.yaml",
    "fixtures/tiny_e2e/security_master.csv",
    "fixtures/tiny_e2e/company_alias.csv",
    "fixtures/tiny_e2e/raw_disclosures.csv",
    "fixtures/tiny_e2e/README.md",
)

_MACHINE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
)


@dataclass(frozen=True)
class PipelineResult:
    exit_code: ExitCode
    last_phase: int | None
    state: str | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _phase1_test_command(root: Path) -> list[str]:
    del root  # Keep recorded paths portable across machines.
    return [sys.executable, "-m", "pytest", "-q", *MANDATORY_TEST_FILES]


def missing_mandatory_test_artifacts(root: str | Path) -> tuple[str, ...]:
    """Return missing Section-19 tests and Section-25 fixture components."""

    project_root = Path(root)
    required = (*MANDATORY_TEST_FILES, *TINY_E2E_FIXTURE_FILES)
    return tuple(path for path in required if not (project_root / path).is_file())


def _scan_production_sources(root: Path) -> tuple[list[str], list[str]]:
    absolute_paths: list[str] = []
    secrets: list[str] = []
    candidates = [root / "src", root / "scripts", root / "config"]
    for base in candidates:
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            if _MACHINE_PATH.search(text):
                absolute_paths.append(relative)
            if _SECRET_ASSIGNMENT.search(text):
                secrets.append(relative)
    return absolute_paths, secrets


def _clean_install_smoke(root: Path) -> tuple[bool, str]:
    """Install the local package into an isolated target without network access."""

    with tempfile.TemporaryDirectory(prefix="supply-chain-alpha-install-") as tmp:
        temporary = Path(tmp)
        target = temporary / "site"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(target),
                str(root),
            ],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if install.returncode != 0:
            details = (install.stderr or install.stdout).strip()
            return False, f"isolated install failed: {details[-2000:]}"
        smoke_code = (
            "import sys; "
            f"sys.path.insert(0, {str(target)!r}); "
            "import supply_chain_alpha; "
            "from importlib.metadata import version; "
            "assert supply_chain_alpha.__version__ == "
            "version('supply-chain-alpha') == '0.2.0'"
        )
        smoke = subprocess.run(
            [sys.executable, "-I", "-c", smoke_code],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if smoke.returncode != 0:
            details = (smoke.stderr or smoke.stdout).strip()
            return False, f"isolated import smoke failed: {details[-2000:]}"
    return True, "local wheel metadata installed and imported from an isolated target"


def run_phase1(project_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = (
        (root / config_path).resolve()
        if not Path(config_path).is_absolute()
        else Path(config_path)
    )
    started = utc_now()
    state = PhaseState.PASS
    blockers: list[str] = []
    notes: list[str] = []
    criteria: dict[str, Any] = {}
    command = _phase1_test_command(root)
    tiny_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/integration/test_tiny_e2e_pipeline.py",
    ]

    try:
        load_config(config_file)
        criteria["project_config_valid"] = True
        missing_artifacts = missing_mandatory_test_artifacts(root)
        criteria["mandatory_test_registry_complete"] = not missing_artifacts
        criteria["tiny_e2e_fixture_present"] = not any(
            path in missing_artifacts for path in TINY_E2E_FIXTURE_FILES
        )
        if missing_artifacts:
            notes.append(f"missing mandatory test artifacts: {list(missing_artifacts)}")
            state = PhaseState.FAIL
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        criteria["phase1_tests_pass"] = completed.returncode == 0
        notes.append(completed.stdout.strip())
        if completed.stderr.strip():
            notes.append(completed.stderr.strip())
        if completed.returncode != 0:
            state = PhaseState.FAIL

        tiny_completed = subprocess.run(
            tiny_command, cwd=root, capture_output=True, text=True, check=False
        )
        criteria["tiny_e2e_preflight_pass"] = tiny_completed.returncode == 0
        notes.append(tiny_completed.stdout.strip())
        if tiny_completed.stderr.strip():
            notes.append(tiny_completed.stderr.strip())
        if tiny_completed.returncode != 0:
            state = PhaseState.FAIL

        absolute_paths, secrets = _scan_production_sources(root)
        criteria["no_machine_specific_production_paths"] = not absolute_paths
        criteria["no_literal_secrets"] = not secrets
        if absolute_paths:
            notes.append(f"machine-specific paths: {absolute_paths}")
            state = PhaseState.FAIL
        if secrets:
            notes.append(f"possible literal secrets: {secrets}")
            state = PhaseState.FAIL

        try:
            installed_version = importlib.metadata.version("supply-chain-alpha")
            criteria["package_installed"] = True
            notes.append(f"installed package version: {installed_version}")
        except importlib.metadata.PackageNotFoundError:
            criteria["package_installed"] = False
            state = PhaseState.FAIL

        clean_install_ok, clean_install_details = _clean_install_smoke(root)
        criteria["package_installs_in_clean_environment"] = clean_install_ok
        notes.append(clean_install_details)
        if not clean_install_ok:
            state = PhaseState.FAIL

        manifest_path = root / "reports" / "source_tree_manifest.json"
        manifest = write_source_tree_manifest(root, manifest_path)
        criteria["source_tree_manifest_written"] = bool(manifest["files"])
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        state = PhaseState.FAIL
        notes.append(f"Phase 1 validation error: {type(exc).__name__}: {exc}")

    finished = utc_now()
    status = build_phase_status(
        phase=1,
        name=PHASE_NAMES[1],
        state=state,
        started_at=started,
        finished_at=finished,
        project_root=root,
        config=config_file,
        inputs=[str(config_file.relative_to(root)), "PROJECT_SPEC.md"],
        outputs=["reports/source_tree_manifest.json", "reports/status/phase_1.json"],
        tests_run=[" ".join(command[1:]), " ".join(tiny_command[1:])],
        metrics={
            "phase1_test_return_code": completed.returncode
            if "completed" in locals()
            else 1,
            "tiny_e2e_test_return_code": tiny_completed.returncode
            if "tiny_completed" in locals()
            else 1,
        },
        criteria=criteria,
        blockers=blockers,
        notes=[note for note in notes if note],
    )
    output = phase_status_path(root, 1)
    write_phase_status(
        output,
        status,
        validation_rerun=output.exists(),
        defect_corrected=output.exists(),
    )
    # Validate the artifact after it is on disk; this is itself a Phase-1 gate.
    return read_phase_status(output)


def _status_exit_code(status: dict[str, Any]) -> ExitCode:
    state = status["state"]
    if state in {PhaseState.PASS.value, PhaseState.SKIPPED_BY_DESIGN.value}:
        return ExitCode.SUCCESS
    if state == PhaseState.BLOCKED.value:
        return ExitCode.EXTERNAL_DATA_BLOCKER
    if status.get("criteria", {}).get("holdout_integrity_violation"):
        return ExitCode.HOLDOUT_INTEGRITY_VIOLATION
    if status["phase"] in {3, 6} and status.get("criteria", {}).get(
        "research_gate_failed"
    ):
        return ExitCode.RESEARCH_GATE_FAILED
    return ExitCode.ENGINEERING_FAILURE


def _assert_status_run_context(
    status: dict[str, Any],
    *,
    expected_config_hash: str,
    expected_source_hash: str,
) -> None:
    if status.get("config_sha256") != expected_config_hash:
        raise RuntimeError(
            f"Phase {status.get('phase')} returned a status for a different config"
        )
    if status.get("source_tree_sha256") != expected_source_hash:
        raise RuntimeError(
            f"Phase {status.get('phase')} returned a status for a different source tree"
        )


def _recorded_outputs_exist(root: Path, status: dict[str, Any]) -> bool:
    root_resolved = root.resolve()

    def confined_path(value: str) -> Path | None:
        relative = Path(value)
        if relative.anchor or relative.is_absolute() or ".." in relative.parts:
            return None
        try:
            candidate = (root_resolved / relative).resolve()
            candidate.relative_to(root_resolved)
        except (OSError, ValueError):
            return None
        return candidate

    outputs = status.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    for value in outputs:
        if not isinstance(value, str) or not value:
            return False
        output = confined_path(value)
        if output is None:
            return False
        if not output.exists():
            return False
    artifact_hashes = status.get("metrics", {}).get("artifact_hashes")
    if artifact_hashes is not None:
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            return False
        for value, expected in artifact_hashes.items():
            if not isinstance(value, str) or not isinstance(expected, str):
                return False
            artifact = confined_path(value)
            if artifact is None:
                return False
            if not artifact.is_file() or sha256_file(artifact) != expected:
                return False
    return True


def _run_external_phase(root: Path, phase: int, config_path: Path) -> dict[str, Any]:
    script = root / PHASE_SCRIPTS[phase]
    started = utc_now()
    if not script.is_file():
        status_path = phase_status_path(root, phase)
        status = build_phase_status(
            phase=phase,
            name=PHASE_NAMES[phase],
            state=PhaseState.FAIL,
            started_at=started,
            finished_at=utc_now(),
            project_root=root,
            config=config_path,
            outputs=[status_path.relative_to(root).as_posix()],
            criteria={"required_phase_script_exists": False},
            blockers=[],
            notes=[
                f"Missing required phase script: {script.relative_to(root).as_posix()}"
            ],
        )
        write_phase_status(status_path, status)
        return read_phase_status(status_path)

    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        cwd=root,
        check=False,
    )
    status_path = phase_status_path(root, phase)
    if not status_path.is_file():
        status = build_phase_status(
            phase=phase,
            name=PHASE_NAMES[phase],
            state=PhaseState.FAIL,
            started_at=started,
            finished_at=utc_now(),
            project_root=root,
            config=config_path,
            outputs=[status_path.relative_to(root).as_posix()],
            criteria={"phase_script_wrote_status": False},
            metrics={"return_code": completed.returncode},
            blockers=[],
            notes=[
                "Production phase script returned without a machine-readable status artifact."
            ],
        )
        write_phase_status(status_path, status)
    status = read_phase_status(status_path)
    expected = int(_status_exit_code(status))
    if completed.returncode != expected:
        raise RuntimeError(
            f"Phase {phase} return code {completed.returncode} disagrees with status state "
            f"{status['state']} (expected {expected})"
        )
    return status


def _upstream_complete(
    root: Path,
    phase: int,
    current_config_hash: str,
    current_source_hash: str,
) -> None:
    for upstream in range(1, phase):
        status_path = phase_status_path(root, upstream)
        if not status_path.is_file():
            raise RuntimeError(f"Missing upstream status: phase {upstream}")
        status = read_phase_status(status_path)
        if status["state"] not in {
            PhaseState.PASS.value,
            PhaseState.SKIPPED_BY_DESIGN.value,
        }:
            raise RuntimeError(f"Upstream phase {upstream} is {status['state']}")
        if status["config_sha256"] != current_config_hash:
            raise RuntimeError(
                f"Upstream phase {upstream} used a different project config"
            )
        if status.get("source_tree_sha256") != current_source_hash:
            raise RuntimeError(
                f"Upstream phase {upstream} used a different source tree"
            )


def write_run_manifest(root: Path, config_path: Path) -> Path:
    config = load_config(config_path)
    source_manifest = source_tree_manifest(root)
    raw_manifest_path = root / config["paths"]["raw_manifest"]
    raw_hash: str | None = None
    if raw_manifest_path.is_file():
        verify_raw_manifest(
            raw_manifest_path,
            inventory_root=root / config["paths"]["raw_dir"] / "official_sources",
        )
        raw_hash = sha256_file(raw_manifest_path)

    distributions: dict[str, str] = {}
    for distribution in (
        "pandas",
        "PyYAML",
        "rapidfuzz",
        "pyarrow",
        "requests",
        "pypdf",
        "pypdfium2",
    ):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = "not_installed"

    states: dict[str, str] = {}
    for phase in PHASE_NAMES:
        path = phase_status_path(root, phase)
        if path.is_file():
            states[str(phase)] = read_phase_status(path)["state"]
    final_paths = [
        item
        for item in (
            "reports/final/FINAL_REPORT.md",
            "reports/final/EXECUTIVE_SUMMARY.md",
            "reports/final/FEASIBILITY_REPORT.md",
            "reports/final/results_summary.json",
            "reports/final/limitations.md",
            "reports/final/reproducibility.md",
            "reports/phase3_coverage/summary.json",
        )
        if (root / item).is_file()
    ]
    payload = {
        "run_timestamp_utc": utc_now().isoformat(),
        "python_version": sys.version,
        "package_versions": distributions,
        "config_sha256": config_sha256(config),
        "source_code_sha256": source_tree_manifest_sha256(source_manifest),
        "raw_data_manifest_sha256": raw_hash,
        "random_seed": config["project"]["random_state"],
        "phase_states": states,
        "final_artifact_paths": final_paths,
    }
    output = root / "reports" / "run_manifest.json"
    atomic_write_json(output, payload)
    return output


def run_pipeline(
    project_root: str | Path,
    *,
    config_path: str | Path = "config/project.yaml",
    through_phase: int = 11,
) -> PipelineResult:
    root = Path(project_root).resolve()
    if through_phase < 1 or through_phase > 11:
        raise ValueError("through_phase must be between 1 and 11")
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    current_config_hash = config_sha256(config)
    current_source_hash = source_tree_manifest_sha256(source_tree_manifest(root))
    last_phase: int | None = None
    last_state: str | None = None

    for phase in range(1, through_phase + 1):
        existing_path = phase_status_path(root, phase)
        if existing_path.is_file():
            existing = read_phase_status(existing_path)
            same_run_context = bool(
                existing["config_sha256"] == current_config_hash
                and existing.get("source_tree_sha256") == current_source_hash
            )
            reusable_outputs = _recorded_outputs_exist(root, existing)
            if (
                existing["state"]
                in {PhaseState.PASS.value, PhaseState.SKIPPED_BY_DESIGN.value}
                and same_run_context
                and reusable_outputs
            ):
                last_phase, last_state = phase, existing["state"]
                continue
            terminal_research_failure = bool(
                existing["state"] == PhaseState.FAIL.value
                and existing.get("criteria", {}).get("research_gate_failed")
            )
            if terminal_research_failure and same_run_context and reusable_outputs:
                write_run_manifest(root, config_file)
                return PipelineResult(
                    _status_exit_code(existing), phase, existing["state"]
                )

        _upstream_complete(
            root,
            phase,
            current_config_hash,
            current_source_hash,
        )
        status = (
            run_phase1(root, config_file)
            if phase == 1
            else _run_external_phase(root, phase, config_file)
        )
        _assert_status_run_context(
            status,
            expected_config_hash=current_config_hash,
            expected_source_hash=current_source_hash,
        )
        if not _recorded_outputs_exist(root, status):
            raise RuntimeError(
                f"Phase {phase} returned missing or hash-mismatched outputs"
            )
        observed_source_hash = source_tree_manifest_sha256(source_tree_manifest(root))
        if observed_source_hash != current_source_hash:
            raise RuntimeError(f"Source tree changed while Phase {phase} was running")
        last_phase, last_state = phase, status["state"]
        exit_code = _status_exit_code(status)
        if exit_code != ExitCode.SUCCESS:
            write_run_manifest(root, config_file)
            return PipelineResult(exit_code, phase, status["state"])

    write_run_manifest(root, config_file)
    return PipelineResult(ExitCode.SUCCESS, last_phase, last_state)


__all__ = [
    "MANDATORY_TEST_FILES",
    "MANDATORY_TEST_REGISTRY",
    "PHASE_NAMES",
    "PHASE_SCRIPTS",
    "TINY_E2E_FIXTURE_FILES",
    "ExitCode",
    "PipelineResult",
    "missing_mandatory_test_artifacts",
    "run_phase1",
    "run_pipeline",
    "write_run_manifest",
]
