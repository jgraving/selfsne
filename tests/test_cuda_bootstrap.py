"""Exercise bootstrap control flow without downloads, sudo, or CUDA hardware."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None,
    reason="Ubuntu bootstrap control flow requires Linux and Bash.",
)


@pytest.fixture
def bootstrap(tmp_path):
    root = tmp_path / "checkout with spaces"
    script = root / "scripts" / "test-cuda-ubuntu.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / script.name
    shutil.copyfile(source, script)
    uv = root / ".cuda-tools" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'echo "uv $*" >> "$TRACE"\n'
        'if [[ $* == *--torch-backend=* ]]; then exit "${INSTALL_EXIT:-0}"; fi\n'
        'if [[ $1 == venv ]]; then\n'
        '  environment=${@: -1}\n'
        '  mkdir -p "$environment/bin"\n'
        '  cp "$FAKE_PYTHON" "$environment/bin/python"\n'
        'fi\n'
    )
    uv.chmod(0o755)
    python = tmp_path / "fake-python"
    python.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'echo "python $*" >> "$TRACE"\n'
        'if [[ $1 == - ]]; then cat >/dev/null; exit "${PROBE_EXIT:-0}"; fi\n'
        'if [[ $* == *"-m build"* ]]; then\n'
        '  mkdir -p "${@: -1}"; touch "${@: -1}/selfsne.whl"\n'
        'fi\n'
        'if [[ $* == *cuda-wheel.xml* ]]; then exit "${WHEEL_EXIT:-0}"; fi\n'
    )
    python.chmod(0o755)
    trace = tmp_path / "trace"
    environment = {
        **os.environ,
        "TRACE": str(trace),
        "FAKE_PYTHON": str(python),
    }

    def run(arguments="", overrides="", **variables):
        wrapper = (
            'source "$1"\n'
            'check_platform() { :; }\n'
            'has_nvidia_gpu() { return 0; }\n'
            'nvidia-smi() { echo "GPU, driver"; }\n'
            'sudo() { echo "sudo $*" >> "$TRACE"; }\n'
            + overrides
            + '\nmain '
            + arguments
            + '\n'
        )
        result = subprocess.run(
            ["bash", "-c", wrapper, "bootstrap-test", str(script)],
            env={**environment, **variables},
            text=True,
            capture_output=True,
        )
        return result, trace.read_text() if trace.exists() else ""

    return run, script


def test_bootstrap_runs_source_and_wheel_without_system_changes(bootstrap):
    run, _ = bootstrap
    result, trace = run("--install-driver")
    assert result.returncode == 0, result.stderr
    assert "sudo " not in trace  # A working driver is reused, even with permission.
    assert "--managed-python --python 3.12" in trace
    assert "--torch-backend=auto --reinstall-package torch" in trace
    assert trace.count("--require-cuda") == 2
    assert "--import-mode=importlib" in trace
    assert "--force-reinstall --no-deps" in trace
    assert trace.rstrip().endswith('checkout with spaces')  # Editable restore.
    assert "CUDA source and wheel tests passed" in result.stdout


def test_missing_gpu_stops_before_installation(bootstrap):
    run, _ = bootstrap
    result, trace = run(overrides="has_nvidia_gpu() { return 1; }")
    assert result.returncode != 0
    assert "No NVIDIA GPU" in result.stderr
    assert not trace


@pytest.mark.parametrize("explicit", [False, True])
def test_driver_535_selects_a_cuda_tagged_build(bootstrap, explicit):
    run, _ = bootstrap
    result, trace = run(
        "--torch-backend=cu121" if explicit else "",
        overrides='nvidia-smi() { echo "535.183.01"; }',
    )
    assert result.returncode == 0, result.stderr
    assert "--torch-backend=cu121" in trace
    assert "torch==2.5.1+cu121" in trace
    assert "--reinstall-package torch" in trace
    assert "uv --no-config pip install" in trace
    assert "--no-sources" in trace
    assert "sudo " not in trace


def test_cpu_only_build_never_triggers_driver_update(bootstrap):
    run, _ = bootstrap
    result, trace = run("--install-driver", PROBE_EXIT="40")
    assert result.returncode != 0
    assert "CPU wheel selection failed" in result.stderr
    assert "sudo " not in trace and "-m pytest" not in trace


def test_missing_driver_never_uses_sudo_without_option(bootstrap):
    run, _ = bootstrap
    result, trace = run(overrides="nvidia-smi() { return 1; }")
    assert result.returncode != 0
    assert "--install-driver" in result.stderr
    assert not trace


@pytest.mark.parametrize("missing_driver", [True, False])
def test_driver_update_stops_for_reboot_before_tests(bootstrap, missing_driver):
    run, _ = bootstrap
    result, trace = run(
        "--install-driver",
        overrides="nvidia-smi() { return 1; }" if missing_driver else "",
        PROBE_EXIT="30",
    )
    assert result.returncode == 10
    assert "sudo apt-get install -y ubuntu-drivers-common" in trace
    assert "sudo ubuntu-drivers devices" in trace
    assert "sudo ubuntu-drivers install" in trace
    assert "Reboot Ubuntu" in result.stdout
    assert "-m pytest" not in trace
    assert "sudo reboot" not in trace


def test_failed_cuda_probe_does_not_pass_or_update_driver_implicitly(bootstrap):
    run, _ = bootstrap
    result, trace = run(PROBE_EXIT="30")
    assert result.returncode != 0
    assert "CUDA initialization failed" in result.stderr
    assert "sudo " not in trace and "-m pytest" not in trace


@pytest.mark.parametrize("status", ["1", "20"])
def test_gpu_execution_failure_does_not_trigger_driver_change(bootstrap, status):
    run, _ = bootstrap
    result, trace = run("--install-driver", PROBE_EXIT=status)
    assert result.returncode != 0
    assert "compatibility diagnosis" in result.stderr
    assert "sudo " not in trace and "-m pytest" not in trace


def test_dependency_install_failure_never_triggers_driver_update(bootstrap):
    run, _ = bootstrap
    result, trace = run("--install-driver", INSTALL_EXIT="5")
    assert result.returncode == 5
    assert "sudo " not in trace and "-m pytest" not in trace


def test_uv_bootstrap_download_stays_in_checkout(bootstrap, tmp_path):
    run, script = bootstrap
    uv = script.parents[1] / ".cuda-tools" / "bin" / "uv"
    backup = tmp_path / "mock-uv"
    uv.rename(backup)
    installer = tmp_path / "mock-installer"
    installer.write_text('cp "$MOCK_UV" "$UV_UNMANAGED_INSTALL/uv"\n')
    result, trace = run(
        overrides='curl() { cp "$MOCK_INSTALLER" "${@: -1}"; }',
        MOCK_UV=str(backup),
        MOCK_INSTALLER=str(installer),
    )
    assert result.returncode == 0, result.stderr
    assert uv.is_file() and "--require-cuda" in trace


def test_failed_wheel_validation_restores_source_and_propagates_failure(bootstrap):
    run, _ = bootstrap
    result, trace = run(WHEEL_EXIT="7")
    assert result.returncode == 7
    assert trace.rstrip().endswith("checkout with spaces")
    assert "CUDA source and wheel tests passed" not in result.stdout


def test_help_does_not_detect_hardware_or_install_anything(bootstrap):
    run, script = bootstrap
    result = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--install-driver" in result.stdout
    _, trace = run("--unknown-option")
    assert not trace
