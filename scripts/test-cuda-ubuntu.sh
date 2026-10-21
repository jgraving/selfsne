#!/usr/bin/env bash
# Maintainer bootstrap: Ubuntu NVIDIA validation, with project-local Python tools.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/test-cuda-ubuntu.sh [--installer=uv|pip] [--install-driver] [--torch-backend=cu121]

Detect NVIDIA hardware/driver, download project-local uv and Python 3.12,
install CUDA PyTorch, and test source plus the built wheel.

--install-driver permits sudo apt-get and Ubuntu's recommended driver installer
ONLY if the driver or CUDA initialization is unavailable. Reboot and rerun
after a driver installation. No automatic reboot, PPA, or CUDA toolkit install.
--torch-backend selects an explicit CUDA wheel index instead of automatic
selection. Driver 535 automatically uses PyTorch 2.5.1 with CUDA 12.1.
--installer=pip uses .venv-cuda-pip and separate reports; uv only provisions
the local Python interpreter. Dependency and wheel installation use pip.
EOF
}

check_platform() {
    if [[ $(uname -s) != Linux ]] || [[ ! -r /etc/os-release ]]; then
        echo 'This bootstrap requires native Ubuntu Linux.' >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ ${ID:-} != ubuntu ]]; then
        echo 'This bootstrap supports Ubuntu; no system changes were made.' >&2
        return 1
    fi
    if [[ -n ${WSL_INTEROP:-} ]] || [[ $(uname -r) == *[Mm]icrosoft* ]]; then
        echo 'WSL uses the Windows host driver; do not install a Linux GPU driver here.' >&2
        return 1
    fi
    if [[ $EUID == 0 ]]; then
        echo 'Run as your normal user, without sudo. Only driver commands use sudo.' >&2
        return 1
    fi
}

has_nvidia_gpu() {
    local vendor device_class device
    for device in /sys/bus/pci/devices/*; do
        [[ -r $device/vendor && -r $device/class ]] || continue
        read -r vendor < "$device/vendor"
        read -r device_class < "$device/class"
        if [[ $vendor == 0x10de && $device_class == 0x03* ]]; then
            return 0
        fi
    done
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1
}

install_driver() {
    if [[ $install_driver_allowed != true ]]; then
        echo 'Driver setup is needed. Rerun with --install-driver to permit Ubuntu driver installation.' >&2
        return 1
    fi
    echo 'Installing the Ubuntu-recommended NVIDIA driver using system packages.'
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers devices
    sudo ubuntu-drivers install
    echo 'Driver installation finished. Reboot Ubuntu, then rerun this script.'
    echo 'If Secure Boot asks you to enroll a key during reboot, complete that step.'
    # A successful package install is not evidence that the new kernel driver loaded.
    return 10
}

bootstrap_uv() {
    if [[ ! -x $tools/bin/uv ]]; then
        if command -v curl >/dev/null; then
            curl --fail --location --silent --show-error https://astral.sh/uv/install.sh -o "$tools/install-uv.sh"
        elif command -v wget >/dev/null; then
            wget -q https://astral.sh/uv/install.sh -O "$tools/install-uv.sh"
        else
            echo 'Install a downloader with: sudo apt-get install curl' >&2
            return 1
        fi
        UV_UNMANAGED_INSTALL="$tools/bin" sh "$tools/install-uv.sh"
    fi
}

probe_cuda() {
    "$python" - <<'PY'
import torch

print(f"PyTorch {torch.__version__}; bundled CUDA {torch.version.cuda}", flush=True)
if torch.version.cuda is None:
    print("A CPU-only PyTorch wheel was installed; this is not a driver failure.", flush=True)
    raise SystemExit(40)
if not torch.cuda.is_available():
    print("CUDA is unavailable; CPU fallback is disabled.", flush=True)
    raise SystemExit(30)
for index in range(torch.cuda.device_count()):
    device = torch.device("cuda", index)
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
    # Driver detection alone cannot establish that the installed wheel supports
    # this GPU's architecture. Execute a kernel and a matrix multiplication.
    try:
        values = torch.ones((32, 32), device=device)
        product = values @ values
        torch.cuda.synchronize(device)
        assert torch.isfinite(product).all() and (product == 32).all()
    except (RuntimeError, AssertionError) as exc:
        print(f"CUDA initialized but GPU execution failed: {exc}", flush=True)
        raise SystemExit(20) from exc
PY
}

main() {
    local install_driver_allowed=false argument torch_backend=auto installer=uv
    for argument in "$@"; do
        case "$argument" in
            --install-driver) install_driver_allowed=true ;;
            --installer=uv|--installer=pip) installer=${argument#*=} ;;
            --torch-backend=auto|--torch-backend=cu121|--torch-backend=cu126|--torch-backend=cu128|--torch-backend=cu130)
                torch_backend=${argument#*=} ;;
            --help|-h) usage; return 0 ;;
            *) usage >&2; return 2 ;;
        esac
    done
    check_platform
    if ! has_nvidia_gpu; then
        echo 'No NVIDIA GPU detected. No driver or Python packages were installed.' >&2
        return 1
    fi
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=name,driver_version --format=csv; then
        echo 'The NVIDIA driver is missing or cannot communicate with the GPU.' >&2
        install_driver
        return $?
    fi

    local root tools python uv wheel wheel_directory restore_command probe_status driver_version torch_requirement=torch
    if [[ $torch_backend == auto ]]; then
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        driver_version=${driver_version%%$'\n'*}
        if [[ $driver_version == 535.* ]]; then
            torch_backend=cu121
            echo 'Driver 535 detected: selecting PyTorch 2.5.1 with CUDA 12.1.'
        fi
    fi
    if [[ $torch_backend == cu121 ]]; then
        # Require the CUDA local-version tag: an installed CPU build must not
        # satisfy this requirement or remain selected from the CPU project index.
        torch_requirement='torch==2.5.1+cu121'
    fi
    if [[ $installer == pip && $torch_backend == auto ]]; then
        echo 'The pip path requires an explicit --torch-backend=cu121, cu126, cu128, or cu130 for this driver.' >&2
        return 2
    fi
    local environment_name=.venv-cuda reports
    if [[ $installer == pip ]]; then environment_name=.venv-cuda-pip; fi
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
    tools="$root/.cuda-tools"
    python="$root/$environment_name/bin/python"
    uv="$tools/bin/uv"
    reports="$tools/reports"
    if [[ $installer == pip ]]; then reports="$reports/pip"; fi
    mkdir -p "$tools/bin" "$tools/tmp" "$reports"
    export UV_CACHE_DIR="$tools/cache"
    export UV_PYTHON_INSTALL_DIR="$tools/python"
    export TMPDIR="$tools/tmp"
    export PIP_CACHE_DIR="$tools/pip-cache"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    bootstrap_uv
    cd -- "$root"
    if [[ ! -x $python ]]; then
        # Refuse existing non-environments rather than deleting user files.
        "$uv" venv --managed-python --python 3.12 --seed "$root/$environment_name"
    fi
    # Ignore the checkout's CPU-only uv index configuration in this dedicated
    # CUDA environment; project dependency metadata still comes from pyproject.
    if [[ $installer == uv ]]; then
        "$uv" --no-config pip install --no-sources --python "$python" "--torch-backend=$torch_backend" --reinstall-package torch --upgrade-package torch "$torch_requirement" -e '.[dev,data]'
        "$uv" pip check --python "$python"
    else
        "$python" -m pip install --upgrade --index-url "https://download.pytorch.org/whl/$torch_backend" "$torch_requirement"
        "$python" -m pip install "$torch_requirement" -e '.[dev,data]'
        "$python" -m pip check
    fi
    if probe_cuda; then
        :
    else
        probe_status=$?
        if [[ $probe_status == 40 ]]; then
            echo 'CPU wheel selection failed. Retry with --torch-backend=cu121 for driver 535; do not update the driver to fix CPU wheel selection.' >&2
            return 1
        fi
        if [[ $probe_status != 30 ]]; then
            echo 'The CUDA probe failed unexpectedly or this GPU could not execute the PyTorch build. Send the output for compatibility diagnosis; no driver changes were made.' >&2
            return 1
        fi
        echo 'CUDA initialization failed. An old driver or unsupported GPU/wheel combination may be responsible.' >&2
        if [[ $install_driver_allowed == true ]]; then
            install_driver
            return $?
        fi
        echo 'For the Ubuntu-recommended driver update, rerun with --install-driver.' >&2
        return 1
    fi
    "$python" -m pytest -m cuda --require-cuda -v --tb=short --junitxml="$reports/cuda-source.xml"

    wheel_directory=$(mktemp -d "$tools/tmp/wheel.XXXXXX")
    "$python" -m build --wheel --outdir "$wheel_directory"
    # Always return the environment to editable source, even if wheel install
    # or validation fails. Expand paths now: main's locals expire before EXIT.
    if [[ $installer == uv ]]; then
        printf -v restore_command '%q ' "$uv" pip install --python "$python" --no-deps -e "$root"
    else
        printf -v restore_command '%q ' "$python" -m pip install --no-deps -e "$root"
    fi
    trap "$restore_command" EXIT
    for wheel in "$wheel_directory"/*.whl; do
        if [[ $installer == uv ]]; then
            "$uv" pip install --python "$python" --force-reinstall --no-deps "$wheel"
        else
            "$python" -m pip install --force-reinstall --no-deps "$wheel"
        fi
    done
    cd -- "$tools/tmp"
    "$python" -m pytest --import-mode=importlib -m cuda --require-cuda -v --tb=short "$root/tests" --junitxml="$reports/cuda-wheel.xml"
    cd -- "$root"
    if [[ $installer == uv ]]; then
        "$uv" pip install --python "$python" --no-deps -e "$root"
    else
        "$python" -m pip install --no-deps -e "$root"
    fi
    trap - EXIT
    echo "CUDA source and wheel tests passed ($installer). Reports: $reports/"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
