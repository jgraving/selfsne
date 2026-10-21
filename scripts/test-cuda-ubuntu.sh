#!/usr/bin/env bash
# Maintainer bootstrap: Ubuntu NVIDIA validation, with project-local Python tools.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/test-cuda-ubuntu.sh [--install-driver]

Detect NVIDIA hardware/driver, download project-local uv and Python 3.12,
install CUDA PyTorch, and test source plus the built wheel.

--install-driver permits sudo apt-get and Ubuntu's recommended driver installer
ONLY if the driver is unavailable or the CUDA probe fails. Reboot and rerun
after a driver installation. No automatic reboot, PPA, or CUDA toolkit install.
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
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; CPU fallback is disabled.")
for index in range(torch.cuda.device_count()):
    device = torch.device("cuda", index)
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
    # Driver detection alone cannot establish that the installed wheel supports
    # this GPU's architecture. Execute a kernel and a matrix multiplication.
    values = torch.ones((32, 32), device=device)
    product = values @ values
    torch.cuda.synchronize(device)
    assert torch.isfinite(product).all() and (product == 32).all()
PY
}

main() {
    local install_driver_allowed=false argument
    for argument in "$@"; do
        case "$argument" in
            --install-driver) install_driver_allowed=true ;;
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

    local root tools python uv wheel wheel_directory restore_command
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
    tools="$root/.cuda-tools"
    python="$root/.venv-cuda/bin/python"
    uv="$tools/bin/uv"
    mkdir -p "$tools/bin" "$tools/tmp" "$tools/reports"
    export UV_CACHE_DIR="$tools/cache"
    export UV_PYTHON_INSTALL_DIR="$tools/python"
    export TMPDIR="$tools/tmp"
    export PIP_CACHE_DIR="$tools/pip-cache"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    bootstrap_uv
    cd -- "$root"
    if [[ ! -x $python ]]; then
        # Refuse existing non-environments rather than deleting user files.
        "$uv" venv --managed-python --python 3.12 "$root/.venv-cuda"
    fi
    "$uv" pip install --python "$python" --torch-backend=auto --reinstall-package torch -e '.[dev,data]'
    "$uv" pip check --python "$python"
    if ! probe_cuda; then
        echo 'CUDA execution failed. An old driver or unsupported GPU/wheel combination may be responsible.' >&2
        if [[ $install_driver_allowed == true ]]; then
            install_driver
            return $?
        fi
        echo 'For the Ubuntu-recommended driver update, rerun with --install-driver.' >&2
        return 1
    fi
    "$python" -m pytest -m cuda --require-cuda -v --tb=short --junitxml="$tools/reports/cuda-source.xml"

    wheel_directory=$(mktemp -d "$tools/tmp/wheel.XXXXXX")
    "$python" -m build --wheel --outdir "$wheel_directory"
    # Always return the environment to editable source, even if wheel install
    # or validation fails. Expand paths now: main's locals expire before EXIT.
    printf -v restore_command '%q ' "$uv" pip install --python "$python" --no-deps -e "$root"
    trap "$restore_command" EXIT
    for wheel in "$wheel_directory"/*.whl; do
        "$uv" pip install --python "$python" --force-reinstall --no-deps "$wheel"
    done
    cd -- "$tools/tmp"
    "$python" -m pytest --import-mode=importlib -m cuda --require-cuda -v --tb=short "$root/tests" --junitxml="$tools/reports/cuda-wheel.xml"
    cd -- "$root"
    "$uv" pip install --python "$python" --no-deps -e "$root"
    trap - EXIT
    echo "CUDA source and wheel tests passed. Reports: $tools/reports/"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
