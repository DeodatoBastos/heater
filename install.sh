#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HEATER_INSTALL_DIR:-$HOME/.local/bin}"

ensure_rust_toolchain() {
    if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
        echo "[heater] Found Rust toolchain: $(rustc --version)"
        return 0
    fi
    echo "[heater] Rust toolchain not found. Installing via rustup..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "[heater] Error: 'curl' is required to install rustup, but was not found in PATH." >&2
        exit 1
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    echo "[heater] Installed Rust toolchain: $(rustc --version)"
}

verify_nvidia_runtime() {
    if ! ldconfig -p 2>/dev/null | grep -q "libnvidia-ml.so.1" && [ ! -e "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1" ]; then
        echo "[heater] Warning: libnvidia-ml.so.1 not detected. Ensure NVIDIA proprietary drivers are installed before running heater." >&2
    fi
}

build_and_install_binary() {
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "[heater] Fetching crates and building release binary..."
    cargo build --release --manifest-path "$repo_root/Cargo.toml"
    mkdir -p "$INSTALL_DIR"
    install -m 0755 "$repo_root/target/release/heater" "$INSTALL_DIR/heater"
    echo "[heater] Installed binary to $INSTALL_DIR/heater"
}

check_path_configuration() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) ;;
        *)
            echo "[heater] Note: $INSTALL_DIR is not in your PATH."
            echo "[heater] Add this line to your shell profile (~/.bashrc or ~/.zshrc):"
            echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
            ;;
    esac
}

main() {
    ensure_rust_toolchain
    verify_nvidia_runtime
    build_and_install_binary
    check_path_configuration
    echo "[heater] Installation complete. Run 'heater --target 65.0' to start."
}

main "$@"
