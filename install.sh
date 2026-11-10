#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HEATER_INSTALL_DIR:-$HOME/.local/bin}"
REPO_ARCHIVE_URL="https://github.com/DeodatoBastos/heater/archive/refs/heads/main.tar.gz"
TMP_BUILD_DIR=""

cleanup_temp_dir() {
    if [ -n "$TMP_BUILD_DIR" ] && [ -d "$TMP_BUILD_DIR" ]; then
        rm -rf "$TMP_BUILD_DIR"
    fi
}
trap cleanup_temp_dir EXIT

download_to_stdout() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -fsSL "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$url"
    else
        echo "[heater] Error: Either 'curl' or 'wget' is required, but neither was found." >&2
        exit 1
    fi
}

ensure_rust_toolchain() {
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
        echo "[heater] Found Rust toolchain: $(rustc --version)"
        return 0
    fi
    echo "[heater] Rust toolchain not found. Installing via rustup..."
    download_to_stdout "https://sh.rustup.rs" | sh -s -- -y --no-modify-path
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    echo "[heater] Installed Rust toolchain: $(rustc --version)"
}

verify_nvidia_runtime() {
    if ! ldconfig -p 2>/dev/null | grep -q "libnvidia-ml.so.1" && [ ! -e "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1" ]; then
        echo "[heater] Warning: libnvidia-ml.so.1 not detected. Ensure NVIDIA proprietary drivers are installed before running heater." >&2
    fi
}

resolve_source_directory() {
    local script_dir=""
    if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
    if [ -n "$script_dir" ] && [ -f "$script_dir/Cargo.toml" ]; then
        echo "$script_dir"
        return 0
    fi
    TMP_BUILD_DIR="$(mktemp -d)"
    echo "[heater] Downloading source archive from GitHub..." >&2
    download_to_stdout "$REPO_ARCHIVE_URL" | tar -xz -C "$TMP_BUILD_DIR" --strip-components=1
    echo "$TMP_BUILD_DIR"
}

build_and_install_binary() {
    local source_dir
    source_dir="$(resolve_source_directory)"
    echo "[heater] Fetching crates and building release binary..."
    cargo build --release --manifest-path "$source_dir/Cargo.toml"
    mkdir -p "$INSTALL_DIR"
    install -m 0755 "$source_dir/target/release/heater" "$INSTALL_DIR/heater"
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
