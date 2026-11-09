#!/usr/bin/env bash
# NALA-Colibri-RS v2.2.0-nala.6.6.6 — Zero-Dependency Multi-Arch Installer (macOS / Linux)
set -euo pipefail

REPO="Master-MD/NALA-Colibri-RS"
VERSION="${NALA_COLIBRI_VERSION:-v2.2.0-nala.6.6.6}"
INSTALL_DIR="${NALA_INSTALL_DIR:-$HOME/.local/bin}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
  darwin-arm64|darwin-aarch64) ASSET="nala-colibri-rs-macos-arm64" ;;
  darwin-x86_64)               ASSET="nala-colibri-rs-macos-x86_64" ;;
  linux-x86_64|linux-amd64)    ASSET="nala-colibri-rs-linux-x86_64" ;;
  linux-aarch64|linux-arm64)   ASSET="nala-colibri-rs-linux-aarch64" ;;
  *)
    echo "[ERROR] Unsupported platform: ${OS}-${ARCH}" >&2
    exit 1
    ;;
esac

mkdir -p "${INSTALL_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
echo "[NALA-Colibri-RS] Downloading ${ASSET} (${VERSION})..."
curl -fsSL "${BASE_URL}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
curl -fsSL "${BASE_URL}/SHA256SUMS.txt" -o "${TMP_DIR}/SHA256SUMS.txt"

echo "[NALA-Colibri-RS] Verifying cryptographic SHA-256 digest..."
EXPECTED="$(grep "  ${ASSET}\$" "${TMP_DIR}/SHA256SUMS.txt" | awk '{print $1}')"
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "${TMP_DIR}/${ASSET}" | awk '{print $1}')"
else
  ACTUAL="$(shasum -a 256 "${TMP_DIR}/${ASSET}" | awk '{print $1}')"
fi

if [ -z "${EXPECTED}" ] || [ "${EXPECTED}" != "${ACTUAL}" ]; then
  echo "[FATAL] SHA-256 mismatch! Expected ${EXPECTED}, got ${ACTUAL}" >&2
  exit 1
fi

install -m 0755 "${TMP_DIR}/${ASSET}" "${INSTALL_DIR}/nala-colibri-rs"
ln -sf "${INSTALL_DIR}/nala-colibri-rs" "${INSTALL_DIR}/coli-rs"
echo "[OK] Installed to ${INSTALL_DIR}/nala-colibri-rs (alias: coli-rs)"
"${INSTALL_DIR}/nala-colibri-rs" --version
