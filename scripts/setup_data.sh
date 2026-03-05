#!/usr/bin/env bash
#
# Download all data sources for the Crosswise crossword solver.
#
# Usage:
#   bash scripts/setup_data.sh
#
# Idempotent — skips files that already exist and pass checksum verification.
# See DATASETS.md for source documentation and licensing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# Colors (if terminal supports them)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Verify SHA256 checksum. Returns 0 if match, 1 if mismatch.
verify_checksum() {
    local file="$1"
    local expected="$2"
    if [[ ! -f "$file" ]]; then
        return 1
    fi
    local actual
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
    [[ "$actual" == "$expected" ]]
}

# Download a file if it doesn't exist or fails checksum.
download_file() {
    local url="$1"
    local dest="$2"
    local checksum="$3"
    local description="$4"
    local curl_flags="${5:---fail -L}"

    if verify_checksum "$dest" "$checksum"; then
        info "$description — already exists, checksum OK ✓"
        return 0
    fi

    info "Downloading $description..."
    mkdir -p "$(dirname "$dest")"
    if ! curl $curl_flags -o "$dest" "$url"; then
        error "Failed to download $description from $url"
        return 1
    fi

    if ! verify_checksum "$dest" "$checksum"; then
        error "Checksum mismatch for $dest"
        error "  Expected: $checksum"
        local actual
        actual=$(shasum -a 256 "$dest" | awk '{print $1}')
        error "  Got:      $actual"
        return 1
    fi

    info "$description — downloaded, checksum OK ✓"
}

# ─── 1. xd clues ────────────────────────────────────────────────────────────

XD_DIR="$DATA_DIR/sources/xd"
XD_TSV="$XD_DIR/clues.tsv"
XD_ZIP="$XD_DIR/xd-clues.zip"
XD_TSV_SHA256="36c83fc7837941ecb7cc3a97faef61bd3ae6fc4b81d5e379cf7b70ef11a9cdd5"

download_xd() {
    if verify_checksum "$XD_TSV" "$XD_TSV_SHA256"; then
        info "xd clues.tsv — already exists, checksum OK ✓"
        return 0
    fi

    info "Downloading xd clues archive..."
    mkdir -p "$XD_DIR"
    if ! curl --fail -L -o "$XD_ZIP" "https://xd.saul.pw/data/xd-clues.zip"; then
        error "Failed to download xd-clues.zip"
        return 1
    fi

    info "Extracting xd clues..."
    unzip -o "$XD_ZIP" -d "$XD_DIR"

    # The zip may contain the TSV under a different name; find it
    if [[ ! -f "$XD_TSV" ]]; then
        # Look for any .tsv file in the extracted contents
        local tsv_file
        tsv_file=$(find "$XD_DIR" -name "*.tsv" -type f | head -1)
        if [[ -n "$tsv_file" && "$tsv_file" != "$XD_TSV" ]]; then
            mv "$tsv_file" "$XD_TSV"
        fi
    fi

    if ! verify_checksum "$XD_TSV" "$XD_TSV_SHA256"; then
        warn "xd clues.tsv checksum does not match expected value"
        warn "  The upstream archive may have been updated."
        warn "  Proceeding anyway — verify manually if needed."
    else
        info "xd clues.tsv — extracted, checksum OK ✓"
    fi

    # Clean up zip
    rm -f "$XD_ZIP"
}

# ─── 2. CrosswordQA ─────────────────────────────────────────────────────────

CQA_DIR="$DATA_DIR/sources/crosswordqa"
CQA_TRAIN="$CQA_DIR/train.csv"
CQA_VALID="$CQA_DIR/valid.csv"
CQA_TRAIN_SHA256="25ada9f848e01f27b511817664fdd0b1725345a1d3b195882f379c38cd3c39fa"
CQA_VALID_SHA256="77bb8bf5bd296f6fb670853ebf593ee23f2b137e8a491f46b33722571cd790d2"

download_crosswordqa() {
    local base_url="https://huggingface.co/datasets/albertxu/CrosswordQA/resolve/main"

    download_file \
        "$base_url/train.csv" \
        "$CQA_TRAIN" \
        "$CQA_TRAIN_SHA256" \
        "CrosswordQA train.csv (246MB)"

    download_file \
        "$base_url/valid.csv" \
        "$CQA_VALID" \
        "$CQA_VALID_SHA256" \
        "CrosswordQA valid.csv (14MB)"
}

# ─── 3. Wordlists ───────────────────────────────────────────────────────────

WL_DIR="$DATA_DIR/wordlists"
NEXUS_FILE="$WL_DIR/xwordlist.dict"
BRODA_FILE="$WL_DIR/peter-broda-wordlist__gridtext__scored__july-25-2023.txt"
NEXUS_SHA256="a945a839a5f1e6f48caf9c8de446e5cd85f3567d7f62afcf54c6b738e8906ff4"
BRODA_SHA256="3859f24735f328a3250025dda67cf98165548fed913aecc29b6ff884738a7c29"

download_wordlists() {
    download_file \
        "https://raw.githubusercontent.com/Crossword-Nexus/collaborative-word-list/main/xwordlist.dict" \
        "$NEXUS_FILE" \
        "$NEXUS_SHA256" \
        "Crossword Nexus wordlist (8MB)"

    # Broda site has expired SSL cert — use --insecure
    download_file \
        "http://peterbroda.me/crosswords/wordlist/lists/peter-broda-wordlist__gridtext__scored__july-25-2023.txt" \
        "$BRODA_FILE" \
        "$BRODA_SHA256" \
        "Broda wordlist (8MB)" \
        "--fail -L --insecure"
}

# ─── Main ────────────────────────────────────────────────────────────────────

main() {
    echo ""
    info "=== Crosswise Data Setup ==="
    echo ""

    download_xd
    echo ""
    download_crosswordqa
    echo ""
    download_wordlists

    echo ""
    info "=== Summary ==="
    echo ""

    local total_ok=0
    local total_missing=0

    for item in "$XD_TSV" "$CQA_TRAIN" "$CQA_VALID" "$NEXUS_FILE" "$BRODA_FILE"; do
        if [[ -f "$item" ]]; then
            local size
            size=$(du -h "$item" | awk '{print $1}')
            info "  ✓ $(basename "$item") ($size)"
            ((total_ok++))
        else
            error "  ✗ $(basename "$item") — MISSING"
            ((total_missing++))
        fi
    done

    echo ""
    if [[ $total_missing -eq 0 ]]; then
        info "All $total_ok data files ready."
        echo ""
        info "All data sources downloaded. Run 'make setup' to build the SQLite clue database."
    else
        warn "$total_missing file(s) missing. Check errors above."
        exit 1
    fi
}

main "$@"
