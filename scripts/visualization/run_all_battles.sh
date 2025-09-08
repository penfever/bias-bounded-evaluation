#!/usr/bin/env bash

# Run all visualization scripts to generate tables and figures
# This script should be run from the visualization directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================================${NC}\n"
}

# Check if we're in the correct directory
if [ ! -f "data_loader.py" ]; then
    print_error "This script must be run from the visualization directory"
    print_error "Current directory: $(pwd)"
    exit 1
fi

# Ensure Python environment is set up
print_status "Checking Python environment..."
if ! python -c "import pandas" 2>/dev/null; then
    print_warning "Required Python packages may not be installed"
    print_warning "Please ensure you have activated the correct conda environment"
    print_warning "Run: source ~/.zshrc && conda activate oumi"
fi

# Start the battles-only pipeline
print_section "Starting Battles-Only Pipeline"
print_status "This will ONLY run ELO pseudo-battles + bootstrap with debug"

# Config and path resolution
# Resolve repository root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Allow overriding DATA_BASE via environment; otherwise default to repo sibling
: "${DATA_BASE:=$REPO_ROOT/sos-addl-data/InDepthAnalysis}"
ELO_ROUNDS=100
print_status "DATA_BASE resolved to: $DATA_BASE"
if [ ! -d "$DATA_BASE" ]; then
  print_warning "DATA_BASE not found at: $DATA_BASE"
  # Try alternative common locations relative to script
  ALT1="$REPO_ROOT/sos-addl-data/InDepthAnalysis"
  ALT2="$REPO_ROOT/sos-artifacts/InDepthAnalysis"
  for ALT in "$ALT1" "$ALT2"; do
    if [ -d "$ALT" ]; then
      DATA_BASE="$ALT"
      print_status "Using alternate DATA_BASE: $DATA_BASE"
      break
    fi
  done
  if [ ! -d "$DATA_BASE" ]; then
    print_error "DATA_BASE does not exist after trying alternates. Set DATA_BASE env var to correct path."
    exit 1
  fi
fi

# Collect settings
settings=("$DATA_BASE"/*-setting*)
if [ ${#settings[@]} -eq 1 ] && [[ "${settings[0]}" == "$DATA_BASE"/*-setting* ]]; then
  print_error "No setting directories matching '*-setting*' found under $DATA_BASE"
  exit 1
fi
print_status "Found ${#settings[@]} setting directories"

# Only run battles + ELO bootstrap with debug
print_section "Running Debiased ELO Battles (debug)"
for judge_dir in "${settings[@]}"; do
  judge_name=$(basename "$judge_dir")
  print_status "Judge: $judge_name"
  for strat_dir in "$judge_dir"/base_debiased*; do
    [ -d "$strat_dir" ] || continue
    approach=${strat_dir##*/}
    # Derive approach suffix: if exactly base_debiased, mark as default; else strip prefix
    if [ "$approach" = "base_debiased" ]; then
      suffix="default"
    else
      suffix=${approach#base_debiased_}
    fi
    print_status "  Approach dir: $approach (suffix=$suffix) -> ELO bootstrap (rounds=$ELO_ROUNDS, debug)"
    set -x
    python generate_debiased_rankings_elo.py \
      --setting-dir "$judge_dir" \
      --approach "$suffix" \
      --debiased-dir "$strat_dir" \
      --metric all \
      --rounds $ELO_ROUNDS \
      --debug \
      || print_warning "Debiased ELO generation failed for $judge_name [$approach]"
    { set +x; } 2>/dev/null
  done
done

print_section "Battles-only run complete"
