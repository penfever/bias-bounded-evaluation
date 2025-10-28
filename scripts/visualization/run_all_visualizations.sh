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

# Start the visualization pipeline
print_section "Starting Visualization Pipeline"
print_status "This will generate all tables and figures using the improved ELO bootstrap where available"

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

# Step 0: Generate original ELO rankings from base_processed
# Step 0: (skipped) Original ELO rankings — we use existing CSVs from factor_scores_original_cis
print_section "Step 0: Using existing original CSVs (no regeneration)"

# Step 1: Generate debiased ELO rankings for each approach
print_section "Step 1: Generating Debiased ELO Rankings"
for judge_dir in "${settings[@]}"; do
  # Enumerate debiased strategy directories
  for strat_dir in "$judge_dir"/base_debiased*; do
    if [ -d "$strat_dir" ]; then
      base_name=$(basename "$strat_dir")
      if [ "$base_name" = "base_debiased" ]; then
        approach_suffix="default"
      else
        approach_suffix=${base_name#base_debiased_}
      fi
      print_status "Debiased ELO ($approach_suffix) for $(basename "$judge_dir")"
      echo "python generate_debiased_rankings_elo.py --setting-dir \"$judge_dir\" --approach \"$approach_suffix\" --debiased-dir \"$strat_dir\" --metric all --rounds $ELO_ROUNDS"
      python generate_debiased_rankings_elo.py \
        --setting-dir "$judge_dir" \
        --approach "$approach_suffix" \
        --debiased-dir "$strat_dir" \
        --metric all \
        --rounds $ELO_ROUNDS \
        || print_warning "Debiased ELO generation failed for $(basename "$judge_dir") [$approach_suffix]"
    fi
  done
done

# Step 1b: Copy ELO ranking CSV files into standard locations (no fallback)
print_section "Step 1b: Copying ELO Ranking CSV Files"
print_status "Copying ELO CSVs into factor_scores_updated_cis (required)..."
python generate_rankings.py || { print_error "Failed to copy ELO ranking CSV files"; exit 1; }

# Step 2: Create comparison visualizations
print_section "Step 2: Creating Comparison Visualizations"
print_status "Generating bias transformation comparison plots..."
if python create_comparisons.py; then
    print_status "✅ Comparison visualizations created successfully"
else
    print_error "Failed to create comparison visualizations"
    exit 1
fi

# Step 3: Generate summary visualizations from JSONL
print_section "Step 3: Creating Summary Visualizations"
print_status "Generating combined A-BB analysis visualizations..."
if python visualize_results.py --use-jsonl; then
    print_status "✅ Summary visualizations created successfully"
else
    print_error "Failed to create summary visualizations"
    exit 1
fi

# Helper: normalize approach name to ELO output dir suffix (POSIX-compatible)
normalize_approach() {
  a="$1"
  case "$a" in
    combined_abb_conservative) printf '%s\n' conservative ;;
    combined_abb_rms)          printf '%s\n' rms ;;
    combined_abb_weighted)     printf '%s\n' weighted ;;
    combined_abb_montecarlo)   printf '%s\n' montecarlo ;;
    abb_*)                     printf '%s\n' "${a#abb_}" ;;
    *)                         printf '%s\n' "$a" ;;
  esac
}

# Step 4: Generate individual judge visualizations (optional)
print_section "Step 4: Creating Individual Judge Visualizations"
print_status "Generating detailed visualizations for each judge..."

# Process each judge with debiased data
for judge_dir in "$DATA_BASE"/*-setting*; do
  judge_name=$(basename "$judge_dir")
    # For each approach, require ELO CSV visualizations; no JSONL fallback
  for strat_dir in "$judge_dir"/base_debiased*; do
    [ -d "$strat_dir" ] || continue
    approach=${strat_dir##*/}
    if [ "$approach" = "base_debiased" ]; then
      suffix="default"
    else
      suffix=${approach#base_debiased_}
    fi
    norm_approach=$(normalize_approach "$suffix")
    # Prefer normalized directory; fall back to raw approach; finally scan for a matching dir
    elo_deb_dir="$judge_dir/tables_debiased_${norm_approach}/tables/factor_scores_updated_cis_elo"
    if [ ! -d "$elo_deb_dir" ]; then
      alt_dir="$judge_dir/tables_debiased_${approach}/tables/factor_scores_updated_cis_elo"
      if [ -d "$alt_dir" ]; then
        elo_deb_dir="$alt_dir"
      else
        found_dir=""
        while IFS= read -r d; do
          if [[ "$d" == *"tables_debiased_${norm_approach}"*"/tables/factor_scores_updated_cis_elo" ]]; then
            found_dir="$d"; break; fi
        done < <(find "$judge_dir" -maxdepth 2 -type d -name "tables_debiased_*" 2>/dev/null | sed 's#$#/tables/factor_scores_updated_cis_elo#' | sort)
        if [ -z "$found_dir" ]; then
          while IFS= read -r d; do
            if [[ "$d" == *"tables_debiased_${approach}"*"/tables/factor_scores_updated_cis_elo" ]]; then
              found_dir="$d"; break; fi
          done < <(find "$judge_dir" -maxdepth 2 -type d -name "tables_debiased_*" 2>/dev/null | sed 's#$#/tables/factor_scores_updated_cis_elo#' | sort)
        fi
        if [ -n "$found_dir" ] && [ -d "$found_dir" ]; then
          elo_deb_dir="$found_dir"
        fi
      fi
    fi
    out_dir="../../figures/individual_judges/${judge_name}_${suffix}"
    mkdir -p "$out_dir"
    if [ -d "$elo_deb_dir" ] && ls "$elo_deb_dir"/*.csv >/dev/null 2>&1; then
      print_status "ELO visualization for $judge_name [$suffix]"
      # Use existing original CSVs directory
      elo_orig_dir="$judge_dir/tables/factor_scores_original_cis"
      if python visualize_bias_transformation.py \
          --elo-debiased-dir "$elo_deb_dir" \
          ${elo_orig_dir:+--elo-original-dir "$elo_orig_dir"} \
          --output-dir "$out_dir" \
          --metrics score correctness_score completeness_score safety_score conciseness_score style_score; then
        print_status "✅ ELO visualizations created for $judge_name [$suffix]"
      else
        print_error "Failed ELO visualization for $judge_name [$suffix]"
        exit 1
      fi
    else
      print_error "No ELO CSVs found for $judge_name [$suffix]. Tried:"
      print_error "  - $judge_dir/tables_debiased_${norm_approach}/tables/factor_scores_updated_cis_elo"
      print_error "  - $judge_dir/tables_debiased_${suffix}/tables/factor_scores_updated_cis_elo"
      [ -n "$found_dir" ] && print_error "  - $found_dir"
      exit 1
    fi
  done
done

# Summary report
print_section "Visualization Pipeline Complete!"

# Count generated files
FIGURES_DIR="../../figures"
TABLES_DIR="../../tables"

if [ -d "$FIGURES_DIR" ]; then
    n_figures=$(find "$FIGURES_DIR" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    print_status "Generated figures: $n_figures PNG files"
    print_status "Figures location: $FIGURES_DIR"
else
    print_warning "Figures directory not found"
fi

if [ -d "$TABLES_DIR" ]; then
    n_tables=$(find "$TABLES_DIR" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
    print_status "Generated tables: $n_tables CSV files"
    print_status "Tables location: $TABLES_DIR"
else
    print_warning "Tables directory not found"
fi

# List output directories
print_section "Output Locations"
echo "📊 Figures:"
if [ -d "$FIGURES_DIR" ]; then
    find "$FIGURES_DIR" -type d -maxdepth 2 | sort | while read -r dir; do
        if [ "$dir" != "$FIGURES_DIR" ]; then
            echo "   - $dir"
        fi
    done
fi

echo -e "\n📋 Tables:"
if [ -d "$TABLES_DIR" ]; then
    find "$TABLES_DIR" -type d -maxdepth 2 | sort | while read -r dir; do
        if [ "$dir" != "$TABLES_DIR" ]; then
            echo "   - $dir"
        fi
    done
fi

# Also check for CSV files in the data directories
echo -e "\n📁 Ranking CSVs in data directories (ELO):"
find "$DATA_BASE" -path "*/tables_debiased_*/tables/factor_scores_updated_cis_elo" -type d 2>/dev/null | while read -r dir; do
    n_csvs=$(ls -1 "$dir"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_csvs" -gt 0 ]; then
        echo "   - $dir ($n_csvs files)"
    fi
done

print_section "All visualizations completed successfully! 🎉"
