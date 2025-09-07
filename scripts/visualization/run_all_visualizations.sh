#!/bin/bash

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
print_status "This will generate all tables and figures from the JSONL data"

# Step 1: Generate ranking CSV files from JSONL data
print_section "Step 1: Generating Ranking CSV Files"
print_status "Converting JSONL data to ranking CSV format..."
if python generate_rankings.py; then
    print_status "✅ Ranking CSV files generated successfully"
else
    print_error "Failed to generate ranking CSV files"
    exit 1
fi

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

# Step 4: Generate individual judge visualizations (optional)
print_section "Step 4: Creating Individual Judge Visualizations"
print_status "Generating detailed visualizations for each judge..."

# Get the data base path
DATA_BASE="../../sos-addl-data/InDepthAnalysis"

# Process each judge with debiased data
for judge_dir in "$DATA_BASE"/*-setting*; do
    # Check for any base_debiased* directories (new or old format)
    if ls "$judge_dir"/base_debiased* >/dev/null 2>&1; then
        judge_name=$(basename "$judge_dir")
        print_status "Processing $judge_name..."
        
        if python visualize_bias_transformation.py \
            --judge-dir "$judge_dir" \
            --output-dir "../../figures/individual_judges/$judge_name"; then
            print_status "✅ Visualizations created for $judge_name"
        else
            print_warning "Failed to create visualizations for $judge_name"
        fi
    fi
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
echo -e "\n📁 Ranking CSVs in data directories:"
find "$DATA_BASE" -path "*/tables_debiased_*/tables/factor_scores_updated_cis" -type d 2>/dev/null | while read -r dir; do
    n_csvs=$(ls -1 "$dir"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_csvs" -gt 0 ]; then
        echo "   - $dir ($n_csvs files)"
    fi
done

print_section "All visualizations completed successfully! 🎉"