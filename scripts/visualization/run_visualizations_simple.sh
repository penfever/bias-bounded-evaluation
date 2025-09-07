#!/bin/bash

# Simple script to run all visualizations in the correct order
# Run from the visualization directory

set -e

echo "Starting visualization pipeline..."
echo "================================"

# 1. Generate ranking CSV files from JSONL
echo "1. Generating ranking CSV files..."
python generate_rankings.py

# 2. Create comparison visualizations
echo -e "\n2. Creating comparison visualizations..."
python create_comparisons.py

# 3. Generate summary visualizations
echo -e "\n3. Creating summary visualizations..."
python visualize_results.py --use-jsonl

echo -e "\n✅ All visualizations complete!"
echo "Figures saved to: ../../figures/"
echo "Tables saved to: ../../tables/"