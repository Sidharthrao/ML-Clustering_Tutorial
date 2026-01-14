#!/bin/bash

# Script to push E-commerce Customer Segmentation project to GitHub
# Run this script from the repository root

REPO_PATH="/Users/sidharthrao/Documents/Documents - Sidharth's MacBook Pro/GitHub/Project-Rogue"
PROJECT_PATH="Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data"

echo "=== Navigating to repository ==="
cd "$REPO_PATH" || { echo "Error: Could not navigate to $REPO_PATH"; exit 1; }
pwd

echo ""
echo "=== Checking git status ==="
git status

echo ""
echo "=== Adding project files ==="
git add "$PROJECT_PATH/"

echo ""
echo "=== Files staged for commit ==="
git status --short

echo ""
echo "=== Committing changes ==="
git commit -m "Add E-commerce Customer Segmentation ML Pipeline

- Complete modular pipeline structure (config, src, scripts, notebooks)
- Data loading and aggregation modules for event-level to customer-level transformation
- Feature engineering (RFM, behavioral, temporal, engagement, price sensitivity)
- Clustering implementation (K-means, DBSCAN, Hierarchical)
- Cluster evaluation and visualization modules
- Comprehensive pipeline notebook with detailed justifications
- Training script and documentation"

echo ""
echo "=== Checking remote repository ==="
git remote -v

echo ""
echo "=== Pushing to GitHub ==="
echo "Note: If this is your first push, you may need to set upstream:"
echo "  git push -u origin main"
echo ""
read -p "Press Enter to push to GitHub (or Ctrl+C to cancel)..."
git push origin main || git push -u origin main

echo ""
echo "=== Done! ==="
echo "Check your GitHub repository to verify the push was successful."

