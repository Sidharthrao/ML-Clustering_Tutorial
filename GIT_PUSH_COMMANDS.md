# Git Push Commands for E-commerce Customer Segmentation Project

## Step 1: Navigate to Repository Root

```bash
cd "/Users/sidharthrao/Documents/Documents - Sidharth's MacBook Pro/GitHub/Project-Rogue"
```

## Step 2: Check Git Status

```bash
git status
```

## Step 3: Add All New Files

Add the new E-commerce clustering project files:

```bash
# Add the entire project directory
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/"

# Or add specific files if you prefer
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/config/"
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/src/"
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/scripts/"
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/notebooks/"
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/requirements.txt"
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/README.md"
```

## Step 4: Commit Changes

```bash
git commit -m "Add E-commerce Customer Segmentation ML Pipeline

- Complete modular pipeline structure (config, src, scripts, notebooks)
- Data loading and aggregation modules for event-level to customer-level transformation
- Feature engineering (RFM, behavioral, temporal, engagement, price sensitivity)
- Clustering implementation (K-means, DBSCAN, Hierarchical)
- Cluster evaluation and visualization modules
- Comprehensive pipeline notebook with detailed justifications
- Training script and documentation"
```

## Step 5: Push to GitHub

If you haven't set up a remote yet:

```bash
# Check if remote exists
git remote -v

# If no remote, add one (replace with your actual GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/Project-Rogue.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/Project-Rogue.git
```

Then push:

```bash
# Push to main branch
git push origin main

# Or if your branch is called master:
git push origin master
```

## Step 6: Verify Push

```bash
# Check remote status
git status

# Or view commits
git log --oneline -5
```

## Important Notes

1. **Before pushing, make sure you have:**
   - A GitHub repository created (if not, create one at github.com)
   - Proper .gitignore file (should exclude venv/, __pycache__/, *.pyc, etc.)

2. **Files that should NOT be committed:**
   - `venv/` directory
   - `__pycache__/` directories
   - `*.pyc` files
   - `models/*.pkl` (if they're large, consider git-lfs)
   - `logs/*.log`
   - `.DS_Store`

3. **If you encounter issues:**
   - Make sure you're authenticated with GitHub (use `gh auth login` or set up SSH keys)
   - Check if you have write permissions to the repository
   - Verify the remote URL is correct

## Quick One-Liner (if everything is set up)

```bash
cd "/Users/sidharthrao/Documents/Documents - Sidharth's MacBook Pro/GitHub/Project-Rogue" && \
git add "Inttrvu/Capstone_Projects/Captsone_Project - Clustering/ECommerce behavior data/" && \
git commit -m "Add E-commerce Customer Segmentation ML Pipeline" && \
git push origin main
```

