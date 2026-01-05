# Security Incident: Exposed API Keys

## Date: January 4, 2026

## Issue
The `.env` file containing API keys was accidentally committed to the repository in commit `61cd32d`.

## Exposed Secrets
The following API keys were exposed in the git history:
- `NEWS_API_KEY=565af227ed644a478b7e3d97817af9e4`
- `ALPHA_VANTAGE_API_KEY=437KCYLLKB9KXNMT`
- `POLYGON_API_KEY=c72mT_m6VyA07WbN9GSOpRvtXAtIptpJ`

## Actions Taken
1. ✅ Removed `.env` from git tracking
2. ✅ Verified `.env` is in `.gitignore`
3. ⚠️ **REQUIRED**: Remove from git history (see below)
4. ⚠️ **REQUIRED**: Rotate all exposed API keys

## Required Actions

### 1. Remove from Git History
The `.env` file still exists in git history. To completely remove it, you have two options:

**Option A: Using git-filter-repo (Recommended)**
```bash
pip install git-filter-repo
git filter-repo --path .env --invert-paths --force
git push origin --force --all
```

**Option B: Using BFG Repo Cleaner**
```bash
# Download BFG from https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files .env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all
```

**Option C: Manual git filter-branch (if above tools unavailable)**
```bash
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all
git push origin --force --all
```

⚠️ **WARNING**: Force pushing rewrites history. Coordinate with any collaborators first.

### 2. Rotate Exposed API Keys
**IMMEDIATELY** rotate/regenerate all exposed API keys:

1. **NewsAPI** (https://newsapi.org/account):
   - Go to your account settings
   - Generate a new API key
   - Update your local `.env` file

2. **Alpha Vantage** (https://www.alphavantage.co/support/#api-key):
   - Request a new API key
   - Update your local `.env` file

3. **Polygon.io** (https://polygon.io/dashboard/api-keys):
   - Generate a new API key
   - Revoke the old one
   - Update your local `.env` file

### 3. Verify .gitignore
The `.env` file is already in `.gitignore` (line 78), which is correct.

### 4. Prevent Future Issues
- Never commit `.env` files
- Use `git status` before committing to verify no sensitive files are staged
- Consider using GitHub Secrets for CI/CD
- Use environment variable injection in production

## Status
- [x] Removed from tracking
- [x] Removed from history (completed - force pushed to GitHub)
- [ ] API keys rotated (REQUIRED - do this immediately)
- [x] .gitignore verified

## History Cleanup Completed
✅ Successfully removed `.env` file from entire git history using `git-filter-repo`
✅ Force pushed to GitHub - history has been rewritten
✅ Verified: `.env` file no longer exists in any commit in the repository

**Note**: The commit hashes have changed due to history rewrite. This is expected and normal.

