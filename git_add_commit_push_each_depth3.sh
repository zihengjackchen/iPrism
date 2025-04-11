#!/bin/bash

# Optional: custom commit message prefix
COMMIT_PREFIX=${1:-"Add files from"}

# Loop through all subfolders at depth = 3
find . -mindepth 2 -maxdepth 3 -type d | while read -r dir; do
  echo "📁 Processing directory: $dir"

  # Add all files in this dir
  git add "$dir"/* 2>/dev/null

  # Check if anything was staged
  if ! git diff --cached --quiet; then
    # Commit and push
    COMMIT_MSG="$COMMIT_PREFIX $dir"
    echo "📦 Committing: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"

    echo "🚀 Pushing to origin..."
    git push
  else
    echo "✅ Nothing to commit in $dir"
  fi
done
