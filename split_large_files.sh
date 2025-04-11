#!/bin/bash

# Directory to search (default: current directory)
SEARCH_DIR=${1:-.}
CHUNK_SIZE=50M

find "$SEARCH_DIR" -type f -size +50M | while read -r file; do
  dir=$(dirname "$file")
  base=$(basename "$file")
  full_path=$(realpath "$file")

  # Split extension and filename
  extension="${base##*.}"
  filename="${base%.*}"

  # Rename original file to *_unsplitted.ext
  renamed="${filename}_unsplitted.${extension}"
  renamed_path="${dir}/${renamed}"

  echo "Renaming $base to $renamed"
  mv "$file" "$renamed_path"

  # Split using original name as prefix
  split -b "$CHUNK_SIZE" "$renamed_path" "${dir}/${filename}.${extension}_chunk_"

  echo "✅ Done splitting $base into chunks prefixed with: ${filename}.${extension}_chunk_"
done
