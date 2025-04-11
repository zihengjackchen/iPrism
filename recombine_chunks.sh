#!/bin/bash

# Directory to search (default: current directory)
SEARCH_DIR=${1:-.}

# Find all chunk_aa files
find "$SEARCH_DIR" -type f -name "*_chunk_aa" | while read -r first_chunk; do
  prefix="${first_chunk%chunk_aa}"       # Path prefix to all chunks
  base_with_ext_chunk="${prefix##*/}"    # e.g., big_model.pth_chunk_
  base_with_ext="${base_with_ext_chunk%_chunk_}"  # e.g., big_model.pth
  dir=$(dirname "$first_chunk")
  output_path="${dir}/${base_with_ext}"

  echo "Recombining chunks to: $output_path"
  cat "${prefix}"* > "$output_path"
  echo "✅ Recombined file saved as: $output_path"
done
