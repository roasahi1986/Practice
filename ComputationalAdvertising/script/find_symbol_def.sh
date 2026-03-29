#!/bin/bash

# Check if input directory is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <directory> <symbol>"
  exit 1
fi

directory="$1"
symbol="$2"

# Check if the directory exists
if [ ! -d "${directory}" ]; then
  echo "Directory '$directory' not found."
  exit 1
fi

# Find all .a files in the directory and its subdirectories
a_files=$(find "$directory" -type f -name "*.a")

# Iterate over each .a file
for a_file in ${a_files}; do
  # Extract symbols from the .a file and check if the specified symbol exists
  nm -C "${a_file}" | grep -v "^ " | grep -q "\<$symbol\>"
  if [ $? -eq 0 ]; then
    echo "Symbol '${symbol}' found in: ${a_file}"
    nm -C "$a_file" | grep "${symbol}"
  fi
done
