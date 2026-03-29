#!/bin/bash

#===============================================================================
# FLAC to MP3 Converter
#
# Converts all .flac files in a directory to .mp3 using ffmpeg.
# Handles filenames with spaces, commas, colons, Japanese characters,
# and other special characters.
#
# Usage:
#   ./flac2mp3.sh <input_directory> [output_directory]
#
# Examples:
#   ./flac2mp3.sh /Users/astrobot/Music/flac
#   ./flac2mp3.sh /Users/astrobot/Music/flac /Users/astrobot/Music/mp3
#
# Prerequisites:
#   - ffmpeg installed (brew install ffmpeg)
#===============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

#===============================================================================
# Configuration
#===============================================================================

# MP3 quality settings
# -b:a 320k = constant 320 kbps bitrate (highest quality CBR)
# -q:a 0 = VBR highest quality (~245 kbps average)
# -q:a 2 = VBR good quality (~190 kbps average)
QUALITY="-b:a 320k"

#===============================================================================
# Prerequisite Checks
#===============================================================================

check_ffmpeg() {
    if ! command -v ffmpeg &> /dev/null; then
        log_error "ffmpeg is not installed. Please install it first."
        echo "  macOS: brew install ffmpeg"
        echo "  Ubuntu: sudo apt install ffmpeg"
        exit 1
    fi
}

check_input_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        log_error "Input directory does not exist: $dir"
        exit 1
    fi
}

#===============================================================================
# Conversion Functions
#===============================================================================

convert_flac_to_mp3() {
    local input_dir="$1"
    local output_dir="$2"
    
    # Counters
    local file_count=0
    local converted_count=0
    local skipped_count=0
    local error_count=0
    
    # Change to the input directory
    cd "$input_dir" || exit 1
    
    # Use a simple glob with proper settings
    shopt -s nullglob
    
    # Count files
    for file in *.flac; do
        ((file_count++))
    done
    
    if [ "$file_count" -eq 0 ]; then
        log_warning "No FLAC files found in $input_dir"
        exit 0
    fi
    
    log_info "Found $file_count FLAC file(s)"
    log_info "Output directory: $output_dir"
    log_info "Quality: $QUALITY"
    echo ""
    
    for file in *.flac; do
        # Skip if not a file
        [ -f "$file" ] || continue
        
        # Get filename without extension
        local filename="${file%.flac}"
        local output_file="${output_dir}/${filename}.mp3"
        
        # Skip if output already exists
        if [ -f "$output_file" ]; then
            log_warning "Skipping (already exists): $filename.mp3"
            ((skipped_count++))
            continue
        fi
        
        # Convert to MP3
        # Using file: protocol to handle special characters (commas, colons, etc.)
        printf '%s\n' "Converting: $file"
        if ffmpeg -nostdin -hide_banner -loglevel error -i "file:./$file" -codec:a libmp3lame $QUALITY "file:$output_file"; then
            ((converted_count++))
        else
            log_error "Failed to convert: $file"
            ((error_count++))
        fi
    done
    
    echo ""
    log_header "Conversion Summary"
    echo "  Converted: $converted_count"
    echo "  Skipped:   $skipped_count"
    echo "  Errors:    $error_count"
    echo ""
    
    if [ "$error_count" -eq 0 ]; then
        log_success "Conversion complete!"
    else
        log_warning "Conversion complete with $error_count error(s)"
    fi
}

#===============================================================================
# Show Help
#===============================================================================

show_help() {
    cat << EOF
FLAC to MP3 Converter - Convert FLAC audio files to MP3

Usage: ./flac2mp3.sh <input_directory> [output_directory]

Arguments:
  input_directory   Directory containing .flac files
  output_directory  Directory for output .mp3 files (defaults to input_directory)

Examples:
  ./flac2mp3.sh /Users/astrobot/Music/flac
  ./flac2mp3.sh /Users/astrobot/Music/flac /Users/astrobot/Music/mp3

Quality Settings (edit script to change):
  -b:a 320k   Constant 320 kbps (current setting)
  -q:a 0      Variable bitrate, highest quality (~245 kbps)
  -q:a 2      Variable bitrate, good quality (~190 kbps)

Features:
  - Handles special characters in filenames (spaces, commas, Japanese, etc.)
  - Preserves metadata (artist, album, etc.)
  - Skips files that already exist
  - Shows progress and summary

Prerequisites:
  - ffmpeg installed

EOF
}

#===============================================================================
# Main
#===============================================================================

# Show help if no arguments or help flag
if [ $# -lt 1 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "help" ]; then
    show_help
    exit 0
fi

# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="${2:-$INPUT_DIR}"

# Run checks
check_ffmpeg
check_input_directory "$INPUT_DIR"

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    log_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR" || { log_error "Cannot create output directory: $OUTPUT_DIR"; exit 1; }
fi

# Run conversion
log_header "FLAC to MP3 Converter"
log_info "Input directory: $INPUT_DIR"

convert_flac_to_mp3 "$INPUT_DIR" "$OUTPUT_DIR"
