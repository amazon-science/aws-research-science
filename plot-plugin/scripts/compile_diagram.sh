#!/bin/bash
# Compile LaTeX/TikZ diagram to high-quality PNG

set -e

TEX_FILE="$1"

if [ -z "$TEX_FILE" ]; then
    echo "Usage: $0 <diagram.tex>"
    exit 1
fi

if [ ! -f "$TEX_FILE" ]; then
    echo "Error: File '$TEX_FILE' not found"
    exit 1
fi

# Get base name without extension
BASE_NAME="${TEX_FILE%.tex}"
OUTPUT_DIR="$(dirname "$TEX_FILE")"

echo "📝 Compiling $TEX_FILE..."

# Check for required tools
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install TeX Live:"
    echo "   macOS: brew install --cask mactex-no-gui"
    echo "   Linux: sudo apt-get install texlive-latex-base texlive-pictures"
    exit 1
fi

if ! command -v convert &> /dev/null; then
    echo "❌ Error: ImageMagick convert not found. Please install:"
    echo "   macOS: brew install imagemagick"
    echo "   Linux: sudo apt-get install imagemagick"
    exit 1
fi

# Compile LaTeX to PDF
echo "  → Running pdflatex..."
pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$TEX_FILE" > /dev/null 2>&1

if [ ! -f "${BASE_NAME}.pdf" ]; then
    echo "❌ Compilation failed. Check LaTeX errors:"
    pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$TEX_FILE"
    exit 1
fi

# Convert PDF to high-quality PNG (300 DPI for publication quality)
echo "  → Converting to PNG (300 DPI)..."
convert -density 300 "${BASE_NAME}.pdf" -quality 100 "${BASE_NAME}.png"

# Clean up auxiliary files
rm -f "${BASE_NAME}.aux" "${BASE_NAME}.log" "${BASE_NAME}.pdf"

echo "✅ Generated: ${BASE_NAME}.png"
echo ""
echo "📊 To view the diagram:"
echo "   Read ${BASE_NAME}.png"
