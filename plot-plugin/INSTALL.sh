#!/bin/bash
# Quick installer for CORAL Plot Plugin

echo "🚀 Installing CORAL Plot Plugin..."
echo ""

# Check Claude Code installed
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code not installed"
    echo "Install: https://claude.com/claude-code"
    exit 1
fi

echo "✅ Claude Code found"

# Check LaTeX
if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  LaTeX not found - required for diagram compilation"
    echo ""
    echo "Install LaTeX:"
    echo "  macOS:  brew install --cask mactex-no-gui"
    echo "  Linux:  sudo apt-get install texlive-latex-base texlive-pictures texlive-latex-extra"
    echo ""
fi

# Check ImageMagick
if ! command -v convert &> /dev/null; then
    echo "⚠️  ImageMagick not found - required for PNG conversion"
    echo ""
    echo "Install ImageMagick:"
    echo "  macOS:  brew install imagemagick"
    echo "  Linux:  sudo apt-get install imagemagick"
    echo ""
fi

# Setup scripts
chmod +x scripts/*.sh 2>/dev/null

echo ""
echo "✅ Plot Plugin installed!"
echo ""
echo "📝 To generate diagrams:"
echo ""
echo "1. Use the /plot command:"
echo '   /plot neural network with 3 layers'
echo ""
echo "2. Or compile manually:"
echo '   ${CLAUDE_PLUGIN_ROOT}/scripts/compile_diagram.sh my_diagram.tex'
echo ""
echo "📊 Examples available in examples/ directory"
echo ""
