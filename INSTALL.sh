#!/bin/bash
# Quick installer for CORAL DS plugin

echo "🚀 Installing CORAL DS Plugin..."
echo ""

# Check Claude Code installed
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code not installed"
    echo "Install: https://claude.com/claude-code"
    exit 1
fi

echo "✅ Claude Code found"

# Check Python and rich
if ! python3 -c "import rich" 2>/dev/null; then
    echo "📦 Installing rich for dashboard..."
    pip install rich --quiet
fi

echo "✅ Dependencies ready"

# Setup scripts
chmod +x scripts/*.sh scripts/*.py 2>/dev/null

echo ""
echo "✅ DS Experiment Plugin installed!"
echo ""
echo "📝 To enable automatic experiment tracking:"
echo ""
echo "1. Add to Claude Code settings (~/.claude/settings.json):"
echo '   {"enabledPlugins": {"ds": true}}'
echo ""
echo "2. Start Claude Code in your project directory"
echo ""
echo "3. The plugin will automatically:"
echo "   - Track all training tasks"
echo "   - Show GPU status on startup"
echo "   - Enable /ds:dash and statusline"
echo ""
echo "📊 Commands:"
echo "   /ds:dash    # View dashboard"
echo "   /ds:help    # Show help"
echo ""
