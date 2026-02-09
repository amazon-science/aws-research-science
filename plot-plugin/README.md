# Plot Plugin: Publication-Quality Diagrams

Generate publication-quality diagrams with iterative refinement using LaTeX/TikZ (and other backends in the future).

## Features

- **Iterative Refinement**: Claude generates, views, and improves diagrams until publication-ready
- **High Quality**: 300 DPI PNG output suitable for papers and presentations
- **Multiple Diagram Types**: Neural networks, flowcharts, architecture diagrams, plots, etc.
- **Automatic Improvement**: Claude assesses and refines diagrams automatically

## Installation

This plugin is part of the CORAL marketplace:

```bash
/plugin marketplace add amazon-science/aws-research-science#plugins
/plugin install plot@coral
```

## Requirements

**LaTeX Distribution:**
```bash
# macOS
brew install --cask mactex-no-gui

# Linux
sudo apt-get install texlive-latex-base texlive-pictures texlive-latex-extra
```

**ImageMagick:**
```bash
# macOS
brew install imagemagick

# Linux
sudo apt-get install imagemagick
```

## Usage

Simply use the `/plot:tex` command with your requirements:

```
/plot:tex neural network with 3 input nodes, 4 hidden nodes, and 2 output nodes
```

```
/plot:tex flowchart for data processing pipeline with error handling
```

```
/plot:tex architecture diagram showing microservices communication
```

Claude will:
1. Generate LaTeX/TikZ code
2. Compile to PNG
3. View and assess the result
4. Make improvements iteratively
5. Continue until publication-quality

## Examples

The `examples/` directory contains sample diagrams:
- `neural_network.tex` - Neural network architecture
- `flowchart.tex` - Process flowchart

To compile an example:
```bash
${CLAUDE_PLUGIN_ROOT}/scripts/compile_diagram.sh examples/neural_network.tex
```

## Manual Compilation

You can also compile diagrams manually:

```bash
# Create your diagram
vim my_diagram.tex

# Compile to PNG
${CLAUDE_PLUGIN_ROOT}/scripts/compile_diagram.sh my_diagram.tex

# View the result
open my_diagram.png  # macOS
xdg-open my_diagram.png  # Linux
```

## Publication Quality Checklist

Claude checks for:
- ✅ Clean, crisp lines
- ✅ Readable font sizes at print scale
- ✅ Professional color scheme
- ✅ Proper alignment and spacing
- ✅ Clear labels and legends
- ✅ Appropriate arrow styles
- ✅ No overlapping text
- ✅ Consistent styling

## Future Backends

Currently uses LaTeX/TikZ. Future versions may support:
- Matplotlib (Python)
- D3.js (JavaScript)
- Graphviz (DOT)
- PlantUML

## Troubleshooting

**"pdflatex not found"**
→ Install LaTeX distribution (see Requirements)

**"convert not found"**
→ Install ImageMagick (see Requirements)

**Compilation errors**
→ Check LaTeX syntax in the `.tex` file
→ Ensure all required TikZ libraries are imported

**Low quality output**
→ The script uses 300 DPI by default
→ Modify `compile_diagram.sh` density setting if needed

## Version

**1.0.0** - Initial release with LaTeX/TikZ backend

## Author

Amazon CORAL Lab
