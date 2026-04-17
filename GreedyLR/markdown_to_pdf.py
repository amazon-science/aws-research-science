#!/usr/bin/env python3
"""
Convert markdown to PDF using available Python libraries
"""

import re
import base64
from pathlib import Path

def markdown_to_html(markdown_content):
    """Convert markdown to HTML with basic formatting"""

    # Convert headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', markdown_content, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Convert bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Convert code blocks
    html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

    # Convert lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)

    # Wrap consecutive list items
    html = re.sub(r'(<li>.*?</li>)(?:\n<li>.*?</li>)*', lambda m: '<ul>' + m.group(0) + '</ul>', html, flags=re.DOTALL)

    # Convert tables
    lines = html.split('\n')
    in_table = False
    new_lines = []

    for line in lines:
        if '|' in line and not line.strip().startswith('*') and not line.strip().startswith('#'):
            if not in_table:
                new_lines.append('<table border="1" style="border-collapse: collapse;">')
                in_table = True

            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last
            if all(cell.replace('-', '').replace(' ', '') == '' for cell in cells):
                # Skip separator rows
                continue

            new_lines.append('<tr>')
            for cell in cells:
                new_lines.append(f'<td style="padding: 8px;">{cell}</td>')
            new_lines.append('</tr>')
        else:
            if in_table:
                new_lines.append('</table>')
                in_table = False
            new_lines.append(line)

    if in_table:
        new_lines.append('</table>')

    html = '\n'.join(new_lines)

    # Convert images to embedded base64
    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2)

        # Try to find the image file
        full_path = Path(image_path)
        if not full_path.exists():
            return f'<p><em>Image not found: {image_path}</em></p>'

        try:
            with open(full_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return f'<img src="data:image/png;base64,{img_base64}" alt="{alt_text}" style="max-width: 100%; height: auto;">'
        except Exception as e:
            return f'<p><em>Error loading image: {image_path} - {str(e)}</em></p>'

    html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, html)

    # Convert links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Convert line breaks
    html = re.sub(r'\n\n', '</p><p>', html)
    html = re.sub(r'^(?!<[h1-6]|<ul|<ol|<table|<pre|<p)', '<p>', html, flags=re.MULTILINE)

    return html

def create_pdf_html():
    """Create a complete HTML document with CSS for PDF conversion"""

    # Read the markdown file
    with open('FINAL_COMPREHENSIVE_README.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Convert to HTML
    html_content = markdown_to_html(markdown_content)

    # Create complete HTML document
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GreedyLR Comprehensive Analysis</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 28px;
        }}

        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
            font-size: 22px;
        }}

        h3 {{
            color: #2c3e50;
            margin-top: 25px;
            font-size: 18px;
        }}

        table {{
            width: 100%;
            margin: 20px 0;
            border-collapse: collapse;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}

        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}

        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}

        code {{
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}

        strong {{
            color: #2c3e50;
            font-weight: bold;
        }}

        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}

        li {{
            margin-bottom: 8px;
        }}

        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .figure-caption {{
            font-style: italic;
            color: #666;
            margin-top: -15px;
            margin-bottom: 20px;
            font-size: 14px;
        }}

        @media print {{
            body {{
                font-size: 12px;
            }}

            h1 {{
                font-size: 24px;
                page-break-after: avoid;
            }}

            h2 {{
                font-size: 18px;
                page-break-after: avoid;
            }}

            h3 {{
                font-size: 16px;
                page-break-after: avoid;
            }}

            img {{
                max-height: 400px;
                page-break-inside: avoid;
            }}

            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

    # Save HTML file
    with open('FINAL_COMPREHENSIVE_README.html', 'w', encoding='utf-8') as f:
        f.write(full_html)

    print("HTML file created: FINAL_COMPREHENSIVE_README.html")
    print("\nTo convert to PDF:")
    print("1. Open the HTML file in Chrome/Safari")
    print("2. Print -> Save as PDF")
    print("3. Or use: wkhtmltopdf FINAL_COMPREHENSIVE_README.html FINAL_COMPREHENSIVE_README.pdf")

if __name__ == "__main__":
    create_pdf_html()