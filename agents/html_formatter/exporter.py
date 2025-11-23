"""
Deterministic HTML exporter for CourseState.
Converts the structured JSON data into a complete HTML page with interactive elements.
"""
from typing import List
from main.state import CourseState, Section, HtmlElement


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not isinstance(text, str):
        return str(text)
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def render_element(element: HtmlElement) -> str:
    """Render a single HTML element based on its type."""
    if element.type == "p":
        return f"<p>{escape_html(element.content)}</p>"
    
    elif element.type == "ul":
        if isinstance(element.content, list):
            items = "".join(f"<li>{escape_html(str(item))}</li>" for item in element.content)
            return f"<ul>{items}</ul>"
        return ""
    
    elif element.type == "quote":
        quote_data = element.content
        if isinstance(quote_data, dict):
            quote_text = escape_html(str(quote_data.get("quote", "")))
            author = escape_html(str(quote_data.get("author", "")))
            return f'<blockquote class="quote"><p>{quote_text}</p><footer>— {author}</footer></blockquote>'
        return ""
    
    elif element.type == "table":
        table_data = element.content
        if isinstance(table_data, dict):
            title = escape_html(str(table_data.get("title", "")))
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            
            html = f'<div class="table-container"><h4>{title}</h4><table>'
            
            # Headers
            if headers:
                html += "<thead><tr>"
                for header in headers:
                    html += f"<th>{escape_html(str(header))}</th>"
                html += "</tr></thead>"
            
            # Rows
            if rows:
                html += "<tbody>"
                for row in rows:
                    html += "<tr>"
                    for cell in row:
                        html += f"<td>{escape_html(str(cell))}</td>"
                    html += "</tr>"
                html += "</tbody>"
            
            html += "</table></div>"
            return html
        return ""
        
    elif element.type == "paragraphs":
        # Render nested blocks
        html = '<div class="content-items">'
        if isinstance(element.content, list):
            for block in element.content:
                html += '<div class="content-item">'
                html += f'<h4 class="item-title"><i class="{block.icon}"></i> {escape_html(block.title)}</h4>'
                html += '<div class="item-body">'
                for sub_element in block.elements:
                    html += render_element(sub_element)
                html += '</div></div>'
        html += '</div>'
        return html
    
    return ""


def render_section(section: Section, section_num: int) -> str:
    """Render a complete section with all its content in a simple, linear format."""
    html = f'<section class="course-section" id="section-{section_num}">'
    html += f'<h3 class="section-title">{escape_html(section.title)}</h3>'
    
    # HTML Structure
    if section.html:
        # Iterate through theory elements
        for element in section.html.theory:
            # Wrap intro in specific class if it's the first p
            if element == section.html.theory[0] and element.type == "p":
                html += '<div class="section-intro">'
                html += render_element(element)
                html += '</div>'
            # Wrap conclusion in specific class if it's the last p
            elif element == section.html.theory[-1] and element.type == "p":
                html += '<div class="section-conclusion">'
                html += render_element(element)
                html += '</div>'
            else:
                html += render_element(element)
    
    html += '</section>'
    return html


def export_to_html(course_state: CourseState, output_path: str) -> None:
    """
    Export the complete course to a clean, readable HTML file.
    
    Args:
        course_state: The complete course state with all content
        output_path: Path where the HTML file should be saved
    """
    html = f"""<!DOCTYPE html>
<html lang="{course_state.language.lower()[:2]}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(course_state.title)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@mdi/font@7.4.47/css/materialdesignicons.min.css">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.8; 
            color: #2c3e50; 
            background: #fafafa;
        }}
        .container {{ 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px;
            background: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.1);
        }}
        
        /* Header */
        header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 60px 40px; 
            text-align: center;
            margin: -20px -20px 40px -20px;
        }}
        h1 {{ 
            font-size: 3em; 
            margin-bottom: 15px;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        .course-meta {{ 
            font-size: 1.1em;
            opacity: 0.95;
            margin: 10px 0;
        }}
        
        /* Module Structure */
        .module {{ 
            margin-bottom: 50px;
            page-break-inside: avoid;
        }}
        .module-title {{ 
            color: #667eea; 
            font-size: 2.2em; 
            margin: 40px 0 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            font-weight: 400;
        }}
        .submodule {{ 
            margin: 30px 0;
            padding-left: 20px;
            border-left: 3px solid #e0e7ff;
        }}
        .submodule-title {{ 
            color: #764ba2; 
            font-size: 1.8em; 
            margin-bottom: 20px;
            font-weight: 400;
        }}
        
        /* Section */
        .course-section {{ 
            margin: 30px 0;
            padding: 25px;
            background: #fefefe;
            border-radius: 8px;
            border: 1px solid #e8e8e8;
        }}
        .section-title {{ 
            color: #2c3e50; 
            font-size: 1.6em; 
            margin-bottom: 20px;
            font-weight: 500;
        }}
        
        /* Content */
        .section-intro {{ 
            background: linear-gradient(to right, #f0f4ff, #fefefe);
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
            font-size: 1.05em;
        }}
        
        .content-items {{ 
            margin: 25px 0;
        }}
        .content-item {{ 
            margin: 25px 0;
            padding: 20px 0;
        }}
        .item-title {{ 
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }}
        .item-title i {{ 
            font-size: 1.2em;
        }}
        .item-body {{ 
            padding-left: 35px;
        }}
        
        .section-conclusion {{ 
            background: linear-gradient(to right, #fefefe, #f0f4ff);
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #764ba2;
            font-style: italic;
        }}
        
        /* Typography */
        p {{ 
            margin: 15px 0;
            text-align: justify;
            font-size: 1.05em;
        }}
        ul {{ 
            margin: 15px 0 15px 25px;
            list-style-type: disc;
        }}
        li {{ 
            margin: 8px 0;
            font-size: 1.05em;
        }}
        
        /* Special Elements */
        .quote {{ 
            background: #fff9e6;
            border-left: 5px solid #f59e0b;
            padding: 20px 25px;
            margin: 25px 0;
            font-style: italic;
            font-size: 1.1em;
            color: #744210;
        }}
        .quote footer {{ 
            text-align: right;
            margin-top: 15px;
            font-weight: bold;
            font-style: normal;
        }}
        
        .table-container {{
            margin: 25px 0;
            overflow-x: auto;
        }}
        .table-container h4 {{
            margin-bottom: 10px;
            color: #667eea;
        }}
        table {{ 
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
        }}
        th, td {{ 
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: left;
        }}
        th {{ 
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ 
            background: #f9f9f9;
        }}
        
        /* Footer */
        footer {{ 
            text-align: center;
            padding: 40px 20px;
            color: #7f8c8d;
            border-top: 2px solid #ecf0f1;
            margin-top: 60px;
            font-size: 0.95em;
        }}
        
        /* Print Styles */
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
            .course-section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{escape_html(course_state.title)}</h1>
            <p class="course-meta">{escape_html(course_state.description)}</p>
            <p class="course-meta">📚 {len(course_state.modules)} Módulos • 
               {sum(len(sm.sections) for m in course_state.modules for sm in m.submodules)} Secciones</p>
        </header>
"""
    
    # Render all modules
    section_counter = 0
    for module_idx, module in enumerate(course_state.modules, 1):
        html += f'<div class="module">'
        html += f'<h2 class="module-title">Módulo {module_idx}: {escape_html(module.title)}</h2>'
        
        for submodule_idx, submodule in enumerate(module.submodules, 1):
            html += f'<div class="submodule">'
            html += f'<h3 class="submodule-title">{module_idx}.{submodule_idx} {escape_html(submodule.title)}</h3>'
            
            for section in submodule.sections:
                section_counter += 1
                html += render_section(section, section_counter)
            
            html += '</div>'
        
        html += '</div>'
    
    html += """
        <footer>
            <p>📚 Curso generado por Course Generator Agent</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
