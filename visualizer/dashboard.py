import os
from jinja2 import Template

def generate_html_report(context, output_file):
    # Load HTML template from the installed package location
    template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')

    with open(template_path, 'r') as f:
        html_template = f.read()

    template = Template(html_template)
    rendered = template.render(**context)

    with open(output_file, 'w') as f:
        f.write(rendered)
