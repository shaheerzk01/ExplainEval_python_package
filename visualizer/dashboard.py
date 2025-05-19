from jinja2 import Template


def generate_html_report(context, output_file):
    html_template = """
    <html>
    <head><title>Model Evaluation Report</title></head>
    <body>
        <h1>Task: {{ task }}</h1>
        {% for k, v in metrics.items() %}
        <h2>{{ k }}</h2>
        <pre>{{ v }}</pre>
        {% endfor %}
    </body>
    </html>
    """
    template = Template(html_template)
    with open(output_file, "w") as f:
        f.write(template.render(**context))
