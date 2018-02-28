{%- extends 'basic.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}


{%- block header -%}
<!DOCTYPE html>
<html>
<head>
{%- block html_head -%}
<meta charset="utf-8" />
{% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
<title>{{nb_title}}</title>

{%- if "widgets" in nb.metadata -%}
<script src="https://unpkg.com/jupyter-js-widgets@2.0.*/dist/embed.js"></script>
{%- endif-%}

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>

{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}

{%- if resources.global_content_filter.no_prompt-%}
div#notebook-container{
  padding: 6ex 12ex 8ex 12ex;
}
{%- endif -%}

@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}

.container {
    width:99% !important;
}

.popover {
    max-width: 100%;
}

.hover {
    background: #cccccc !important;
}

</style>

<!-- Loading mathjax macro -->
{{ mathjax() }}
{%- endblock html_head -%}
</head>
{%- endblock header -%}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">
{{ super() }}
    </div>
  </div>

<script>
    $(document).ready(function(){
        // activate the bootstrap popovers
        $('[data-toggle="popover"]').popover();
        
        // make the level-0 index color on hover
        // based on https://codepen.io/chriscoyier/pen/wLGDz
        $("td, th").hover(function() {
            if ($(this).parent().has('th[rowspan]').length == 0) {
                $(this)
                    .parent()
                    .prevAll('tr:has(th[rowspan]):first')
                    .find('th[rowspan]')
                    .addClass("hover");
            } 
        }, function() { 
            $(this)
                .parent()
                .prevAll('tr:has(th[rowspan]):first')
                .find('th[rowspan]')
                .removeClass("hover");
        });
    });
</script>

</body>
{%- endblock body %}

{% block footer %}
{{ super() }}
</html>
{% endblock footer %}
