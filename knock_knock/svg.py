import re
import html
import io
import xml.etree.ElementTree as ET

import PIL
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt

from . import table, layout

before_svg_template = '''\
<head>
<title>{title}</title>
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<style>
.popover {{
    max-width: 100%;
}}
</style>

</head>
<body>
<div style="height:5000px;text-align:center">\
'''.format

after_svg = '''\
</div>
<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>

<script>
    $(document).ready(function(){
        $('[id^=length_range]').hide();

        $('[data-toggle="popover"]').popover();

        $('[data-toggle="popover"]').on('show.bs.popover', function() {
          $("path", this).attr('stroke-opacity', '0.5');
        });

        $('[data-toggle="popover"]').on('hide.bs.popover', function() {
          $("path", this).attr('stroke-opacity', '0.0');
        });
    });

    document.onkeydown = function(evt) {
        evt = evt || window.event;
        if (evt.keyCode == 27) {
            $('[data-toggle="popover"').popover('hide')
            $('[id^=length_range]').hide();

            $('[id^=outcome_category]').find('path').css('stroke-opacity', '1');

            $('[id^=line_highlighted]').find('path').css('stroke-width', '1');
            $('[id^=line_nonhighlighted]').find('path').css('stroke-width', '1');

            $('[id^=line_highlighted]').find('path').css('stroke-opacity', '1');
            $('[id^=line_nonhighlighted_6]').find('path').css('stroke-opacity', '0.6');
            $('[id^=line_nonhighlighted_3]').find('path').css('stroke-opacity', '0.3');
            $('[id^=line_nonhighlighted_05]').find('path').css('stroke-opacity', '0.05');
        }
    };
</script>

</body>\
'''
toggle_function_template = '''\
javascript:(function(){{
    $all_ranges = $('[id^=length_range]');
    $clicked_ranges = $('[id^=length_range_{outcome_string}]');

    $all_ranges.hide();
    $clicked_ranges.show();

    $all_lines = $('[id^=line_highlighted], [id^=line_nonhighlighted]').find('path');

    $all_lines.css({{
        'stroke-width': '1',
        'stroke-opacity': '0.3'
    }});

    /* For lines with initial opacity below 0.3, keep them below 0.3. */
    $('[id^=line_nonhighlighted_05]').find('path').css('stroke-opacity', '0.05');

    $clicked_lines = $('[id^=line_highlighted_{outcome_string}]').find('path');
    $clicked_lines.css({{
        'stroke-width': '2',
        'stroke-opacity': '1'
    }});

    $('[id^=outcome_category]').find('path').css('stroke-opacity', '0.5');
    $('[id^=outcome_{outcome_string}]').find('path').css('stroke-opacity', '1');

}} )();'''.format

toggle_zoom_function_template = '''\
javascript:(function(){{
    /* axes are id'd starting with axes_1 */

    $next_ax = $('#axes_' + ({panel_i} + 1 + 1));
    $next_lines = $('[id^=zoom_dotted_line_' + {panel_i} + ']');

    if ($next_ax.css('visibility') == 'hidden') {{
        $next_ax.css('visibility', 'visible');
        $next_lines.css('visibility', 'visible');
    }} else {{
        $next_ax.css('visibility', 'hidden');
        $next_lines.css('visibility', 'hidden');
    }};

    for (i = {panel_i} + 1 + 1; i < {num_panels}; i++) {{
        $ax = $('#axes_' + (i + 1));
        $lines = $('[id^=zoom_dotted_line_' + (i - 1) + ']');
        $ax.css('visibility', 'hidden');
        $lines.css('visibility', 'hidden');
    }};
}} )();'''.format

def decorate_outcome_browser(exp):
    fig = exp.plot_outcome_stratified_lengths()
    num_panels = len(fig.axes)

    # Write matplotlib svg to a string.
    with io.StringIO() as write_fh:
        fig.savefig(write_fh, format='svg', bbox_inches='tight')
        contents = write_fh.getvalue()

    plt.close(fig)    

    # Read matplotib svg into an ElementTree.
    with io.StringIO(contents) as read_fh:
        d = ET.parse(read_fh)

    # ElementTree's handling of namespaces is confusing.
    # Learned this approach from http://blog.tomhennigan.co.uk/post/46945128556/elementtree-and-xmlns
    # but don't really understand it.

    default_namespace = 'http://www.w3.org/2000/svg'
    xlink_namespace = 'http://www.w3.org/1999/xlink'
    namespaces = {
        '': default_namespace,
        'xlink': xlink_namespace,
    }

    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

    elements_to_decorate = {
        'length_range': [],
        'outcome': [],
        'zoom_toggle_top': [],
    }

    for element in d.iter(f'{{{default_namespace}}}g'):
        if 'id' in element.attrib:
            match = re.match('axes_(?P<panel_i_plus_one>\d+)', element.attrib['id'])
            if match:
                panel_i = int(match.group('panel_i_plus_one')) - 1
                if panel_i > 0:
                    element.attrib['style'] = 'visibility: hidden'
            
            match = re.match('zoom_dotted_line', element.attrib['id'])
            if match:
                element.attrib['style'] = 'visibility: hidden'

            for prefix in elements_to_decorate:
                if element.attrib['id'].startswith(prefix):
                    elements_to_decorate[prefix].append(element)

    def decorate_with_popover(group, container_selector='body', inline_images=False):
        path = group.find(f'{{{default_namespace}}}path', namespaces)
        path.attrib.update({
            'style': 'fill:#000000;stroke:#000000;stroke-linejoin:miter;',
            'fill-opacity': '0.00',
            'stroke-opacity': '0.00',
        })
        
        match = re.match('length_range_(?P<outcome>.+)_(?P<start>\d+)_(?P<end>\d+)', group.attrib['id'])
        
        sanitized_outcome, start, end = match.groups()

        if sanitized_outcome == 'all':
            fns = exp.fns
        else:
            outcome = layout.sanitized_string_to_outcome(sanitized_outcome)
            fns = exp.outcome_fns(outcome)

        fn = fns['length_range_figure'](start, end)

        if fn.exists() or not inline_images:
            if inline_images:
                URI, width, height = table.fn_to_URI(fn)
            else:
                relative_path = fn.relative_to(exp.fns['dir'])
                URI = str(relative_path)
                if fn.exists():
                    with PIL.Image.open(fn) as im:
                        width, height = im.size
                else:
                    width, height = 100, 100

            width = width * 0.75
            height = height * 0.75

            attrib = {
                'data-toggle': 'popover',
                'data-container': container_selector,
                'data-trigger': 'hover click',
                'data-html': 'true',
                'data-placement': 'bottom',
                'data-content': f'<img width={width} height={height} src={URI}>',
            }

            decorator = ET.Element(f'{{{default_namespace}}}a', attrib=attrib)
            decorator.append(path)
            group.remove(path)
            group.append(decorator)

    def decorate_with_toggle(group):
        path = group.find(f'{{{default_namespace}}}path', namespaces)
        
        match = re.match('outcome_(?P<outcome>.+)', group.attrib['id'])
        outcome_string = match.group('outcome')
        
        attrib = {
            f'{{{xlink_namespace}}}href': toggle_function_template(outcome_string=outcome_string),
        }
        
        decorator = ET.Element(f'{{{default_namespace}}}a', attrib=attrib)
        decorator.append(path)
        group.remove(path)
        group.append(decorator)
    
    def decorate_with_zoom_toggle(group):
        path = group.find(f'{{{default_namespace}}}path', namespaces)
        
        match = re.match('zoom_toggle_top_(?P<panel_i>.+)', group.attrib['id'])
        panel_i = match.group('panel_i')
        
        attrib = {
            f'{{{xlink_namespace}}}href': toggle_zoom_function_template(panel_i=panel_i, num_panels=num_panels),
        }
        
        decorator = ET.Element(f'{{{default_namespace}}}a', attrib=attrib)
        decorator.append(path)
        group.remove(path)
        group.append(decorator)

    for group in elements_to_decorate['length_range']:
        decorate_with_popover(group)
        
    for group in elements_to_decorate['outcome']:
        decorate_with_toggle(group)
    
    for group in elements_to_decorate['zoom_toggle_top']:
        decorate_with_zoom_toggle(group)

    with exp.fns['outcome_browser'].open('w') as fh:
        fh.write(before_svg_template(title=exp.name))
        d.write(fh, encoding='unicode')
        fh.write(after_svg)

def length_plot_with_popovers(exp, outcome=None, standalone=False, container_selector='body', x_lims=None, y_lims=None, inline_images=True):
    fig = exp.length_distribution_figure(outcome=outcome, show_ranges=True, x_lims=x_lims, y_lims=y_lims)

    with io.StringIO() as buf:
        fig.savefig(buf, format='svg', bbox_inches='tight')
        # skip matplotlib header lines
        lines = buf.getvalue().splitlines(keepends=True)[4:]

    plt.close(fig)

    length_range_pattern = re.compile('<g id="length_range_(?P<outcome>.+)_(?P<start>\d+)_(?P<end>\d+)"')
    output_lines = []

    if outcome is None:
        title = exp.name
    else:
        category, subcategory = outcome
        title = f'{category}: {subcategory}'
    if standalone:
        output_lines.append(before_svg.format(title=title))

    if outcome is None:
        fns = exp.fns
    else:
        fns = exp.outcome_fns(outcome)

    line_i = 0
    while line_i < len(lines):
        line = lines[line_i]
        output_lines.append(line)

        match = re.search(length_range_pattern, line)
        if match:
            outcome, start, end = match.groups()
            fn = fns['length_range_figures'] / f'{start}_{end}.png'
            if fn.exists():
                if inline_images:
                    URI, width, height = table.fn_to_URI(fn)
                else:
                    relative_path = fn.relative_to(fns['dir'])
                    URI = str(relative_path)
                    with PIL.Image.open(fn) as im:
                        width, height = im.size

                before_path = before_path_template.format(URI=URI,
                                                          width=width,
                                                          height=height,
                                                          container_selector=container_selector,
                                                         )
                output_lines.append(before_path)
                line_i += 1
                line = lines[line_i]
                while not line.endswith('/>\n'):
                    output_lines.append(line)
                    line_i += 1
                    try:
                        line = lines[line_i]
                    except IndexError:
                        print(line_i, len(lines))
                        print(lines[-1])
                        raise

                output_lines.append(after_path)
        line_i += 1

    if standalone:
        output_lines.append(after_svg)

    return '\n'.join(output_lines)
