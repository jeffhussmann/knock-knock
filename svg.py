import io
import matplotlib.pyplot as plt

from . import  table

before_path_template = '''\
<a
    data-toggle="popover" 
    data-container="{container_selector}"
    data-trigger="hover click"
    data-html="true"
    data-placement="bottom"
    data-content="&lt;img width={width} height={height} src={URI}&gt;"
>'''

after_path = '''\
"
style="fill:#000000;stroke:#000000;stroke-linejoin:miter;"
fill-opacity="0.01"
stroke-opacity="0.05"
/>
</a>'''

before_svg = '''\
<head>

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

<style>
.popover {
    max-width: 100%;
}
</style>
</head>
<body>
<div style="height:5000px;">'''

after_svg = '''\
</div>
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<script>
    $(document).ready(function(){
        $('[data-toggle="popover"]').popover();
        $('[data-toggle="popover"]').on('show.bs.popover', function() {
          $("path", this).attr('stroke-opacity', '0.5');
        });
        $('[data-toggle="popover"]').on('hide.bs.popover', function() {
          $("path", this).attr('stroke-opacity', '0.05');
        });
    });
</script>
</body>'''

def length_plot_with_popovers(exp, standalone=False, container_selector='body'):
    fig = exp.length_distribution_figure(show_ranges=True)

    with io.StringIO() as buf:
        fig.savefig(buf, format='svg', bbox_inches='tight')
        # skip matplotlib header lines
        lines = buf.getvalue().splitlines(keepends=True)[4:]

    plt.close(fig)

    def extract_length_range(line):
        range_string = line[line.index('length_range_') + len('length_range_'):][:2 * 5 + 1]
        start, end = map(int, range_string.split('_'))
        return start, end

    output_lines = []

    if standalone:
        output_lines.append(before_svg)

    line_i = 0
    while line_i < len(lines):
        line = lines[line_i]
        output_lines.append(line)
        if '<g id="length_range_' in line:
            start, end = extract_length_range(line)
            im = exp.span_to_Image(start, end, num_examples=3)
            URI, width, height = table.Image_to_png_URI(im)

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
