import functools
import io
import base64

import pandas as pd
import bokeh.palettes
import IPython.display
import nbconvert
import nbformat.v4 as nbf
import PIL

import Sequencing.ipython

import pacbio_experiment
import svg

totals_row_label = (' ', 'Total reads')

def load_counts(conditions=None):
    exps = pacbio_experiment.get_all_experiments(conditions)

    counts = {exp.name: exp.load_outcome_counts() for exp in exps}
    df = pd.DataFrame(counts).fillna(0)

    totals = df.sum(axis=0)
    df.loc[totals_row_label, :] = totals

    sort_order = {}
    for exp in exps:
        sort_order.update(exp.load_outcome_sort_order())

    sort_order[totals_row_label] = (0, 0)

    df['_sort_order'] = df.index.map(lambda k: sort_order[k])
    df = df.sort_values('_sort_order').drop('_sort_order', axis=1).astype(int)
    df.index.names = (None, None)

    return df

def png_bytes_to_URI(bs):
    encoded = base64.b64encode(bs).decode('UTF-8')
    URI = "'data:image/png;base64,{0}'".format(encoded)
    return URI

def fn_to_URI(fn):
    im = PIL.Image.open(fn)
    im.load()
    return Image_to_png_URI(im)

def Image_to_png_URI(im):
    with io.BytesIO() as buf:
        im.save(buf, format='png')
        png_bytes = buf.getvalue()
    URI = png_bytes_to_URI(png_bytes)
    return URI, im.width, im.height

link_template = '''\
<a 
    data-toggle="popover" 
    data-trigger="hover"
    data-html="true"
    data-placement="auto"
    data-content="<img width={width} height={height} src={URI}>"
    onclick="$('#{modal_id}').appendTo('body').modal()"
    style="text-decoration:none; color:black"
>
    {text}
</a>
'''
modal_template = '''\
<div class="modal fade" tabindex="-1" id="{modal_id}" role="dialog">
    <div class="modal-dialog" style="width:95%; margin:auto;">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">{title}</h2>
            </div>
            <div class="modal-body" style="height:5000px">
                <div class="text-center">
                    {contents}
                </div>
            </div>
        </div>
    </div>
</div>
'''

class ModalMaker(object):
    def __init__(self):
        self.current_number = 0

    def get_next_id(self):
        next_id = 'modal_{:06d}'.format(self.current_number)
        self.current_number += 1
        return next_id
        
    def make_length(self, exp):
        modal_id = self.get_next_id()
        
        svg_text = svg.length_plot_with_popovers(exp, container_selector='#{0}'.format(modal_id))
        modal_div = modal_template.format(modal_id=modal_id, contents=svg_text, title=exp.name)
        
        return modal_div, modal_id

    def make_outcome(self, exp, outcome):
        modal_id = self.get_next_id()
        outcome_fns = exp.outcome_fns(outcome)

        outcome_string = '_'.join(outcome)
        title = '{0}: {1}'.format(exp.name, outcome_string)
        
        URI, width, height = fn_to_URI(outcome_fns['lengths_figure'])
        lengths_img = '<img src={0} width={1}, height={2}>'.format(URI, width, height)
        
        URI, width, height = fn_to_URI(outcome_fns['combined_figure'])
        reads_img = '<img src={0} width={1}, height={2}>'.format(URI, width, height)
        
        contents = lengths_img + reads_img
        modal_div = modal_template.format(modal_id=modal_id, contents=contents, title=title)
        
        return modal_div, modal_id
        
def make_table(conditions=None):
    df = load_counts(conditions)
    totals = df.loc[totals_row_label]

    modal_maker = ModalMaker()

    def link_maker(val, col, row):
        if val == 0:
            return ''
        else:
            outcome = row
            exp_name = col

            exp = pacbio_experiment.PacbioExperiment(exp_name)
            outcome_fns = exp.outcome_fns(outcome)
            
            fraction = val / float(totals[col])
            
            if row == totals_row_label:
                modal_div, modal_id = modal_maker.make_length(exp)

                hover_image_fn = str(exp.fns['lengths_figure'])
                hover_URI, width, height = fn_to_URI(hover_image_fn)

                link = link_template.format(text='{:,}'.format(val),
                                            modal_id=modal_id,
                                            URI=hover_URI,
                                            width=width,
                                            height=height,
                                           )
                
                return link + modal_div
            else:
                modal_div, modal_id = modal_maker.make_outcome(exp, outcome)
                
                hover_image_fn = str(outcome_fns['first_example'])
                hover_URI, width, height = fn_to_URI(hover_image_fn)

                link = link_template.format(text='{:.2%}'.format(fraction),
                                            modal_id=modal_id,
                                            URI=hover_URI,
                                            width=width,
                                            height=height,
                                           )
                
                return link + modal_div
    
    def bind_link_maker(row):
        return {col: functools.partial(link_maker, col=col, row=row) for col in df}

    styled = df.style
    
    styles = [
        dict(selector="th", props=[("border", "1px solid black")]),
        dict(selector="tr:hover", props=[("background-color", "#cccccc")]),
    ]
    
    for row in df.index:
        sl = pd.IndexSlice[[row], :]
        styled = styled.format(bind_link_maker(row), subset=sl)
        
    styled = styled.set_properties(**{'border': '1px solid black'})
    for col in df:
        exp = pacbio_experiment.PacbioExperiment(col)
        styled = styled.bar(subset=pd.IndexSlice[:, col], color=exp.color)
        
    styled.set_table_styles(styles)
    
    return styled

def generate_html(title='table', conditions=None):
    nb = nbf.new_notebook()

    cell_contents = '''\
import table

conditions = {0}
table.make_table(conditions)
'''.format(conditions)

    nb['cells'] = [nbf.new_code_cell(cell_contents)]

    nb['metadata'] = {'title': title}

    exporter = nbconvert.HTMLExporter()
    exporter.template_file = 'modal_template.tpl'

    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600, kernel_name='python3.6')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open('table_{0}.html'.format(title), 'w') as fh:
        fh.write(body)
