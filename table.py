import functools
import base64

import pandas as pd
import bokeh.palettes
import IPython.display
import nbconvert
import nbformat.v4 as nbf

import Sequencing.ipython

import pacbio_experiment

totals_row_with_priority = ('000:  ', '000: Total reads')
totals_row = tuple([pacbio_experiment.priority_string_to_description(s) for s in totals_row_with_priority])

def load_counts(conditions=None):
    exps = pacbio_experiment.get_all_experiments(conditions)

    counts = {exp.name: exp.load_outcome_counts() for exp in exps}
    df = pd.DataFrame(counts).fillna(0)

    totals = df.sum(axis=0)
    df.loc[totals_row_with_priority, :] = totals

    df = df.sort_index().astype(int)

    new_levels = [
        [pacbio_experiment.priority_string_to_description(s) for s in l]
        for l in df.index.levels
    ]

    new_index = df.index.set_levels(new_levels)

    outcome_lookup = {new: old for new, old in zip(new_index, df.index)}

    df.index = new_index
    df.index.names = (None, None)

    return df, outcome_lookup

def fn_to_URI(fn):
    contents = open(fn, 'rb').read()
    encoded_data = base64.b64encode(contents).decode('UTF-8')
    return "'data:image/png;base64,{0}'".format(encoded_data)

outcome_link_template = '''\
<a 
    data-toggle="popover" 
    data-trigger="hover"
    data-html="true"
    data-placement="auto"
    data-content="<img width=1000 src={URI}>"
    onclick="$('#{modal_id}').appendTo('body').modal()"
    style="text-decoration:none; color:black"
>
    {fraction:.2%}
</a>
'''

dataset_link_template = '''\
<a
    onclick="$('#{modal_id}').appendTo('body').modal()"
    style="text-decoration:none; color:black"
>
    {count:,}
</a>
'''

modal_template = '''\
<div class="modal fade" tabindex="-1" id="{modal_id}" role="dialog">
    <div class="modal-dialog" style="width:1250px; margin:auto">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">{title}</h4>
            </div>
            <div class="modal-body">
                <div class="text-center">
                    <img src={URI}>
                </div>
            </div>
        </div>
    </div>
</div>
'''

class ModalMaker(object):
    def __init__(self):
        self.current_number = 0
        
    def make(self, image_fn, title):
        modal_id = 'modal_{:06d}'.format(self.current_number)
        self.current_number += 1
        
        URI = fn_to_URI(image_fn)
        modal_div = modal_template.format(modal_id=modal_id, URI=URI, title=title)
        
        return modal_div, modal_id
        
def make_table(conditions=None):
    df, outcome_lookup = load_counts(conditions)
    totals = df.loc[totals_row]

    modal_maker = ModalMaker()

    def link_maker(val, col, row):
        if val == 0:
            return ''
        else:
            outcome = outcome_lookup[row]
            exp = pacbio_experiment.PacbioExperiment(col)
            outcome_fns = exp.outcome_fns(outcome)
            
            fraction = val / float(totals[col])
            
            if row == totals_row:
                title = col
                modal_image_fn = str(exp.fns['lengths_figure'])
                modal_div, modal_id = modal_maker.make(modal_image_fn, title)
                link = dataset_link_template.format(modal_id=modal_id, count=val)
                
                return link + modal_div
            else:
                fields = [pacbio_experiment.priority_string_to_description(s) for s in outcome]
                outcome_string = '_'.join(fields)
                title = '{0}: {1}'.format(col, outcome_string)
                modal_image_fn = str(outcome_fns['combined_figure'])
                modal_div, modal_id = modal_maker.make(modal_image_fn, title)
                
                hover_image_fn = str(outcome_fns['first_example'])
                hover_URI = fn_to_URI(hover_image_fn)
                link = outcome_link_template.format(fraction=fraction, modal_id=modal_id, URI=hover_URI)
                
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

    ep = nbconvert.preprocessors.ExecutePreprocessor(kernel_name='python3.6')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open('table_{0}.html'.format(title), 'w') as fh:
        fh.write(body)
