import functools
import io
import base64
import os
from pathlib import Path

import pandas as pd
import nbconvert
import nbformat.v4 as nbf
import PIL

from . import experiment
from . import svg

totals_row_label = (' ', 'Total reads')

def load_counts(base_dir, conditions=None, drop_outcomes=None):
    if drop_outcomes is None:
        drop_outcomes = []

    exps = experiment.get_all_experiments(base_dir, conditions)

    counts = {(exp.group, exp.name): exp.load_outcome_counts() for exp in exps}
    no_outcomes = [k for k, v in counts.items() if v is None]
    if no_outcomes:
        raise ValueError('Can\'t find outcome counts for {0}'.format(no_outcomes))

    df = pd.DataFrame(counts).fillna(0).drop(drop_outcomes)

    totals = df.sum(axis=0)
    totals_row = pd.DataFrame.from_dict({totals_row_label: totals}, orient='index')
    
    # Sort order for outcome is defined in the relevant layout module.
    layout_modules = {exp.layout_module for exp in exps}
    
    if len(layout_modules) > 1:
        raise ValueError('Can\'t make table for experiments with inconsistent layout modules.')
    
    layout_module = layout_modules.pop()
    
    df['_sort_order'] = df.index.map(layout_module.order)
    df = df.sort_values('_sort_order').drop('_sort_order', axis=1)
    
    df = pd.concat([totals_row, df]).astype(int)
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

link_without_modal_template = '''\
<a 
    data-toggle="popover" 
    data-trigger="hover"
    data-html="true"
    data-placement="auto"
    data-content="<img width={width} height={height} src={URI}>"
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
        
        contents = '<div> {0} </div> <div> {1} </div>'.format(lengths_img, reads_img)
        modal_div = modal_template.format(modal_id=modal_id, contents=contents, title=title)
        
        return modal_div, modal_id
        
def make_table(base_dir, conditions=None, drop_outcomes=None):
    df = load_counts(base_dir, conditions, drop_outcomes)
    totals = df.loc[totals_row_label]

    modal_maker = ModalMaker()

    def link_maker(val, col, row):
        if val == 0:
            html = ''
        else:
            outcome = row
            exp_group, exp_name = col

            exp = experiment.Experiment(base_dir, exp_group, exp_name)
            outcome_fns = exp.outcome_fns(outcome)
            
            fraction = val / float(totals[col])
            
            if row == totals_row_label:
                #modal_div, modal_id = modal_maker.make_length(exp)

                hover_image_fn = str(exp.fns['lengths_figure'])
                hover_URI, width, height = fn_to_URI(hover_image_fn)

                link = link_without_modal_template.format(text='{:,}'.format(val),
                                                          URI=hover_URI,
                                                          width=width,
                                                          height=height,
                                           )
                
                html = link + modal_div
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
                
                html = link + modal_div

        return html
    
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
        exp_group, exp_name = col
        exp = experiment.Experiment(base_dir, exp_group, exp_name)
        # Note: as of pandas 0.22, col needs to be in brackets here so that
        # apply is ultimately called on a df, not a series, to prevent
        # TypeError: _bar_left() got an unexpected keyword argument 'axis'
        styled = styled.bar(subset=pd.IndexSlice[:, [col]], color=exp.color)
        
    styled.set_table_styles(styles)
    
    return styled

def make_table_new(base_dir, conditions=None, drop_outcomes=None, include_images=True):
    df = load_counts(base_dir, conditions, drop_outcomes)
    totals = df.loc[totals_row_label]

    df = df.T
    
    modal_maker = ModalMaker()

    def link_maker(val, outcome, exp_group, exp_name, include_images):
        if val == 0:
            html = ''
        else:
            exp = experiment.Experiment(base_dir, exp_group, exp_name)
            outcome_fns = exp.outcome_fns(outcome)
            
            fraction = val / totals[(exp_group, exp_name)]
            
            if outcome == totals_row_label:
                text = '{:,}'.format(val)
                if include_images:
                    #modal_div, modal_id = modal_maker.make_length(exp)

                    hover_image_fn = str(exp.fns['lengths_figure'])
                    hover_URI, width, height = fn_to_URI(hover_image_fn)
                
                    link = link_without_modal_template.format(text=text,
                                                              URI=hover_URI,
                                                              width=width,
                                                              height=height,
                                                             )

                    html = link# + modal_div
                else:
                    html = text

            else:
                text = '{:.2%}'.format(fraction)
                if include_images:
                    modal_div, modal_id = modal_maker.make_outcome(exp, outcome)
                    hover_image_fn = str(outcome_fns['first_example'])
                    hover_URI, width, height = fn_to_URI(hover_image_fn)
                    
                    link = link_template.format(text=text,
                                                modal_id=modal_id,
                                                URI=hover_URI,
                                                width=width,
                                                height=height,
                                               )
                    html = link + modal_div
                else:
                    html = text

        return html
    
    def bind_link_maker(exp_group, exp_name):
        return {outcome: functools.partial(link_maker, outcome=outcome, exp_group=exp_group, exp_name=exp_name, include_images=include_images) for outcome in df}

    styled = df.style
    
    styles = [
        dict(selector="th", props=[("border", "1px solid black")]),
        dict(selector="tr:hover", props=[("background-color", "#cccccc")]),
    ]
    
    for exp_group, exp_name in df.index:
        sl = pd.IndexSlice[[(exp_group, exp_name)], :]
        styled = styled.format(bind_link_maker(exp_group, exp_name), subset=sl)
    
    styled = styled.set_properties(**{'border': '1px solid black'})
    for exp_group, exp_name in df.index:
        exp = experiment.Experiment(base_dir, exp_group, exp_name)
        # Note: as of pandas 0.22, col needs to be in brackets here so that
        # apply is ultimately called on a df, not a series, to prevent
        # TypeError: _bar_left() got an unexpected keyword argument 'axis'
        styled = styled.bar(subset=pd.IndexSlice[[(exp_group, exp_name)], :], axis=1, color=exp.color)
        
    styled.set_table_styles(styles)
    
    return styled

def generate_html(base_dir, fn, conditions=None, drop_outcomes=None):
    nb = nbf.new_notebook()

#    cell_contents = '''\
#import knockin.table
#
#conditions = {conditions}
#drop_outcomes = {drop_outcomes}
#knockin.table.make_table_new('{base_dir}', conditions, drop_outcomes, include_images=False)
#'''.format(conditions=conditions, base_dir=base_dir, drop_outcomes=drop_outcomes)
    
    cell_contents = '''\
import knockin.table

conditions = {conditions}
drop_outcomes = {drop_outcomes}
knockin.table.make_table_new('{base_dir}', conditions, drop_outcomes, include_images=True)
'''.format(conditions=conditions, base_dir=base_dir, drop_outcomes=drop_outcomes)

    nb['cells'] = [nbf.new_code_cell(cell_contents)]

    nb['metadata'] = {'title': fn}

    exporter = nbconvert.HTMLExporter(exclude_input=True, exclude_output_prompt=True)
    template_path = Path(os.path.realpath(__file__)).parent / 'modal_template.tpl'
    exporter.template_file = str(template_path)

    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open(fn, 'w') as fh:
        fh.write(body)