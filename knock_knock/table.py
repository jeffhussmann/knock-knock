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

totals_all_row_label = (' ', 'Total reads')

totals_relevant_row_label = (' ', 'Total relevant reads')

def load_counts(base_dir, conditions=None, exclude_malformed=False, exclude_empty=True):
    exps = experiment.get_all_experiments(base_dir, conditions)

    counts = {}
    no_outcomes = []

    for exp in exps:
        exp_counts = exp.load_outcome_counts()
        if exp_counts is None:
            no_outcomes.append((exp.group, exp.name))
        else:
            counts[exp.group, exp.name] = exp_counts

    if no_outcomes:
        no_outcomes_string = '\n'.join(f'\t{group}: {name}' for group, name in no_outcomes)
        print(f'Warning: can\'t find outcome counts for\n{no_outcomes_string}') 

    df = pd.DataFrame(counts).fillna(0)

    # Sort order for outcomes is defined in the relevant layout module.
    layout_modules = {exp.layout_module for exp in exps}
    
    if len(layout_modules) > 1:
        raise ValueError('Can\'t make table for experiments with inconsistent layout modules.')
    
    layout_module = layout_modules.pop()
    
    df = df.reindex(layout_module.full_index, fill_value=0)

    if exclude_malformed:
        df = df.drop('malformed layout', axis='index', level=0, errors='ignore')
        totals_row_label = totals_relevant_row_label
    else:
        totals_row_label = totals_all_row_label

    if exclude_empty:
        empty_rows = df.index[df.sum(axis=1) == 0].values
        df = df.drop(empty_rows, axis='index', errors='ignore')
    
    totals = df.sum(axis=0)
    totals_row = pd.DataFrame.from_dict({totals_row_label: totals}, orient='index')
    
    df = pd.concat([totals_row, df]).astype(int)
    df.index.names = (None, None)

    # Sort by group and sample name.
    df = df.sort_index(axis=1)

    return df

def calculate_performance_metrics(base_dir, conditions=None):
    full_counts = load_counts(base_dir, conditions=conditions)
    counts = full_counts.drop([totals_all_row_label, totals_relevant_row_label], axis='index', errors='ignore').sum(level=0)

    not_real_cell_categories = [
        'malformed layout',
    ]

    real_cells = counts.drop(not_real_cell_categories)

    all_edit_categories = [cat for cat in real_cells.index if cat != 'WT']

    all_integration_categories = [
        'HDR',
        'truncated misintegration',
        'blunt misintegration',
        'complex misintegration',
        'concatenated misintegration',
    ]

    # reindex to handle possibly missing keys
    HDR_counts = real_cells.reindex(['HDR'], fill_value=0).loc['HDR']
    edit_counts = real_cells.reindex(all_edit_categories, fill_value=0)
    integration_counts = real_cells.reindex(all_integration_categories, fill_value=0)

    performance_metrics = pd.DataFrame({
        'HDR_rate': HDR_counts / real_cells.sum(),
        'specificity_edits': HDR_counts / edit_counts.sum(),
        'specificity_integrations': HDR_counts / integration_counts.sum(),
    })

    return performance_metrics

def png_bytes_to_URI(png_bytes):
    encoded = base64.b64encode(png_bytes).decode('UTF-8')
    URI = f"'data:image/png;base64,{encoded}'"
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

def fig_to_png_URI(fig):
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png', bbox_inches='tight')
        png_bytes = buffer.getvalue()
        im = PIL.Image.open(buffer)
        im.load()
       
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
    href="{URL}"
    target="_blank"
>
    {text}
</a>
'''

modal_template = '''\
<div class="modal" tabindex="-1" id="{modal_id}" role="dialog">
    <div class="modal-dialog" style="width:90%; margin:auto">
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
        
    def make_length(self, exp, outcome=None, inline_images=True):
        modal_id = self.get_next_id()
        
        svg_text = svg.length_plot_with_popovers(exp, outcome=outcome, container_selector=f'#{modal_id}', inline_images=inline_images)
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
        
def make_table(base_dir,
               conditions=None,
               inline_images=False,
               show_details=True,
              ):
    df = load_counts(base_dir, conditions=conditions, exclude_malformed=(not show_details))
    if show_details:
        totals_row_label = totals_all_row_label
    else:
        totals_row_label = totals_relevant_row_label
    totals_row_label_collapsed = totals_row_label[1]

    totals = df.loc[totals_row_label]

    df = df.T

    # Hack to give the html the information it needs to build links to diagram htmls
    df.index = pd.MultiIndex.from_tuples([(g, f'{g}/{n}') for g, n in df.index.values])
    
    if not show_details:
        level_0 = list(df.columns.levels[0])
        level_0[0] = totals_row_label[1]
        df.columns = df.columns.set_levels(level_0, level=0)

        df = df.sum(axis=1, level=0)

    exps = experiment.get_all_experiments(base_dir, conditions=conditions, as_dictionary=True)

    modal_maker = ModalMaker()

    def link_maker(val, outcome, exp_group, exp_name):
        if val == 0:
            html = ''
        else:
            exp = exps[exp_group, exp_name]
            
            fraction = val / totals[(exp_group, exp_name)]

            if outcome == totals_row_label or outcome == totals_row_label_collapsed:
                text = '{:,}'.format(val)
                if False:
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
                
                #modal_div, modal_id = modal_maker.make_outcome(exp, outcome)

                hover_image_fn = exp.outcome_fns(outcome)['first_example']
                click_html_fn = exp.outcome_fns(outcome)['diagrams_html']
                
                if inline_images:
                    hover_URI, width, height = fn_to_URI(hover_image_fn)
                else:
                    relative_path = hover_image_fn.relative_to(exp.base_dir / 'results')
                    hover_URI = str(relative_path)
                    if hover_image_fn.exists():
                        with PIL.Image.open(hover_image_fn) as im:
                            width, height = im.size
                            width = width * 0.75
                            height = height * 0.75
                    else:
                        width, height = 100, 100

                relative_path = click_html_fn.relative_to(exp.base_dir / 'results')
                link = link_without_modal_template.format(text=text,
                                                          #modal_id=modal_id,
                                                          URI=hover_URI,
                                                          width=width,
                                                          height=height,
                                                          URL=str(relative_path),
                                                         )
                html = link# + modal_div

        return html
    
    def bind_link_maker(exp_group, exp_name):
        bound = {}
        for outcome in df:
            bound[outcome] = functools.partial(link_maker, outcome=outcome, exp_group=exp_group, exp_name=exp_name)

        return bound

    styled = df.style

    styles = [
        dict(selector="th", props=[("border", "1px solid black")]),
        dict(selector="tr:hover", props=[("background-color", "#cccccc")]),
    ]
    
    for exp_group, group_and_name in df.index:
        _, exp_name = group_and_name.split('/')
        sl = pd.IndexSlice[[(exp_group, group_and_name)], :]
        styled = styled.format(bind_link_maker(exp_group, exp_name), subset=sl)
    
    styled = styled.set_properties(**{'border': '1px solid black'})
    for exp_group, group_and_name in df.index:
        _, exp_name = group_and_name.split('/')
        exp = experiment.Experiment(base_dir, exp_group, exp_name)
        # Note: as of pandas 0.22, col needs to be in brackets here so that
        # apply is ultimately called on a df, not a series, to prevent
        # TypeError: _bar_left() got an unexpected keyword argument 'axis'
        styled = styled.bar(subset=pd.IndexSlice[[(exp_group, group_and_name)], :], axis=1, color=exp.color)
        
    styled.set_table_styles(styles)
    
    return styled

def generate_html(base_dir, fn, conditions=None, show_details=True):
    nb = nbf.new_notebook()

    cell_contents = f'''\
import knock_knock.table

conditions = {conditions}
knock_knock.table.make_table('{base_dir}', conditions, show_details={show_details})
'''
    
    nb['cells'] = [nbf.new_code_cell(cell_contents)]

    nb['metadata'] = {'title': str(fn.name)}

    exporter = nbconvert.HTMLExporter(exclude_input=True, exclude_output_prompt=True)
    template_path = Path(os.path.realpath(__file__)).parent / 'modal_template.tpl'
    exporter.template_file = str(template_path)

    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open(fn, 'w') as fh:
        fh.write(body)
