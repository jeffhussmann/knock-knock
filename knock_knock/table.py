import base64
import functools
import io
import logging
import os
import zipfile

from collections import defaultdict
from pathlib import Path

import pandas as pd
import nbconvert
import nbformat.v4 as nbf
import PIL
import tqdm

import knock_knock.arrayed_experiment_group
import knock_knock.experiment
import knock_knock.svg

totals_all_row_label = (' ', 'Total reads')
totals_relevant_row_label = (' ', 'Total relevant reads')

def load_counts(base_dir,
                conditions=None,
                exclude_malformed=False,
                exclude_empty=True,
                sort_samples=True,
                groups_to_exclude=None,
                arrayed=False,
               ):

    if groups_to_exclude is None:
        groups_to_exclude = set()

    if arrayed:
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir, conditions=conditions) 

    else:
        exps = knock_knock.experiment.get_all_experiments(base_dir, conditions, groups_to_exclude=groups_to_exclude)

    counts = {}
    no_outcomes = []

    for name_tuple, exp in exps.items():
        if exp.category_counts is None:
            no_outcomes.append(name_tuple)
        else:
            counts[name_tuple] = exp.category_counts

    if no_outcomes:
        no_outcomes_string = '\n'.join(f'\t{": ".join(name_tuple)}' for name_tuple in no_outcomes)
        logging.warning(f'Warning: can\'t find outcome counts for\n{no_outcomes_string}') 

    df = pd.DataFrame(counts).fillna(0)

    # Sort order for outcomes is defined in the relevant layout module.
    full_indexes = {tuple(exp.categorizer.full_index()) for exp in exps.values()}
    
    if len(full_indexes) > 1:
        print(full_indexes)
        raise ValueError('Can\'t make table for experiments with inconsistent layout modules.')
    
    full_index = full_indexes.pop()
    
    df = df.reindex(full_index, fill_value=0)

    if exclude_malformed:
        df = df.drop(['malformed layout', 'nonspecific amplification', 'bad sequence'], axis='index', level=0, errors='ignore')
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

    if sort_samples:
        # Sort by group and sample name.
        df = df.sort_index(axis=1)

    return df

def calculate_performance_metrics(base_dir, conditions=None, arrayed=False):
    full_counts = load_counts(base_dir, conditions=conditions, arrayed=arrayed)
    counts = full_counts.drop([totals_all_row_label, totals_relevant_row_label], axis='index', errors='ignore').groupby(level=0).sum()

    not_real_cell_categories = [
        'malformed layout',
    ]

    real_cells = counts.drop(not_real_cell_categories, errors='ignore')

    all_edit_categories = [cat for cat in real_cells.index if cat != 'WT']

    all_integration_categories = [
        'HDR',
        'blunt misintegration',
        'complex misintegration',
        'concatenated misintegration',
        'incomplete HDR',
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
    id="{id}"
    data-toggle="popover" 
    data-trigger="hover"
    data-html="true"
    data-placement="auto"
    data-content="<img width={width} height={height} src={URI}>"
    onclick="$('#{modal_id} .modal-body').html('<iframe width=&quot;100%&quot; height=&quot;100%&quot; frameborder=&quot;0&quot; scrolling=&quot;no&quot; allowtransparency=&quot;true&quot; src=&quot;{iframe_URL}&quot;></iframe>'); $('#{modal_id}').appendTo('body').modal();"
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

bare_link_template = '''\
<a 
    style="text-decoration:none; color:black"
>
    {text}
</a>
'''

modal_template = '''\
<div class="modal" tabindex="-1" id="{modal_id}" role="dialog">
    <div class="modal-dialog" style="width:90%; margin:auto">
        <div class="modal-content">
            <div class="modal-body" style="height:15000px">
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
        
        svg_text = knock_knock.svg.length_plot_with_popovers(exp, outcome=outcome, container_selector=f'#{modal_id}', inline_images=inline_images)
        modal_div = modal_template.format(modal_id=modal_id, contents=svg_text, title=exp.name)
        
        return modal_div, modal_id

    def make_outcome(self):
        modal_id = self.get_next_id()
        modal_div = modal_template.format(modal_id=modal_id)
        
        return modal_div, modal_id
        
def make_table(base_dir,
               conditions=None,
               include_images=True,
               inline_images=False,
               show_details=True,
               sort_samples=True,
               arrayed=False,
               vmax_multiple=1,
              ):

    df = load_counts(base_dir,
                     conditions=conditions,
                     exclude_malformed=(not show_details),
                     sort_samples=sort_samples,
                     arrayed=arrayed,
                    )

    if show_details:
        totals_row_label = totals_all_row_label
    else:
        totals_row_label = totals_relevant_row_label
    totals_row_label_collapsed = totals_row_label[1]

    totals = df.loc[totals_row_label]

    df = df.T

    # Hack to give the html the information it needs to build links to diagram htmls
    if arrayed:
        df.index = pd.MultiIndex.from_tuples([(b, f'{b}/{g}/{s}') for b, g, s in df.index.values])
    else:
        df.index = pd.MultiIndex.from_tuples([(g, f'{g}/{n}') for g, n in df.index.values])
    
    if not show_details:
        level_0 = list(df.columns.levels[0])
        level_0[0] = totals_row_label[1]
        df.columns = df.columns.set_levels(level_0, level=0)

        df = df.groupby(axis=1, level=0, sort=False).sum()

    if arrayed:
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir)
    else:
        exps = knock_knock.experiment.get_all_experiments(base_dir, conditions=conditions, as_dictionary=True)

    modal_maker = ModalMaker()

    def link_maker(val, outcome, name_tuple_string):
        name_tuple = tuple(name_tuple_string.split('/'))

        if val == 0:
            html = ''
        else:
            exp = exps[name_tuple]
            
            fraction = val / totals[name_tuple]

            if outcome == totals_row_label or outcome == totals_row_label_collapsed:
                text = f'{val:,}'

                if include_images:
                    hover_image_fn = exp.fns['lengths_figure']

                    relative_path = hover_image_fn.relative_to(exp.base_dir / 'results')

                    hover_URI = str(relative_path)
                    if hover_image_fn.exists():
                        with PIL.Image.open(hover_image_fn) as im:
                            width, height = im.size
                            width = width * 0.75
                            height = height * 0.75
                    else:
                        width, height = 100, 100

                    link = link_without_modal_template.format(text=text,
                                                             URI=hover_URI,
                                                             width=width,
                                                             height=height,
                                                            )
                else:
                    link = bare_link_template.format(text=text)


                html = link

            else:
                text = f'{fraction:.2%}'

                if include_images:
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

                    modal_div, modal_id = modal_maker.make_outcome()

                    link = link_template.format(id=f'{"_".join(name_tuple)}_{outcome}',
                                                text=text,
                                                modal_id=modal_id,
                                                iframe_URL=relative_path,
                                                URI=hover_URI,
                                                width=width,
                                                height=height,
                                                URL=str(relative_path),
                                               )
                else:
                    link = bare_link_template.format(text=text)
                    modal_div = ''

                html = link + modal_div

        return html
    
    def bind_link_maker(name_tuple_string):
        bound = {}
        for outcome in df:
            bound[outcome] = functools.partial(link_maker, outcome=outcome, name_tuple_string=name_tuple_string)

        return bound

    styled = df.style

    styles = [
        dict(selector="th", props=[("border", "1px solid black")]),
        dict(selector="tr:hover", props=[("background-color", "#cccccc")]),
    ]
    
    for exp_group, name_tuple_string in df.index:
        # rsplit here is future-proofing against handling ArrayedGroup/screens better.
        sl = pd.IndexSlice[[(exp_group, name_tuple_string)], :]
        styled = styled.format(bind_link_maker(name_tuple_string), subset=sl)
    
    styled = styled.set_properties(**{'border': '1px solid black'})
    for exp_group, name_tuple_string in df.index:
        name_tuple = tuple(name_tuple_string.split('/'))
        exp = exps[name_tuple]
        # Note: as of pandas 0.22, col needs to be in brackets here so that
        # apply is ultimately called on a df, not a series, to prevent
        # TypeError: _bar_left() got an unexpected keyword argument 'axis'

        subset_slice = pd.IndexSlice[[(exp_group, name_tuple_string)], :]

        styled = styled.bar(subset=subset_slice,
                            axis=1,
                            color=exp.color,
                            vmin=0,
                            vmax=df.loc[exp_group, name_tuple_string].max() * vmax_multiple,
                           )

    styled.set_table_styles(styles)
        
    return styled

def generate_html(base_dir, fn,
                  conditions=None,
                  show_details=True,
                  include_images=True,
                  sort_samples=True,
                  arrayed=False,
                  vmax_multiple=1,
                 ):

    fn = Path(fn)
    logo_fn = Path(os.path.realpath(__file__)).parent / 'logo_v2.png'
    logo_URI, logo_width, logo_height = fn_to_URI(logo_fn)

    nb = nbf.new_notebook()

    documentation_cell_contents = f'''\
<a href="https://github.com/jeffhussmann/knock-knock" target="_blank"><img width={logo_width} height={logo_height} src={logo_URI} alt="knock-knock" align="left"></a>
<br clear="all">

knock-knock is a tool for exploring, categorizing, and quantifying the sequence outcomes produced by genome editing experiments.

<a href="https://github.com/jeffhussmann/knock-knock/blob/master/docs/visualization.md#interactive-exploration-of-outcomes" target="_blank">How to use this table</a>

<a href="https://github.com/jeffhussmann/knock-knock/blob/master/docs/visualization.md" target="_blank">How to interpret read diagrams</a>
'''
    documentation_cell = nbf.new_markdown_cell(documentation_cell_contents)

    table = make_table(base_dir,
                       conditions,
                       show_details=show_details,
                       include_images=include_images,
                       sort_samples=sort_samples,
                       arrayed=arrayed,
                       vmax_multiple=vmax_multiple,
                      )

    table_cell = nbf.new_code_cell('',
                                   outputs=[
                                       nbf.nbbase.NotebookNode(
                                           output_type='display_data',
                                           metadata=nbf.nbbase.NotebookNode(),
                                           data={
                                               'text/html': table._repr_html_(),
                                           },
                                       ),
                                   ],
                                  )

    nb['cells'] = [
        documentation_cell,
        table_cell,
    ]

    nb['metadata'] = {
        'title': fn.stem,
        'include_images': include_images,
    }

    # Note: with nbconvert==6.3.0, can't call the template file 'index.html.j2'
    # or it will silently fail to use to the template, possible related to
    # https://github.com/jupyter/nbconvert/issues/1558.
    template_path = Path(os.path.realpath(__file__)).parent / 'table_template' / 'table.html.j2'
    exporter = nbconvert.HTMLExporter(exclude_input=True,
                                      exclude_output_prompt=True,
                                      template_file=str(template_path),
                                     )

    body, resources = exporter.from_notebook_node(nb)
    with open(fn, 'w') as fh:
        fh.write(body)

def make_self_contained_zip(base_dir,
                            conditions,
                            table_name,
                            include_images=True,
                            include_details=True,
                            sort_samples=True,
                            arrayed=False,
                            vmax_multiple=1,
                           ):

    base_dir = Path(base_dir)
    results_dir = base_dir / 'results'
    fn_prefix = results_dir / table_name
    fns_to_zip = []

    logging.info('Generating csv table...')
    csv_fn = fn_prefix.with_suffix('.csv')
    df = load_counts(base_dir, conditions, exclude_empty=False, arrayed=arrayed).T
    df.to_csv(csv_fn)
    fns_to_zip.append(csv_fn)

    logging.info('Generating high-level html table...')
    html_fn = fn_prefix.with_suffix('.html')
    generate_html(base_dir, html_fn, conditions,
                  show_details=False,
                  include_images=include_images,
                  sort_samples=sort_samples,
                  arrayed=arrayed,
                  vmax_multiple=vmax_multiple,
                 )
    fns_to_zip.append(html_fn)

    if include_details:
        logging.info('Generating detailed html table...')
        html_fn = fn_prefix.parent / (f'{fn_prefix.name}_with_details.html')
        generate_html(base_dir, html_fn, conditions,
                      show_details=True,
                      include_images=include_images,
                      sort_samples=sort_samples,
                      arrayed=arrayed,
                      vmax_multiple=vmax_multiple,
                     )
        fns_to_zip.append(html_fn)

    logging.info('Generating performance metrics...')
    pms_fn = fn_prefix.parent / (f'{fn_prefix.name}_performance_metrics.csv')
    pms = calculate_performance_metrics(base_dir, conditions, arrayed=arrayed)
    pms.to_csv(pms_fn)
    fns_to_zip.append(pms_fn)

    if arrayed:
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir, conditions=conditions)
    else:
        exps = knock_knock.experiment.get_all_experiments(base_dir, conditions)

    exps_missing_files = defaultdict(list)

    if include_images:
        for exp in exps.values():
            def add_fn(fn):
                if not fn.exists():
                    exps_missing_files[exp.group, exp.sample_name].append(fn)
                else:
                    if fn.is_dir():
                        for child_fn in fn.iterdir():
                            fns_to_zip.append(child_fn)
                    else:
                        fns_to_zip.append(fn)
            
            add_fn(exp.fns['outcome_browser'])
            add_fn(exp.fns['lengths_figure'])

            if exp.categories_by_frequency is not None:
                for outcome in exp.categories_by_frequency:
                    outcome_fns = exp.outcome_fns(outcome)
                    if include_details:
                        add_fn(outcome_fns['diagrams_html'])
                        add_fn(outcome_fns['first_example'])
                    add_fn(outcome_fns['length_ranges_dir'])

                categories = set(c for c, s in exp.categories_by_frequency)
                for category in categories:
                    outcome_fns = exp.outcome_fns(category)
                    add_fn(outcome_fns['diagrams_html'])
                    add_fn(outcome_fns['first_example'])

    if len(exps_missing_files) > 0:
        logging.warning(f'Warning: {len(exps_missing_files)} experiment(s) are missing output files:')
        for group, exp_name in sorted(exps_missing_files):
            logging.warning(f'\t{group} {exp_name}')
            for fn in exps_missing_files[group, exp_name]:
                logging.warning(f'\t\t{fn}')

    zip_fn = fn_prefix.with_suffix('.zip')
    archive_base = Path(fn_prefix.name)
    with zipfile.ZipFile(zip_fn, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_fh:
        description = 'Zipping table files'
        for fn in tqdm.tqdm(fns_to_zip, desc=description):
            arcname = archive_base / fn.relative_to(results_dir)
            zip_fh.write(fn, arcname=arcname)
