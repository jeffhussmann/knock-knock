import base64
import functools
import io
import os
import zipfile

from pathlib import Path

import pandas as pd
import nbconvert
import nbformat.v4 as nbf
import PIL
import tqdm

from . import experiment
from . import svg

totals_all_row_label = (' ', 'Total reads')

totals_relevant_row_label = (' ', 'Total relevant reads')

def load_counts(base_dir,
                conditions=None,
                exclude_malformed=False,
                exclude_empty=True,
                sort_samples=True,
                groups_to_exclude=None,
               ):

    if groups_to_exclude is None:
        groups_to_exclude = set()

    exps = experiment.get_all_experiments(base_dir, conditions, groups_to_exclude=groups_to_exclude)

    counts = {}
    no_outcomes = []

    for (group, name), exp in exps.items():
        if exp.category_counts is None:
            no_outcomes.append((group, name))
        else:
            counts[group, name] = exp.category_counts

    if no_outcomes:
        no_outcomes_string = '\n'.join(f'\t{group}: {name}' for group, name in no_outcomes)
        print(f'Warning: can\'t find outcome counts for\n{no_outcomes_string}') 

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
        
        svg_text = svg.length_plot_with_popovers(exp, outcome=outcome, container_selector=f'#{modal_id}', inline_images=inline_images)
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
              ):
    df = load_counts(base_dir, conditions=conditions, exclude_malformed=(not show_details), sort_samples=sort_samples)
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
                text = '{:.2%}'.format(fraction)

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

                    link = link_template.format(text=text,
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

def generate_html(base_dir, fn, conditions=None, show_details=True, include_images=True, sort_samples=True):
    logo_fn = Path(os.path.realpath(__file__)).parent / 'logo_v2.png'
    logo_URI, logo_width, logo_height = fn_to_URI(logo_fn)

    nb = nbf.new_notebook()

    documentation_cell_contents = f'''\
<a target="_blank" href="https://github.com/jeffhussmann/knock-knock" rel="nofollow"><img width={logo_width} height={logo_height} src={logo_URI} alt="knock-knock" align="left"></a>
<br clear="all">

knock-knock is a tool for exploring, categorizing, and quantifying the full spectrum of sequence outcomes produced by CRISPR knock-in experiments.

<a href="https://github.com/jeffhussmann/knock-knock/blob/master/docs/visualization.md#interactive-exploration-of-outcomes" target="_blank">How to use this table</a>

<a href="https://github.com/jeffhussmann/knock-knock/blob/master/docs/visualization.md" target="_blank">How to interpret read diagrams</a>
'''

    table_cell_contents = f'''\
import knock_knock.table

conditions = {conditions}
knock_knock.table.make_table('{base_dir}',
                             conditions,
                             show_details={show_details},
                             include_images={include_images},
                             sort_samples={sort_samples},
                            )
'''
    
    nb['cells'] = [
        nbf.new_markdown_cell(documentation_cell_contents),
        nbf.new_code_cell(table_cell_contents),
    ]

    nb['metadata'] = {
        'title': str(fn.name),
        'include_images': include_images,
    }

    exporter = nbconvert.HTMLExporter(exclude_input=True, exclude_output_prompt=True)
    template_path = Path(os.path.realpath(__file__)).parent / 'modal_template.tpl'
    exporter.template_file = str(template_path)

    ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})

    body, resources = exporter.from_notebook_node(nb)
    with open(fn, 'w') as fh:
        fh.write(body)

def make_self_contained_zip(base_dir, conditions, table_name,
                            include_images=True,
                            include_details=True,
                            sort_samples=True,
                           ):
    base_dir = Path(base_dir)
    results_dir = base_dir / 'results'
    fn_prefix = results_dir / table_name
    fns_to_zip = []

    print('Generating csv table...')
    csv_fn = fn_prefix.with_suffix('.csv')
    df = load_counts(base_dir, conditions, exclude_empty=False).T
    df.to_csv(csv_fn)
    fns_to_zip.append(csv_fn)

    print('Generating high-level html table...')
    html_fn = fn_prefix.with_suffix('.html')
    generate_html(base_dir, html_fn, conditions, show_details=False, include_images=include_images, sort_samples=sort_samples)
    fns_to_zip.append(html_fn)

    if include_details:
        print('Generating detailed html table...')
        html_fn = fn_prefix.parent / (f'{fn_prefix.name}_with_details.html')
        generate_html(base_dir, html_fn, conditions, show_details=True, include_images=include_images, sort_samples=sort_samples)
        fns_to_zip.append(html_fn)

    print('Generating performance metrics...')
    pms_fn = fn_prefix.parent / (f'{fn_prefix.name}_performance_metrics.csv')
    pms = calculate_performance_metrics(base_dir, conditions)
    pms.to_csv(pms_fn)
    fns_to_zip.append(pms_fn)

    exps = experiment.get_all_experiments(base_dir, conditions)

    exps_missing_files = set()

    if include_images:
        for exp in exps.values():
            def add_fn(fn):
                if not fn.exists():
                    exps_missing_files.add((exp.group, exp.sample_name))
                else:
                    if fn.is_dir():
                        for child_fn in fn.iterdir():
                            fns_to_zip.append(child_fn)
                    else:
                        fns_to_zip.append(fn)
            
            add_fn(exp.fns['outcome_browser'])
            add_fn(exp.fns['lengths_figure'])

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

    if exps_missing_files:
        print(f'Warning: {len(exps_missing_files)} experiment(s) are missing output files:')
        for group, exp_name in sorted(exps_missing_files):
            print(f'\t{group} {exp_name}')

    zip_fn = fn_prefix.with_suffix('.zip')
    archive_base = Path(fn_prefix.name)
    with zipfile.ZipFile(zip_fn, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_fh:
        description = 'Zipping table files'
        for fn in tqdm.tqdm(fns_to_zip, desc=description):
            arcname = archive_base / fn.relative_to(results_dir)
            zip_fh.write(fn, arcname=arcname)
