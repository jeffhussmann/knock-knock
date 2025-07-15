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

import knock_knock.svg

logger = logging.getLogger(__name__)

totals_all_row_label = (' ', 'Total reads')
totals_relevant_row_label = (' ', 'Total relevant reads')

def load_counts(base_dir,
                conditions=None,
                exclude_non_relevant=False,
                exclude_empty=True,
                sort_samples=True,
                groups_to_exclude=None,
                arrayed=False,
               ):

    if groups_to_exclude is None:
        groups_to_exclude = set()

    if arrayed:
        import knock_knock.arrayed_experiment_group
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir, conditions=conditions) 

    else:
        import knock_knock.experiment
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
        logger.warning(f'Warning: can\'t find outcome counts for\n{no_outcomes_string}') 

    df = pd.DataFrame(counts).fillna(0)

    # Sort order for outcomes is defined in the relevant layout module.
    full_indexes = {tuple(exp.categorizer.full_index()) for exp in exps.values()}
    
    if len(full_indexes) > 1:
        print(full_indexes)
        raise ValueError('Can\'t make table for experiments with inconsistent layout modules.')

    full_index = full_indexes.pop()

    df = df.reindex(full_index, fill_value=0)

    
    if exclude_non_relevant:
        all_non_relevant_categories = {tuple(sorted(exp.categorizer.non_relevant_categories)) for exp in exps.values()}

        if len(all_non_relevant_categories) > 1:
            print(all_non_relevant_categories)
            raise ValueError('Can\'t make table for experiments with inconsistent layout modules.')

        non_relevant_categories = list(all_non_relevant_categories.pop())

        df = df.drop(non_relevant_categories, axis='index', level=0, errors='ignore')

        totals_row_label = totals_relevant_row_label

    else:
        totals_row_label = totals_all_row_label

    if exclude_empty:
        empty_rows = df.index[df.sum(axis=1) == 0].values
        df = df.drop(empty_rows, axis='index', errors='ignore')
    
    totals = df.sum(axis=0)
    totals_row = pd.DataFrame.from_dict({totals_row_label: totals}, orient='index')
    
    df = pd.concat([totals_row, df]).astype(int)

    if sort_samples:
        # Sort by group and sample name.
        df = df.sort_index(axis=1)

    return df

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

outcome_browser_link_template = '''\
<a
    href="{URI}"
    target="_blank"
>
    {sample_name}
</a>
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
        
        svg_text = knock_knock.svg.length_plot_with_popovers(exp,
                                                             outcome=outcome,
                                                             container_selector=f'#{modal_id}',
                                                             inline_images=inline_images,
                                                            )

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
                     exclude_non_relevant=(not show_details),
                     sort_samples=sort_samples,
                     arrayed=arrayed,
                    )

    if df.shape == (0, 0):
        return None

    if show_details:
        totals_row_label = totals_all_row_label
    else:
        totals_row_label = totals_relevant_row_label

    totals_row_label_collapsed = totals_row_label[1]

    totals = df.loc[totals_row_label]

    df = df.T

    if arrayed:
        df.index = pd.MultiIndex.from_tuples([(batch, (batch, group, sample)) for batch, group, sample in df.index.values])
    else:
        df.index = pd.MultiIndex.from_tuples([(g, f'{g}/{n}') for g, n in df.index.values])
    
    if not show_details:
        level_0 = list(df.columns.levels[0])
        level_0[0] = totals_row_label[1]
        df.columns = df.columns.set_levels(level_0, level=0)

        df = df.T.groupby(level=0, sort=False).sum().T

    if arrayed:
        import knock_knock.arrayed_experiment_group
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir, conditions=conditions)
    else:
        exps = knock_knock.experiment.get_all_experiments(base_dir, conditions=conditions, as_dictionary=True)

    modal_maker = ModalMaker()

    def link_maker(val, outcome, name_tuple):
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
    
    def bind_link_maker(name_tuple):
        bound = {}
        for outcome in df:
            bound[outcome] = functools.partial(link_maker, outcome=outcome, name_tuple=name_tuple)

        return bound

    styled = df.style

    for exp_group, name_tuple in df.index:
        sl = pd.IndexSlice[[(exp_group, name_tuple)], :]
        styled.format(bind_link_maker(name_tuple), subset=sl)
    
    for exp_group, name_tuple in df.index:
        exp = exps[name_tuple]
        # Note: as of pandas 0.22, col needs to be in brackets here so that
        # apply is ultimately called on a df, not a series, to prevent
        # TypeError: _bar_left() got an unexpected keyword argument 'axis'

        subset_slice = pd.IndexSlice[[(exp_group, name_tuple)], :]

        styled.bar(subset=subset_slice,
                   axis=1,
                   color=exp.color,
                   vmin=0,
                   vmax=df.loc[subset_slice].max(axis=None) * vmax_multiple,
                  )

    def make_outcome_browser_link(name_tuple):
        sample_name = name_tuple[-1]

        exp = exps[name_tuple]
        outcome_browser_fn = exp.fns['outcome_browser']
        relative_path = outcome_browser_fn.relative_to(exp.base_dir / 'results')

        return outcome_browser_link_template.format(URI=relative_path, sample_name=sample_name)

    styled.format_index(axis=0,
                        level=1,
                        formatter=make_outcome_browser_link,
                       )

    styled.set_table_attributes('style="border-collapse: separate"')

    pre_styles = [
        {
            # Cap the bottom of row labels on the last row.
            'selector': 'tbody tr:nth-last-child(1) th.level1, tbody tr:nth-child(1) th.level0',
            'props': [
                ('border-bottom', '1px solid black'),
            ],
        },
        {
            # Cap the bottom of upper left corer.
            'selector': 'thead tr:nth-last-child(1) th',
            'props': [
                ('border-bottom', '1px solid black'),
            ],
        },
        {
            # level 1 needs left and right since it is sticky.
            'selector': 'tbody tr th.row_heading.level1',
            'props': [
                ('border-left', '1px solid black'),
                ('border-right', '1px solid black'),
            ],
        },
        {
            # level 0 needs left.
            'selector': 'tbody tr th.row_heading.level0',
            'props': [
                ('border-left', '1px solid black'),
            ],
        },
        {
            # Cap the top of all row labels.
            'selector': 'tbody tr th',
            'props': [
                ('border-top', '1px solid black'),
            ],
        },
        {
            # Cap the far right side of the upper left corner. (nth-last-child(1) doesn't work...)
            'selector': 'thead tr th.blank:nth-child(2)',
            'props': [
                ('border-right', '1px solid black'),
            ],
        },
        {
            # Top and right of all col heading cells.
            'selector': 'thead tr th.col_heading',
            'props': [
                ('border-right', '1px solid black'),
                ('border-top', '1px solid black'),
            ],
        },
        {
            # Top and right of all data cells.
            'selector': 'td',
            'props': [
                ('border-right', '1px solid black'),
                ('border-top', '1px solid black'),
            ],
        },
        {
            # Cap the bottom of last row of data cells.
            'selector': 'tbody tr:nth-last-child(1) td',
            'props': [
                ('border-bottom', '1px solid black'),
            ],
        },
         {
            # Cap the bottom of the column headings.
            'selector': 'thead tr:nth-last-child(1) th.col_heading',
            'props': [
                ('border-bottom', '1px solid black'),
            ],
        },
            # Hover background for data rows
        {
            'selector': 'tr:hover',
            'props': [
                ('background-color', '#cccccc'),
            ],
        },
        {
            # No hover background for header rows
            'selector': 'thead tr:hover',
            'props': [
                ('background-color', 'white'),
            ],
        },
    ]

    post_styles = [
        {
            # Overwrite max-width set by .set_sticky(axis=1).
            'selector': 'tbody tr th.level0, tbody tr th.level1',
            'props': [
                ('max-width', 'none'),
            ],
        },
        {
            # Make upper left corner opaque.
            'selector': 'thead tr th.blank, thead tr th.col_heading',
            'props': [
                ('background-color', 'white'),
            ],
        },
        {
            # Set height of the top level of column labels...
            'selector': 'thead tr:nth-child(1) th',
            'props': [
                ('height', '50px'),
            ],
        },
        {
            # ... and adjust sticky top of 2nd level to match this height.
            'selector': 'thead tr:nth-child(2) th',
            'props': [
                ('top', '50px'),
            ],
        },
    ]

    styled.set_table_styles(pre_styles)
        
    styled.set_sticky(axis=1)
    styled.set_sticky(axis=0, levels=1)

    styled.set_table_styles(post_styles, overwrite=False)

    return styled

def generate_html(base_dir, fn,
                  conditions=None,
                  show_details=True,
                  include_images=True,
                  include_documentation=False,
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

    if table is None:
        return None

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

    cells = []

    if include_documentation:
        cells.append(documentation_cell)

    cells.append(table_cell)

    nb['cells'] = cells

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
                            show_progress=False,
                           ):

    base_dir = Path(base_dir)
    results_dir = base_dir / 'results'
    fn_prefix = results_dir / table_name
    fns_to_zip = set()

    def add_fn(fn):
        if not fn.exists():
            logger.warning(f'{fn} is missing')
        else:
            if fn.is_dir():
                for child_fn in fn.iterdir():
                    fns_to_zip.add(child_fn)
            else:
                fns_to_zip.add(fn)

    logger.info('Generating csv table...')
    csv_fn = fn_prefix.with_suffix('.csv')
    df = load_counts(base_dir, conditions, exclude_empty=False, arrayed=arrayed).T
    df.to_csv(csv_fn)
    add_fn(csv_fn)

    logger.info('Generating high-level html table...')
    html_fn = fn_prefix.with_suffix('.html')
    generate_html(base_dir,
                  html_fn,
                  conditions,
                  show_details=False,
                  include_images=include_images,
                  sort_samples=sort_samples,
                  arrayed=arrayed,
                  vmax_multiple=vmax_multiple,
                 )
    add_fn(html_fn)

    if include_details:
        logger.info('Generating detailed html table...')
        html_fn = fn_prefix.parent / f'{fn_prefix.name}_with_details.html'
        generate_html(base_dir,
                      html_fn,
                      conditions,
                      show_details=True,
                      include_images=include_images,
                      sort_samples=sort_samples,
                      arrayed=arrayed,
                      vmax_multiple=vmax_multiple,
                     )
        add_fn(html_fn)

    if arrayed:
        import knock_knock.arrayed_experiment_group
        exps = knock_knock.arrayed_experiment_group.get_all_experiments(base_dir, conditions=conditions)
    else:
        exps = knock_knock.experiment.get_all_experiments(base_dir, conditions)

    missing_files = {
        'experiment': defaultdict(list),
        'batch': defaultdict(list),
        'group': defaultdict(list),
    }

    batches = {}
    groups = {}

    if include_images:
        for exp in exps.values():
            def add_fn(fn):
                if not fn.exists():
                    missing_files['experiment'][exp.group_name, exp.sample_name].append(fn)
                else:
                    if fn.is_dir():
                        for child_fn in fn.iterdir():
                            fns_to_zip.add(child_fn)
                    else:
                        fns_to_zip.add(fn)

            group = exp.experiment_group
            batch = group.batch

            batches[batch.batch_name] = batch
            groups[batch.batch_name, group.group_name] = group

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

        for batch in batches.values():
            def add_fn(fn):
                if not fn.exists():
                    missing_files['batch'][batch.batch_name].append(fn)
                else:
                    fns_to_zip.add(fn)

            add_fn(batch.fns['group_name_to_sanitized_group_name'])
            add_fn(batch.fns['performance_metrics'])

            for fn in batch.pegRNA_conversion_fractions_fns():
                add_fn(fn)

        for group in groups.values():
            def add_fn(fn):
                if not fn.exists():
                    missing_files['group'][group.group_name].append(fn)
                else:
                    fns_to_zip.add(fn)

            add_fn(group.fns['pegRNA_conversion_fractions'])

            add_fn(group.fns['partial_incorporation_figure_high_threshold'])
            add_fn(group.fns['partial_incorporation_figure_low_threshold'])

            add_fn(group.fns['deletion_boundaries_figure'])

            add_fn(group.fns['single_flap_rejoining_boundaries_figure'])
            add_fn(group.fns['single_flap_rejoining_boundaries_figure_normalized'])
            add_fn(group.fns['single_flap_rejoining_boundaries_figure_individual_samples'])
            add_fn(group.fns['single_flap_rejoining_boundaries_figure_individual_samples_normalized'])
            
    for kind, missing_files_for_kind in missing_files.items():
        if len(missing_files_for_kind) > 0:
            list_sources = (len(missing_files_for_kind) <= 10)

            logger.warning(f'{len(missing_files_for_kind)} {kind}{"s are" if len(missing_files_for_kind) > 1 else " is"} missing output files{":" if list_sources else "."}')

            if list_sources:
                for key in sorted(missing_files_for_kind):
                    logger.warning(f'\t{key}')
                    if len(missing_files_for_kind[key]) <= 10:
                        for fn in missing_files_for_kind[key]:
                            logger.warning(f'\t\t{fn}')

    zip_fn = fn_prefix.with_suffix('.zip')

    archive_base = Path(fn_prefix.name)

    with zipfile.ZipFile(zip_fn, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_fh:

        if show_progress:
            description = 'Zipping table files'
            fns_to_zip = tqdm.tqdm(fns_to_zip, desc=description)

        for fn in fns_to_zip:
            arcname = archive_base / fn.relative_to(results_dir)
            zip_fh.write(fn, arcname=arcname)
