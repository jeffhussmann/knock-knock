import bokeh
import bokeh.plotting
import bokeh.io
import numpy as np

from . import experiment

bokeh.io.output_notebook()

def range_picker(base_dir, inital_dataset=None):
    exps = {exp.name: exp for exp in experiment.get_all_experiments(base_dir)}
    if inital_dataset is None:
        inital_dataset = list(exps)[0]

    name_to_color = {name: exps[name].color for name in exps}

    all_ys = {name: exps[name].read_lengths for name in exps}
    max_length = max(len(ys) for ys in all_ys.values())
    max_y = max(max(ys) for ys in all_ys.values()) * 1.02
    xs = np.arange(max_length)
    for name in all_ys:
        padded = np.zeros(max_length, dtype=int)
        ys = all_ys[name]
        padded[:len(ys)] = ys
        all_ys[name] = padded

    f = bokeh.plotting.Figure(
        plot_width=1500,
        plot_height=600,
        lod_threshold=20000,
        active_scroll='wheel_zoom',
    )
    
    source = bokeh.models.ColumnDataSource(data={
        'x': xs,
        'y': all_ys[inital_dataset],
        **all_ys,
    })
    
    line = f.line(x='x', y='y', source=source, color=name_to_color[inital_dataset])

    f.x_range = bokeh.models.Range1d(start=0, end=max_length)
    f.y_range = bokeh.models.Range1d(start=0, end=max(all_ys[inital_dataset] * 1.02))

    bound_at_zero = bokeh.models.CustomJS.from_coffeescript('cb_obj.start = 0 if cb_obj.start < 0')
    fix_at_zero = bokeh.models.CustomJS.from_coffeescript('cb_obj.start = 0 if cb_obj.start != 0')
    y_bound_at_max = bokeh.models.CustomJS.from_coffeescript('cb_obj.end = {max} if cb_obj.end > {max}'.format(max=max_y))
    x_bound_at_max = bokeh.models.CustomJS.from_coffeescript('cb_obj.end = {max} if cb_obj.end > {max}'.format(max=max_length))

    f.x_range.js_on_change('start', bound_at_zero)
    f.x_range.js_on_change('end', x_bound_at_max)
    f.y_range.js_on_change('start', fix_at_zero)
    f.y_range.js_on_change('end', y_bound_at_max)

    q_data_source = bokeh.models.ColumnDataSource(data=dict(
        top=[],
        bottom=[],
        left=[],
        right=[],
    ))

    q = f.quad(
        top='top',
        bottom='bottom',
        left='left',
        right='right',
        source=q_data_source,
        color='black',
        alpha=0.15,
    )

    save_code = '''\
    data = data_source.data
    # learned from http://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side

    length = data.left.length
    lines = (data.left[i] + '\t' + data.right[i] for i in [0...length])

    csv_content = "data:text/csv;charset=utf-8," + lines.join('\\n')
    encoded = encodeURI(csv_content)

    link = document.createElement('a')
    link.setAttribute('href', encoded)
    link.setAttribute('download', 'manual_length_ranges.csv')
    link.click()
    '''
    save_button = bokeh.models.Button(label='save x ranges')
    save_button.js_on_click(bokeh.models.CustomJS.from_coffeescript(save_code, args={'data_source': q.data_source}))

    t = bokeh.models.BoxSelectTool(callback=bokeh.models.CustomJS.from_coffeescript('''
        data = data_source.data
        console.log data
        data['bottom'].push 0
        data['top'].push {max_y}
        data['left'].push Math.round(cb_data.geometry.x0)
        data['right'].push Math.round(cb_data.geometry.x1)
        data_source.change.emit()
    '''.format(max_y=max_y), args={'data_source': q.data_source}))

    f.add_tools(t)
    f.grid.visible = False

    select = bokeh.models.Select(title='Dataset:',
                                 value=inital_dataset,
                                 options=sorted(exps),
                                )
    
    select.js_on_change('value', bokeh.models.CustomJS.from_coffeescript(
        '''\
        data = data_source.data
        name = cb_obj.value
        data['y'] = data[name]

        name_to_color = {name_to_color}
        line.glyph.line_color = name_to_color[name]

        y_range.end = Math.max(...data['y']) * 1.02

        data_source.change.emit()

        q_data_source.data[key] = [] for key in ['top', 'bottom', 'left', 'right']
        q_data_source.change.emit()
        '''.format(name_to_color=name_to_color),
        args=dict(data_source=source, line=line, y_range=f.y_range, q_data_source=q_data_source)))

    bokeh.io.show(bokeh.layouts.row(children=[f, bokeh.layouts.column(children=[select, save_button])]))
