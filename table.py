import pandas as pd
import functools
import bokeh.palettes
import Sequencing.ipython
import base64
import pacbio
import IPython.display

colors = bokeh.palettes.Category20c_20
col_to_color = {}
for i, donor in enumerate(['PCR', 'Plasmid', 'ssDNA', 'CT']):
    for replicate in range(3):
        col_to_color['{0}-{1}'.format(donor, replicate + 1)] = colors[4 * i  + replicate]

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
        
def make_table(target, must_contain=None):
    if must_contain is None:
        must_contain = []

    df = pacbio.load_counts(target)
    totals = df.loc[(' ', 'Total reads')]

    modal_maker = ModalMaker()

    def link_maker(val, col, row):
        if val == 0:
            return ''
        else:
            outcome = row
            fns = pacbio.make_fns(target, col, outcome)
            
            fraction = val / float(totals[col])
            
            if row == (' ', 'Total reads'):
                title = col
                modal_image_fn = str(fns['all_lengths'])
                modal_div, modal_id = modal_maker.make(modal_image_fn, title)
                link = dataset_link_template.format(modal_id=modal_id, count=val)
                
                return link + modal_div
            elif row == ('malformed layout', 'no alignments detected'):
                return '{:.2%}'.format(fraction)
            else:
                outcome_string = '_'.join(outcome)
                title = '{0}: {1}'.format(col, outcome_string)
                modal_image_fn = str(fns['figure'])
                modal_div, modal_id = modal_maker.make(modal_image_fn, title)
                
                hover_image_fn = str(fns['first'])
                hover_URI = fn_to_URI(hover_image_fn)
                link = outcome_link_template.format(fraction=fraction, modal_id=modal_id, URI=hover_URI)
                
                return link + modal_div
    
    def bind_link_maker(row):
        return {col: functools.partial(link_maker, col=col, row=row) for col in df}


    df = df[[c for c in df.columns if all(s in c for s in must_contain)]]

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
        #styled = styled.bar(subset=pd.IndexSlice[:, col], color=col_to_color[col[1].split('_')[-1]])
        styled = styled.bar(subset=pd.IndexSlice[:, col], color=col_to_color.get(col.split('_')[-1], 'grey'))
        
    styled.set_table_styles(styles)

    return styled