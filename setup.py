from setuptools import setup
from pathlib import Path

example_data_fns = []
base_dir = Path('knock_knock/example_data/data')
for fn in base_dir.glob('**/*'):
    example_data_fns.append(str(fn.relative_to('knock_knock')))

target_csv_fn = Path('knock_knock/example_data/targets/targets.csv')
example_data_fns.append(str(target_csv_fn.relative_to('knock_knock')))

setup(
    name='knock_knock',
    version='0.1.5',

    author='Jeff Hussmann',
    author_email='jeff.hussmann@gmail.com',

    packages=[
        'knock_knock',
    ],

    package_data={
        'knock_knock': [
            'modal_template.tpl',
        ] + example_data_fns,
    },

    scripts=[
        'knock_knock/knock_knock',
    ],

    install_requires=[
        'bokeh>=0.12.14',
        'biopython>=1.70',
        'ipywidgets>=7.1.2',
        'matplotlib>=2.1.2',
        'nbconvert>=5.3.1',
        'nbformat>=4.4.0',
        'numpy>=1.14.2',
        'pandas>=0.22.0',
        'Pillow>=5.0.0',
        'pysam>=0.14',
        'PyYAML>=3.12',
        'hits>=0.0.6',
        'tqdm>=4.31.1',
    ],

    python_requires='>=3.6',
)
