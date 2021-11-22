from setuptools import setup
from pathlib import Path

test_fns = []
data_dir = Path('knock_knock/example_data/data')
for fn in data_dir.glob('**/*'):
    test_fns.append(str(fn.relative_to('knock_knock')))

targets_dir = Path('knock_knock/example_data/targets')
for fn in targets_dir.glob('*.csv'):
    test_fns.append(str(fn.relative_to('knock_knock')))

setup(
    name='knock_knock',
    version='0.3.3',
    url='https://pypi.org/projects/knock-knock',

    author='Jeff Hussmann',
    author_email='jeff.hussmann@gmail.com',

    packages=[
        'knock_knock',
    ],

    package_data={
        'knock_knock': [
            'modal_template.tpl',
        ] + test_fns,
    },

    scripts=[
        'knock_knock/knock-knock',
    ],

    install_requires=[
        'bokeh>=0.12.14',
        'biopython>=1.78',
        'ipywidgets>=7.1.2',
        'matplotlib>=2.1.2',
        'nbconvert>=5.3.1',
        'nbformat>=4.4.0',
        'numpy>=1.14.2',
        'pandas>=0.22.0',
        'Pillow>=5.0.0',
        'pysam>=0.14',
        'PyYAML>=3.12',
        'hits>=0.1',
        'tqdm>=4.31.1',
    ],

    python_requires='>=3.6',

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
