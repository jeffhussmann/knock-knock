from setuptools import setup

setup(
    name='knock_knock',
    version='0.1',
    
    packages=[
        'knockin',
    ],

    package_data={
        'knockin': ['modal_template.tpl'],
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
        'hits>=0.0.1',
    ],
)
