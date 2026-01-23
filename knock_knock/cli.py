import argparse
import logging
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml
import tqdm

import knock_knock
import knock_knock.utilities

logger = logging.getLogger(__name__)

def check_blastn(require_precise_version=False):
    try:
        output = subprocess.check_output(['blastn', '-version'])

        if require_precise_version and b'2.7.1' not in output:
            print('blastn 2.7.1 is required and couldn\'t be found')
            sys.exit(1)

    except:
        print('blastn is required and couldn\'t be found')
        sys.exit(1)

def process_experiment(args):
    import knock_knock.experiment

    check_blastn()

    identifer = knock_knock.experiment.ExperimentIdentifier(args.base_dir,
                                                            args.batch_name,
                                                            args.sample_name,
                                                           )

    stages = args.stages.split(',')

    knock_knock.experiment.process_experiment(identifer,
                                              stages,
                                              progress=args.progress,
                                             )

def parallel(args):
    import knock_knock.experiment
    import knock_knock.parallel

    check_blastn()

    if args.batch:
        args.conditions['batch'] = args.batch

    exps = knock_knock.experiment.get_all_experiments(args.base_dir, args.conditions)

    if len(exps) == 0:
        print('No experiments satify conditions:')
        print(args.conditions)
        sys.exit(1)

    pool = knock_knock.parallel.get_pool(num_processes=args.max_procs,
                                         use_logger_thread=args.use_logger_thread,
                                         log_dir=args.base_dir,
                                        )

    stages = args.stages.split(',')

    arg_tuples = [
        (
            exp.identifier,
            stages,
            None,
            exp_i,
            len(exps),
        )
        for exp_i, exp in enumerate(exps.values())
    ]

    with pool:
        pool.starmap(knock_knock.experiment.process_experiment, arg_tuples)

def make_tables(args):
    import knock_knock.experiment
    import knock_knock.table

    if args.batches:
        batches_to_include = args.batches.split(',')
    else:
        batches_to_include = None

    # TODO: group by experiment_type
    #    for experiment_type in experiment_types:
    #        logger.info(f'Making {batch_name} {experiment_type}')

    #        conditions = {
    #            'batch': [batch_name],
    #            'experiment_type': experiment_type,
    #        }
    if batches_to_include is not None:

        conditions = {
            'batch': batches_to_include,
        }

        knock_knock.table.make_self_contained_zip(args.base_dir,
                                                  conditions,
                                                  args.title,
                                                  sort_samples=not args.unsorted,
                                                  vmax_multiple=args.vmax_multiple,
                                                 )
    else:
        batches = knock_knock.experiment.get_all_batches(args.base_dir)

        for batch_name in batches:
            logger.info(f'Making {batch_name}')

            conditions = {'batch': batch_name}

            knock_knock.table.make_self_contained_zip(args.base_dir,
                                                      conditions,
                                                      batch_name,
                                                      sort_samples=not args.unsorted,
                                                      vmax_multiple=args.vmax_multiple,
                                                     )

def build_strategies(args):
    import knock_knock.build_strategies

    knock_knock.utilities.configure_logging_to_file(args.base_dir, 'knock_knock')

    knock_knock.build_strategies.build_strategies(args.base_dir, args.batch_name)

def build_indices(args):
    import knock_knock.build_strategies

    knock_knock.build_strategies.download_genome_and_build_indices(args.base_dir,
                                                                   args.genome_name,
                                                                   args.num_threads
                                                                  )

def install_example_data(args):
    package_dir = Path(knock_knock.__file__).resolve().parent
    
    subdirs_to_copy = ['data', 'strategies']

    for subdir in subdirs_to_copy:
        src = package_dir / 'example_data' / subdir
        dest = args.base_dir / subdir

        if dest.exists():
            print(f'Can\'t install to {args.base_dir}, {dest} already exists')
            sys.exit(1)

        shutil.copytree(src, dest)

    logger.info(f'Example data installed in {args.base_dir}')

def print_citation(args):
    citation = '''
        Hera Canaj, Jeffrey A. Hussmann, Han Li, Kyle A. Beckman, Leeanne Goodrich,
        Nathan H. Cho, Yucheng J. Li, Daniel A Santos, Aaron McGeever, Edna M Stewart,
        Veronica Pessino, Mohammad A Mandegar, Cindy Huang, Li Gan, Barbara Panning,
        Bo Huang, Jonathan S. Weissman and Manuel D. Leonetti.  "Deep profiling reveals
        the complexity of integration outcomes in CRISPR knock-in experiments."
        https://www.biorxiv.org/content/10.1101/841098v1 (2019).
    '''
    print(textwrap.dedent(citation))

def main():
    parser = argparse.ArgumentParser(prog='knock-knock')

    parser.add_argument('--version', action='version', version=knock_knock.__version__)

    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands')
    subparsers.required = True

    def add_base_dir_arg(parser):
        parser.add_argument('base_dir', type=Path, help='the base directory to store input data, reference annotations, and analysis output for a project')

    parser_process_sample = subparsers.add_parser('process-sample', help='process a single sample')
    add_base_dir_arg(parser_process_sample)
    parser_process_sample.add_argument('batch_name', help='batch name')
    parser_process_sample.add_argument('sample_name', help='sample name')
    parser_process_sample.add_argument('--stages', default='preprocess,align,categorize,generate_example_diagrams,generate_summary_figures')
    parser_process_sample.add_argument('--progress', const=tqdm.tqdm, action='store_const', help='show progress bars')
    parser_process_sample.set_defaults(func=process_experiment)

    parser_parallel = subparsers.add_parser('parallel', help='process multiple samples in parallel')
    add_base_dir_arg(parser_parallel)
    parser_parallel.add_argument('max_procs', type=int, help='maximum number of samples to process at once')
    parser_parallel.add_argument('--batch', help='if specified, the single batch name to process; if not specified, all batches will be processed')
    parser_parallel.add_argument('--conditions', type=yaml.safe_load, default={}, help='if specified, conditions that samples must satisfy to be processed, given as yaml; if not specified, all samples will be processed')
    parser_parallel.add_argument('--stages', default='preprocess,align,categorize,generate_example_diagrams,generate_summary_figures')
    parser_parallel.add_argument('--use_logger_thread', action='store_true', help='write a consolidated log')
    parser_parallel.add_argument('--progress', const=tqdm.tqdm, action='store_const', help='show progress bars')
    parser_parallel.set_defaults(func=parallel)

    parser_table = subparsers.add_parser('table', help='generate tables of outcome frequencies')
    add_base_dir_arg(parser_table)
    parser_table.add_argument('--batches', help='if specified, a comma-separated list of batches to include; if not specified, all batches in base_dir will be generated')
    parser_table.add_argument('--title', default='knock_knock_table', help='if specified, a title for output files')
    parser_table.add_argument('--unsorted', action='store_true', help='don\'t sort samples')
    parser_table.add_argument('--vmax_multiple', type=float, default=1, help='fractional value that corresponds to full horizontal bar')
    parser_table.set_defaults(func=make_tables)

    parser_strategies = subparsers.add_parser('build-strategies', help='build annotations of editing strategies')
    add_base_dir_arg(parser_strategies)
    parser_strategies.add_argument('batch_name', help='batch name')
    parser_strategies.set_defaults(func=build_strategies)

    parser_indices = subparsers.add_parser('build-indices', help='download a reference genome and build alignment indices')
    add_base_dir_arg(parser_indices)
    parser_indices.add_argument('genome_name', help='name of genome to download')
    parser_indices.add_argument('--num-threads', type=int, default=8, help='number of threads to use for index building')
    parser_indices.set_defaults(func=build_indices)

    parser_install_data = subparsers.add_parser('install-example-data', help='install example data into user-specified project directory')
    add_base_dir_arg(parser_install_data)
    parser_install_data.set_defaults(func=install_example_data)

    parser_citation = subparsers.add_parser('whos-there', help='print citation information')
    parser_citation.set_defaults(func=print_citation)

    args = parser.parse_args()

    args.func(args)