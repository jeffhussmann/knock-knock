#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml
import tqdm

import knock_knock

def check_blastn(require_precise_version=False):
    try:
        output = subprocess.check_output(['blastn', '-version'])

        if require_precise_version and b'2.7.1' not in output:
            print('blastn 2.7.1 is required and couldn\'t be found')
            sys.exit(1)

    except:
        print('blastn is required and couldn\'t be found')
        sys.exit(1)

def parallel(args):
    import knock_knock.experiment

    check_blastn()

    if args.group:
        args.conditions['batch'] = args.group

    exps = knock_knock.experiment.get_all_experiments(args.project_directory, args.conditions)

    if len(exps) == 0:
        print('No experiments satify conditions:')
        print(args.conditions)
        sys.exit(1)

    def process_stage(stage):
        with multiprocessing.Pool(processes=args.max_procs, maxtasksperchild=1) as process_pool:
            arg_tuples = []

            for _, exp in exps.items():
                arg_tuple = (exp.base_dir, exp.batch, exp.sample_name, stage, args.progress, True)
                arg_tuples.append(arg_tuple)

            process_pool.starmap(knock_knock.experiment.process_experiment_stage, arg_tuples)

    stages = args.stages.split(',')
    for stage in stages:
        process_stage(stage)

def process(args):
    import knock_knock.experiment

    check_blastn()

    stages = args.stages.split(',')

    for stage in stages:
        knock_knock.experiment.process_experiment_stage(args.project_directory,
                                                        args.group,
                                                        args.sample,
                                                        stage,
                                                        progress=args.progress,
                                                        print_timestamps=True,
                                                       )

def process_arrayed(args):
    import knock_knock.arrayed_experiment_group

    batch = knock_knock.arrayed_experiment_group.get_batch(args.project_directory, args.batch_name)
    group = batch.groups[args.group_name]
    group.process(num_processes=args.max_procs,
                  verbose=not args.silent,
                  generate_example_diagrams=args.generate_example_diagrams,
                 )

def parallel_arrayed(args):
    import knock_knock.arrayed_experiment_group

    batch = knock_knock.arrayed_experiment_group.get_batch(args.project_directory, args.batch_name)

    batch.write_sanitized_group_name_lookup_table()

    parallel_command = [
        'parallel',
        '--max-procs', str(args.max_procs),
        'knock-knock', 'arrayed', 'process',
        args.project_directory,
        args.batch_name,
        '--max_procs', str(args.max_procs_per_group),
        '--generate_example_diagrams' if args.generate_example_diagrams else '',
        '--silent',
        ':::',
    ] + sorted(batch.groups)

    if args.progress_bar:
        parallel_command.insert(1, '--bar')

    subprocess.run(parallel_command)

    batch.write_performance_metrics()
    batch.write_pegRNA_conversion_fractions()

def make_tables(args):
    import knock_knock.arrayed_experiment_group
    import knock_knock.experiment
    import knock_knock.table

    if args.batches:
        batches_to_include = args.batches.split(',')
    else:
        batches_to_include = None

    if args.arrayed:
        batches = knock_knock.arrayed_experiment_group.get_all_batches(args.project_directory)

        if batches_to_include is not None:
            batches = {name: batch for name, batch in batches.items() if name in batches_to_include}

        for batch_name, batch in batches.items():
            experiment_types = {group.experiment_type for group in batch.groups.values()}
            for experiment_type in experiment_types:
                logging.info(f'Making {batch_name} {experiment_type}')

                conditions = {
                    'batch': [batch_name],
                    'experiment_type': experiment_type,
                }

                knock_knock.table.make_self_contained_zip(args.project_directory,
                                                          conditions,
                                                          f'{batch_name}_{experiment_type}',
                                                          sort_samples=not args.unsorted,
                                                          arrayed=True,
                                                          vmax_multiple=args.vmax_multiple,
                                                         )

    else:
        if batches_to_include is not None:

            conditions = {
                'batch': batches_to_include,
            }

            if args.experiment_type is not None:
                conditions['experiment_type'] = args.experiment_type

            knock_knock.table.make_self_contained_zip(args.project_directory,
                                                      conditions,
                                                      args.title,
                                                      sort_samples=not args.unsorted,
                                                      arrayed=args.arrayed,
                                                      vmax_multiple=args.vmax_multiple,
                                                     )
        else:
            batches = knock_knock.experiment.get_all_batches(args.project_directory)

            for batch_name in batches:
                logging.info(f'Making {batch_name}')

                conditions = {'batch': batch_name}

                knock_knock.table.make_self_contained_zip(args.project_directory,
                                                          conditions,
                                                          batch_name,
                                                          sort_samples=not args.unsorted,
                                                          arrayed=False,
                                                          vmax_multiple=args.vmax_multiple,
                                                         )

def build_targets(args):
    import knock_knock.build_targets

    knock_knock.build_targets.build_target_infos_from_csv(args.project_directory,
                                                          defer_HA_identification=args.defer_HA_identification,
                                                         )

def build_manual_target(args):
    import knock_knock.build_targets

    knock_knock.build_targets.build_manual_target(args.project_directory, args.target_name)

def build_indices(args):
    import knock_knock.build_targets

    knock_knock.build_targets.download_genome_and_build_indices(args.project_directory,
                                                                args.genome_name,
                                                                args.num_threads
                                                               )

def install_example_data(args):
    package_dir = Path(os.path.realpath(knock_knock.__file__)).parent
    subdirs_to_copy = ['data', 'targets']
    for subdir in subdirs_to_copy:
        src = package_dir / 'example_data' / subdir
        dest = args.project_directory / subdir

        if dest.exists():
            print(f'Can\'t install to {args.project_directory}, {dest} already exists')
            sys.exit(1)

        shutil.copytree(str(src), str(dest))

    logging.info(f'Example data installed in {args.project_directory}')

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

    def add_project_directory_arg(parser):
        parser.add_argument('project_directory', type=Path, help='the base directory to store input data, reference annotations, and analysis output for a project')

    parser_process = subparsers.add_parser('process', help='process a single sample')
    add_project_directory_arg(parser_process)
    parser_process.add_argument('group', help='group name')
    parser_process.add_argument('sample', help='sample name')
    parser_process.add_argument('--progress', const=tqdm.tqdm, action='store_const', help='show progress bars')
    parser_process.add_argument('--stages', default='preprocess,align,categorize,generate_figures')
    parser_process.set_defaults(func=process)

    parser_parallel = subparsers.add_parser('parallel', help='process multiple samples in parallel')
    add_project_directory_arg(parser_parallel)
    parser_parallel.add_argument('max_procs', type=int, help='maximum number of samples to process at once')
    parser_parallel.add_argument('--group', help='if specified, the single group name to process; if not specified, all groups will be processed')
    parser_parallel.add_argument('--conditions', type=yaml.safe_load, default={}, help='if specified, conditions that samples must satisfy to be processed, given as yaml; if not specified, all samples will be processed')
    parser_parallel.add_argument('--stages', default='preprocess,align,categorize,generate_figures')
    parser_parallel.add_argument('--progress', const=tqdm.tqdm, action='store_const', help='show progress bars')
    parser_parallel.set_defaults(func=parallel)

    parser_table = subparsers.add_parser('table', help='generate tables of outcome frequencies')
    add_project_directory_arg(parser_table)
    parser_table.add_argument('--batches', help='if specified, a comma-separated list of batches to include; if not specified, all batches in project_directory will be generated')
    parser_table.add_argument('--title', default='knock_knock_table', help='if specified, a title for output files')
    parser_table.add_argument('--unsorted', action='store_true', help='don\'t sort samples')
    parser_table.add_argument('--arrayed', action='store_true', help='samples are organized as arrayed_experiment_groups')
    parser_table.add_argument('--vmax_multiple', type=float, default=1, help='fractional value that corresponds to full horizontal bar')
    parser_table.set_defaults(func=make_tables)

    parser_targets = subparsers.add_parser('build-targets', help='build annotations of target locii')
    add_project_directory_arg(parser_targets)
    parser_targets.add_argument('--defer_HA_identification', action='store_true', help='don\'t try to identiy homology arms')
    parser_targets.set_defaults(func=build_targets)

    parser_manual_target = subparsers.add_parser('build-manual-target', help='build a single target from a hand-annotated genbank file')
    add_project_directory_arg(parser_manual_target)
    parser_manual_target.add_argument('target_name', help='sample name')
    parser_manual_target.set_defaults(func=build_manual_target)

    parser_indices = subparsers.add_parser('build-indices', help='download a reference genome and build alignment indices')
    add_project_directory_arg(parser_indices)
    parser_indices.add_argument('genome_name', help='name of genome to download')
    parser_indices.add_argument('--num-threads', type=int, default=8, help='number of threads to use for index building')
    parser_indices.set_defaults(func=build_indices)

    parser_install_data = subparsers.add_parser('install-example-data', help='install example data into user-specified project directory')
    add_project_directory_arg(parser_install_data)
    parser_install_data.set_defaults(func=install_example_data)

    parser_citation = subparsers.add_parser('whos-there', help='print citation information')
    parser_citation.set_defaults(func=print_citation)

    parser_arrayed = subparsers.add_parser('arrayed', help='process using ArrayedExperimentGroups')
    parser_arrayed_subparsers = parser_arrayed.add_subparsers(dest='subcommand', title='subcommands')
    parser_arrayed_subparsers.required = True

    parser_arrayed_process = parser_arrayed_subparsers.add_parser('process', help='process a group')
    add_project_directory_arg(parser_arrayed_process)
    parser_arrayed_process.add_argument('batch_name')
    parser_arrayed_process.add_argument('group_name')
    parser_arrayed_process.add_argument('--max_procs', type=int, default=6)
    parser_arrayed_process.add_argument('--silent', action='store_true')
    parser_arrayed_process.add_argument('--generate_example_diagrams', action='store_true')
    parser_arrayed_process.set_defaults(func=process_arrayed)

    parser_arrayed_parallel = parser_arrayed_subparsers.add_parser('parallel', help='process groups in parallel')
    add_project_directory_arg(parser_arrayed_parallel)
    parser_arrayed_parallel.add_argument('batch_name')
    parser_arrayed_parallel.add_argument('--max_procs', type=int, default=10)
    parser_arrayed_parallel.add_argument('--max_procs_per_group', type=int, default=3)
    parser_arrayed_parallel.add_argument('--generate_example_diagrams', action='store_true')
    parser_arrayed_parallel.add_argument('--progress_bar', action='store_true')
    parser_arrayed_parallel.set_defaults(func=parallel_arrayed)

    args = parser.parse_args()

    args.func(args)