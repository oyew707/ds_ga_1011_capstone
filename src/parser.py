"""
-------------------------------------------------------
File for parsing command line arguments.
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "11/22/24"
-------------------------------------------------------
"""

# Imports
import argparse
import os
from src.logger import getlogger

# Constants
log = getlogger(__name__, 'info')


def validate_args(args):
    """
    -------------------------------------------------------
    Validates the parsed arguments from the command line
    -------------------------------------------------------
    Parameters:
       args - Parsed arguments from command line (argparse.Namespace)
    -------------------------------------------------------
    """
    log.debug(f'Validating arguments: {args}')
    # Validate random seed
    random_seed = args.random_seed
    assert random_seed >= 0 and isinstance(random_seed, int), "Random seed must be a positive integer"

    # Validate maximum epochs
    max_epochs = args.maximum_epochs
    assert max_epochs > 0 and isinstance(max_epochs, int), "Maximum epochs must be a positive integer"

    # Validate batch size
    batch_size = args.batch_size
    assert batch_size > 0 and isinstance(batch_size, int), "Batch size must be a positive integer"

    # Validate llm extraction model
    llm_model = args.llm_model
    assert llm_model in ['gpt', 'gemma', 'llama'], "LLM model must be 'gpt' or 'gemma' or 'llama'"

    # Validate execution mode
    execution_mode = args.execution_mode
    assert execution_mode in {'train', 'evaluate', 'analyze'}, "Execution mode must be 'train' or 'evaluate'"
    assert execution_mode == 'train', "Only training mode is supported at this time"

    # Validate data model
    data_model = args.data_model
    assert data_model in {'wikitext'}, "Data model must be one of 'wikitext'"

    # Validate Overcomplete size
    overcomplete_size = args.over_complete_size
    assert 256 > overcomplete_size > 0 and isinstance(overcomplete_size,
                                                      int), "Overcomplete size must be a positive integer less than 256"

    # Validate L1 coefficient
    l1_coefficient_start = args.l1_coefficient_start
    l1_coefficient_end = args.l1_coefficient_end
    assert l1_coefficient_end >= l1_coefficient_start > 0, ("L1 coefficient start must be a positive float and smaller "
                                                            "than L1 coefficient end")

    # Validate L1 coefficient scheduler
    l1_coefficient_scheduler = args.l1_coefficient_scheduler
    assert l1_coefficient_scheduler in {'linear', 'exponential',
                                        'cosine'}, ("L1 coefficient scheduler must be 'linear' or 'exponential' or "
                                                    "'cosine'")

    # Validate warmup epochs
    warmup_epochs = args.warmup_epochs
    assert max_epochs >= warmup_epochs > 0 and isinstance(warmup_epochs,
                                                          int), "Warmup epochs must be a positive integer"


def parse_args() -> argparse.Namespace:
    """
    -------------------------------------------------------
    Defines Argument parser from command line arguments
    -------------------------------------------------------
    Returns:
       args - Parsed arguments from command line (argparse.Namespace)
    -------------------------------------------------------
    """
    # Define argument parser
    parser = argparse.ArgumentParser()

    # Add parsing arguments
    parser.add_argument(
        '-em', '--execution-mode',
        type=str, default='train')
    parser.add_argument(
        '-rn', '--run-name',
        type=str, default='default')
    parser.add_argument(
        '-lm', '--llm-model',
        type=str, default='gpt')
    parser.add_argument(
        '-dm', '--data-model',
        type=str, default='wikitext')
    parser.add_argument(
        '-bs', '--batch-size',
        type=int, default=64)
    parser.add_argument(
        '-lr', '--learning-rate',
        type=float, default=1e-4)
    parser.add_argument(
        '-me', '--maximum-epochs',
        type=int, default=256)
    parser.add_argument(
        '-op', '--output-path',
        type=str, default='output')
    parser.add_argument(
        '-rs', '--random-seed',
        type=int, default=1234)
    parser.add_argument(
        '-ocs', '--over-complete-size',
        type=int, default=5)
    parser.add_argument(
        '-lcs', '--l1-coefficient-start',
        type=float, default=1e-4)
    parser.add_argument(
        '-lce', '--l1-coefficient-end',
        type=float, default=1)
    parser.add_argument(
        '-lcs', '--l1-coefficient-scheduler',
        type=str, default='linear')
    parser.add_argument(
        '-we', '--warmup-epochs',
        type=int, default=64)

    # Get parsed arguments
    args = parser.parse_args()

    validate_args(args)

    # Define run path
    run_path = os.path.join(args.output_path, args.run_name)
    setattr(args, 'run_path', run_path)

    return args
