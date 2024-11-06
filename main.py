"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "10/28/24"
-------------------------------------------------------
"""

# Imports
import argparse
import os
import random

import numpy as np
import torch
from copy import deepcopy
from src.dataset import (
    TextDataset,
    DataConfig,
    GPT2ActivationExtractor,
    GemmaActivationExtractor,
    LlamaActivationExtractor)
from src.evaluation import plot_training_metrics
from src.logger import getlogger
from src.model import AutoencoderConfig, SparseAutoencoder
from src.trainer import MonosemanticityTrainer, TrainingConfig

# Constants
log = getlogger(__name__, 'info')
extractor_map = {
    'gpt': GPT2ActivationExtractor,
    'gemma': GemmaActivationExtractor,
    'llama': LlamaActivationExtractor
}
model_name_map = {
    'gpt': 'gpt2',
    'gemma': 'google/gemma-2-2b',
    'llama': 'meta-llama/Llama-3.2-1B'
}
data_name_map = {
    'wikitext': 'RealTimeData/wikitext_latest',
}


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
    assert execution_mode in {'train', 'evaluate'}, "Execution mode must be 'train' or 'evaluate'"
    assert execution_mode == 'train', "Only training mode is supported at this time"

    # Validate data model
    data_model = args.data_model
    assert data_model in {'wikitext'}, "Data model must be one of 'wikitext'"

    # Validate Overcomplete size
    overcomplete_size = args.over_complete_size
    assert 256 > overcomplete_size > 0 and isinstance(overcomplete_size,
                                                      int), "Overcomplete size must be a positive integer less than 256"


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

    # Get parsed arguments
    args = parser.parse_args()

    validate_args(args)

    # Define run path
    run_path = os.path.join(args.output_path, args.run_name)
    setattr(args, 'run_path', run_path)

    return args


def main():
    """
    -------------------------------------------------------
    Main function for training and evaluating the sparse autoencoder model
    -------------------------------------------------------
    """
    # Parse arguments
    log.debug('Parsing arguments')
    args = parse_args()

    # set random seed
    log.debug(f'Setting random seed: {args.random_seed}')
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Define data configuration and dataset
    data_config = DataConfig(
        batch_size=args.batch_size,
        dataset_name=data_name_map[args.data_model],
        model_name=model_name_map[args.llm_model],
        text_column="text",
        use_flash_attention=False
    )
    extractor = extractor_map[args.llm_model](data_config)
    if args.execution_mode == "train":
        # training data
        train_config = deepcopy(data_config)
        train_config.split = "train[:90%]"
        dataset = TextDataset(extractor.tokenizer, train_config)
        dataloader = dataset.get_dataloader(batch_size=args.batch_size)
        # validation data
        validation_config = deepcopy(data_config)
        validation_config.split = "train[90%:]"
        validation_dataset = TextDataset(extractor.tokenizer, validation_config)
        validation_dataloader = validation_dataset.get_dataloader(batch_size=args.batch_size)
    else:
        log.error('Evaluation mode not supported yet')
        raise NotImplementedError

    # Define autoencoder configuration and model
    activation_dim = extractor._get_final_layer().normalized_shape[0]
    log.info(f'Activation dimension: {activation_dim}')
    model_config = AutoencoderConfig(
        input_dim=activation_dim,
        hidden_dim=activation_dim * args.over_complete_size,
    )
    model = SparseAutoencoder(model_config)

    if args.execution_mode == "train":
        log.info('Training the model')
        # Define training configuration and optimizer
        trainer_config = TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.maximum_epochs,
            mixed_precision='fp16',
            run_path=args.run_path,
            learning_rate=args.learning_rate
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
        trainer = MonosemanticityTrainer(model, optimizer=optimizer, extractor=extractor, train_config=trainer_config)

        # Train the model
        results = trainer.train(dataloader, validation_dataloader)
        plot_training_metrics(results, args.run_path)
    else:
        # Evaluate the model
        log.info('Evaluating the model')
        log.error('Evaluation mode not supported yet')
        pass


if __name__ == '__main__':
    main()
