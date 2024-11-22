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
from src.feature_analysis import FeatureTracker

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
    assert l1_coefficient_end >= l1_coefficient_start > 0, "L1 coefficient start must be a positive float and smaller than L1 coefficient end"

    # Validate L1 coefficient scheduler
    l1_coefficient_scheduler = args.l1_coefficient_scheduler
    assert l1_coefficient_scheduler in {'linear', 'exponential', 'cosine'}, "L1 coefficient scheduler must be 'linear' or 'exponential' or 'cosine'"

    # Validate warmup epochs
    warmup_epochs = args.warmup_epochs
    assert max_epochs >= warmup_epochs > 0 and isinstance(warmup_epochs, int), "Warmup epochs must be a positive integer"

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
    elif args.execution_mode == "analyze":
        # Configure test data
        test_config = deepcopy(data_config)
        test_config.split = "test[:10%]"  # Use smaller subset for analysis
        test_dataset = TextDataset(extractor.tokenizer, test_config)
        test_dataloader = test_dataset.get_dataloader(batch_size=args.batch_size)
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
            learning_rate=args.learning_rate,
            l1_coefficient_start=args.l1_coefficient_start,
            l1_coefficient_end=args.l1_coefficient_end,
            l1_coefficient_scheduler=args.l1_coefficient_scheduler,
            warmup_epochs=args.warmup_epochs
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
        trainer = MonosemanticityTrainer(model, optimizer=optimizer, extractor=extractor, train_config=trainer_config)

        # Train the model
        results = trainer.train(dataloader, validation_dataloader)
        plot_training_metrics(results, args.run_path)
    elif args.execution_mode == "analyze":
        log.info('Analyzing the model')
        # load model
        model.load_state_dict(torch.load(os.path.join(args.run_path, 'model.pkl')))
        model.eval()

        # Define feature tracker
        tracker = FeatureTracker(model, extractor, args.run_path)

        # Load state
        tracker.load_state("feature_analysis.pkl")

        tracker.analyze(dataloader)

        # Get features that activate for many different words
        interesting_features = tracker.get_interesting_features(top_k=10)
        for feature_idx, word_count in interesting_features:
            log.info(f"Feature {feature_idx} activates for {word_count} different words")
            tracker.print_feature_analysis(feature_idx)
            log.info('-'*25)

        # Get specific words that activate few features
        specific_words = tracker.get_specific_words(top_k=10)
        for word, feature_count in specific_words:
            log.info(f"Word '{word}' activates {feature_count} different features")
            tracker.print_word_analysis(word)
            log.info('-' * 25)

        # Save state
        tracker.save_state("feature_analysis.pkl")
    else:
        # Evaluate the model
        log.info('Evaluating the model')
        log.error('Evaluation mode not supported yet')
        pass


if __name__ == '__main__':
    main()
