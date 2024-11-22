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
from src.parser import parse_args

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
data_path_map = {
    'wikitext': 'Salesforce/wikitext',
    'commoncrawl': 'allenai/c4',
}
data_name_map = {
    'wikitext': 'wikitext-103-raw-v1',
    'commoncrawl': 'en',
}


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
        dataset_config=data_path_map[args.data_model],
        model_name=model_name_map[args.llm_model],
        text_column="text",
        use_flash_attention=False
    )
    extractor = extractor_map[args.llm_model](data_config)

    # Define autoencoder configuration and model
    activation_dim = extractor._get_final_layer().normalized_shape[0]
    log.info(f'Activation dimension: {activation_dim}')
    model_config = AutoencoderConfig(
        input_dim=activation_dim,
        hidden_dim=activation_dim * args.over_complete_size,
    )
    model = SparseAutoencoder(model_config)

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
            schedule_type=args.l1_coefficient_scheduler,
            warmup_epochs=args.warmup_epochs
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
        trainer = MonosemanticityTrainer(model, optimizer=optimizer, extractor=extractor, train_config=trainer_config)

        # Train the model
        results = trainer.train(dataloader, validation_dataloader)
        plot_training_metrics(results, args.run_path)

    elif args.execution_mode == "analyze":
        # Configure test data
        test_config = deepcopy(data_config)
        test_config.split = "validation"  # Use smaller subset for analysis
        test_dataset = TextDataset(extractor.tokenizer, test_config)
        test_dataloader = test_dataset.get_dataloader(batch_size=args.batch_size)

        log.info('Analyzing the model')
        # load model
        model.load_state_dict(torch.load(os.path.join(args.run_path, 'model.pkl')))
        model.eval()

        # Define feature tracker
        tracker = FeatureTracker(model, extractor, args.run_path)

        # Load state
        tracker.load_state("feature_analysis.pkl")

        tracker.analyze(test_dataloader)

        # Get features that activate for many different words
        interesting_features = tracker.get_interesting_features(top_k=10)
        for feature_idx, word_count in interesting_features:
            log.info(f"Feature {feature_idx} activates for {word_count} different words")
            tracker.print_feature_analysis(feature_idx)
            log.info('-' * 25)

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
