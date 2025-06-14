"""
-------------------------------------------------------
Module for analyzing feature-word associations in model outputs
-------------------------------------------------------
Author:  einsteinoyewole
ID:      eo2233@nyu.edu
__updated__ = "11/21/24"
-------------------------------------------------------
"""


# Imports
import torch
from typing import List, Tuple, Dict, Set, Any, Optional
from collections import defaultdict, Counter
import pickle
import os
from src.logger import getlogger
from src.model import SparseAutoencoder
from src.dataset import BaseActivationExtractor
from tqdm import tqdm
from itertools import islice

# Constants
log = getlogger(__name__, 'info')
NUM_BATCHES = 64

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class FeatureTracker:
    """
    -------------------------------------------------------
    Tracks feature-word associations across model outputs
    -------------------------------------------------------
    Parameters:
        model - Sparse Autoencoder model (SparseAutoencoder)
        extractor - Activation extractor for input data (BaseActivationExtractor)
        feature_activation_threshold - Activation threshold for feature selection (float)
        save_dir - Directory to save feature-word associations (str)
    -------------------------------------------------------
    """

    def __init__(self, model: SparseAutoencoder, extractor: BaseActivationExtractor, save_dir: str, feature_activation_threshold: Optional[float]= None):
        assert feature_activation_threshold is None or 0 <= feature_activation_threshold <= 1, "Feature threshold must be in [0, 1]"
        self.device = torch.device(get_device())
        self.model = model.to(self.device)
        self.extractor = extractor
        self.save_dir = save_dir
        self.feature_threshold = feature_activation_threshold
        if not os.path.exists(save_dir):
            log.debug(f"Creating directory {save_dir}")
            os.makedirs(save_dir)

        # Dictionary mapping feature indices to Counter objects of words
        self.feature_words = defaultdict(Counter)
        # Dictionary mapping words to Counter objects of features
        self.word_features = defaultdict(Counter)

    def process_prompt(self, prompts: List[str]) -> Dict[Any, Set[str]]:
        """
        -------------------------------------------------------
        Extracts feature-word associations from model outputs
        -------------------------------------------------------
        Parameters:
            prompts - Input prompt for model inference (str)
        Returns:
            Dictionary mapping feature indices to sets of associated words (Dict[str, Set[str]])
        -------------------------------------------------------
        """
        # Extract activations
        with torch.no_grad():
            output_extraction = self.extractor.extract_activations(prompts, top_k=1)
            activations = output_extraction['activations']
            tokens = output_extraction['top_tokens']

            # Get feature activations through autoencoder
            _, features = self.model(activations)

        log.debug(f"Features shape: {features.shape}; Feature norm stats: {features.norm(dim=1).mean():.4f}")

        if self.feature_threshold is None:
            sample_size = min(1000000, features.numel())
            sampled_features = features.flatten()[torch.randperm(features.numel())[:sample_size]]
            self.feature_threshold = torch.quantile(sampled_features, 0.75)
            log.debug(f'{self.feature_threshold=} {torch.mean(sampled_features)=} {torch.min(sampled_features)=} {torch.max(sampled_features)=}')
        
        # # Skip if the quantile threshold equals the mean (indicates all values are the same)
        # log.debug(f'{self.feature_threshold=} <= {features.mean()=}')
        # if torch.all(torch.isclose(self.feature_threshold, features.mean(), atol=1e-4)):
        #     log.warning("Skipping - there are no learned features")
        #     return defaultdict(list)
    
        # Track associations for significantly active features
        new_associations = defaultdict(list)
        
        # Process each item in the batch
        batch_size = features.size(0)
        for batch_idx in range(batch_size):
            # Get features for this batch item
            batch_features = features[batch_idx].t()  # Shape: [sequence_length, hidden_dim]
            log.debug(f"Feature activation stats: min={batch_features.min():.4f}, max={batch_features.max():.4f}, mean={batch_features.mean():.4f}")
            
            # Decode tokens for this batch
            batch_tokens = tokens[batch_idx].tolist()
            batch_words = self.extractor.decode_tokens(batch_tokens)
            
            # Make sure we only process up to the number of words we have
            seq_length = min(batch_features.shape[0], len(batch_words))
            
            # Iterate over each word
            for word_idx in range(seq_length):
                word_features = batch_features[word_idx]
                # Get features that are significantly active for this word
                active_features = torch.where(word_features > self.feature_threshold)[0]
                log.debug(f"{batch_words[word_idx]=} {active_features=} {word_features=}")
                word = batch_words[word_idx]
                
                # Update counters for each active feature
                for feature_idx in active_features:
                    feature_idx = feature_idx.item()
                    self.feature_words[feature_idx][word] += 1
                    self.word_features[word][feature_idx] += 1
                    new_associations[feature_idx].append(word)

        return dict(new_associations)

    def get_top_words_for_feature(self, feature_idx: int, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        -------------------------------------------------------
        Get the top words associated with a feature
        -------------------------------------------------------
        Parameters:
            feature_idx - Index of the feature (int)
            top_k - Number of top words to return (int)
        Returns:
            List of top words associated with the feature (List[Tuple[str, int]])
        -------------------------------------------------------
        """
        assert feature_idx in self.feature_words, f"No data for feature {feature_idx}"
        assert top_k > 0, "Top_k must be positive"
        return self.feature_words[feature_idx].most_common(top_k)

    def get_top_features_for_word(self, word: str, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        -------------------------------------------------------
        Get the top features associated with a word
        -------------------------------------------------------
        Parameters:
            word - Word to get top features for (str)
            top_k - Number of top features to return (int)
        Returns:
            List of top features associated with the word (List[Tuple[int, int]])
        -------------------------------------------------------
        """
        assert word in self.word_features, f"No data for word '{word}'"
        assert top_k > 0, "Top_k must be positive"

        return self.word_features[word].most_common(top_k)

    def save_state(self, filename: str) -> None:
        """
        -------------------------------------------------------
        Save the current state of the feature tracker
        -------------------------------------------------------
        Parameters:
            filename - Name of the file to save the state to (str)
        -------------------------------------------------------
        """
        save_path = os.path.join(self.save_dir, filename)
        log.debug(f"Saving state to {save_path}")
        state = {
            'feature_words': dict(self.feature_words),
            'word_features': dict(self.word_features),
            'feature_threshold': self.feature_threshold
        }
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filename: str) -> None:
        """
        -------------------------------------------------------
        Load a saved state of the feature tracker
        -------------------------------------------------------
        Parameters:
            filename - Name of the file to load the state from (str)
        -------------------------------------------------------
        """
        load_path = os.path.join(self.save_dir, filename)
        log.debug(f"Loading state from {load_path}")
        if not os.path.exists(load_path):
            log.warning(f"File {load_path} does not exist")
            return

        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.feature_words = defaultdict(Counter, state['feature_words'])
        self.word_features = defaultdict(Counter, state['word_features'])
        self.feature_threshold = state['feature_threshold']

    def get_interesting_features(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """
        -------------------------------------------------------
        Get features that are associated with the most amount of unique words
        -------------------------------------------------------
        Parameters:
            top_k: Number of top features to return
        Returns:
            List of (feature_idx, unique_word_count) tuples
        -------------------------------------------------------
        """
        log.debug(f"Getting top {top_k} interesting features from {len(self.feature_words)} total features")
        # Sort features by number of unique words they activate for
        sorted_features = sorted(
            self.feature_words.items(),
            key=lambda x: len(x[1]),  # x[1] is the key i.e. Counter object
            reverse=True  # Most words first
        )

        # Return top_k features and their word counts
        results = [(feature_idx, len(word_counter))
                for feature_idx, word_counter in sorted_features[:top_k]]

        if results:
            log.debug(f"Top feature has {results[0][1]} unique words")
        return results

    def get_specific_words(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        -------------------------------------------------------
        Get words that are associated with the fewest unique features
        (potentially more semantically focused/specific words)
        -------------------------------------------------------
        Parameters:
            top_k: Number of top words to return
        Returns:
            List of (word, feature_count) tuples
        -------------------------------------------------------
        """
        log.debug(f"Getting top {top_k} specific words from {len(self.word_features)} total words")
        # Sort words by number of unique features they activate
        sorted_words = sorted(
            self.word_features.items(),
            key=lambda x: len(x[1]),
            reverse=False  # Fewest features first
        )

        # Return top_k words and their feature counts
        results = [(word, len(feature_counter))
                for word, feature_counter in sorted_words[:top_k]]

        if results:
            log.debug(f"Top word activates {results[0][1]} unique features")

        return results

    def print_feature_analysis(self, feature_idx: int, top_k: int = 10) -> None:
        """
        -------------------------------------------------------
        Print detailed analysis of a specific feature
        -------------------------------------------------------
        Parameters:
            feature_idx: Index of the feature to analyze
            top_k: Number of top words to show
        -------------------------------------------------------
        """
        if feature_idx not in self.feature_words:
            log.warn(f"No data for feature {feature_idx}")
            return

        word_counter = self.feature_words[feature_idx]
        total_words = len(word_counter)
        total_activations = sum(word_counter.values())

        log.info(f"\nAnalysis for Feature {feature_idx}:")
        log.info(f"Total unique words: {total_words}")
        log.info(f"Total activations: {total_activations}")
        log.info(f"\nTop {top_k} most common words:")
        for word, count in word_counter.most_common(top_k):
            log.info(f"  {word}: {count} times")

    def print_word_analysis(self, word: str, top_k: int = 10) -> None:
        """
        -------------------------------------------------------
        Print detailed analysis of a specific word
        -------------------------------------------------------
        Parameters:
            word: Word to analyze
            top_k: Number of top features to show
        -------------------------------------------------------
        """
        if word not in self.word_features:
            log.warn(f"No data for word '{word}'")
            return

        feature_counter = self.word_features[word]
        total_features = len(feature_counter)
        total_activations = sum(feature_counter.values())

        log.info(f"\nAnalysis for word '{word}':")
        log.info(f"Total unique features: {total_features}")
        log.info(f"Total activations: {total_activations}")
        log.info(f"\nTop {top_k} most active features:")
        for feature_idx, count in feature_counter.most_common(top_k):
            log.info(f"  Feature {feature_idx}: {count} times")

    def analyze(self, dataloader: torch.utils.data.DataLoader, text_column: str):
        """
        -------------------------------------------------------
        Analyze feature-word associations in a dataset
        -------------------------------------------------------
        Parameters:
            dataloader - DataLoader for the dataset to analyze
        -------------------------------------------------------
        """
        log.info("Starting feature analysis on dataset")
        with torch.no_grad():
            for batch in tqdm(islice(dataloader, NUM_BATCHES)):
                # Convert batch items to list of texts
                # log.debug(f'{batch=}')
                texts = batch[text_column]
                self.process_prompt(texts)


        log.info("Completed feature analysis on dataset")
