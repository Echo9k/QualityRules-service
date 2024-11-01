# src/standardize_descriptions.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging

# Import NLTK and download necessary resources
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re


# Initialize logging
logger = logging.getLogger(__name__)

class StandardizeDescriptions:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.6):
        """
        Initialize the StandardizeDescriptions class with a lower threshold.
        
        Parameters:
            model_name (str): The name of the transformer model to use for embeddings.
            threshold (float): Similarity threshold for matching descriptions.
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        logger.info(f"Model {model_name} loaded successfully with threshold {threshold}.")

    def preprocess_text(self, text):
        """
        Preprocess text by lowering case, removing punctuation, lemmatizing, and removing stopwords.
        
        Parameters:
            text (str): The text to preprocess.
        
        Returns:
            str: Preprocessed text.
        """
        text = text.lower()  # Lowercase
        text = re.sub(r'\W+', ' ', text)  # Remove punctuation
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words])
        return text

    def standardize(self, user_descriptions, standard_descriptions):
        """
        Standardize user descriptions by finding their closest standard description.
        
        Parameters:
            user_descriptions (list): List of user descriptions.
            standard_descriptions (list): List of standard descriptions.
        
        Returns:
            matched_df (DataFrame): DataFrame with user descriptions, matched standard descriptions, and similarity scores.
        """
        # Preprocess descriptions
        user_descriptions = [self.preprocess_text(desc) for desc in user_descriptions]
        standard_descriptions = [self.preprocess_text(desc) for desc in standard_descriptions]
        
        # Generate embeddings
        user_embeddings = self.generate_embeddings(user_descriptions)
        standard_embeddings = self.generate_embeddings(standard_descriptions)
        
        # Match user descriptions to standard descriptions
        matches = []
        for idx, user_embedding in enumerate(user_embeddings):
            user_desc = user_descriptions[idx]
            best_match, score = self.find_best_match(user_embedding, standard_embeddings, standard_descriptions)
            
            # Debugging: Print scores for non-matching descriptions
            if best_match is None:
                logger.debug(f"No match for '{user_desc}'.")
                for std_idx, std_embedding in enumerate(standard_embeddings):
                    similarity = util.cos_sim(user_embedding, std_embedding)[0][0].item()
                    logger.debug(f"Similarity with '{standard_descriptions[std_idx]}': {similarity:.2f}")
            
            matches.append((user_desc, best_match, score))
        
        # Convert results to a DataFrame
        matched_df = pd.DataFrame(matches, columns=['User Description', 'Standard Description', 'Similarity Score'])
        logger.info("Standardization completed successfully.")
        
        return matched_df