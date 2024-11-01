# src/main.py

import logging
import os
from src.standardize_descriptions import StandardizeDescriptions
import yaml
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file.
    
    Parameters:
        config_path (str): Path to the configuration file.
    
    Returns:
        config (dict): Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set file paths from config
    user_desc_path = config['user_desc_path']
    standard_desc_path = config['standard_desc_path']
    output_path = config['output_path']
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize StandardizeDescriptions
    standardizer = StandardizeDescriptions(
        model_name=config['model_name'],
        threshold=config['similarity_threshold']
    )
    
    # Perform standardization
    logger.info("Starting standardization process...")
    
    # Load the user and standard descriptions from the same file
    standard_df = pd.read_csv(standard_desc_path)
    user_descriptions = standard_df['user_description'].tolist()
    standard_descriptions = standard_df['standard_description'].tolist()
    
    # Standardize descriptions
    standardized_df = standardizer.standardize(user_descriptions, standard_descriptions)
    
    # Save output to CSV
    output_file = os.path.join(output_path, 'standardized_descriptions.csv')
    standardized_df.to_csv(output_file, index=False)
    logger.info(f"Standardized descriptions saved to {output_file}")

    # Print or inspect the result
    print("Standardization complete. Results:")
    print(standardized_df)

if __name__ == "__main__":
    main()