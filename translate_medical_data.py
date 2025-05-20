import os
from datasets import load_dataset
import pandas as pd
import logging
from tqdm import tqdm  # For progress bar
from dotenv import load_dotenv
from googletrans import Translator
import time

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def translate_dataset(num_samples=10, output_dir=None):
    """
    Translate English medical text to Thai and save results to CSV.
    
    Args:
        num_samples: Number of samples to translate
        output_dir: Directory to save output (if None, uses current directory)
    """
    try:
        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Loading first {num_samples} samples from the dataset...")
        # Load samples from the dataset
        dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split=f"train[:{num_samples}]")
        
        # Use the 'Question' field for translation
        texts = dataset["Question"]
        
        logging.info("Loading the English-to-Thai translation model...")
        # Initialize Google Translator
        logging.info("Initializing Google Translator...")
        translator = Translator()
        
        # Translate texts
        logging.info("Translating texts...")
        translated_texts = []
        
        # Process one by one with delay to avoid rate limiting
        for i in tqdm(range(len(texts))):
            text = texts[i]
            
            # Skip very long texts to avoid translator issues
            if len(text) > 5000:
                logging.warning(f"Text at index {i} is too long ({len(text)} chars). Truncating.")
                text = text[:5000]
                
            try:
                # Translate with retries
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        translated = translator.translate(text, src='en', dest='th')
                        translated_texts.append(translated.text)
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            logging.warning(f"Translation retry {retry+1}/{max_retries}: {str(e)}")
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise e
                            
                # Add a small delay to avoid hitting rate limits
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error translating text {i}: {str(e)}")
                # Add empty translation for failed texts to maintain alignment
                translated_texts.append(f"[ERROR: {str(e)}]")
                time.sleep(5)  # Longer wait after an error
        
        # Create DataFrame and save to CSV
        logging.info("Creating DataFrame...")
        df = pd.DataFrame({
            "original_question_en": texts,
            "translated_question_th": translated_texts,
            "complex_cot": dataset["Complex_CoT"],
            "response": dataset["Response"]
        })
        
        output_path = os.path.join(output_dir, "translated_medical_o1_input_en_to_th.csv")
        logging.info(f"Saving results to {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return output_path
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    output_path = translate_dataset(num_samples=10, output_dir="./output")
    print(f"Translation complete. Results saved to: {output_path}")
