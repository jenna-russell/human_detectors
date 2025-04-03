import argparse
import pandas as pd
from models import Evader, Detector
from helpers import clean_article_with_headings
from tqdm import tqdm

def load_data(file_path):
    """Load data from various file formats into a DataFrame"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        # For text files, read as single text
        with open(file_path, 'r') as f:
            text = f.read()
        df = pd.DataFrame({'text': [text]})
    
    # Ensure DataFrame has a 'text' column
    if 'text' not in df.columns:
        raise ValueError("Input file must contain a 'text' column or be a text file")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Test evasion techniques on AI-generated text')
    parser.add_argument('--text', type=str, help='Text to evade detection')
    parser.add_argument('--file', type=str, help='File containing text(s) to evade')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06', help='Model to use for evasion')
    parser.add_argument('--key_file', type=str, default="", help='File containing API key')
    parser.add_argument('--api_key', type=str, default="", help='API key')
    parser.add_argument('--publication', type=str, required=True, help='Target publication style')
    parser.add_argument('--examples', type=str, help='File containing example texts')
    parser.add_argument('--test_detection', action='store_true', help='Test detection after evasion')
    parser.add_argument('--output', type=str, default="./evaded_results.csv", help='Output file path for results (CSV)')
    args = parser.parse_args()

    # Initialize evader
    evader = Evader(
        llm=args.model,
        key_file=args.key_file,
        api_key=args.api_key
    )

    # Get data to evade
    if args.text:
        df = pd.DataFrame({'text': [args.text]})
    elif args.file:
        df = load_data(args.file)
    else:
        print("Please provide either --text or --file argument")
        return

    # Load examples if provided
    examples = None
    if args.examples:
        with open(args.examples, 'r') as f:
            examples = f.read()

    # Initialize results lists
    evaded_texts = []

    # Process each text
    print("\nEvading detection...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
       
        # Evade
        evaded_text = evader.evade(text, args.publication, examples)
        evaded_texts.append(evaded_text)

        # Print results for single text
        if len(df) == 1:
            print("\nEvaded Text:")
            print("-" * 50)
            print(evaded_text)

    # Add results to DataFrame
    df['evaded_text'] = evaded_texts

    # Test detection if requested
    if args.test_detection:
        print("\nTesting detection on evaded texts...")
        detector = Detector(
            llm=args.model,
            key_file=args.key_file,
            api_key=args.api_key
        )
        
        classifications = []
        explanations = []
        
        for text in tqdm(evaded_texts, desc="Testing detection"):
            answer, explanation = detector.detect(text)
            classifications.append(answer)
            explanations.append(explanation)
        
        df['classification'] = classifications
        df['explanation'] = explanations

        # Print summary for multiple texts
        if len(df) > 1:
            print("\nDetection Results Summary:")
            print("-" * 50)
            print(f"Total texts analyzed: {len(df)}")
            print("\nClassification distribution:")
            print(df['classification'].value_counts())

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 