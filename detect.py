import argparse
import json
import pandas as pd
from models import Detector
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
    parser = argparse.ArgumentParser(description='Detect AI-generated text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text(s) to analyze')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06', help='Model to use for detection')
    parser.add_argument('--key_file', type=str, default="", help='File containing API key')
    parser.add_argument('--api_key', type=str, default="", help='API key')
    parser.add_argument('--no_explain', action='store_false', help='Disable explanation generation')
    parser.add_argument('--no_guide', action='store_false', help='Disable detection guide')
    parser.add_argument('--output', type=str, default="./results.csv", help='Output file path for results (CSV)')
    args = parser.parse_args()

    # Initialize detector
    detector = Detector(
        llm=args.model,
        explain=args.no_explain,
        key_file=args.key_file,
        api_key=args.api_key,
        guide=args.no_guide
    )

    # Get data to analyze
    if args.text:
        df = pd.DataFrame({'text': [args.text]})
    elif args.file:
        df = load_data(args.file)
    else:
        print("Please provide either --text or --file argument")
        return

    # Initialize results lists
    classifications = []
    explanations = []

    # Process each text
    print("\nAnalyzing texts...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        
        # Clean text if it's from a text file
        if args.file and args.file.endswith('.txt'):
            title = args.file.split('/')[-1].replace('.txt', '')
            subtitle = ""  # You can add subtitle handling if needed
            text = clean_article_with_headings(title, subtitle, text)

        # Detect
        answer, explanation = detector.detect(text)
        classifications.append(answer)
        explanations.append(explanation)

        # Print results for single text
        if len(df) == 1:
            print("\nDetection Results:")
            print("-" * 50)
            print(f"Classification: {answer}")
            if explanation:
                print("\nExplanation:")
                print(explanation)

    # Add results to DataFrame
    df['classification'] = classifications
    if not args.no_explain:
        df['explanation'] = explanations

    # Save results if output file specified
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    elif len(df) > 1:
        # Print summary for multiple texts
        print("\nSummary:")
        print("-" * 50)
        print(f"Total texts analyzed: {len(df)}")
        print("\nClassification distribution:")
        print(df['classification'].value_counts())
        print("\nDetailed results:")
        for idx, row in df.iterrows():
            print(f"\nText {idx + 1}:")
            print(f"Classification: {row['classification']}")
            if not args.no_explain:
                print(f"Explanation: {row['explanation']}")

if __name__ == "__main__":
    main() 