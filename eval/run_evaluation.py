import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add parent directory to path so we can import src.qabot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qabot import process_pdf_and_query

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_dir = os.path.join(base_dir, 'test_set')
    qa_file = os.path.join(test_set_dir, 'qa_pairs.json')
    raw_results_file = os.path.join(base_dir, 'results_raw.json')
    failures_file = os.path.join(base_dir, 'failures.json')

    if not os.path.exists(qa_file):
        print(f"Error: {qa_file} not found.")
        return

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    results = []
    failures = []

    print(f"Starting evaluation of {len(qa_pairs)} questions...")

    for i, qa in enumerate(qa_pairs):
        pdf_path = os.path.join(test_set_dir, qa['pdf_file'])
        question = qa['question']
        print(f"[{i+1}/{len(qa_pairs)}] Processing {qa['id']} on {qa['pdf_file']}...")
        
        start_time = time.time()
        try:
            # We assume process_pdf_and_query works as a pure function.
            # Note: in qabot.py, it creates the vector store per call.
            prediction = process_pdf_and_query(pdf_path, question)
            latency = time.time() - start_time
            
            # If the pipeline returned a string starting with "Error", it failed internally
            if prediction.startswith("Error processing your request:"):
                raise Exception(prediction)
            
            results.append({
                "id": qa['id'],
                "pdf_file": qa['pdf_file'],
                "question": question,
                "ground_truth": qa['ground_truth'],
                "prediction": prediction,
                "latency_seconds": latency
            })
            print(f"  -> Success in {latency:.2f}s")
        except Exception as e:
            latency = time.time() - start_time
            print(f"  -> Failed: {str(e)}")
            failures.append({
                "id": qa['id'],
                "pdf_file": qa['pdf_file'],
                "question": question,
                "ground_truth": qa['ground_truth'],
                "error": str(e),
                "latency_seconds": latency
            })

    # Save results
    with open(raw_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    if failures:
        with open(failures_file, 'w', encoding='utf-8') as f:
            json.dump(failures, f, indent=4)
        print(f"\nEvaluation complete. {len(results)} successes, {len(failures)} failures.")
    else:
        # Clear failures file if it existed
        if os.path.exists(failures_file):
            os.remove(failures_file)
        print(f"\nEvaluation complete. {len(results)} successes, 0 failures.")
        
    print(f"Results saved to {raw_results_file}")

if __name__ == "__main__":
    main()
