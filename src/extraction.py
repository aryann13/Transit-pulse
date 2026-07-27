import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Define valid categories so LLM maps to our expected taxonomy
VALID_CATEGORIES = [
    "AC Failure", "Cleanliness", "Staff Behaviour", 
    "Food Quality", "Water Availability", "Overcrowding", 
    "Delay-related", "Other"
]

SYSTEM_PROMPT = f"""You are an intelligent Named Entity Recognition (NER) system for Indian Railways.
Your task is to extract specific information from the following passenger complaint.

Categories must strictly be one of: {', '.join(VALID_CATEGORIES)}
If an entity is not explicitly mentioned, return null.

Return STRICTLY a JSON object with these exact keys:
- "extracted_train_number": (string or null) The train number.
- "extracted_coach_number": (string or null) The coach identifier (e.g., A1, B2, S5).
- "extracted_issue_category": (string or null) The closest matching category from the allowed list.
"""


def extract_single(client, complaint_text):
    """Sends one complaint to the LLM and returns the parsed JSON dict, or None on failure."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Complaint: '{complaint_text}'"}
            ],
            temperature=0.0,  # 0.0 for maximum consistency in extraction
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        print(f"  LLM error: {e}")
        return None


def is_match(true_val, pred_val):
    """Helper to check match (handling case-insensitivity and None)."""
    if pd.isna(true_val) or true_val == 'None' or true_val is None:
        return pd.isna(pred_val) or pred_val == 'None' or pred_val is None
    if pd.isna(pred_val) or pred_val == 'None' or pred_val is None:
        return False
        
    str_true = str(true_val).strip().lower()
    str_pred = str(pred_val).strip().lower()
    
    # Pandas often converts integer columns with NaNs to floats (e.g. 12951.0 instead of 12951)
    if str_true.endswith('.0'):
        str_true = str_true[:-2]
        
    return str_true == str_pred


def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        print("ERROR: Please set a valid GROQ_API_KEY in your .env file.")
        return

    client = Groq(api_key=api_key)

    input_file = 'data/synthetic_complaints.csv'
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run Phase 1 first.")
        return

    print("Loading synthetic data...")
    df = pd.read_csv(input_file)

    # ================================================================
    # STEP 1: VALIDATION SAMPLE — 20-row accuracy check (unchanged)
    # ================================================================
    sample_size = min(20, len(df))
    test_df = df.sample(n=sample_size, random_state=42).copy()
    
    print(f"\n{'='*60}")
    print(f"STEP 1: Validation — LLM Extraction on {sample_size} sample complaints")
    print(f"{'='*60}\n")

    validation_results = []

    for index, row in test_df.iterrows():
        complaint_text = row['complaint_text']
        parsed = extract_single(client, complaint_text)
        
        if parsed:
            validation_results.append({
                "complaint_id": row['complaint_id'],
                "complaint_text": complaint_text,
                "true_train_number": str(row['true_train_number']) if pd.notna(row['true_train_number']) else None,
                "true_coach_number": str(row['true_coach_number']) if pd.notna(row['true_coach_number']) else None,
                "true_issue_category": str(row['true_issue_category']) if pd.notna(row['true_issue_category']) else None,
                "extracted_train_number": str(parsed.get('extracted_train_number')) if parsed.get('extracted_train_number') else None,
                "extracted_coach_number": str(parsed.get('extracted_coach_number')) if parsed.get('extracted_coach_number') else None,
                "extracted_issue_category": str(parsed.get('extracted_issue_category')) if parsed.get('extracted_issue_category') else None,
            })
            print(f"  Validated {row['complaint_id']}")
        else:
            print(f"  FAILED {row['complaint_id']}")
            
        time.sleep(2)  # Respect rate limits

    # Calculate Accuracy
    total = len(validation_results)
    train_correct = sum(is_match(r['true_train_number'], r['extracted_train_number']) for r in validation_results)
    coach_correct = sum(is_match(r['true_coach_number'], r['extracted_coach_number']) for r in validation_results)
    issue_correct = sum(is_match(r['true_issue_category'], r['extracted_issue_category']) for r in validation_results)

    train_acc = (train_correct / total) * 100 if total > 0 else 0
    coach_acc = (coach_correct / total) * 100 if total > 0 else 0
    issue_acc = (issue_correct / total) * 100 if total > 0 else 0

    print(f"\n--- EXTRACTION ACCURACY RESULTS (on {total} samples) ---")
    print(f"Train Number Accuracy:   {train_correct}/{total} ({train_acc:.1f}%)")
    print(f"Coach Number Accuracy:   {coach_correct}/{total} ({coach_acc:.1f}%)")
    print(f"Issue Category Accuracy: {issue_correct}/{total} ({issue_acc:.1f}%)")
    print("-----------------------------------")

    # Save validation results
    val_df = pd.DataFrame(validation_results)
    val_df.to_csv('data/extraction_validation.csv', index=False)
    print(f"Saved validation results to data/extraction_validation.csv")

    # Save accuracy metrics as JSON for the dashboard to display later
    accuracy_data = {
        "sample_size": total,
        "train_number_accuracy": round(train_acc, 1),
        "coach_number_accuracy": round(coach_acc, 1),
        "issue_category_accuracy": round(issue_acc, 1),
    }
    with open('data/extraction_accuracy.json', 'w') as f:
        json.dump(accuracy_data, f, indent=2)
    print(f"Saved accuracy metrics to data/extraction_accuracy.json")

    # ================================================================
    # STEP 2: FULL DATASET EXTRACTION — All 200 complaints
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Full Extraction — Processing ALL {len(df)} complaints")
    print(f"{'='*60}")
    print("This will take ~7 minutes (2-second delay per API call).\n")

    full_results = []
    failed_count = 0

    for i, (index, row) in enumerate(df.iterrows()):
        complaint_text = row['complaint_text']
        parsed = extract_single(client, complaint_text)

        if parsed:
            full_results.append({
                "complaint_id": row['complaint_id'],
                "timestamp": row['timestamp'],
                "complaint_text": complaint_text,
                # Ground-truth columns (for reference/comparison only)
                "true_train_number": str(row['true_train_number']) if pd.notna(row['true_train_number']) else None,
                "true_coach_number": str(row['true_coach_number']) if pd.notna(row['true_coach_number']) else None,
                "true_issue_category": str(row['true_issue_category']) if pd.notna(row['true_issue_category']) else None,
                # LLM-extracted columns (these drive the pipeline from now on)
                "extracted_train_number": str(parsed.get('extracted_train_number')) if parsed.get('extracted_train_number') else None,
                "extracted_coach_number": str(parsed.get('extracted_coach_number')) if parsed.get('extracted_coach_number') else None,
                "extracted_issue_category": str(parsed.get('extracted_issue_category')) if parsed.get('extracted_issue_category') else None,
            })
        else:
            # On failure, keep the row but with null extracted values
            full_results.append({
                "complaint_id": row['complaint_id'],
                "timestamp": row['timestamp'],
                "complaint_text": complaint_text,
                "true_train_number": str(row['true_train_number']) if pd.notna(row['true_train_number']) else None,
                "true_coach_number": str(row['true_coach_number']) if pd.notna(row['true_coach_number']) else None,
                "true_issue_category": str(row['true_issue_category']) if pd.notna(row['true_issue_category']) else None,
                "extracted_train_number": None,
                "extracted_coach_number": None,
                "extracted_issue_category": None,
            })
            failed_count += 1

        # Progress indicator every 10 rows
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(df)} complaints extracted...")

        time.sleep(2)  # Respect rate limits

    # Save full extraction results
    full_df = pd.DataFrame(full_results)
    output_path = 'data/extracted_complaints.csv'
    full_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"FULL EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {len(full_results)}")
    print(f"Successful:      {len(full_results) - failed_count}")
    print(f"Failed:          {failed_count}")
    print(f"Saved to:        {output_path}")
    print(f"\nPhase 2 Complete!")


if __name__ == "__main__":
    main()
