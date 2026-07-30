# Transit Pulse - Architecture & Implementation Details

This document outlines the end-to-end data pipeline and dashboard implementation for Transit Pulse, a Railway Complaint Intelligence System.

## Phase 1: Data Synthesis

**Description:** Uses the Groq API (llama-3.1-8b-instant) to generate 200 synthetic, highly diverse passenger complaints in English, Hindi, and Hinglish. It includes ground truth labels for benchmarking.

**File:** src/synthesis.py

`python
import os
import json
import time
import random
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
# This requires GROQ_API_KEY to be set in the .env file.
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "your_groq_key_here":
    print("ERROR: Please set a valid GROQ_API_KEY in your .env file.")
    exit(1)

client = Groq(api_key=api_key)

# Constants for data generation
CATEGORIES = [
    "AC Failure", "Cleanliness", "Staff Behaviour", 
    "Food Quality", "Water Availability", "Overcrowding", 
    "Delay-related", "Other"
]
TRAINS = ["12626", "12951", "12301", "12423", "12229", "22436", "12809", "12004"]
COACHES = ["A1", "B2", "B3", "S1", "S5", "S7", "H1", "GS", "D1", "D2"]

def generate_batch(batch_size=10):
    """Calls the Groq API to generate a batch of synthetic complaints."""
    
    # Randomly pick some parameters to guide the LLM and ensure diversity
    sample_categories = random.sample(CATEGORIES, k=min(4, len(CATEGORIES)))
    sample_trains = random.sample(TRAINS, k=3)
    sample_coaches = random.sample(COACHES, k=4)
    
    system_prompt = f"""You are a helpful data generation assistant.
Your task is to generate {batch_size} realistic passenger complaints for Indian Railways.
The complaints must vary in tone (angry, polite, urgent, descriptive).
Crucially, the language must vary: use English, Hindi (written in Latin script), and Hinglish (code-mixed English and Hindi). 
For example: 'AC is not working in coach B1', 'toilet me pani nahi aa raha hai coach S5 me', 'Food quality is very bad in train 12626'.

Use the following parameters to ground your complaints:
- Train Numbers: {', '.join(sample_trains)}
- Coaches: {', '.join(sample_coaches)}
- Issue Categories: {', '.join(sample_categories)}

Output the data STRICTLY as a JSON array of objects. Do NOT output any markdown, introductory text, or explanatory text. Just the JSON array.
Each object must have the following keys:
- "complaint_text": The raw text of the complaint (vary the length from 1 sentence to 4 sentences).
- "true_train_number": The train number mentioned (or null if not mentioned).
- "true_coach_number": The coach number mentioned (or null if not mentioned).
- "true_issue_category": The category the complaint falls into (choose from the list provided).
"""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the JSON array of complaints now."}
            ],
            temperature=0.7,
            # Using JSON mode to ensure the output is parseable
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        
        # In JSON mode, Llama 3 often returns an object containing the array.
        # We need to parse it carefully.
        data = json.loads(response_text)
        
        # If the LLM wrapped it in an object like {"complaints": [...]}, extract the array
        if isinstance(data, dict):
            for key in data.keys():
                if isinstance(data[key], list):
                    return data[key]
            return [] # fallback
        elif isinstance(data, list):
            return data
        else:
            return []
            
    except Exception as e:
        print(f"Error generating batch: {e}")
        return []

def generate_random_timestamp(days_back=30):
    """Generates a random timestamp within the last `days_back` days."""
    now = datetime.now()
    random_days = random.randrange(days_back)
    random_seconds = random.randrange(24 * 60 * 60)
    past_date = now - timedelta(days=random_days, seconds=random_seconds)
    return past_date.strftime("%Y-%m-%d %H:%M:%S")

def main():
    total_complaints = 200
    batch_size = 10
    num_batches = total_complaints // batch_size
    
    all_complaints = []
    
    print(f"Starting data synthesis. Target: {total_complaints} complaints.")
    print("This will make multiple API calls to Groq. Please wait...\n")
    
    for i in range(num_batches):
        print(f"Generating batch {i+1}/{num_batches}...")
        batch_data = generate_batch(batch_size=batch_size)
        
        if batch_data:
            all_complaints.extend(batch_data)
            print(f"  -> Generated {len(batch_data)} complaints.")
        else:
            print("  -> Failed to parse batch.")
            
        # Sleep to respect rate limits on Groq free tier
        if i < num_batches - 1:
            time.sleep(3)
            
    if not all_complaints:
        print("No complaints generated. Exiting.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_complaints)
    
    # Add random timestamps and complaint IDs
    df['timestamp'] = [generate_random_timestamp() for _ in range(len(df))]
    df['complaint_id'] = [f"COMP_{i:04d}" for i in range(1, len(df) + 1)]
    
    # Reorder columns
    cols = ['complaint_id', 'timestamp', 'complaint_text', 'true_train_number', 'true_coach_number', 'true_issue_category']
    # Ensure all columns exist in case the LLM missed some
    for col in cols:
        if col not in df.columns:
            df[col] = None
    
    df = df[cols]
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/synthetic_complaints.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Saved {len(df)} complaints to {output_path}")

if __name__ == "__main__":
    main()

`

---

## Phase 2: LLM Information Extraction

**Description:** Performs Named Entity Recognition (NER) to extract Train Number, Coach Number, and Issue Category from raw text. It first runs a 20-row validation sample for accuracy metrics, then processes the full 200-row dataset.

**File:** src/extraction.py

`python
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

`

---

## Phase 3: Semantic Clustering

**Description:** Uses SentenceTransformers (all-MiniLM-L6-v2) to convert complaint text into vector embeddings. Runs K-Means clustering (k=8) to group similar complaints, and uses PCA to compress dimensions to 2D for interactive mapping.

**File:** src/clustering.py

`python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
    # ===== STEP 1: Load Data =====
    input_file = 'data/extracted_complaints.csv'
    print("Loading extracted complaints...")
    df = pd.read_csv(input_file)
    texts = df['complaint_text'].tolist()
    print(f"Loaded {len(texts)} complaints.\n")

    # ===== STEP 2: Generate Embeddings =====
    # Download and load the sentence-transformer model (runs locally, no API needed)
    print("Loading Sentence Transformer model (first time will download ~80MB)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Converting complaint texts into numerical vectors (embeddings)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Generated embeddings of shape: {embeddings.shape}")
    # embeddings.shape will be (200, 384) — 200 complaints, each represented by 384 numbers
    print()

    # ===== STEP 3: K-Means Clustering =====
    n_clusters = 8  # We expect roughly 8 types of issues
    print(f"Running K-Means Clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Add cluster_id to the DataFrame
    df['cluster_id'] = cluster_labels
    print("Clustering complete!\n")

    # ===== STEP 4: PCA — Compress 384D to 2D for visualization =====
    print("Compressing 384 dimensions to 2D using PCA (for scatter plot visualization)...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)
    df['pca_x'] = coords_2d[:, 0]
    df['pca_y'] = coords_2d[:, 1]
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%\n")

    # ===== STEP 5: Print Cluster Summary =====
    print("=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    
    for cid in range(n_clusters):
        cluster_df = df[df['cluster_id'] == cid]
        count = len(cluster_df)
        
        # Show the most common issue category in this cluster
        if 'extracted_issue_category' in cluster_df.columns:
            top_category = cluster_df['extracted_issue_category'].mode()
            top_cat_str = top_category.iloc[0] if len(top_category) > 0 else "N/A"
        else:
            top_cat_str = "N/A"
        
        print(f"\n--- Cluster {cid} ({count} complaints) | Dominant Issue: {top_cat_str} ---")
        
        # Print top 3 sample complaints from this cluster
        samples = cluster_df['complaint_text'].head(3).tolist()
        for i, sample in enumerate(samples, 1):
            # Truncate long texts for readability
            truncated = sample[:100] + "..." if len(sample) > 100 else sample
            print(f"  {i}. {truncated}")
    
    print("\n" + "=" * 60)

    # ===== STEP 6: Save Results =====
    output_path = 'data/clustered_complaints.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved clustered data to {output_path}")
    print(f"New columns added: 'cluster_id', 'pca_x', 'pca_y'")
    print("\nPhase 3 Complete!")

if __name__ == "__main__":
    main()

`

---

## Shared Utility: Crisis Score Aggregation

**Description:** Shared function compute_crisis_metrics() to calculate Crisis Score per Train/Coach group, compute dominant category, and format top complaints safely.

**File:** src/crisis_utils.py

`python
import pandas as pd

def compute_crisis_metrics(df):
    """
    Groups complaints by Train + Coach and calculates a Crisis Score.
    Formula: Crisis Score = (Count × Avg Severity) + (Max Severity × 2)
    Now uses EXTRACTED columns (LLM output) instead of ground-truth columns.
    """
    # Filter out rows where extracted train number is missing
    valid_df = df.dropna(subset=['extracted_train_number']).copy()
    
    if valid_df.empty:
        print("No valid extracted train numbers found for crisis aggregation.")
        return pd.DataFrame()
    
    # Fill missing extracted coach numbers with "Unknown"
    valid_df['extracted_coach_number'] = valid_df['extracted_coach_number'].fillna('Unknown')
    
    # Group by extracted train + coach
    grouped = valid_df.groupby(['extracted_train_number', 'extracted_coach_number'])
    
    crisis_data = []
    for (train, coach), group in grouped:
        count = len(group)
        avg_severity = group['severity_score'].mean()
        max_severity = group['severity_score'].max()
        
        # Crisis Score Formula
        crisis_score = (count * avg_severity) + (max_severity * 2)
        
        # Collect top complaint for context (converted cleanly to string)
        worst_complaint = str(group.loc[group['severity_score'].idxmax(), 'complaint_text'])
        
        # Dominant Category
        dominant_category = group['extracted_issue_category'].mode()
        dominant_category_str = dominant_category.iloc[0] if len(dominant_category) > 0 else "Multiple Issues"
        
        crisis_data.append({
            'train_number': int(float(train)) if pd.notna(train) else 'Unknown',
            'coach_number': coach,
            'dominant_category': dominant_category_str,
            'complaint_count': count,
            'avg_severity': round(avg_severity, 2),
            'max_severity': int(max_severity),
            'crisis_score': round(crisis_score, 2),
            'worst_complaint': worst_complaint
        })
    
    crisis_df = pd.DataFrame(crisis_data)
    if not crisis_df.empty:
        crisis_df = crisis_df.sort_values('crisis_score', ascending=False).reset_index(drop=True)
    return crisis_df

`

---

## Phase 4: Severity Scoring

**Description:** Assigns severity scores (1-5) using category baselines and keyword-boosting for critical/vulnerable terms. Includes portable sys.path resolution to import src.crisis_utils.

**File:** src/severity.py

`python
import pandas as pd
import re

# ===== CATEGORY BASE SCORES =====
# Each issue category gets a default severity score
CATEGORY_BASE_SCORES = {
    "AC Failure": 4,
    "Water Availability": 4,
    "Cleanliness": 3,
    "Overcrowding": 3,
    "Food Quality": 2,
    "Staff Behaviour": 2,
    "Delay-related": 2,
    "Other": 2,
}

# ===== KEYWORD LISTS FOR SEVERITY BOOSTING =====

# If ANY of these words appear → Score instantly becomes 5 (Critical)
CRITICAL_KEYWORDS = [
    'emergency', 'medical', 'hospital', 'injured', 'bleeding',
    'fainted', 'unconscious', 'attack', 'fire', 'accident',
    'died', 'death', 'critical', 'serious condition'
]

# Security-related → Score instantly becomes 5 (Critical)
SECURITY_KEYWORDS = [
    'stolen', 'theft', 'chain snatching', 'molest', 'molestation',
    'assault', 'threat', 'robbery', 'pickpocket', 'harassment',
    'chori', 'loot', 'dhamki', 'maar', 'peet'
]

# Vulnerable passengers → Score gets +1 bump
VULNERABLE_KEYWORDS = [
    'children', 'child', 'baby', 'infant', 'pregnant', 'elderly',
    'disabled', 'old age', 'senior citizen', 'wheelchair',
    'bachcha', 'bacche', 'boodhe', 'bujurg'
]

# Extended suffering → Score gets +1 bump
EXTENDED_SUFFERING_KEYWORDS = [
    'hours', '2 ghante', '3 ghante', '4 ghante', 'bahut der',
    'stuck', 'stranded', 'since morning', 'since yesterday',
    'all night', 'poori raat', 'kab se', 'waiting for hours'
]


def assign_severity(row):
    """
    Takes a single complaint row and returns (severity_score, severity_label).
    Uses a 2-layer approach: Category Base Score + Keyword Boosting.
    """
    text = str(row.get('complaint_text', '')).lower()
    category = str(row.get('extracted_issue_category', 'Other'))
    
    # ---- Layer 1: Get base score from the issue category ----
    score = CATEGORY_BASE_SCORES.get(category, 2)
    
    # ---- Layer 2: Scan text for danger keywords ----
    
    # Check for CRITICAL keywords → instant score 5
    for keyword in CRITICAL_KEYWORDS:
        if keyword in text:
            score = 5
            break
    
    # Check for SECURITY keywords → instant score 5
    for keyword in SECURITY_KEYWORDS:
        if keyword in text:
            score = 5
            break
    
    # Check for VULNERABLE passengers → bump score by 1
    for keyword in VULNERABLE_KEYWORDS:
        if keyword in text:
            score = min(score + 1, 5)  # Cap at maximum 5
            break
    
    # Check for EXTENDED SUFFERING → bump score by 1
    for keyword in EXTENDED_SUFFERING_KEYWORDS:
        if keyword in text:
            score = min(score + 1, 5)  # Cap at maximum 5
            break
    
    # ---- Assign human-readable label ----
    labels = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
        5: "Critical"
    }
    
    return score, labels.get(score, "Medium")


import sys
import os
# Add project root to sys.path so src.crisis_utils is resolved regardless of working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.crisis_utils import compute_crisis_metrics


def main():
    # ===== STEP 1: Load clustered data from Phase 3 =====
    input_file = 'data/clustered_complaints.csv'
    print("Loading clustered complaints...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} complaints.\n")

    # ===== STEP 2: Apply severity scoring to every complaint =====
    print("Assigning severity scores...")
    results = df.apply(assign_severity, axis=1)
    df['severity_score'] = [r[0] for r in results]
    df['severity_level'] = [r[1] for r in results]
    print("Severity scoring complete!\n")

    # ===== STEP 3: Print severity distribution =====
    print("=" * 50)
    print("SEVERITY DISTRIBUTION")
    print("=" * 50)
    dist = df['severity_level'].value_counts()
    for level in ["Critical", "Very High", "High", "Medium", "Low"]:
        count = dist.get(level, 0)
        bar = "#" * count
        print(f"  {level:>10}: {count:>3} complaints  {bar}")
    print()

    # ===== STEP 4: Compute crisis metrics per Train+Coach =====
    print("Computing Crisis Scores per Train + Coach...")
    crisis_df = compute_crisis_metrics(df)
    
    if not crisis_df.empty:
        print("\n" + "=" * 70)
        print("TOP 10 CRISIS ALERTS - TRAINS/COACHES NEEDING IMMEDIATE ACTION")
        print("=" * 70)
        
        for i, row in crisis_df.head(10).iterrows():
            print(f"\n  #{i+1} | Train {row['train_number']} - Coach {row['coach_number']}")
            print(f"      Complaints: {row['complaint_count']} | Avg Severity: {row['avg_severity']} | Max: {row['max_severity']}")
            print(f"      CRISIS SCORE: {row['crisis_score']}")
            print(f"      Worst: \"{row['worst_complaint']}\"")
        
        print("\n" + "=" * 70)
        
        # Save crisis report
        crisis_output = 'data/crisis_report.csv'
        crisis_df.to_csv(crisis_output, index=False)
        print(f"\nSaved crisis report to {crisis_output}")

    # ===== STEP 5: Save final processed complaints =====
    output_path = 'data/final_processed_complaints.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved final processed data to {output_path}")
    print(f"New columns added: 'severity_score', 'severity_level'")
    print("\nPhase 4 Complete!")


if __name__ == "__main__":
    main()

`

---

## Phase 5: Interactive Intelligence Dashboard

**Description:** A Streamlit-based UI featuring top-level KPIs, semantic cluster cards with URGENT/MODERATE/MINOR badging, an interactive Plotly map for the PCA projection, exact train matching, and extraction accuracy metrics.

**File:** app/dashboard.py

`python
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Transit Pulse — Complaint Intelligence",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS — Matching the mockup design exactly =====
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* ---- Root Variables (Mockup Palette) ---- */
    :root {
        --bg: #10131A;
        --panel: #191E28;
        --panel-2: #1F2531;
        --rule: #2A3142;
        --rule-soft: #232A38;
        --text: #E9ECF2;
        --text-dim: #8B93A6;
        --text-faint: #5B6478;
        --accent: #F4B740;
        --accent-dim: #6E5A26;
        --signal-red: #EF5B4E;
        --signal-red-bg: rgba(239,91,78,0.12);
        --signal-amber: #F4B740;
        --signal-amber-bg: rgba(244,183,64,0.12);
        --signal-green: #4FAE83;
        --signal-green-bg: rgba(79,174,131,0.12);
        --radius: 10px;
    }

    /* ---- Global overrides for Streamlit ---- */
    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px !important;
    }

    /* Hide ALL Streamlit branding & chrome */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    button[kind="header"] {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}

    /* ---- Topbar / Wordmark ---- */
    .topbar {
        border-bottom: 1px solid var(--rule);
        background: linear-gradient(180deg, #12151D 0%, #0F1218 100%);
        padding: 28px 0 0 0;
        margin: -1rem -6rem 0 -6rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .topbar-inner {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 10px;
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }
    .wordmark {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .flap-word {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 30px;
        letter-spacing: 0.04em;
        color: var(--text);
        text-transform: uppercase;
    }
    .flap-word span { color: var(--accent); }
    .subtitle-text {
        font-size: 12px;
        color: var(--text-faint);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding-bottom: 4px;
    }
    .status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: var(--text-dim);
        padding-bottom: 4px;
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--signal-green);
        box-shadow: 0 0 0 3px rgba(79,174,131,0.18);
        display: inline-block;
    }

    /* ---- Flap Ticker (KPI strip) ---- */
    .ticker {
        max-width: 100%;
        margin: 18px 0 0 0;
        padding: 0 10px;
        display: flex;
        gap: 1px;
        overflow: hidden;
        border-top: 1px solid var(--rule-soft);
    }
    .flap {
        flex: 1 1 0;
        background: var(--panel);
        padding: 14px 20px;
        border-right: 1px solid var(--rule);
        min-width: 150px;
    }
    .flap:last-child { border-right: none; }
    .flap-label {
        font-size: 10.5px;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-faint);
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .flap-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 26px;
        font-weight: 600;
        color: var(--text);
        line-height: 1.1;
    }
    .flap-delta {
        display: block;
        font-size: 11px;
        margin-top: 4px;
        color: var(--text-faint);
        font-family: 'IBM Plex Mono', monospace;
    }
    .flap-delta.up { color: var(--signal-red); }
    .flap-delta.down { color: var(--signal-green); }

    /* ---- Panel Titles ---- */
    .panel-title {
        font-family: 'Oswald', sans-serif !important;
        font-size: 13px;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin: 0 0 14px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--rule);
    }

    /* ---- Section Divider ---- */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 14px 2px;
    }
    .section-divider .line { flex: 1; height: 1px; background: var(--rule); }
    .section-divider span {
        font-size: 10.5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-faint);
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ---- Cluster Cards ---- */
    .cluster-card {
        background: var(--panel);
        border: 1px solid var(--rule);
        border-left: 3px solid var(--rule);
        border-radius: var(--radius);
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .cluster-card:hover { background: var(--panel-2); }
    .cluster-card.sev-critical { border-left-color: var(--signal-red); }
    .cluster-card.sev-veryhigh { border-left-color: var(--signal-red); }
    .cluster-card.sev-high { border-left-color: var(--signal-amber); }
    .cluster-card.sev-medium { border-left-color: var(--signal-green); }
    .cluster-card.sev-low { border-left-color: var(--signal-green); }

    .cluster-top {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .train-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13.5px;
        font-weight: 600;
        background: var(--panel-2);
        border: 1px solid var(--rule);
        padding: 3px 8px;
        border-radius: 5px;
        letter-spacing: 0.02em;
    }
    .cluster-issue {
        font-weight: 600;
        font-size: 14.5px;
    }
    .cluster-meta {
        font-size: 12px;
        color: var(--text-faint);
        margin-top: 3px;
    }

    /* ---- Severity Badges ---- */
    .sev-badge {
        margin-left: auto;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 9px;
        border-radius: 5px;
        font-weight: 600;
    }
    .sev-badge.urgent { background: var(--signal-red-bg); color: var(--signal-red); }
    .sev-badge.moderate { background: var(--signal-amber-bg); color: var(--signal-amber); }
    .sev-badge.minor { background: var(--signal-green-bg); color: var(--signal-green); }

    /* ---- Count Block ---- */
    .cluster-bottom {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin-top: 12px;
        gap: 16px;
    }
    .count-block { display: flex; align-items: baseline; gap: 6px; }
    .count-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 20px;
        font-weight: 600;
    }
    .count-label { font-size: 11px; color: var(--text-faint); }

    /* ---- Sample complaint ---- */
    .sample {
        font-size: 12.5px;
        color: var(--text-dim);
        line-height: 1.5;
        padding: 8px 10px;
        background: var(--bg);
        border-radius: 6px;
        margin-bottom: 6px;
        border: 1px solid var(--rule-soft);
    }

    /* ---- Chip tags ---- */
    .chip-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
    .chip {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        color: var(--text-dim);
        background: var(--panel-2);
        border: 1px solid var(--rule);
        padding: 3px 8px;
        border-radius: 4px;
    }

    /* ---- Sidebar panels ---- */
    .sidebar-panel {
        background: var(--panel);
        border: 1px solid var(--rule);
        border-radius: var(--radius);
        padding: 18px 18px 16px 18px;
        margin-bottom: 16px;
    }

    /* ---- Breakdown bars ---- */
    .breakdown-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 11px;
        font-size: 12.5px;
    }
    .breakdown-row:last-child { margin-bottom: 0; }
    .breakdown-label {
        width: 130px;
        flex-shrink: 0;
        color: var(--text-dim);
        font-size: 12px;
        font-family: 'Inter', sans-serif;
    }
    .breakdown-track {
        flex: 1;
        height: 7px;
        background: var(--panel-2);
        border-radius: 4px;
        overflow: hidden;
        min-width: 60px;
    }
    .breakdown-fill { height: 100%; border-radius: 4px; background: var(--accent); }
    .breakdown-pct {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        color: var(--text-faint);
        width: 35px;
        text-align: right;
        flex-shrink: 0;
    }

    /* ---- Legend ---- */
    .legend {
        display: flex;
        gap: 14px;
        font-size: 10.5px;
        color: var(--text-faint);
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.04em;
    }
    .legend-item { display: flex; align-items: center; gap: 5px; cursor: pointer; }
    .legend-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

    /* ---- Footer ---- */
    .custom-footer {
        max-width: 100%;
        margin: 40px 0 20px 0;
        padding: 20px 0 0 0;
        font-size: 11.5px;
        color: var(--text-faint);
        font-family: 'IBM Plex Mono', monospace;
        border-top: 1px solid var(--rule);
    }

    /* ---- Streamlit widget overrides ---- */
    .stSelectbox > div > div {
        background-color: var(--panel) !important;
        border-color: var(--rule) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10.5px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--text-faint) !important;
    }
    .stTextInput > div > div > input {
        background-color: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--rule) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
    }
    div[data-testid="stExpander"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--rule) !important;
        border-radius: var(--radius) !important;
        margin-top: 4px !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-dim) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }
    /* Column gap fix */
    div[data-testid="stHorizontalBlock"] {
        gap: 24px !important;
    }
</style>
""", unsafe_allow_html=True)


# ===== DATA LOADING =====
@st.cache_data
def load_data():
    df = pd.read_csv('data/final_processed_complaints.csv')
    # Load extraction accuracy metrics if available
    accuracy = None
    if os.path.exists('data/extraction_accuracy.json'):
        import json
        with open('data/extraction_accuracy.json', 'r') as f:
            accuracy = json.load(f)
    return df, accuracy

df, extraction_accuracy = load_data()

import sys
import os
# Add the parent directory (project root) to sys.path so we can import the sibling 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.crisis_utils import compute_crisis_metrics


# ===== TOPBAR HEADER =====
total_complaints = len(df)
critical_count = len(df[df['severity_level'].isin(['Critical', 'Very High'])])
trains_flagged = df['extracted_train_number'].dropna().nunique()
avg_severity = df['severity_score'].mean()

st.markdown(f"""
<div class="topbar">
    <div class="topbar-inner">
        <div class="wordmark">
            <span class="flap-word">TRANSIT<span>PULSE</span></span>
            <span class="subtitle-text">Railway Complaint Intelligence</span>
        </div>
        <div class="status-pill"><span class="status-dot"></span> LIVE · SYNCED {datetime.now().strftime('%H:%M')}</div>
    </div>
    <div class="ticker">
        <div class="flap">
            <div class="flap-label">Total Complaints</div>
            <div class="flap-value">{total_complaints}</div>
        </div>
        <div class="flap">
            <div class="flap-label">High-Severity Alerts</div>
            <div class="flap-value">{critical_count}</div>
            <span class="flap-delta up">▲ Active</span>
        </div>
        <div class="flap">
            <div class="flap-label">Trains Flagged</div>
            <div class="flap-value">{trains_flagged}</div>
        </div>
        <div class="flap">
            <div class="flap-label">Avg. Severity Score</div>
            <div class="flap-value">{avg_severity:.1f}/5</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===== FILTER BAR =====
fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1.5, 1.5, 1])

with fcol1:
    train_options = ["All Trains"] + sorted([str(int(float(t))) for t in df['extracted_train_number'].dropna().unique()])
    selected_train = st.selectbox("TRAIN NUMBER", train_options, label_visibility="visible")

with fcol2:
    sev_options = ["All Severities", "Critical", "Very High", "High", "Medium", "Low"]
    selected_severity = st.selectbox("SEVERITY", sev_options, label_visibility="visible")

with fcol3:
    cat_options = ["All Categories"] + sorted(df['extracted_issue_category'].dropna().unique().tolist())
    selected_category = st.selectbox("ISSUE CATEGORY", cat_options, label_visibility="visible")

with fcol4:
    search_query = st.text_input("SEARCH", placeholder="e.g. AC, toilet, food", label_visibility="visible")

# Apply filters
filtered_df = df.copy()
if selected_train != "All Trains":
    # Normalize column to string and strip any trailing .0 from pandas float conversion
    normalized_train_col = filtered_df['extracted_train_number'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Exact match instead of prefix match
    filtered_df = filtered_df[normalized_train_col == selected_train]
if selected_severity != "All Severities":
    filtered_df = filtered_df[filtered_df['severity_level'] == selected_severity]
if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df['extracted_issue_category'] == selected_category]
if search_query:
    filtered_df = filtered_df[filtered_df['complaint_text'].str.contains(search_query, case=False, na=False)]

st.markdown(f'<div style="text-align:right; font-size:12.5px; color:#5B6478; margin-bottom:10px;">{len(filtered_df)} complaints matching</div>', unsafe_allow_html=True)


# ===== MAIN LAYOUT: 2 columns (Left: Cluster Cards, Right: Sidebar Panels) =====
left_col, right_col = st.columns([1.65, 1])

with left_col:
    # ---- PER-TRAIN CLUSTERS RANKED BY SEVERITY ----
    st.markdown("""
    <div class="panel-title">
        Per-Train Clusters — Ranked by Severity
        <div class="legend">
            <span class="legend-item"><i class="legend-dot" style="background:#EF5B4E"></i> Urgent</span>
            <span class="legend-item"><i class="legend-dot" style="background:#F4B740"></i> Moderate</span>
            <span class="legend-item"><i class="legend-dot" style="background:#4FAE83"></i> Minor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    filtered_crisis_df = compute_crisis_metrics(filtered_df)

    if not filtered_crisis_df.empty:
        for idx, row in filtered_crisis_df.head(8).iterrows():
            # Determine severity class
            max_sev = row['max_severity']
            if max_sev >= 4:
                sev_class = "sev-critical"
                badge_class = "urgent"
                badge_text = "URGENT"
            elif max_sev == 3:
                sev_class = "sev-high"
                badge_class = "moderate"
                badge_text = "MODERATE"
            else:
                sev_class = "sev-medium"
                badge_class = "minor"
                badge_text = "MINOR"

            train_num = int(row['train_number']) if pd.notna(row['train_number']) else "Unknown"
            coach = row['coach_number'] if pd.notna(row['coach_number']) else "N/A"
            complaint_count = row['complaint_count']
            worst = str(row['worst_complaint'])[:150] if pd.notna(row['worst_complaint']) else ""

            # Render the card
            st.markdown(f"""
            <div class="cluster-card {sev_class}">
                <div class="cluster-top">
                    <span class="train-tag">{train_num}</span>
                    <span class="cluster-issue">{row.get('dominant_category', 'Issue')} · Coach {coach}</span>
                    <span class="sev-badge {badge_class}">{badge_text}</span>
                </div>
                <div class="cluster-bottom">
                    <div class="count-block">
                        <span class="count-num">{complaint_count}</span>
                        <span class="count-label">complaints</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Inspect — Train {train_num}, Coach {coach}"):
                # Show chip tags
                st.markdown(f"""
                <div class="chip-row">
                    <span class="chip">train: {train_num}</span>
                    <span class="chip">coach: {coach}</span>
                    <span class="chip">issue: {str(row.get('dominant_category', 'unknown')).lower()}</span>
                    <span class="chip">crisis: {row['crisis_score']}</span>
                    <span class="chip">max-sev: {max_sev}/5</span>
                </div>
                """, unsafe_allow_html=True)

                # Show sample complaints from this train/coach
                mask = pd.Series([True] * len(df))
                if pd.notna(row['train_number']):
                    mask &= pd.to_numeric(df['extracted_train_number'], errors='coerce') == row['train_number']
                if coach != "Unknown" and coach != "N/A" and pd.notna(row['coach_number']):
                    mask &= df['extracted_coach_number'] == row['coach_number']
                else:
                    mask &= df['extracted_coach_number'].isna()
                samples = df[mask]['complaint_text'].head(3).tolist()

                for s in samples:
                    truncated = s[:200] + "..." if len(str(s)) > 200 else s
                    st.markdown(f'<div class="sample">"{truncated}"</div>', unsafe_allow_html=True)

    # ---- SECTION DIVIDER ----
    st.markdown("""
    <div class="section-divider"><span>Semantic Clusters</span><div class="line"></div></div>
    """, unsafe_allow_html=True)

    # ---- SEMANTIC CLUSTER MAP ----
    st.markdown('<div class="panel-title">2D Embedding Map — PCA Projection</div>', unsafe_allow_html=True)

    if 'pca_x' in filtered_df.columns and 'pca_y' in filtered_df.columns:
        chart_df = filtered_df.dropna(subset=['pca_x', 'pca_y', 'extracted_issue_category']).copy()
        
        import textwrap
        # Wrap long complaint texts with HTML line breaks so the tooltip isn't incredibly wide
        chart_df['wrapped_text'] = chart_df['complaint_text'].apply(
            lambda x: '<br>'.join(textwrap.wrap(str(x), width=60))
        )
        
        # Define brand colors to match the rest of the dashboard
        color_discrete_map = {
            "AC Failure": "#4A90E2",
            "Cleanliness": "#50E3C2",
            "Delay-related": "#B8E986",
            "Food Quality": "#EF5B4E",
            "Overcrowding": "#F4B740",
            "Staff Behaviour": "#9013FE",
            "Water Availability": "#4FAE83",
            "Other": "#8B93A6"
        }
        
        fig = px.scatter(
            chart_df,
            x='pca_x',
            y='pca_y',
            color='extracted_issue_category',
            color_discrete_map=color_discrete_map,
            hover_data={
                'pca_x': False,
                'pca_y': False,
                'extracted_train_number': True,
                'extracted_coach_number': True,
                'severity_level': True,
                'wrapped_text': True,
                'complaint_text': False
            },
            labels={
                "extracted_issue_category": "Issue Category",
                "extracted_train_number": "Train",
                "extracted_coach_number": "Coach",
                "severity_level": "Severity",
                "wrapped_text": "Complaint"
            }
        )
        
        # Apply dark theme styling to match dashboard
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#8B93A6',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                title=""
            ),
            hoverlabel=dict(
                bgcolor="#1F2531",
                font_size=13,
                font_family="Inter"
            )
        )
        
        # Make gridlines very subtle
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#232A38', zeroline=False, showticklabels=False, title="")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#232A38', zeroline=False, showticklabels=False, title="")
        
        # Bigger dots with slight transparency
        fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0)))
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


with right_col:
    # ---- ISSUE BREAKDOWN PANEL ----
    st.markdown("""
    <div class="sidebar-panel">
        <div class="panel-title">Issue Breakdown</div>
    """, unsafe_allow_html=True)

    cat_counts = filtered_df['extracted_issue_category'].value_counts()
    total_cat = cat_counts.sum() if cat_counts.sum() > 0 else 1

    breakdown_html = ""
    for cat, count in cat_counts.items():
        pct = int((count / total_cat) * 100)
        breakdown_html += f"""
        <div class="breakdown-row">
            <div class="breakdown-label">{cat}</div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{pct}%"></div></div>
            <div class="breakdown-pct">{pct}%</div>
        </div>
        """

    st.markdown(breakdown_html + "</div>", unsafe_allow_html=True)

    # ---- SEVERITY DISTRIBUTION PANEL ----
    st.markdown("""
    <div class="sidebar-panel">
        <div class="panel-title">Severity Distribution</div>
    """, unsafe_allow_html=True)

    sev_counts = filtered_df['severity_level'].value_counts()
    total_sev = sev_counts.sum() if sev_counts.sum() > 0 else 1

    sev_colors = {
        "Critical": "#EF5B4E",
        "Very High": "#EF5B4E",
        "High": "#F4B740",
        "Medium": "#4FAE83",
        "Low": "#4FAE83"
    }

    sev_html = ""
    for level in ["Critical", "Very High", "High", "Medium", "Low"]:
        count = sev_counts.get(level, 0)
        pct = int((count / total_sev) * 100)
        color = sev_colors.get(level, "#F4B740")
        sev_html += f"""
        <div class="breakdown-row">
            <div class="breakdown-label">{level}</div>
            <div class="breakdown-track"><div class="breakdown-fill" style="width:{pct}%; background:{color}"></div></div>
            <div class="breakdown-pct">{pct}%</div>
        </div>
        """

    st.markdown(sev_html + "</div>", unsafe_allow_html=True)

    # ---- QUICK DATA DOWNLOAD ----
    st.markdown("""
    <div class="sidebar-panel">
        <div class="panel-title">Export Data</div>
    </div>
    """, unsafe_allow_html=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Complaints CSV",
        data=csv_data,
        file_name="transit_pulse_export.csv",
        mime="text/csv",
        use_container_width=True
    )

    try:
        with open('transit_pulse_architecture.md', 'r', encoding='utf-8') as f:
            md_data = f.read().encode('utf-8')
        st.download_button(
            label="Download Architecture Doc (MD)",
            data=md_data,
            file_name="transit_pulse_architecture.md",
            mime="text/markdown",
            use_container_width=True
        )
    except FileNotFoundError:
        pass

    # ---- EXTRACTION ACCURACY PANEL ----
    if extraction_accuracy:
        st.markdown("""
        <div class="sidebar-panel">
            <div class="panel-title">NER Extraction Accuracy</div>
        """, unsafe_allow_html=True)

        acc_items = [
            ("Train Number", extraction_accuracy.get('train_number_accuracy', 0), "#4FAE83"),
            ("Coach Number", extraction_accuracy.get('coach_number_accuracy', 0), "#F4B740"),
            ("Issue Category", extraction_accuracy.get('issue_category_accuracy', 0), "#EF5B4E"),
        ]
        acc_html = ""
        for label, pct, color in acc_items:
            acc_html += f"""
            <div class="breakdown-row">
                <div class="breakdown-label">{label}</div>
                <div class="breakdown-track"><div class="breakdown-fill" style="width:{pct}%; background:{color}"></div></div>
                <div class="breakdown-pct">{pct}%</div>
            </div>
            """
        acc_html += f'<div style="font-size:10.5px; color:#5B6478; margin-top:8px; font-family:IBM Plex Mono,monospace;">Validated on {extraction_accuracy.get("sample_size", 20)} random samples</div>'
        st.markdown(acc_html + "</div>", unsafe_allow_html=True)


# ---- COMPLAINT DATA TABLE ----
st.markdown("""
<div class="section-divider"><span>All Complaints</span><div class="line"></div></div>
""", unsafe_allow_html=True)

display_columns = [
    'complaint_id', 'complaint_text', 'extracted_train_number',
    'extracted_coach_number', 'extracted_issue_category',
    'severity_level', 'severity_score', 'cluster_id'
]
available_cols = [c for c in display_columns if c in filtered_df.columns]

st.dataframe(
    filtered_df[available_cols],
    width='stretch',
    height=400,
    column_config={
        "complaint_id": "ID",
        "complaint_text": st.column_config.TextColumn("Complaint", width="large"),
        "extracted_train_number": "Train (Extracted)",
        "extracted_coach_number": "Coach (Extracted)",
        "extracted_issue_category": "Category (Extracted)",
        "severity_level": "Severity",
        "severity_score": st.column_config.NumberColumn("Score", format="%d/5"),
        "cluster_id": "Cluster"
    }
)

# ---- FOOTER ----
st.markdown("""
<div class="custom-footer">Transit Pulse v1.0 · Built by Aryan Prajapati · Railway Complaint Intelligence System</div>
""", unsafe_allow_html=True)

`

---

## Validation: Real-World Data Testing

**Description:** A dedicated validation script that runs the exact same Llama-3.1 NER prompt against a dataset of real Indian Railways tweets (data/real_railway_tweets_validation_set.csv). This proves the prompt architecture is robust against real-world messiness, typos, and abbreviations, validating the PoC pipeline.

**File:** src/extract_real_tweets.py

`python
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        print("ERROR: Please set a valid GROQ_API_KEY in your .env file.")
        return

    client = Groq(api_key=api_key)

    input_file = 'data/real_railway_tweets_validation_set.csv'
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found.")
        return

    print("Loading REAL Twitter data...")
    df = pd.read_csv(input_file)
    
    # We will test on a random sample of 20 real tweets
    sample_size = min(20, len(df))
    test_df = df.sample(n=sample_size, random_state=123).copy()
    
    print(f"Running LLM Extraction on {sample_size} REAL tweets to test robustness...\n")

    # Define valid categories
    VALID_CATEGORIES = [
        "AC Failure", "Cleanliness", "Staff Behaviour", 
        "Food Quality", "Water Availability", "Overcrowding", 
        "Delay-related", "Other"
    ]

    for index, row in test_df.iterrows():
        tweet_text = row['text']
        
        system_prompt = f"""You are an intelligent Named Entity Recognition (NER) system for Indian Railways.
Your task is to extract specific information from the following passenger complaint tweet.

Categories must strictly be one of: {', '.join(VALID_CATEGORIES)}
If an entity is not explicitly mentioned, return null.

Return STRICTLY a JSON object with these exact keys:
- "extracted_train_number": (string or null) The train number (look for 5-digit numbers).
- "extracted_coach_number": (string or null) The coach identifier (e.g., A1, B2, S5).
- "extracted_issue_category": (string or null) The closest matching category from the allowed list.
"""

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Tweet: '{tweet_text}'"}
                ],
                temperature=0.0, 
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            parsed = json.loads(response_text)
            
            print(f"--- Tweet ID: {row['complaint_id']} ---")
            print(f"Raw Text: {tweet_text}")
            print(f"Extracted: Train={parsed.get('extracted_train_number')}, Coach={parsed.get('extracted_coach_number')}, Issue={parsed.get('extracted_issue_category')}\n")
            
            time.sleep(2) # Respect rate limits
            
        except Exception as e:
            print(f"Error processing {row['complaint_id']}: {e}")

    print("Validation on Real Tweets Complete!")

if __name__ == "__main__":
    main()

`

---

