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
