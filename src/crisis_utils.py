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
