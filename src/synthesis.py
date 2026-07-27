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
