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
