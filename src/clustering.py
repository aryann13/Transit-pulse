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
