"""
Compare Aarron and Akhil facial embeddings
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compare_aarron_and_akhil():
    """Compare facial embeddings between Aarron and Akhil"""
    
    print("🔄 COMPARING AARRON vs AKHIL FACIAL EMBEDDINGS")
    print("=" * 55)
    
    # Load Aarron's embedding
    try:
        with open('photos/aarron.jpg_embedding.json', 'r') as f:
            aarron_data = json.load(f)
        aarron_embedding = aarron_data['embedding']
        print(f"✅ Loaded Aarron's embedding: {len(aarron_embedding)} dimensions")
    except Exception as e:
        print(f"❌ Error loading Aarron's data: {e}")
        return
    
    # Load Akhil's embedding
    try:
        with open('akhil.jpg_embedding.json', 'r') as f:
            akhil_data = json.load(f)
        akhil_embedding = akhil_data['embedding']
        print(f"✅ Loaded Akhil's embedding: {len(akhil_embedding)} dimensions")
    except Exception as e:
        print(f"❌ Error loading Akhil's data: {e}")
        return
    
    # Calculate similarity
    aarron_array = np.array(aarron_embedding).reshape(1, -1)
    akhil_array = np.array(akhil_embedding).reshape(1, -1)
    
    similarity = cosine_similarity(aarron_array, akhil_array)[0][0]
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"🔢 Cosine Similarity: {similarity:.6f}")
    print(f"📈 Percentage: {similarity * 100:.2f}%")
    
    # Interpretation
    print(f"\n🎯 INTERPRETATION:")
    if similarity > 0.8:
        print(f"🟢 Very High Similarity (>80%) - Very likely the same person!")
    elif similarity > 0.6:
        print(f"🟡 Moderate Similarity (60-80%) - Possibly the same person or similar features")
    elif similarity > 0.4:
        print(f"🟠 Low Similarity (40-60%) - Some similar features but likely different people")
    else:
        print(f"🔴 Very Low Similarity (<40%) - Clearly different people")
    
    # Individual statistics
    print(f"\n📈 INDIVIDUAL STATISTICS:")
    
    aarron_stats = np.array(aarron_embedding)
    akhil_stats = np.array(akhil_embedding)
    
    print(f"\n👤 Aarron's Embedding:")
    print(f"   Min: {aarron_stats.min():.6f}")
    print(f"   Max: {aarron_stats.max():.6f}")
    print(f"   Mean: {aarron_stats.mean():.6f}")
    print(f"   Std: {aarron_stats.std():.6f}")
    
    print(f"\n👤 Akhil's Embedding:")
    print(f"   Min: {akhil_stats.min():.6f}")
    print(f"   Max: {akhil_stats.max():.6f}")
    print(f"   Mean: {akhil_stats.mean():.6f}")
    print(f"   Std: {akhil_stats.std():.6f}")
    
    # Save comparison result
    comparison_result = {
        'aarron_file': 'photos/aarron.jpg_embedding.json',
        'akhil_file': 'akhil.jpg_embedding.json',
        'similarity': similarity,
        'similarity_percentage': similarity * 100,
        'model_used': 'VGG-Face',
        'embedding_dimensions': len(aarron_embedding),
        'interpretation': 'Very High' if similarity > 0.8 else 'Moderate' if similarity > 0.6 else 'Low' if similarity > 0.4 else 'Very Low'
    }
    
    with open('aarron_vs_akhil_comparison.json', 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    print(f"\n💾 Comparison saved to: aarron_vs_akhil_comparison.json")

if __name__ == "__main__":
    compare_aarron_and_akhil()
