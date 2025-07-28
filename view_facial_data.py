"""
Simple Facial Data Viewer
Easy access to your stored facial embeddings
"""

import json
import os

def simple_data_viewer():
    """Simple interface to view your facial data"""
    
    print("👤 YOUR FACIAL DATA VIEWER")
    print("=" * 40)
    
    # Find all your facial data files
    json_files = [f for f in os.listdir('.') if f.endswith('_embedding.json') or f == 'all_embeddings.json']
    
    if not json_files:
        print("📂 No facial data found!")
        return
    
    print(f"📁 Found {len(json_files)} facial data files:")
    for i, filename in enumerate(json_files, 1):
        size_kb = round(os.path.getsize(filename) / 1024, 1)
        print(f"  {i}. {filename} ({size_kb} KB)")
    
    print("\n🔍 QUICK ACCESS METHODS:")
    print("-" * 30)
    
    # Method 1: Load specific person's data
    akhil_file = "akhil.jpg_embedding.json"
    if os.path.exists(akhil_file):
        with open(akhil_file, 'r') as f:
            akhil_data = json.load(f)
        
        print(f"👤 Akhil's Data:")
        print(f"   📊 Embedding dimensions: {len(akhil_data['embedding'])}")
        print(f"   🤖 Model used: {akhil_data.get('model_used', 'Unknown')}")
        print(f"   ✅ Status: {'Success' if akhil_data['success'] else 'Failed'}")
        
        # Show sample Python code to access this data
        print(f"\n💻 Python code to load Akhil's embedding:")
        print(f"   import json")
        print(f"   with open('{akhil_file}', 'r') as f:")
        print(f"       data = json.load(f)")
        print(f"   embedding = data['embedding']  # 4096-dimensional vector")
    
    # Method 2: Load batch data
    batch_file = "all_embeddings.json"
    if os.path.exists(batch_file):
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        print(f"\n📊 Batch Data (all_embeddings.json):")
        print(f"   📸 Total images: {len(batch_data)}")
        successful = sum(1 for item in batch_data if item.get('success', False))
        print(f"   ✅ Successful extractions: {successful}")
        
        print(f"\n💻 Python code to load batch data:")
        print(f"   import json")
        print(f"   with open('{batch_file}', 'r') as f:")
        print(f"       all_data = json.load(f)")
        print(f"   for person_data in all_data:")
        print(f"       if person_data['success']:")
        print(f"           embedding = person_data['embedding']")
        print(f"           print(f'Loaded embedding for {{person_data[\"image_path\"]}}')")

def extract_embedding_only(filename):
    """Extract just the embedding vector from a file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Batch file
            embeddings = []
            for item in data:
                if item.get('success', False):
                    embeddings.append(item['embedding'])
            return embeddings
        else:
            # Single file
            if data.get('success', False):
                return data['embedding']
            else:
                return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

if __name__ == "__main__":
    simple_data_viewer()
    
    print(f"\n🎯 HOW TO ACCESS YOUR DATA:")
    print(f"=" * 40)
    print(f"1. 📁 Location: C:\\Users\\aarro\\Desktop\\Final_project\\")
    print(f"2. 📄 Main files:")
    print(f"   • akhil.jpg_embedding.json - Akhil's facial data")
    print(f"   • all_embeddings.json - All processed faces")
    print(f"3. 📊 Data format: JSON with 4096-dimensional vectors")
    print(f"4. 🔧 Use Python's json library to load and process")
    
    print(f"\n🚀 QUICK EXAMPLES:")
    print(f"# Load Akhil's embedding")
    print(f"import json")
    print(f"with open('akhil.jpg_embedding.json', 'r') as f:")
    print(f"    data = json.load(f)")
    print(f"embedding = data['embedding']  # List of 4096 numbers")
    print(f"")
    print(f"# Compare two people")
    print(f"from sklearn.metrics.pairwise import cosine_similarity")
    print(f"import numpy as np")
    print(f"similarity = cosine_similarity(")
    print(f"    np.array(embedding1).reshape(1, -1),")
    print(f"    np.array(embedding2).reshape(1, -1)")
    print(f")[0][0]")
    print(f"print(f'Similarity: {{similarity:.2f}}')")
