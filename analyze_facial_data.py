"""
Facial Data Access and Analysis Tool
View, analyze, and manage your stored facial embeddings
"""

import json
import os
import numpy as np
from datetime import datetime

def load_embedding_file(filename):
    """Load and display facial embedding data from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def analyze_single_embedding(filename):
    """Analyze a single embedding file"""
    print(f"\n🔍 Analyzing: {filename}")
    print("-" * 50)
    
    data = load_embedding_file(filename)
    if not data:
        return
    
    if data.get('success', False):
        embedding = data['embedding']
        print(f"✅ Status: Successfully extracted")
        print(f"📊 Embedding dimensions: {data.get('embedding_size', len(embedding))}")
        print(f"🤖 Model used: {data.get('model_used', 'Unknown')}")
        print(f"👤 Faces detected: {data.get('faces_detected', 'Unknown')}")
        print(f"📸 Image path: {data.get('image_path', 'Unknown')}")
        print(f"📅 Timestamp: {data.get('timestamp', 'Unknown')}")
        
        # Statistics about the embedding
        embedding_array = np.array(embedding)
        print(f"\n📈 Embedding Statistics:")
        print(f"  Min value: {embedding_array.min():.6f}")
        print(f"  Max value: {embedding_array.max():.6f}")
        print(f"  Mean value: {embedding_array.mean():.6f}")
        print(f"  Standard deviation: {embedding_array.std():.6f}")
        
        print(f"\n🔢 First 10 embedding values:")
        print(f"  {embedding[:10]}")
        
    else:
        print(f"❌ Status: Failed")
        print(f"🚨 Error: {data.get('error', 'Unknown error')}")

def analyze_batch_embeddings(filename):
    """Analyze batch embedding file (multiple people)"""
    print(f"\n📊 Analyzing Batch File: {filename}")
    print("-" * 50)
    
    data = load_embedding_file(filename)
    if not data:
        return
    
    if isinstance(data, list):
        total_files = len(data)
        successful = sum(1 for item in data if item.get('success', False))
        failed = total_files - successful
        
        print(f"📈 Batch Processing Summary:")
        print(f"  Total images processed: {total_files}")
        print(f"  ✅ Successful extractions: {successful}")
        print(f"  ❌ Failed extractions: {failed}")
        print(f"  📊 Success rate: {(successful/total_files)*100:.1f}%")
        
        print(f"\n👤 Individual Results:")
        for i, item in enumerate(data, 1):
            image_name = os.path.basename(item.get('image_path', f'Image_{i}'))
            if item.get('success', False):
                dimensions = item.get('embedding_size', len(item.get('embedding', [])))
                print(f"  {i:2d}. ✅ {image_name} - {dimensions} dimensions")
            else:
                error = item.get('error', 'Unknown error')
                print(f"  {i:2d}. ❌ {image_name} - Error: {error}")

def compare_two_embeddings(file1, file2):
    """Compare embeddings from two different files"""
    print(f"\n🔄 Comparing Embeddings")
    print("-" * 30)
    
    data1 = load_embedding_file(file1)
    data2 = load_embedding_file(file2)
    
    if not (data1 and data2):
        print("❌ Could not load one or both files")
        return
    
    # Extract embeddings
    if isinstance(data1, list):
        emb1 = data1[0]['embedding'] if data1[0].get('success') else None
        name1 = os.path.basename(data1[0].get('image_path', file1))
    else:
        emb1 = data1['embedding'] if data1.get('success') else None
        name1 = os.path.basename(data1.get('image_path', file1))
    
    if isinstance(data2, list):
        emb2 = data2[0]['embedding'] if data2[0].get('success') else None
        name2 = os.path.basename(data2[0].get('image_path', file2))
    else:
        emb2 = data2['embedding'] if data2.get('success') else None
        name2 = os.path.basename(data2.get('image_path', file2))
    
    if emb1 and emb2:
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        emb1_array = np.array(emb1).reshape(1, -1)
        emb2_array = np.array(emb2).reshape(1, -1)
        similarity = cosine_similarity(emb1_array, emb2_array)[0][0]
        
        print(f"👤 Comparing: {name1} vs {name2}")
        print(f"🔢 Similarity Score: {similarity:.4f}")
        print(f"📊 Percentage: {similarity*100:.2f}%")
        
        if similarity > 0.8:
            print("🎯 Result: Very likely the same person!")
        elif similarity > 0.6:
            print("🤔 Result: Possibly the same person")
        else:
            print("🚫 Result: Likely different people")
    else:
        print("❌ Could not extract embeddings from one or both files")

def export_embedding_summary():
    """Create a summary report of all facial data"""
    
    # Find all JSON files
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    summary = {
        'report_date': datetime.now().isoformat(),
        'total_files': len(json_files),
        'files_analyzed': [],
        'statistics': {
            'total_embeddings': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'models_used': set(),
            'embedding_dimensions': set()
        }
    }
    
    print(f"\n📋 Creating Summary Report...")
    print("-" * 30)
    
    for filename in json_files:
        print(f"📄 Processing: {filename}")
        data = load_embedding_file(filename)
        
        if data:
            file_info = {
                'filename': filename,
                'file_size_kb': round(os.path.getsize(filename) / 1024, 2),
                'type': 'batch' if isinstance(data, list) else 'single'
            }
            
            if isinstance(data, list):
                successful = sum(1 for item in data if item.get('success', False))
                file_info['total_images'] = len(data)
                file_info['successful'] = successful
                file_info['failed'] = len(data) - successful
                
                summary['statistics']['total_embeddings'] += len(data)
                summary['statistics']['successful_extractions'] += successful
                summary['statistics']['failed_extractions'] += len(data) - successful
                
                # Collect model and dimension info
                for item in data:
                    if item.get('success'):
                        summary['statistics']['models_used'].add(item.get('model_used', 'Unknown'))
                        summary['statistics']['embedding_dimensions'].add(item.get('embedding_size', 0))
            else:
                if data.get('success'):
                    file_info['successful'] = 1
                    file_info['failed'] = 0
                    summary['statistics']['successful_extractions'] += 1
                    summary['statistics']['models_used'].add(data.get('model_used', 'Unknown'))
                    summary['statistics']['embedding_dimensions'].add(data.get('embedding_size', 0))
                else:
                    file_info['successful'] = 0
                    file_info['failed'] = 1
                    summary['statistics']['failed_extractions'] += 1
                
                summary['statistics']['total_embeddings'] += 1
            
            summary['files_analyzed'].append(file_info)
    
    # Convert sets to lists for JSON serialization
    summary['statistics']['models_used'] = list(summary['statistics']['models_used'])
    summary['statistics']['embedding_dimensions'] = list(summary['statistics']['embedding_dimensions'])
    
    # Save summary
    with open('facial_data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary Report Generated!")
    print(f"💾 Saved as: facial_data_summary.json")
    
    return summary

def main():
    """Main function to analyze all facial data"""
    
    print("🔍 FACIAL DATA ANALYSIS TOOL")
    print("=" * 50)
    
    # List all JSON files
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("📂 No facial data files found!")
        return
    
    print(f"📁 Found {len(json_files)} facial data files:")
    for i, filename in enumerate(json_files, 1):
        size_kb = round(os.path.getsize(filename) / 1024, 2)
        print(f"  {i}. {filename} ({size_kb} KB)")
    
    # Analyze each file
    for filename in json_files:
        if 'summary' not in filename:  # Skip summary files
            if 'all_embeddings' in filename or 'batch' in filename:
                analyze_batch_embeddings(filename)
            else:
                analyze_single_embedding(filename)
    
    # Generate summary report
    summary = export_embedding_summary()
    
    print(f"\n🎯 FINAL STATISTICS:")
    print(f"📊 Total embeddings processed: {summary['statistics']['total_embeddings']}")
    print(f"✅ Successful extractions: {summary['statistics']['successful_extractions']}")
    print(f"❌ Failed extractions: {summary['statistics']['failed_extractions']}")
    print(f"🤖 Models used: {', '.join(summary['statistics']['models_used'])}")
    print(f"📏 Embedding dimensions: {', '.join(map(str, summary['statistics']['embedding_dimensions']))}")

if __name__ == "__main__":
    main()
