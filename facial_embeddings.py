"""
Facial Embedding Extraction using DeepFace
This script demonstrates how to extract facial embeddings from photos
"""

import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image
import json

def extract_facial_embedding(image_path, model_name='VGG-Face'):
    """
    Extract facial embedding from a single image
    
    Args:
        image_path (str): Path to the image file
        model_name (str): Model to use for embedding extraction
                         Options: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 
                                 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'
    
    Returns:
        dict: Contains embedding vector and metadata
    """
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Using model: {model_name}")
        
        # Extract embedding using DeepFace
        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=True,  # Set to False if you want to process images without clear faces
            detector_backend='opencv'  # Options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
        )
        
        # DeepFace returns a list of embeddings (one per face detected)
        if result:
            embedding = result[0]['embedding']
            
            return {
                'success': True,
                'embedding': embedding,
                'embedding_size': len(embedding),
                'model_used': model_name,
                'faces_detected': len(result),
                'image_path': image_path
            }
        else:
            return {
                'success': False,
                'error': 'No faces detected in the image',
                'image_path': image_path
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image_path': image_path
        }

def process_multiple_images(image_folder, output_file='embeddings.json', model_name='VGG-Face'):
    """
    Process multiple images and save embeddings to a JSON file
    
    Args:
        image_folder (str): Path to folder containing images
        output_file (str): Path to save the embeddings JSON file
        model_name (str): Model to use for embedding extraction
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    results = []
    
    if not os.path.exists(image_folder):
        print(f"Error: Folder {image_folder} does not exist")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No supported image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        print(f"\nProcessing {i}/{len(image_files)}: {filename}")
        
        result = extract_facial_embedding(image_path, model_name)
        results.append(result)
        
        if result['success']:
            print(f"✅ Success: Extracted {result['embedding_size']}-dimensional embedding")
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {len(results) - successful}")

def compare_embeddings(embedding1, embedding2):
    """
    Compare two facial embeddings using cosine similarity
    
    Args:
        embedding1, embedding2: Facial embedding vectors
    
    Returns:
        float: Cosine similarity score (0-1, higher means more similar)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert to numpy arrays and reshape
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

# Example usage
if __name__ == "__main__":
    print("Facial Embedding Extraction Tool")
    print("=" * 40)
    
    # Example 1: Single image processing
    print("\nExample 1: Single Image Processing")
    print("-" * 35)
    
    # You need to provide a path to an actual image
    sample_image = "photos\\aarron.jpg"  # Use relative path to photos folder

    if os.path.exists(sample_image):
        result = extract_facial_embedding(sample_image, model_name='VGG-Face')
        if result['success']:
            print(f"✅ Embedding extracted successfully!")
            print(f"📊 Embedding dimensions: {result['embedding_size']}")
            print(f"🔢 First 5 values: {result['embedding'][:5]}")
            
            # Save individual embedding to facial_embeddings folder
            filename = os.path.basename(sample_image).replace('.jpg', '_embedding.json')
            embedding_path = os.path.join('facial_embeddings', filename)
            with open(embedding_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Embedding saved to: {embedding_path}")
        else:
            print(f"❌ Failed to extract embedding: {result['error']}")
    else:
        print(f"📁 Sample image '{sample_image}' not found.")
        print("🔄 Please add an image file to test the extraction.")
        print("📋 Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
    
    # Example 2: Batch processing
    print("\nExample 2: Batch Processing")
    print("-" * 30)
    
    # Create a photos folder if it doesn't exist
    photos_folder = "photos"
    if not os.path.exists(photos_folder):
        os.makedirs(photos_folder)
        print(f"📁 Created '{photos_folder}' folder - add your photos here!")
    
    # Check if there are photos to process
    image_files = [f for f in os.listdir(photos_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if image_files:
        print(f"📸 Found {len(image_files)} images in '{photos_folder}' folder")
        user_input = input("Process all images? (y/n): ").lower().strip()
        if user_input == 'y':
            process_multiple_images(photos_folder, "facial_embeddings/all_embeddings.json")
    else:
        print(f"📂 No images found in '{photos_folder}' folder")
        print("📋 Add photos (.jpg, .png, etc.) to the 'photos' folder to batch process them")
    
    print("\nTo use this script:")
    print("1. Place your image(s) in the project folder")
    print("2. Update the image path in the script")
    print("3. Run: python facial_embeddings.py")
