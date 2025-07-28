"""
Simple demo to extract facial embeddings from any image
"""

from deepface import DeepFace
import os

def quick_demo():
    """Quick demo to extract embeddings from the test image"""
    
    print("🚀 FACIAL EMBEDDING EXTRACTION DEMO")
    print("=" * 50)
    
    # Check if test image exists, if not create a simple one or skip
    test_image = "test_image.jpg"
    if not os.path.exists(test_image):
        print("❌ Test image not found.")
        print("💡 To test with a real photo:")
        print("   1. Add your photo to this folder (e.g., 'my_photo.jpg')")
        print("   2. Update the script to use your photo filename")
        print("   3. Run the script again")
        return
    
    try:
        print(f"📸 Processing: {test_image}")
        print("⏳ This may take a moment for the first run (downloading models)...")
        
        # Extract embedding using VGG-Face model
        result = DeepFace.represent(
            img_path=test_image,
            model_name='VGG-Face',
            enforce_detection=False,  # Set to False for test image since it's just gray
            detector_backend='opencv'
        )
        
        if result:
            embedding = result[0]['embedding']
            
            print("\n✅ SUCCESS! Facial embedding extracted:")
            print(f"📊 Embedding dimensions: {len(embedding)}")
            print(f"🔢 First 10 values: {embedding[:10]}")
            print(f"📈 Embedding range: {min(embedding):.3f} to {max(embedding):.3f}")
            
            # Save embedding to file
            import json
            with open('demo_embedding.json', 'w') as f:
                json.dump({
                    'image': test_image,
                    'model': 'VGG-Face',
                    'embedding': embedding,
                    'dimensions': len(embedding)
                }, f, indent=2)
            
            print(f"💾 Embedding saved to: demo_embedding.json")
            
        else:
            print("❌ No embedding could be extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try using enforce_detection=False or a different detector")

if __name__ == "__main__":
    quick_demo()
    
    print("\n" + "="*50)
    print("📝 TO USE WITH YOUR OWN PHOTOS:")
    print("1. Copy your photo to this folder")
    print("2. Update the image path in facial_embeddings.py")
    print("3. Run: python facial_embeddings.py")
    print("="*50)
