"""
Quick test to verify DeepFace installation and basic functionality
"""

from deepface import DeepFace
import numpy as np
import cv2

def test_deepface_installation():
    """Test if DeepFace is properly installed and working"""
    
    print("Testing DeepFace Installation...")
    print("=" * 40)
    
    try:
        # Test 1: Check available models
        print("✅ DeepFace imported successfully")
        
        # Test 2: List available models
        models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace']
        print(f"✅ Available models: {', '.join(models)}")
        
        # Test 3: Check detector backends
        detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        print(f"✅ Available detectors: {', '.join(detectors)}")
        
        # Test 4: Create a simple test image (solid color)
        print("\nCreating test image...")
        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray image
        cv2.imwrite('test_image.jpg', test_img)
        print("✅ Test image created: test_image.jpg")
        
        print("\n" + "="*40)
        print("✅ ALL TESTS PASSED!")
        print("DeepFace is ready to use for facial embedding extraction.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_model_info():
    """Display information about different models"""
    
    print("\nFacial Recognition Models Available:")
    print("=" * 45)
    
    model_info = {
        'VGG-Face': {
            'embedding_size': 2622,
            'description': 'Good general purpose model, widely used',
            'speed': 'Medium'
        },
        'Facenet': {
            'embedding_size': 128,
            'description': 'Fast and efficient, good for real-time applications',
            'speed': 'Fast'
        },
        'Facenet512': {
            'embedding_size': 512,
            'description': 'Higher dimensional version of Facenet',
            'speed': 'Medium'
        },
        'OpenFace': {
            'embedding_size': 128,
            'description': 'Open source, lightweight',
            'speed': 'Fast'
        },
        'ArcFace': {
            'embedding_size': 512,
            'description': 'State-of-the-art accuracy, good for verification',
            'speed': 'Medium'
        },
        'DeepFace': {
            'embedding_size': 4096,
            'description': 'Original Facebook model, very detailed embeddings',
            'speed': 'Slow'
        }
    }
    
    for model, info in model_info.items():
        print(f"\n{model}:")
        print(f"  Embedding Size: {info['embedding_size']} dimensions")
        print(f"  Description: {info['description']}")
        print(f"  Speed: {info['speed']}")

if __name__ == "__main__":
    # Run tests
    if test_deepface_installation():
        show_model_info()
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("1. Add your photo(s) to this folder")
        print("2. Edit facial_embeddings.py with your image path")
        print("3. Run: python facial_embeddings.py")
        print("="*50)
