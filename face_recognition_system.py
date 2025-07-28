"""
Comprehensive Face Recognition System
Complete example for extracting, comparing, and managing facial embeddings
"""

import os
import json
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from datetime import datetime

class FaceEmbeddingSystem:
    """A complete facial embedding extraction and comparison system"""
    
    def __init__(self, model='VGG-Face', detector='opencv'):
        """
        Initialize the face embedding system
        
        Args:
            model: Model to use for embedding extraction
            detector: Face detection backend to use
        """
        self.model = model
        self.detector = detector
        self.embeddings_db = {}
        self.load_database()
        
        print(f"🚀 Face Embedding System Initialized")
        print(f"📊 Model: {model}")
        print(f"🔍 Detector: {detector}")
    
    def extract_embedding(self, image_path, person_name=None):
        """
        Extract facial embedding from an image
        
        Args:
            image_path: Path to the image file
            person_name: Optional name to associate with this embedding
            
        Returns:
            dict: Result containing embedding and metadata
        """
        try:
            if not os.path.exists(image_path):
                return {'success': False, 'error': f'Image not found: {image_path}'}
            
            print(f"📸 Processing: {os.path.basename(image_path)}")
            
            # Extract embedding
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model,
                enforce_detection=True,
                detector_backend=self.detector
            )
            
            if result:
                embedding = result[0]['embedding']
                
                embedding_data = {
                    'success': True,
                    'embedding': embedding,
                    'dimensions': len(embedding),
                    'model': self.model,
                    'detector': self.detector,
                    'image_path': image_path,
                    'person_name': person_name,
                    'timestamp': datetime.now().isoformat(),
                    'faces_detected': len(result)
                }
                
                # Auto-save to database if person name provided
                if person_name:
                    self.save_embedding(person_name, embedding_data)
                
                print(f"✅ Success! Extracted {len(embedding)}-dimensional embedding")
                return embedding_data
            else:
                return {'success': False, 'error': 'No faces detected'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_embedding(self, person_name, embedding_data):
        """Save embedding to database"""
        if person_name not in self.embeddings_db:
            self.embeddings_db[person_name] = []
        
        self.embeddings_db[person_name].append(embedding_data)
        self.save_database()
        print(f"💾 Saved embedding for: {person_name}")
    
    def compare_embeddings(self, embedding1, embedding2):
        """
        Compare two embeddings using cosine similarity
        
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return similarity
    
    def find_matches(self, embedding, threshold=0.6):
        """
        Find matching faces in the database
        
        Args:
            embedding: Embedding to search for
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            list: List of matches with similarity scores
        """
        matches = []
        
        for person_name, person_embeddings in self.embeddings_db.items():
            for stored_embedding in person_embeddings:
                if stored_embedding.get('success', False):
                    similarity = self.compare_embeddings(
                        embedding, 
                        stored_embedding['embedding']
                    )
                    
                    if similarity >= threshold:
                        matches.append({
                            'person_name': person_name,
                            'similarity': similarity,
                            'timestamp': stored_embedding.get('timestamp', 'Unknown'),
                            'image_path': stored_embedding.get('image_path', 'Unknown')
                        })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def identify_person(self, image_path, threshold=0.6):
        """
        Identify a person from an image by comparing against database
        
        Args:
            image_path: Path to the image to identify
            threshold: Minimum similarity threshold
            
        Returns:
            dict: Identification result
        """
        # Extract embedding from new image
        result = self.extract_embedding(image_path)
        
        if not result['success']:
            return result
        
        # Find matches
        matches = self.find_matches(result['embedding'], threshold)
        
        if matches:
            best_match = matches[0]
            return {
                'success': True,
                'identified_as': best_match['person_name'],
                'confidence': best_match['similarity'],
                'all_matches': matches,
                'embedding_dimensions': result['dimensions']
            }
        else:
            return {
                'success': True,
                'identified_as': 'Unknown',
                'confidence': 0.0,
                'all_matches': [],
                'embedding_dimensions': result['dimensions'],
                'message': f'No matches found above threshold {threshold}'
            }
    
    def save_database(self):
        """Save embeddings database to file"""
        # Create facial_embeddings folder if it doesn't exist
        if not os.path.exists('facial_embeddings'):
            os.makedirs('facial_embeddings')
        
        with open('facial_embeddings/face_database.json', 'w') as f:
            json.dump(self.embeddings_db, f, indent=2)
    
    def load_database(self):
        """Load embeddings database from file"""
        database_path = 'facial_embeddings/face_database.json'
        if os.path.exists(database_path):
            try:
                with open(database_path, 'r') as f:
                    self.embeddings_db = json.load(f)
                print(f"📂 Loaded database with {len(self.embeddings_db)} people")
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                self.embeddings_db = {}
        else:
            self.embeddings_db = {}
    
    def list_people(self):
        """List all people in the database"""
        if not self.embeddings_db:
            print("📂 Database is empty")
            return
        
        print("👥 People in database:")
        for person_name, embeddings in self.embeddings_db.items():
            print(f"  - {person_name}: {len(embeddings)} photo(s)")
    
    def get_database_stats(self):
        """Get database statistics"""
        total_people = len(self.embeddings_db)
        total_photos = sum(len(embeddings) for embeddings in self.embeddings_db.values())
        
        return {
            'total_people': total_people,
            'total_photos': total_photos,
            'database_size_mb': os.path.getsize('face_database.json') / (1024*1024) if os.path.exists('face_database.json') else 0
        }

# Example usage and demo functions
def demo_basic_usage():
    """Demonstrate basic usage of the face embedding system"""
    
    print("🎯 FACE EMBEDDING SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    face_system = FaceEmbeddingSystem(model='VGG-Face')
    
    # Example 1: Extract embedding from single image
    print("\\n1️⃣ Single Image Processing:")
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        result = face_system.extract_embedding(test_image, person_name="Test Person")
        
        if result['success']:
            print(f"✅ Embedding extracted: {result['dimensions']} dimensions")
            print(f"📅 Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Example 2: Database statistics
    print("\\n2️⃣ Database Status:")
    stats = face_system.get_database_stats()
    print(f"👥 People: {stats['total_people']}")
    print(f"📸 Photos: {stats['total_photos']}")
    print(f"💾 Database size: {stats['database_size_mb']:.2f} MB")
    
    face_system.list_people()

def demo_face_recognition():
    """Demonstrate face recognition workflow"""
    
    print("\\n🔍 FACE RECOGNITION WORKFLOW")
    print("=" * 50)
    
    face_system = FaceEmbeddingSystem()
    
    # This would be used with actual photos:
    print("📝 To use with your photos:")
    print("1. Copy photos to this folder")
    print("2. Register known people:")
    print("   face_system.extract_embedding('person1.jpg', 'John Doe')")
    print("   face_system.extract_embedding('person2.jpg', 'Jane Smith')")
    print("")
    print("3. Identify unknown person:")
    print("   result = face_system.identify_person('unknown.jpg')")
    print("   print(f'Identified as: {result[\"identified_as\"]} with {result[\"confidence\"]:.2f} confidence')")
    
    return face_system

if __name__ == "__main__":
    print("🚀 FACIAL EMBEDDING EXTRACTION SYSTEM")
    print("=" * 55)
    
    # Run basic demo
    demo_basic_usage()
    
    # Show recognition workflow
    face_system = demo_face_recognition()
    
    print("\\n" + "="*55)
    print("📋 USAGE INSTRUCTIONS:")
    print("1. Place your photos in this folder")
    print("2. Run the following Python code:")
    print("")
    print("   # Initialize system")
    print("   face_system = FaceEmbeddingSystem()")
    print("")
    print("   # Register known people")
    print("   face_system.extract_embedding('photo1.jpg', 'Person Name')")
    print("")
    print("   # Identify unknown person")
    print("   result = face_system.identify_person('unknown.jpg')")
    print("   print(result)")
    print("")
    print("Available models: VGG-Face, Facenet, Facenet512, ArcFace")
    print("Available detectors: opencv, mtcnn, retinaface, ssd, dlib")
    print("="*55)
