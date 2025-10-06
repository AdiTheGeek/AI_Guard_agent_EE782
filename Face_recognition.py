"""
AI Guard - Face Recognition System for Video
This system recognizes specific faces in video streams in real-time.
"""

import cv2
import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime

class FaceRecognitionGuard:
    def __init__(self, known_faces_dir="known_faces", encodings_file="face_encodings.pkl"):
        """
        Initialize the Face Recognition Guard System
        
        Args:
            known_faces_dir: Directory containing images of known faces
            encodings_file: File to save/load face encodings
        """
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create directory if it doesn't exist
        Path(known_faces_dir).mkdir(exist_ok=True)
        
    def register_face_from_images(self):
        """
        Register faces from images in the known_faces directory.
        Expects images named as: person_name.jpg or person_name_1.jpg
        """
        print(f"Scanning {self.known_faces_dir} for face images...")
        
        supported_formats = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in supported_formats:
            image_files.extend(Path(self.known_faces_dir).glob(f"*{ext}"))
            image_files.extend(Path(self.known_faces_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {self.known_faces_dir}")
            print(f"Please add images named as 'your_name.jpg'")
            return False
        
        for image_path in image_files:
            print(f"Processing {image_path.name}...")
            
            # Extract name from filename
            name = image_path.stem.rsplit('_', 1)[0]  # Remove _1, _2 etc if present
            
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Use the first face found
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                print(f"  ✓ Registered face for {name}")
            else:
                print(f"  ✗ No face found in {image_path.name}")
        
        # Save encodings
        self.save_encodings()
        print(f"\nTotal faces registered: {len(self.known_face_names)}")
        return len(self.known_face_names) > 0
    
    def register_face_from_webcam(self, name, num_samples=5):
        """
        Register a face by capturing images from webcam
        
        Args:
            name: Name of the person
            num_samples: Number of face samples to capture
        """
        print(f"Capturing {num_samples} samples for {name}")
        print("Press SPACE to capture, ESC to cancel")
        
        video_capture = cv2.VideoCapture(0)
        samples_captured = 0
        temp_encodings = []
        
        while samples_captured < num_samples:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Samples: {samples_captured}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Register Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Registration cancelled")
                video_capture.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == 32:  # SPACE
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_encodings = face_recognition.face_encodings(rgb_frame)
                
                if len(face_encodings) > 0:
                    temp_encodings.append(face_encodings[0])
                    samples_captured += 1
                    print(f"  Captured sample {samples_captured}/{num_samples}")
                    
                    # Save image
                    img_path = Path(self.known_faces_dir) / f"{name}_{samples_captured}.jpg"
                    cv2.imwrite(str(img_path), frame)
                else:
                    print("  No face detected, try again")
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Average the encodings for better accuracy
        if temp_encodings:
            avg_encoding = np.mean(temp_encodings, axis=0)
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(name)
            self.save_encodings()
            print(f"✓ Successfully registered {name}")
            return True
        
        return False
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Encodings saved to {self.encodings_file}")
    
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_names)} face(s)")
            return True
        return False
    
    def recognize_faces_in_video(self, video_source=0, confidence_threshold=0.6):
        """
        Recognize faces in video stream
        
        Args:
            video_source: 0 for webcam, or path to video file
            confidence_threshold: Lower = stricter matching (0.6 is default)
        """
        if not self.known_face_encodings:
            print("No faces registered! Please register faces first.")
            return
        
        print("\nStarting face recognition...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        video_capture = cv2.VideoCapture(video_source)
        
        # Process every other frame for speed
        frame_count = 0
        face_locations = []
        face_names = []
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every other frame
            if frame_count % 2 == 0:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=confidence_threshold
                    )
                    name = "Unknown"
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            name = f"{name} ({confidence:.2f})"
                    
                    face_names.append(name)
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Determine color based on recognition
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('AI Guard - Face Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the AI Guard system"""
    print("=" * 50)
    print("AI GUARD - Face Recognition System")
    print("=" * 50)
    
    guard = FaceRecognitionGuard()
    
    # Try to load existing encodings
    if not guard.load_encodings():
        print("\nNo existing face data found.")
        print("\nHow would you like to register your face?")
        print("1. From webcam (recommended)")
        print("2. From images in 'known_faces' folder")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            name = input("Enter your name: ").strip()
            if name:
                guard.register_face_from_webcam(name, num_samples=5)
        elif choice == '2':
            print(f"\nPlace your images in '{guard.known_faces_dir}' folder")
            print("Name them as: your_name.jpg")
            input("Press Enter when ready...")
            guard.register_face_from_images()
        else:
            print("Invalid choice")
            return
    
    if guard.known_face_encodings:
        print(f"\nRegistered faces: {', '.join(guard.known_face_names)}")
        print("\nStarting face recognition system...")
        print("Options:")
        print("1. Use webcam (live)")
        print("2. Use video file")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            guard.recognize_faces_in_video(video_source=0)
        elif choice == '2':
            video_path = input("Enter video file path: ").strip()
            guard.recognize_faces_in_video(video_source=video_path)
    else:
        print("No faces registered. Exiting.")


if __name__ == "__main__":
    main()