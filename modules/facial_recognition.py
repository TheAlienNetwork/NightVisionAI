import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional
import tempfile
from datetime import datetime
import io

class FacialRecognitionSystem:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.face_tolerance = 0.6
        self.temp_dir = tempfile.mkdtemp()
        # Initialize OpenCV face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except AttributeError:
            # Fallback for systems where cv2.data is not available
            import os
            cascade_path = os.path.join(cv2.__path__[0], 'data')
            self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
            self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_eye.xml'))
    
    def detect_faces_in_image(self, image_data: bytes) -> Tuple[List, List, List]:
        """Detect faces in image and return locations, face features, and face images"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using Haar cascade
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            face_locations = []
            face_features = []
            face_images = []
            
            for (x, y, w, h) in faces:
                # Convert to face_recognition format (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))
                
                # Extract face region
                face_region = image[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                # Calculate simple features (histogram and edge features)
                hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
                edges = cv2.Canny(face_gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / edges.size
                
                # Create feature vector
                features = {
                    'histogram': hist.flatten().tolist(),
                    'edge_ratio': edge_ratio,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w / h if h > 0 else 0
                }
                
                face_features.append(features)
                face_images.append(face_region)
            
            return face_locations, face_features, face_images
            
        except Exception as e:
            st.error(f"Error detecting faces: {str(e)}")
            return [], [], []
    
    def detect_faces_in_video(self, video_data: bytes, frame_skip: int = 30) -> List[Dict]:
        """Detect faces in video frames"""
        try:
            # Save video to temporary file
            temp_path = os.path.join(self.temp_dir, "temp_video.mp4")
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Open video capture
            cap = cv2.VideoCapture(temp_path)
            faces_detected = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame to improve performance
                if frame_number % frame_skip == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        
                        # Convert to face_recognition format and extract features
                        face_locations = []
                        face_features = []
                        
                        for (x, y, w, h) in faces:
                            face_locations.append((y, x + w, y + h, x))
                            face_region = gray[y:y+h, x:x+w]
                            
                            # Calculate features
                            hist = cv2.calcHist([face_region], [0], None, [256], [0, 256])
                            edges = cv2.Canny(face_region, 50, 150)
                            edge_ratio = np.sum(edges > 0) / edges.size
                            
                            features = {
                                'histogram': hist.flatten().tolist(),
                                'edge_ratio': edge_ratio,
                                'width': w,
                                'height': h,
                                'aspect_ratio': w / h if h > 0 else 0
                            }
                            face_features.append(features)
                        
                        faces_detected.append({
                            'frame_number': frame_number,
                            'timestamp': timestamp,
                            'face_locations': face_locations,
                            'face_features': face_features,
                            'num_faces': len(face_locations)
                        })
                
                frame_number += 1
            
            cap.release()
            os.remove(temp_path)
            
            return faces_detected
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return []
    
    def compare_faces(self, known_features: Dict, unknown_features: Dict) -> Tuple[bool, float]:
        """Compare two face feature sets"""
        try:
            # Compare histograms using correlation
            hist1 = np.array(known_features['histogram'])
            hist2 = np.array(unknown_features['histogram'])
            
            hist_correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # Compare aspect ratios
            aspect_diff = abs(known_features['aspect_ratio'] - unknown_features['aspect_ratio'])
            aspect_similarity = max(0, 1 - aspect_diff / 2)  # Normalize aspect ratio difference
            
            # Compare edge ratios
            edge_diff = abs(known_features['edge_ratio'] - unknown_features['edge_ratio'])
            edge_similarity = max(0, 1 - edge_diff)
            
            # Combine similarities with weights
            overall_similarity = (hist_correlation * 0.7 + aspect_similarity * 0.15 + edge_similarity * 0.15)
            
            # Convert to percentage and determine match
            confidence = max(0, overall_similarity * 100)
            matches = confidence >= (self.face_tolerance * 100)
            
            return matches, confidence
            
        except Exception as e:
            st.error(f"Error comparing faces: {str(e)}")
            return False, 0.0
    
    def identify_faces(self, face_features: List[Dict], reference_database: List[Dict]) -> List[Dict]:
        """Identify faces against a reference database"""
        identifications = []
        
        for i, features in enumerate(face_features):
            best_match = None
            best_confidence = 0
            
            for ref_person in reference_database:
                if 'face_features' not in ref_person:
                    continue
                
                matches, confidence = self.compare_faces(ref_person['face_features'], features)
                
                if matches and confidence > best_confidence:
                    best_match = ref_person
                    best_confidence = confidence
            
            identification = {
                'face_index': i,
                'identified_person': best_match['name'] if best_match else 'Unknown',
                'confidence': best_confidence,
                'match_found': best_match is not None,
                'person_id': best_match.get('id') if best_match else None
            }
            
            identifications.append(identification)
        
        return identifications
    
    def create_face_database_entry(self, name: str, face_features: Dict, 
                                  metadata: Dict = None) -> Dict:
        """Create a new face database entry"""
        entry = {
            'id': len(self.known_faces) + 1,
            'name': name,
            'face_features': face_features,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        return entry
    
    def add_to_database(self, name: str, image_data: bytes, metadata: Dict = None) -> bool:
        """Add a person to the face recognition database"""
        try:
            face_locations, face_features, face_images = self.detect_faces_in_image(image_data)
            
            if not face_features:
                st.error("No faces detected in the image")
                return False
            
            if len(face_features) > 1:
                st.warning("Multiple faces detected. Using the first face.")
            
            # Use the first face features
            features = face_features[0]
            
            # Create database entry
            entry = self.create_face_database_entry(name, features, metadata)
            
            # Add to known faces (in memory for this session)
            self.known_faces.append(features)
            self.known_names.append(name)
            
            return True
            
        except Exception as e:
            st.error(f"Error adding face to database: {str(e)}")
            return False
    
    def search_faces_in_files(self, file_list: List[Dict], reference_database: List[Dict]) -> List[Dict]:
        """Search for known faces across multiple files"""
        search_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_info in enumerate(file_list):
            status_text.text(f"Searching faces in {file_info.get('filename', 'Unknown')}")
            
            try:
                if file_info.get('category') == 'image':
                    # Process image
                    face_locations, face_features, face_images = self.detect_faces_in_image(
                        file_info.get('file_data', b'')
                    )
                    
                    if face_features:
                        identifications = self.identify_faces(face_features, reference_database)
                        
                        result = {
                            'file_id': file_info.get('id'),
                            'filename': file_info.get('filename'),
                            'file_type': 'image',
                            'faces_found': len(face_features),
                            'face_locations': face_locations,
                            'identifications': identifications,
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        search_results.append(result)
                
                elif file_info.get('category') == 'video':
                    # Process video
                    video_faces = self.detect_faces_in_video(file_info.get('file_data', b''))
                    
                    if video_faces:
                        # Process each frame's faces
                        frame_results = []
                        for frame_data in video_faces:
                            face_features = frame_data['face_features']
                            identifications = self.identify_faces(face_features, reference_database)
                            
                            frame_result = {
                                'frame_number': frame_data['frame_number'],
                                'timestamp': frame_data['timestamp'],
                                'faces_found': frame_data['num_faces'],
                                'identifications': identifications
                            }
                            
                            frame_results.append(frame_result)
                        
                        result = {
                            'file_id': file_info.get('id'),
                            'filename': file_info.get('filename'),
                            'file_type': 'video',
                            'total_frames_processed': len(video_faces),
                            'frame_results': frame_results,
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        
                        search_results.append(result)
            
            except Exception as e:
                st.error(f"Error processing {file_info.get('filename')}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(file_list))
        
        status_text.text("Face search complete!")
        return search_results
    
    def generate_face_report(self, search_results: List[Dict]) -> Dict:
        """Generate comprehensive face recognition report"""
        if not search_results:
            return {}
        
        total_files = len(search_results)
        total_faces = sum(result.get('faces_found', 0) for result in search_results)
        files_with_faces = len([r for r in search_results if r.get('faces_found', 0) > 0])
        
        # Count identifications
        identified_persons = {}
        unknown_faces = 0
        
        for result in search_results:
            if result.get('file_type') == 'image':
                identifications = result.get('identifications', [])
                for ident in identifications:
                    if ident.get('match_found'):
                        person_name = ident.get('identified_person', 'Unknown')
                        identified_persons[person_name] = identified_persons.get(person_name, 0) + 1
                    else:
                        unknown_faces += 1
            
            elif result.get('file_type') == 'video':
                for frame_result in result.get('frame_results', []):
                    identifications = frame_result.get('identifications', [])
                    for ident in identifications:
                        if ident.get('match_found'):
                            person_name = ident.get('identified_person', 'Unknown')
                            identified_persons[person_name] = identified_persons.get(person_name, 0) + 1
                        else:
                            unknown_faces += 1
        
        report = {
            'summary': {
                'total_files_processed': total_files,
                'files_with_faces': files_with_faces,
                'total_faces_detected': total_faces,
                'known_persons_identified': len(identified_persons),
                'unknown_faces': unknown_faces
            },
            'identified_persons': identified_persons,
            'processing_timestamp': datetime.now().isoformat(),
            'detailed_results': search_results
        }
        
        return report
    
    def visualize_face_detections(self, image_data: bytes, face_locations: List, 
                                identifications: List[Dict] = None) -> np.ndarray:
        """Create visualization of face detections on image"""
        try:
            # Convert bytes to image array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
            
            # Draw rectangles around faces
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Draw rectangle
                cv2.rectangle(image_array, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add label if identification available
                if identifications and i < len(identifications):
                    ident = identifications[i]
                    label = f"{ident.get('identified_person', 'Unknown')} ({ident.get('confidence', 0):.1f}%)"
                    cv2.putText(image_array, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            return image_array
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None

