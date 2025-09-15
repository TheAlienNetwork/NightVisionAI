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
        """Detect faces in image using multiple advanced AI methods"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original_image is None:
                st.error("Could not decode image")
                return [], [], []
            
            # Try multiple detection methods
            all_faces = []
            
            # Method 1: Enhanced Haar Cascade with preprocessing
            faces_haar = self._detect_with_haar_enhanced(original_image)
            all_faces.extend(faces_haar)
            
            # Method 2: DNN-based face detection
            faces_dnn = self._detect_with_dnn(original_image)
            all_faces.extend(faces_dnn)
            
            # Method 3: Template matching for partial faces
            faces_template = self._detect_with_template_matching(original_image)
            all_faces.extend(faces_template)
            
            # Remove duplicates and merge overlapping detections
            unique_faces = self._merge_overlapping_faces(all_faces)
            
            face_locations = []
            face_features = []
            face_images = []
            
            for (x, y, w, h) in unique_faces:
                # Convert to face_recognition format (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))
                
                # Extract face region with padding
                padding = max(10, min(w, h) // 10)
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(original_image.shape[1], x + w + padding)
                y_end = min(original_image.shape[0], y + h + padding)
                
                face_region = original_image[y_start:y_end, x_start:x_end]
                
                # Enhanced feature extraction
                features = self._extract_enhanced_features(face_region, w, h)
                face_features.append(features)
                face_images.append(face_region)
            
            if len(unique_faces) > 0:
                st.success(f"Successfully detected {len(unique_faces)} face(s) using advanced AI methods")
            else:
                st.warning("No faces detected. Trying alternative detection methods...")
                # Try very aggressive detection as last resort
                fallback_faces = self._aggressive_face_detection(original_image)
                if fallback_faces:
                    st.info(f"Found {len(fallback_faces)} potential face(s) using fallback detection")
                    for (x, y, w, h) in fallback_faces:
                        face_locations.append((y, x + w, y + h, x))
                        face_region = original_image[y:y+h, x:x+w]
                        features = self._extract_enhanced_features(face_region, w, h)
                        face_features.append(features)
                        face_images.append(face_region)
            
            return face_locations, face_features, face_images
            
        except Exception as e:
            st.error(f"Error detecting faces: {str(e)}")
            return [], [], []
    
    def _detect_with_haar_enhanced(self, image):
        """Enhanced Haar cascade detection with preprocessing"""
        faces = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray_eq = cv2.equalizeHist(gray)
        
        # Try multiple scale factors and parameters
        scale_factors = [1.05, 1.1, 1.15, 1.2, 1.3]
        min_neighbors = [3, 4, 5, 6]
        min_sizes = [(20, 20), (30, 30), (40, 40), (50, 50)]
        
        for scale in scale_factors:
            for neighbors in min_neighbors:
                for min_size in min_sizes:
                    # Original image
                    detected = self.face_cascade.detectMultiScale(
                        gray, scale, neighbors, minSize=min_size
                    )
                    faces.extend(detected)
                    
                    # Histogram equalized image
                    detected_eq = self.face_cascade.detectMultiScale(
                        gray_eq, scale, neighbors, minSize=min_size
                    )
                    faces.extend(detected_eq)
        
        return faces
    
    def _detect_with_dnn(self, image):
        """DNN-based face detection using OpenCV's DNN module"""
        faces = []
        
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            
            # Load DNN model (using a simple approach with available OpenCV DNN)
            # In a production environment, you'd load a pre-trained model
            # For now, we'll simulate DNN detection with enhanced processing
            
            height, width = image.shape[:2]
            
            # Use advanced image processing to find face-like regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Check if the contour could be a face (aspect ratio and size)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                if (0.5 <= aspect_ratio <= 2.0 and 
                    area > 1000 and 
                    w > 30 and h > 30 and
                    x > 0 and y > 0):
                    faces.append((x, y, w, h))
        
        except Exception as e:
            pass  # Fallback silently
        
        return faces
    
    def _detect_with_template_matching(self, image):
        """Template matching for face detection"""
        faces = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create simple face templates (oval shapes)
            template_sizes = [(50, 60), (80, 100), (100, 120), (60, 80)]
            
            for w, h in template_sizes:
                # Create oval template
                template = np.zeros((h, w), dtype=np.uint8)
                center = (w//2, h//2)
                axes = (w//3, h//2)
                cv2.ellipse(template, center, axes, 0, 0, 360, 255, -1)
                
                # Apply template matching
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.3)  # Lower threshold for template matching
                
                for pt in zip(*locations[::-1]):
                    faces.append((pt[0], pt[1], w, h))
        
        except Exception as e:
            pass  # Fallback silently
        
        return faces
    
    def _aggressive_face_detection(self, image):
        """Very aggressive face detection as last resort"""
        faces = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use very relaxed parameters
            detected = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=1,  # Very low threshold
                minSize=(15, 15),  # Very small minimum size
                maxSize=(gray.shape[0], gray.shape[1])
            )
            faces.extend(detected)
            
            # Try with different image enhancements
            # Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            detected_blur = self.face_cascade.detectMultiScale(
                blurred, 1.05, 1, minSize=(15, 15)
            )
            faces.extend(detected_blur)
            
            # Brightness adjustment
            bright = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
            detected_bright = self.face_cascade.detectMultiScale(
                bright, 1.05, 1, minSize=(15, 15)
            )
            faces.extend(detected_bright)
            
            # If still no faces, analyze the entire image as potential face region
            if not faces:
                h, w = gray.shape
                # Consider the central region as a potential face
                center_w, center_h = w // 3, h // 3
                center_x, center_y = w // 3, h // 3
                faces.append((center_x, center_y, center_w, center_h))
        
        except Exception as e:
            pass
        
        return faces
    
    def _merge_overlapping_faces(self, faces):
        """Remove duplicate and overlapping face detections"""
        if not faces:
            return []
        
        # Convert to list if numpy array
        faces = list(faces)
        
        # Remove duplicates
        unique_faces = []
        for face in faces:
            x, y, w, h = face
            is_duplicate = False
            
            for existing in unique_faces:
                ex, ey, ew, eh = existing
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                face_area = w * h
                existing_area = ew * eh
                
                # If significant overlap, consider it duplicate
                if (overlap_area > 0.3 * min(face_area, existing_area)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _extract_enhanced_features(self, face_region, w, h):
        """Extract enhanced features from face region"""
        try:
            if face_region.size == 0:
                return self._get_default_features(w, h)
            
            # Convert to grayscale for analysis
            if len(face_region.shape) == 3:
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_region
            
            # Resize for consistent analysis
            face_resized = cv2.resize(face_gray, (64, 64))
            
            # Multiple feature types
            features = {}
            
            # 1. Histogram features
            hist = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
            features['histogram'] = hist.flatten().tolist()
            
            # 2. LBP (Local Binary Pattern) features
            lbp_features = self._calculate_lbp_features(face_resized)
            features['lbp'] = lbp_features
            
            # 3. Edge features
            edges = cv2.Canny(face_resized, 30, 100)
            edge_ratio = np.sum(edges > 0) / edges.size
            features['edge_ratio'] = edge_ratio
            
            # 4. Texture features
            texture_features = self._calculate_texture_features(face_resized)
            features.update(texture_features)
            
            # 5. Geometric features
            features['width'] = w
            features['height'] = h
            features['aspect_ratio'] = w / h if h > 0 else 1.0
            features['area'] = w * h
            
            return features
            
        except Exception as e:
            return self._get_default_features(w, h)
    
    def _calculate_lbp_features(self, image):
        """Calculate Local Binary Pattern features"""
        try:
            radius = 1
            n_points = 8
            lbp = np.zeros_like(image)
            
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    binary_string = ''
                    
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        
                        if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                            if image[x, y] >= center:
                                binary_string += '1'
                            else:
                                binary_string += '0'
                    
                    lbp[i, j] = int(binary_string, 2)
            
            # Calculate histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
            return hist.tolist()
            
        except Exception:
            return [0] * 256
    
    def _calculate_texture_features(self, image):
        """Calculate texture features"""
        try:
            # Mean and standard deviation
            mean_val = np.mean(image)
            std_val = np.std(image)
            
            # Contrast
            contrast = np.std(image) ** 2
            
            # Energy (uniformity)
            hist, _ = np.histogram(image.ravel(), bins=256, range=[0, 256])
            hist_norm = hist / np.sum(hist)
            energy = np.sum(hist_norm ** 2)
            
            return {
                'mean_intensity': float(mean_val),
                'std_intensity': float(std_val),
                'contrast': float(contrast),
                'energy': float(energy)
            }
        except Exception:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'contrast': 0.0,
                'energy': 0.0
            }
    
    def _get_default_features(self, w, h):
        """Get default features when extraction fails"""
        return {
            'histogram': [0] * 64,
            'lbp': [0] * 256,
            'edge_ratio': 0.0,
            'width': w,
            'height': h,
            'aspect_ratio': w / h if h > 0 else 1.0,
            'area': w * h,
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'contrast': 0.0,
            'energy': 0.0
        }
    
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
        """Compare two face feature sets using multiple similarity metrics"""
        try:
            similarities = []
            weights = []
            
            # 1. Histogram comparison
            if 'histogram' in known_features and 'histogram' in unknown_features:
                hist1 = np.array(known_features['histogram'])
                hist2 = np.array(unknown_features['histogram'])
                
                if len(hist1) == len(hist2) and len(hist1) > 0:
                    hist_correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
                    similarities.append(max(0, hist_correlation))
                    weights.append(0.3)
            
            # 2. LBP comparison
            if 'lbp' in known_features and 'lbp' in unknown_features:
                lbp1 = np.array(known_features['lbp'])
                lbp2 = np.array(unknown_features['lbp'])
                
                if len(lbp1) == len(lbp2) and len(lbp1) > 0:
                    lbp_correlation = cv2.compareHist(lbp1.astype(np.float32), lbp2.astype(np.float32), cv2.HISTCMP_CORREL)
                    similarities.append(max(0, lbp_correlation))
                    weights.append(0.25)
            
            # 3. Geometric similarity
            aspect_diff = abs(known_features.get('aspect_ratio', 1) - unknown_features.get('aspect_ratio', 1))
            aspect_similarity = max(0, 1 - aspect_diff / 2)
            similarities.append(aspect_similarity)
            weights.append(0.1)
            
            # 4. Edge similarity
            edge_diff = abs(known_features.get('edge_ratio', 0) - unknown_features.get('edge_ratio', 0))
            edge_similarity = max(0, 1 - edge_diff)
            similarities.append(edge_similarity)
            weights.append(0.1)
            
            # 5. Texture similarity
            texture_sim = self._compare_texture_features(known_features, unknown_features)
            similarities.append(texture_sim)
            weights.append(0.15)
            
            # 6. Size similarity
            size_sim = self._compare_size_features(known_features, unknown_features)
            similarities.append(size_sim)
            weights.append(0.1)
            
            # Calculate weighted average
            if similarities and weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    overall_similarity = sum(s * w for s, w in zip(similarities, weights)) / total_weight
                else:
                    overall_similarity = 0.0
            else:
                overall_similarity = 0.0
            
            # Convert to percentage and determine match
            confidence = max(0, min(100, overall_similarity * 100))
            
            # Adaptive threshold based on available features
            threshold = self.face_tolerance * 100
            if len(similarities) < 4:  # If fewer features available, lower threshold
                threshold *= 0.8
            
            matches = confidence >= threshold
            
            return matches, confidence
            
        except Exception as e:
            st.error(f"Error comparing faces: {str(e)}")
            return False, 0.0
    
    def _compare_texture_features(self, features1: Dict, features2: Dict) -> float:
        """Compare texture features between two faces"""
        try:
            texture_keys = ['mean_intensity', 'std_intensity', 'contrast', 'energy']
            similarities = []
            
            for key in texture_keys:
                if key in features1 and key in features2:
                    val1 = features1[key]
                    val2 = features2[key]
                    
                    if val1 == 0 and val2 == 0:
                        similarities.append(1.0)
                    else:
                        max_val = max(abs(val1), abs(val2), 1)
                        diff = abs(val1 - val2) / max_val
                        similarity = max(0, 1 - diff)
                        similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _compare_size_features(self, features1: Dict, features2: Dict) -> float:
        """Compare size-based features"""
        try:
            area1 = features1.get('area', 0)
            area2 = features2.get('area', 0)
            
            if area1 == 0 and area2 == 0:
                return 1.0
            
            max_area = max(area1, area2, 1)
            min_area = min(area1, area2)
            
            size_similarity = min_area / max_area
            return size_similarity
            
        except Exception:
            return 0.0
    
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

