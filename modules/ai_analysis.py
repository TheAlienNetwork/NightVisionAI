import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
import base64
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import tempfile
import imagehash
import hashlib
import re
from collections import Counter
import math

class AIAnalysisSystem:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
    def analyze_image_with_ai(self, image_data: bytes, analysis_type: str = "comprehensive") -> Dict:
        """Analyze image using local computer vision and pattern recognition"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform comprehensive computer vision analysis
            analysis_result = {
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'Local Computer Vision',
                'findings': {},
                'confidence_scores': {},
                'technical_analysis': {}
            }
            
            # Basic image properties
            height, width = gray.shape
            analysis_result['technical_analysis']['image_dimensions'] = {'width': width, 'height': height}
            analysis_result['technical_analysis']['image_size'] = len(image_data)
            
            # Color analysis
            color_analysis = self._analyze_colors(image)
            analysis_result['findings']['color_analysis'] = color_analysis
            analysis_result['confidence_scores']['color_analysis'] = 0.9
            
            # Object detection using OpenCV
            objects_detected = self.detect_objects_opencv(image_data)
            analysis_result['findings']['object_detection'] = objects_detected
            analysis_result['confidence_scores']['object_detection'] = 0.8
            
            # Edge and texture analysis
            edge_analysis = self._analyze_edges_and_texture(gray)
            analysis_result['findings']['edge_texture_analysis'] = edge_analysis
            analysis_result['confidence_scores']['edge_texture_analysis'] = 0.85
            
            # Brightness and contrast analysis
            lighting_analysis = self._analyze_lighting(gray)
            analysis_result['findings']['lighting_analysis'] = lighting_analysis
            analysis_result['confidence_scores']['lighting_analysis'] = 0.9
            
            # Specific analysis based on type
            if analysis_type == "surveillance":
                surveillance_analysis = self._perform_surveillance_analysis(image, gray)
                analysis_result['findings']['surveillance_specific'] = surveillance_analysis
                analysis_result['confidence_scores']['surveillance_specific'] = 0.75
            
            elif analysis_type == "forensic":
                forensic_analysis = self._perform_forensic_analysis(image, gray)
                analysis_result['findings']['forensic_specific'] = forensic_analysis
                analysis_result['confidence_scores']['forensic_specific'] = 0.8
            
            elif analysis_type == "anomaly_detection":
                anomaly_analysis = self._detect_anomalies(image, gray)
                analysis_result['findings']['anomaly_specific'] = anomaly_analysis
                analysis_result['confidence_scores']['anomaly_specific'] = 0.7
            
            # Generate summary
            analysis_result['summary'] = self._generate_analysis_summary(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            st.error(f"Local image analysis failed: {str(e)}")
            return {'error': str(e), 'analysis_type': analysis_type}
    
    def analyze_video_with_ai(self, video_data: bytes, frame_interval: int = 30) -> List[Dict]:
        """Analyze video by extracting frames and analyzing key moments"""
        try:
            # Save video temporarily
            temp_path = os.path.join(self.temp_dir, "temp_video.mp4")
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Extract frames
            cap = cv2.VideoCapture(temp_path)
            frame_analyses = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze every nth frame
                if frame_number % frame_interval == 0:
                    # Convert frame to bytes
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Analyze frame using local analysis
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    frame_analysis = self.analyze_image_with_ai(frame_bytes, "surveillance")
                    frame_analysis['frame_number'] = frame_number
                    frame_analysis['timestamp'] = timestamp
                    
                    frame_analyses.append(frame_analysis)
                
                frame_number += 1
            
            cap.release()
            os.remove(temp_path)
            
            return frame_analyses
            
        except Exception as e:
            st.error(f"Video analysis failed: {str(e)}")
            return []
    
    def detect_objects_opencv(self, image_data: bytes) -> Dict:
        """Detect objects using OpenCV's built-in classifiers"""
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            detections = {}
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            detections['faces'] = {
                'count': len(faces),
                'locations': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in faces]
            }
            
            # Eye detection
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            detections['eyes'] = {
                'count': len(eyes),
                'locations': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in eyes]
            }
            
            # Car detection (if cascade file exists)
            try:
                car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
                cars = car_cascade.detectMultiScale(gray, 1.3, 5)
                detections['vehicles'] = {
                    'count': len(cars),
                    'locations': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in cars]
                }
            except:
                detections['vehicles'] = {'count': 0, 'locations': [], 'note': 'Vehicle cascade not available'}
            
            # Image properties
            detections['image_properties'] = {
                'width': image.shape[1],
                'height': image.shape[0],
                'channels': image.shape[2],
                'mean_brightness': np.mean(gray),
                'brightness_std': np.std(gray)
            }
            
            detections['timestamp'] = datetime.now().isoformat()
            
            return detections
            
        except Exception as e:
            st.error(f"OpenCV object detection failed: {str(e)}")
            return {'error': str(e)}
    
    def calculate_perceptual_hash(self, image_data: bytes) -> Dict:
        """Calculate perceptual hashes for image similarity detection"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            hashes = {
                'average_hash': str(imagehash.average_hash(image)),
                'perceptual_hash': str(imagehash.phash(image)),
                'difference_hash': str(imagehash.dhash(image)),
                'wavelet_hash': str(imagehash.whash(image))
            }
            
            return hashes
            
        except Exception as e:
            st.error(f"Perceptual hash calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def compare_images_similarity(self, image1_data: bytes, image2_data: bytes) -> Dict:
        """Compare two images for similarity using perceptual hashing"""
        try:
            # Calculate hashes for both images
            hash1 = self.calculate_perceptual_hash(image1_data)
            hash2 = self.calculate_perceptual_hash(image2_data)
            
            if 'error' in hash1 or 'error' in hash2:
                return {'error': 'Failed to calculate hashes'}
            
            # Calculate similarity scores
            similarities = {}
            
            for hash_type in hash1.keys():
                if hash_type != 'error':
                    h1 = imagehash.hex_to_hash(hash1[hash_type])
                    h2 = imagehash.hex_to_hash(hash2[hash_type])
                    
                    # Calculate Hamming distance (lower = more similar)
                    distance = h1 - h2
                    similarity_percentage = max(0, (64 - distance) / 64 * 100)  # Assuming 64-bit hash
                    
                    similarities[hash_type] = {
                        'distance': distance,
                        'similarity_percentage': similarity_percentage
                    }
            
            # Calculate average similarity
            avg_similarity = np.mean([s['similarity_percentage'] for s in similarities.values()])
            
            return {
                'similarities': similarities,
                'average_similarity': avg_similarity,
                'is_likely_match': avg_similarity > 80,
                'is_possible_match': avg_similarity > 60,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Image comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_image_manipulation(self, image_data: bytes) -> Dict:
        """Detect potential image manipulation using AI and computer vision"""
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            manipulation_indicators = []
            
            # Check for JPEG compression artifacts
            # Analyze 8x8 blocks for uniformity (JPEG artifact detection)
            block_variances = []
            for y in range(0, gray.shape[0] - 8, 8):
                for x in range(0, gray.shape[1] - 8, 8):
                    block = gray[y:y+8, x:x+8]
                    block_variances.append(np.var(block))
            
            if block_variances:
                variance_std = np.std(block_variances)
                if variance_std < 20:
                    manipulation_indicators.append({
                        'type': 'Compression Artifacts',
                        'confidence': 0.7,
                        'description': 'Image shows signs of heavy compression which may indicate manipulation'
                    })
            
            # Edge analysis for inconsistencies
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Analyze noise patterns
            noise_estimate = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            
            # Check for unnatural smoothness (possible airbrushing)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            smoothness = np.mean(np.abs(gray.astype(float) - blur.astype(float)))
            
            if smoothness < 5:
                manipulation_indicators.append({
                    'type': 'Excessive Smoothing',
                    'confidence': 0.6,
                    'description': 'Image appears to have unnatural smoothing which may indicate editing'
                })
            
            # Use local analysis for advanced manipulation detection
            local_forensic_analysis = self._perform_local_forensic_analysis(image, gray)
            
            result = {
                'manipulation_indicators': manipulation_indicators,
                'technical_analysis': {
                    'edge_density': edge_density,
                    'noise_estimate': noise_estimate,
                    'smoothness_score': smoothness,
                    'variance_std': variance_std if block_variances else None
                },
                'local_forensic_analysis': local_forensic_analysis,
                'overall_risk_level': self._calculate_manipulation_risk_local(manipulation_indicators, local_forensic_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            st.error(f"Manipulation detection failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_manipulation_risk(self, indicators: List[Dict], ai_analysis: Dict) -> str:
        """Calculate overall manipulation risk level (legacy method)"""
        risk_score = 0
        
        # Technical indicators
        high_confidence_indicators = len([i for i in indicators if i.get('confidence', 0) > 0.7])
        medium_confidence_indicators = len([i for i in indicators if 0.4 < i.get('confidence', 0) <= 0.7])
        
        risk_score += high_confidence_indicators * 3
        risk_score += medium_confidence_indicators * 1
        
        # Local analysis contribution (replacing AI analysis)
        if 'error' not in ai_analysis and ai_analysis.get('findings'):
            local_findings = ai_analysis.get('findings', {})
            if isinstance(local_findings, dict):
                if local_findings.get('compression_artifacts', {}).get('unusual_patterns', False):
                    risk_score += 2
                if local_findings.get('noise_pattern', {}).get('inconsistent', False):
                    risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return 'High'
        elif risk_score >= 3:
            return 'Medium'
        elif risk_score >= 1:
            return 'Low'
        else:
            return 'Minimal'
    
    def analyze_suspicious_activity(self, image_data: bytes) -> Dict:
        """Analyze image for suspicious activities and behaviors using local analysis"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use computer vision analysis
            cv_analysis = self.detect_objects_opencv(image_data)
            
            # Analyze for suspicious patterns
            suspicious_patterns = self._detect_suspicious_patterns(image, gray)
            
            # Motion analysis (if multiple faces/objects detected)
            motion_analysis = self._analyze_potential_motion(gray)
            
            # Combine results
            result = {
                'local_suspicious_activity_analysis': suspicious_patterns,
                'computer_vision_detections': cv_analysis,
                'motion_indicators': motion_analysis,
                'overall_threat_level': self._assess_threat_level_local(suspicious_patterns, cv_analysis),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'suspicious_activity'
            }
            
            return result
            
        except Exception as e:
            st.error(f"Suspicious activity analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_threat_level(self, analysis_result: Dict) -> str:
        """Assess overall threat level based on analysis results (legacy method)"""
        threat_score = 0
        
        findings = analysis_result.get('findings', [])
        if isinstance(findings, list):
            for finding in findings:
                confidence = finding.get('confidence', 0)
                severity = finding.get('severity', 'low')
                
                if severity == 'high':
                    threat_score += confidence * 3
                elif severity == 'medium':
                    threat_score += confidence * 2
                else:
                    threat_score += confidence * 1
        
        if threat_score >= 2.5:
            return 'High'
        elif threat_score >= 1.5:
            return 'Medium'
        elif threat_score >= 0.5:
            return 'Low'
        else:
            return 'Minimal'
    
    def batch_analyze_images(self, image_files: List[Dict], analysis_types: List[str]) -> List[Dict]:
        """Batch analyze multiple images with specified analysis types"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_file in enumerate(image_files):
            status_text.text(f"Analyzing {image_file.get('filename', 'Unknown')}...")
            
            try:
                image_results = {
                    'filename': image_file.get('filename'),
                    'file_id': image_file.get('id'),
                    'analyses': {}
                }
                
                image_data = image_file.get('file_data', b'')
                
                for analysis_type in analysis_types:
                    if analysis_type == 'comprehensive':
                        image_results['analyses']['comprehensive'] = self.analyze_image_with_ai(image_data, 'comprehensive')
                    elif analysis_type == 'surveillance':
                        image_results['analyses']['surveillance'] = self.analyze_image_with_ai(image_data, 'surveillance')
                    elif analysis_type == 'forensic':
                        image_results['analyses']['forensic'] = self.analyze_image_with_ai(image_data, 'forensic')
                    elif analysis_type == 'object_detection':
                        image_results['analyses']['object_detection'] = self.detect_objects_opencv(image_data)
                    elif analysis_type == 'manipulation_detection':
                        image_results['analyses']['manipulation_detection'] = self.detect_image_manipulation(image_data)
                    elif analysis_type == 'suspicious_activity':
                        image_results['analyses']['suspicious_activity'] = self.analyze_suspicious_activity(image_data)
                    elif analysis_type == 'perceptual_hash':
                        image_results['analyses']['perceptual_hash'] = self.calculate_perceptual_hash(image_data)
                
                results.append(image_results)
                
            except Exception as e:
                st.error(f"Error analyzing {image_file.get('filename')}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(image_files))
        
        status_text.text("Batch analysis complete!")
        return results
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze color distribution and properties"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # Calculate dominant colors
            dominant_colors = self._get_dominant_colors(image, k=5)
            
            # Calculate color variance
            color_variance = {
                'blue': float(np.var(image[:,:,0])),
                'green': float(np.var(image[:,:,1])),
                'red': float(np.var(image[:,:,2]))
            }
            
            return {
                'dominant_colors': dominant_colors,
                'color_variance': color_variance,
                'avg_brightness': float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                'color_diversity': len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[List[int]]:
        """Get dominant colors using basic clustering"""
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # Sample for performance
            if len(data) > 10000:
                indices = np.random.choice(len(data), 10000, replace=False)
                data = data[indices]
            
            # Simple k-means clustering approximation
            # Use quantization instead of full k-means for performance
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert back to uint8
            centers = np.uint8(centers)
            
            return centers.tolist()
        except Exception:
            # Fallback: return simple color averages
            return []
    
    def _analyze_edges_and_texture(self, gray: np.ndarray) -> Dict:
        """Analyze edges and texture patterns"""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Texture analysis using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(magnitude)
            
            return {
                'edge_density': float(edge_density),
                'texture_sharpness': float(laplacian_var),
                'average_gradient': float(avg_gradient),
                'edge_count': int(np.sum(edges > 0))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_lighting(self, gray: np.ndarray) -> Dict:
        """Analyze lighting conditions"""
        try:
            # Basic lighting statistics
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Calculate lighting uniformity
            # Divide image into grid and analyze variance
            h, w = gray.shape
            grid_size = 8
            grid_variances = []
            
            for i in range(0, h, h//grid_size):
                for j in range(0, w, w//grid_size):
                    if i+h//grid_size < h and j+w//grid_size < w:
                        block = gray[i:i+h//grid_size, j:j+w//grid_size]
                        grid_variances.append(np.var(block))
            
            lighting_uniformity = 1.0 / (1.0 + np.std(grid_variances)) if grid_variances else 0
            
            return {
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(brightness_std),
                'lighting_uniformity': float(lighting_uniformity),
                'dynamic_range': float(np.max(gray) - np.min(gray))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_surveillance_analysis(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """Perform surveillance-specific analysis"""
        try:
            analysis = {}
            
            # Motion blur detection (approximate)
            kernel = np.ones((1, 15), np.float32) / 15
            motion_blur = cv2.filter2D(gray, -1, kernel)
            motion_score = np.mean(np.abs(gray.astype(float) - motion_blur.astype(float)))
            analysis['motion_blur_score'] = float(motion_score)
            
            # Crowd density estimation (rough approximation)
            # Use contour detection as proxy for objects/people
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            person_sized_objects = len([c for c in contours if cv2.contourArea(c) > 500])
            analysis['estimated_object_count'] = person_sized_objects
            
            # Lighting quality for surveillance
            analysis['surveillance_lighting_quality'] = 'good' if 50 < np.mean(gray) < 200 else 'poor'
            
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_forensic_analysis(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """Perform forensic-specific analysis"""
        try:
            analysis = {}
            
            # JPEG artifact detection
            # Look for 8x8 block patterns typical of JPEG compression
            block_artifacts = self._detect_block_artifacts(gray)
            analysis['jpeg_artifacts'] = block_artifacts
            
            # Noise pattern analysis
            noise_pattern = self._analyze_noise_pattern(gray)
            analysis['noise_analysis'] = noise_pattern
            
            # Timestamp consistency (rough check)
            # Look for digital overlay patterns
            analysis['has_potential_timestamp'] = self._detect_digital_overlay(gray)
            
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_anomalies(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect visual anomalies"""
        try:
            anomalies = []
            
            # Unusual brightness regions
            mean_brightness = np.mean(gray)
            bright_threshold = mean_brightness + 2 * np.std(gray)
            dark_threshold = mean_brightness - 2 * np.std(gray)
            
            bright_regions = np.sum(gray > bright_threshold)
            dark_regions = np.sum(gray < dark_threshold)
            
            if bright_regions > 0.1 * gray.size:
                anomalies.append({
                    'type': 'unusual_bright_regions',
                    'severity': 'medium',
                    'description': 'Large areas with unusual brightness detected'
                })
            
            if dark_regions > 0.1 * gray.size:
                anomalies.append({
                    'type': 'unusual_dark_regions',
                    'severity': 'medium',
                    'description': 'Large areas with unusual darkness detected'
                })
            
            # Edge discontinuities
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.3:
                anomalies.append({
                    'type': 'high_edge_density',
                    'severity': 'low',
                    'description': 'Unusually high number of edges detected'
                })
            
            return {'anomalies': anomalies, 'anomaly_count': len(anomalies)}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_analysis_summary(self, analysis_result: Dict) -> str:
        """Generate a human-readable summary"""
        try:
            findings = analysis_result.get('findings', {})
            summary_parts = []
            
            # Color analysis summary
            if 'color_analysis' in findings:
                color_data = findings['color_analysis']
                if 'avg_brightness' in color_data:
                    brightness = color_data['avg_brightness']
                    if brightness > 180:
                        summary_parts.append("Image appears bright and well-lit.")
                    elif brightness < 80:
                        summary_parts.append("Image appears dark with low lighting.")
                    else:
                        summary_parts.append("Image has moderate lighting conditions.")
            
            # Object detection summary
            if 'object_detection' in findings:
                obj_data = findings['object_detection']
                if obj_data.get('faces', {}).get('count', 0) > 0:
                    face_count = obj_data['faces']['count']
                    summary_parts.append(f"Detected {face_count} face(s) in the image.")
            
            # Edge and texture summary
            if 'edge_texture_analysis' in findings:
                edge_data = findings['edge_texture_analysis']
                if edge_data.get('edge_density', 0) > 0.2:
                    summary_parts.append("Image contains high detail with many edges and textures.")
                elif edge_data.get('edge_density', 0) < 0.05:
                    summary_parts.append("Image appears smooth with minimal texture detail.")
            
            return " ".join(summary_parts) if summary_parts else "Analysis completed with standard results."
        
        except Exception:
            return "Summary generation failed."
    
    def _detect_suspicious_patterns(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect patterns that might indicate suspicious activity"""
        try:
            patterns = []
            
            # Detect clustering of objects/people
            contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            
            if len(large_contours) > 5:
                patterns.append({
                    'type': 'crowd_gathering',
                    'confidence': 0.6,
                    'description': 'Multiple large objects or people detected in close proximity'
                })
            
            # Detect unusual movement patterns (blur analysis)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                patterns.append({
                    'type': 'motion_blur',
                    'confidence': 0.7,
                    'description': 'Image shows signs of motion blur, indicating fast movement'
                })
            
            return {'patterns': patterns, 'pattern_count': len(patterns)}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_potential_motion(self, gray: np.ndarray) -> Dict:
        """Analyze indicators of motion in the image"""
        try:
            # Motion blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Gradient analysis for directional blur
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            motion_indicators = {
                'blur_score': float(laplacian_var),
                'is_blurry': laplacian_var < 100,
                'horizontal_motion': float(np.mean(np.abs(grad_x))),
                'vertical_motion': float(np.mean(np.abs(grad_y)))
            }
            
            return motion_indicators
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_threat_level_local(self, suspicious_patterns: Dict, cv_analysis: Dict) -> str:
        """Assess threat level based on local analysis"""
        try:
            threat_score = 0
            
            # Suspicious patterns contribution
            patterns = suspicious_patterns.get('patterns', [])
            for pattern in patterns:
                confidence = pattern.get('confidence', 0)
                if pattern.get('type') == 'crowd_gathering':
                    threat_score += confidence * 2
                elif pattern.get('type') == 'motion_blur':
                    threat_score += confidence * 1
            
            # Face detection contribution
            face_count = cv_analysis.get('faces', {}).get('count', 0)
            if face_count > 10:
                threat_score += 1.5
            elif face_count > 5:
                threat_score += 1.0
            
            # Determine threat level
            if threat_score >= 3.0:
                return 'High'
            elif threat_score >= 1.5:
                return 'Medium'
            elif threat_score >= 0.5:
                return 'Low'
            else:
                return 'Minimal'
        except Exception:
            return 'Unknown'
    
    def _perform_local_forensic_analysis(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """Perform local forensic analysis"""
        try:
            analysis = {}
            
            # Digital artifact detection
            analysis['compression_artifacts'] = self._detect_compression_artifacts(gray)
            analysis['noise_pattern'] = self._analyze_noise_pattern(gray)
            analysis['edge_consistency'] = self._analyze_edge_consistency(gray)
            
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_manipulation_risk_local(self, indicators: List[Dict], forensic_analysis: Dict) -> str:
        """Calculate manipulation risk using local analysis"""
        try:
            risk_score = 0
            
            # Technical indicators
            high_confidence_indicators = len([i for i in indicators if i.get('confidence', 0) > 0.7])
            risk_score += high_confidence_indicators * 2
            
            # Forensic analysis contribution
            if forensic_analysis.get('compression_artifacts', {}).get('unusual_patterns', False):
                risk_score += 1
            if forensic_analysis.get('noise_pattern', {}).get('inconsistent', False):
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 4:
                return 'High'
            elif risk_score >= 2:
                return 'Medium'
            elif risk_score >= 1:
                return 'Low'
            else:
                return 'Minimal'
        except Exception:
            return 'Unknown'
    
    def _detect_block_artifacts(self, gray: np.ndarray) -> Dict:
        """Detect JPEG block artifacts"""
        try:
            # Analyze 8x8 blocks for JPEG artifacts
            h, w = gray.shape
            block_variances = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8]
                    block_variances.append(np.var(block))
            
            if block_variances:
                variance_std = np.std(block_variances)
                return {
                    'block_variance_std': float(variance_std),
                    'has_artifacts': variance_std < 20
                }
            
            return {'block_variance_std': 0, 'has_artifacts': False}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_noise_pattern(self, gray: np.ndarray) -> Dict:
        """Analyze noise patterns in the image"""
        try:
            # Estimate noise using Gaussian blur difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(float) - blurred.astype(float)
            noise_std = np.std(noise)
            
            return {
                'noise_level': float(noise_std),
                'is_noisy': noise_std > 10
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_digital_overlay(self, gray: np.ndarray) -> bool:
        """Detect potential digital overlays like timestamps"""
        try:
            # Look for consistent rectangular regions with high contrast
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours that might be text overlays
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum size for text
                    # Check if contour is roughly rectangular
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if area / hull_area > 0.8:  # Roughly rectangular
                        return True
            
            return False
        except Exception:
            return False
    
    def _detect_compression_artifacts(self, gray: np.ndarray) -> Dict:
        """Detect compression artifacts"""
        try:
            # Simple compression artifact detection
            # Look for blocking artifacts typical of lossy compression
            artifacts = self._detect_block_artifacts(gray)
            
            return {
                'unusual_patterns': artifacts.get('has_artifacts', False),
                'compression_score': artifacts.get('block_variance_std', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_edge_consistency(self, gray: np.ndarray) -> Dict:
        """Analyze edge consistency for manipulation detection"""
        try:
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze edge density in different regions
            h, w = gray.shape
            regions = [
                edges[0:h//2, 0:w//2],  # Top-left
                edges[0:h//2, w//2:w],  # Top-right
                edges[h//2:h, 0:w//2],  # Bottom-left
                edges[h//2:h, w//2:w]   # Bottom-right
            ]
            
            edge_densities = [np.sum(region > 0) / region.size for region in regions]
            edge_variance = np.var(edge_densities)
            
            return {
                'edge_consistency': float(1.0 / (1.0 + edge_variance)),
                'edge_variance': float(edge_variance)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def generate_ai_analysis_report(self, analysis_results: List[Dict]) -> Dict:
        """Generate comprehensive AI analysis report"""
        if not analysis_results:
            return {'error': 'No analysis results provided'}
        
        report = {
            'report_id': f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_timestamp': datetime.now().isoformat(),
            'total_files_analyzed': len(analysis_results),
            'analysis_summary': {},
            'key_findings': [],
            'threat_assessment': {'high': 0, 'medium': 0, 'low': 0, 'minimal': 0},
            'detailed_results': analysis_results
        }
        
        # Analyze results
        analysis_types_performed = set()
        suspicious_files = []
        high_confidence_findings = []
        
        for result in analysis_results:
            filename = result.get('filename', 'Unknown')
            analyses = result.get('analyses', {})
            
            for analysis_type, analysis_data in analyses.items():
                analysis_types_performed.add(analysis_type)
                
                # Check for high-confidence findings
                if analysis_type == 'suspicious_activity':
                    threat_level = analysis_data.get('overall_threat_level', 'Minimal')
                    report['threat_assessment'][threat_level.lower()] += 1
                    
                    if threat_level in ['High', 'Medium']:
                        suspicious_files.append({
                            'filename': filename,
                            'threat_level': threat_level,
                            'analysis_type': analysis_type
                        })
                
                elif analysis_type == 'manipulation_detection':
                    risk_level = analysis_data.get('overall_risk_level', 'Minimal')
                    if risk_level in ['High', 'Medium']:
                        high_confidence_findings.append({
                            'filename': filename,
                            'finding_type': 'Potential Manipulation',
                            'risk_level': risk_level
                        })
                
                elif analysis_type == 'object_detection':
                    faces_detected = analysis_data.get('faces', {}).get('count', 0)
                    if faces_detected > 5:
                        high_confidence_findings.append({
                            'filename': filename,
                            'finding_type': 'Multiple Faces Detected',
                            'count': faces_detected
                        })
        
        report['analysis_summary'] = {
            'analysis_types_performed': list(analysis_types_performed),
            'suspicious_files_count': len(suspicious_files),
            'high_confidence_findings_count': len(high_confidence_findings)
        }
        
        report['key_findings'] = suspicious_files + high_confidence_findings
        
        return report
