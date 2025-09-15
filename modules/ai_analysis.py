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
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class AIAnalysisSystem:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
    def analyze_image_with_ai(self, image_data: bytes, analysis_type: str = "comprehensive") -> Dict:
        """Analyze image using AI for various purposes"""
        try:
            # Convert image to base64 for API
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            analysis_prompts = {
                "comprehensive": """
                Analyze this image comprehensively for investigative purposes. Focus on:
                1. People: Count, descriptions, activities, clothing, interactions
                2. Objects: Weapons, vehicles, suspicious items, evidence
                3. Environment: Location type, lighting conditions, time indicators
                4. Activities: What's happening, suspicious behavior, timeline clues
                5. Anomalies: Anything unusual, out of place, or potentially relevant to an investigation
                Provide detailed observations in JSON format with confidence scores.
                """,
                
                "surveillance": """
                Analyze this image from a surveillance perspective. Look for:
                1. Security threats or suspicious activities
                2. Person identification features (clothing, physical characteristics)
                3. Vehicle information (license plates, make/model, color)
                4. Behavioral patterns that might indicate criminal activity
                5. Environmental factors that could affect investigation
                Respond with JSON containing findings and confidence levels.
                """,
                
                "forensic": """
                Perform forensic analysis of this image. Examine:
                1. Evidence of tampering or manipulation
                2. Digital artifacts or compression issues
                3. Metadata inconsistencies visible in the image
                4. Timeline indicators (clocks, shadows, weather)
                5. Potential evidence items or traces
                Provide forensic assessment in JSON format with analysis confidence.
                """,
                
                "anomaly_detection": """
                Detect anomalies and unusual elements in this image:
                1. Objects that don't belong in the scene
                2. Inconsistent lighting or shadows
                3. Image quality inconsistencies
                4. Suspicious modifications or alterations
                5. Unusual patterns or behaviors
                Return JSON with anomaly descriptions and severity levels.
                """
            }
            
            prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
            
            response = openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            analysis_result['analysis_type'] = analysis_type
            analysis_result['timestamp'] = datetime.now().isoformat()
            analysis_result['model_used'] = 'gpt-5'
            
            return analysis_result
            
        except Exception as e:
            st.error(f"AI image analysis failed: {str(e)}")
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
                    
                    # Analyze frame
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
            
            # Use AI for advanced manipulation detection
            ai_analysis = self.analyze_image_with_ai(image_data, "forensic")
            
            result = {
                'manipulation_indicators': manipulation_indicators,
                'technical_analysis': {
                    'edge_density': edge_density,
                    'noise_estimate': noise_estimate,
                    'smoothness_score': smoothness,
                    'variance_std': variance_std if block_variances else None
                },
                'ai_forensic_analysis': ai_analysis,
                'overall_risk_level': self._calculate_manipulation_risk(manipulation_indicators, ai_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            st.error(f"Manipulation detection failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_manipulation_risk(self, indicators: List[Dict], ai_analysis: Dict) -> str:
        """Calculate overall manipulation risk level"""
        risk_score = 0
        
        # Technical indicators
        high_confidence_indicators = len([i for i in indicators if i.get('confidence', 0) > 0.7])
        medium_confidence_indicators = len([i for i in indicators if 0.4 < i.get('confidence', 0) <= 0.7])
        
        risk_score += high_confidence_indicators * 3
        risk_score += medium_confidence_indicators * 1
        
        # AI analysis contribution
        if 'error' not in ai_analysis and ai_analysis.get('findings'):
            ai_findings = ai_analysis.get('findings', {})
            if 'tampering' in str(ai_findings).lower() or 'manipulation' in str(ai_findings).lower():
                risk_score += 2
        
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
        """Analyze image for suspicious activities and behaviors"""
        try:
            # Use AI for suspicious activity detection
            prompt = """
            Analyze this image for potentially suspicious activities or behaviors. Look for:
            1. Criminal activities (theft, assault, drug dealing, vandalism)
            2. Suspicious behaviors (loitering, surveillance, unusual gatherings)
            3. Security threats (weapons, dangerous items, threatening gestures)
            4. Traffic violations or dangerous driving behaviors
            5. Unusual or out-of-place activities for the environment
            
            For each finding, provide:
            - Activity description
            - Confidence level (0-1)
            - Severity level (low/medium/high)
            - Reasoning for the assessment
            
            Respond in JSON format with detailed findings.
            """
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            response = openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=800
            )
            
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Add computer vision analysis
            cv_analysis = self.detect_objects_opencv(image_data)
            
            # Combine results
            result = {
                'ai_suspicious_activity_analysis': analysis_result,
                'computer_vision_detections': cv_analysis,
                'overall_threat_level': self._assess_threat_level(analysis_result),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'suspicious_activity'
            }
            
            return result
            
        except Exception as e:
            st.error(f"Suspicious activity analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _assess_threat_level(self, analysis_result: Dict) -> str:
        """Assess overall threat level based on analysis results"""
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
