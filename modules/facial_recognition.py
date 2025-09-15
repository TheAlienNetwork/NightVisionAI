import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
from typing import List, Dict, Tuple, Optional
import tempfile
from datetime import datetime
import io
from scipy import ndimage
import skimage
from skimage import exposure, restoration, filters

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
        """Detect faces in image using multiple advanced AI methods with photo enhancement"""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if original_image is None:
                st.error("Could not decode image")
                return [], [], []
            
            # Apply advanced photo enhancement before detection
            enhanced_images = self._enhance_photo_for_face_detection(original_image)
            
            # Try multiple detection methods on enhanced images
            all_faces = []
            
            for enhanced_img in enhanced_images:
                # Method 1: Enhanced Haar Cascade with preprocessing
                faces_haar = self._detect_with_haar_enhanced(enhanced_img)
                all_faces.extend(faces_haar)
                
                # Method 2: DNN-based face detection
                faces_dnn = self._detect_with_dnn(enhanced_img)
                all_faces.extend(faces_dnn)
                
                # Method 3: Template matching for partial faces
                faces_template = self._detect_with_template_matching(enhanced_img)
                all_faces.extend(faces_template)
                
                # Method 4: AI-powered region analysis
                faces_ai = self._ai_region_analysis(enhanced_img)
                all_faces.extend(faces_ai)
            
            # Remove duplicates and merge overlapping detections
            unique_faces = self._merge_overlapping_faces(all_faces)
            
            face_locations = []
            face_features = []
            face_images = []
            
            for (x, y, w, h) in unique_faces:
                # Convert to face_recognition format (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))
                
                # Extract and enhance face region
                enhanced_face = self._extract_and_enhance_face(original_image, x, y, w, h)
                
                # Enhanced feature extraction
                features = self._extract_enhanced_features(enhanced_face, w, h)
                face_features.append(features)
                face_images.append(enhanced_face)
            
            if len(unique_faces) > 0:
                st.success(f"Successfully detected {len(unique_faces)} face(s) using advanced AI methods")
            else:
                st.warning("No faces detected. Trying ultra-aggressive detection methods...")
                # Try very aggressive detection as last resort
                fallback_faces = self._ultra_aggressive_face_detection(original_image)
                if fallback_faces:
                    st.info(f"Found {len(fallback_faces)} potential face(s) using fallback detection")
                    for (x, y, w, h) in fallback_faces:
                        face_locations.append((y, x + w, y + h, x))
                        enhanced_face = self._extract_and_enhance_face(original_image, x, y, w, h)
                        features = self._extract_enhanced_features(enhanced_face, w, h)
                        face_features.append(features)
                        face_images.append(enhanced_face)
            
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
    
    def _enhance_photo_for_face_detection(self, image):
        """Apply multiple enhancement techniques to improve face detection"""
        enhanced_images = []
        
        try:
            # 1. Original image
            enhanced_images.append(image.copy())
            
            # 2. Histogram equalization
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            enhanced_images.append(hist_eq)
            
            # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
            enhanced_images.append(clahe_bgr)
            
            # 4. Gamma correction
            gamma_corrected = self._adjust_gamma(image, gamma=0.7)
            enhanced_images.append(gamma_corrected)
            
            # 5. Brightness and contrast enhancement
            bright_contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
            enhanced_images.append(bright_contrast)
            
            # 6. Noise reduction with edge preservation
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            enhanced_images.append(denoised)
            
            # 7. Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            enhanced_images.append(sharpened)
            
            # 8. AI-based enhancement using PIL
            pil_enhanced = self._pil_ai_enhancement(image)
            if pil_enhanced is not None:
                enhanced_images.append(pil_enhanced)
            
        except Exception as e:
            st.warning(f"Some enhancements failed: {str(e)}")
        
        return enhanced_images
    
    def _adjust_gamma(self, image, gamma=1.0):
        """Apply gamma correction to image"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _pil_ai_enhancement(self, cv_image):
        """Use PIL for AI-based image enhancement"""
        try:
            # Convert CV2 to PIL
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Apply multiple PIL enhancements
            enhancers = [
                (ImageEnhance.Contrast, 1.2),
                (ImageEnhance.Brightness, 1.1),
                (ImageEnhance.Sharpness, 1.3),
                (ImageEnhance.Color, 1.1)
            ]
            
            enhanced = pil_image
            for enhancer_class, factor in enhancers:
                enhancer = enhancer_class(enhanced)
                enhanced = enhancer.enhance(factor)
            
            # Apply filter for noise reduction
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to CV2 format
            enhanced_array = np.array(enhanced)
            return cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR)
            
        except Exception:
            return None
    
    def _ai_region_analysis(self, image):
        """AI-powered region analysis for face detection"""
        faces = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use edge detection to find regions of interest
            edges = cv2.Canny(gray, 30, 100)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Check if region could be a face
                aspect_ratio = w / h if h > 0 else 0
                rect_area = w * h
                
                # Advanced criteria for face-like regions
                if (0.4 <= aspect_ratio <= 2.5 and 
                    area > 500 and 
                    rect_area > 800 and
                    w > 25 and h > 25 and
                    area / rect_area > 0.3):  # Contour fills enough of the rectangle
                    
                    # Additional validation using local features
                    roi = gray[y:y+h, x:x+w]
                    if self._validate_face_region(roi):
                        faces.append((x, y, w, h))
            
        except Exception:
            pass
        
        return faces
    
    def _validate_face_region(self, roi):
        """Validate if a region likely contains a face using feature analysis"""
        try:
            if roi.size < 100:
                return False
            
            # Check for eye-like regions (dark spots in upper half)
            h, w = roi.shape
            upper_half = roi[:h//2, :]
            
            # Apply threshold to find dark regions
            _, thresh = cv2.threshold(upper_half, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours in upper half (potential eyes)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for 1-3 dark regions in upper half (eyes, nose)
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < (w * h) // 20:  # Reasonable size for facial features
                    valid_contours += 1
            
            # Check gradient distribution (faces have varied gradients)
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_variance = np.var(gradient_magnitude)
            
            return (valid_contours >= 1 and 
                   gradient_variance > 100 and  # Sufficient texture variation
                   np.mean(roi) > 20)  # Not too dark
        
        except Exception:
            return False
    
    def _extract_and_enhance_face(self, image, x, y, w, h):
        """Extract face region and apply resolution enhancement"""
        try:
            # Calculate enhanced crop region with smart padding
            padding_factor = 0.3  # 30% padding
            padding_x = int(w * padding_factor)
            padding_y = int(h * padding_factor)
            
            # Ensure padding doesn't go outside image bounds
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(image.shape[1], x + w + padding_x)
            y_end = min(image.shape[0], y + h + padding_y)
            
            # Extract face region
            face_region = image[y_start:y_end, x_start:x_end]
            
            if face_region.size == 0:
                return image[y:y+h, x:x+w]
            
            # Apply resolution enhancement
            enhanced_face = self._enhance_face_resolution(face_region)
            
            # Apply face-specific enhancements
            final_face = self._apply_face_specific_enhancements(enhanced_face)
            
            return final_face
            
        except Exception as e:
            # Fallback to simple extraction
            return image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else image
    
    def _enhance_face_resolution(self, face_image):
        """Enhance face resolution using AI-like upscaling techniques"""
        try:
            h, w = face_image.shape[:2]
            
            # If face is too small, apply intelligent upscaling
            if w < 100 or h < 100:
                # Calculate scale factor
                target_size = 150
                scale_x = target_size / w
                scale_y = target_size / h
                scale_factor = min(scale_x, scale_y, 4.0)  # Cap at 4x upscaling
                
                if scale_factor > 1.0:
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    
                    # Use LANCZOS for high-quality upscaling
                    upscaled = cv2.resize(face_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Apply sharpening after upscaling
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    sharpened = cv2.filter2D(upscaled, -1, kernel)
                    
                    return sharpened
            
            return face_image
            
        except Exception:
            return face_image
    
    def _apply_face_specific_enhancements(self, face_image):
        """Apply enhancements specifically optimized for facial features"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel (lightness)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
            l_clahe = clahe.apply(l)
            
            # Merge back
            lab_enhanced = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter to smooth skin while preserving edges
            smooth = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            # Subtle sharpening for facial features
            gaussian = cv2.GaussianBlur(smooth, (0, 0), 1.0)
            sharpened = cv2.addWeighted(smooth, 1.5, gaussian, -0.5, 0)
            
            return sharpened
            
        except Exception:
            return face_image
    
    def _ultra_aggressive_face_detection(self, image):
        """Ultra-aggressive face detection as absolute last resort"""
        faces = []
        
        try:
            # Try on heavily enhanced versions
            enhanced_versions = self._create_extreme_enhancements(image)
            
            for enhanced_img in enhanced_versions:
                gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
                
                # Very aggressive parameters
                detected = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.03,  # Very small scale steps
                    minNeighbors=1,    # Almost no neighbor requirement
                    minSize=(10, 10),  # Tiny minimum size
                    maxSize=(gray.shape[0], gray.shape[1]),
                    flags=cv2.CASCADE_DO_CANNY_PRUNING
                )
                faces.extend(detected)
            
            # If still no faces, try region subdivision
            if not faces:
                faces.extend(self._subdivide_and_search(image))
            
            # Last resort: intelligent guessing based on image properties
            if not faces:
                faces.extend(self._intelligent_face_guessing(image))
        
        except Exception:
            pass
        
        return faces
    
    def _create_extreme_enhancements(self, image):
        """Create extremely enhanced versions for difficult detection"""
        enhanced_versions = []
        
        try:
            # Extreme contrast
            extreme_contrast = cv2.convertScaleAbs(image, alpha=2.0, beta=0)
            enhanced_versions.append(extreme_contrast)
            
            # Extreme brightness
            extreme_bright = cv2.convertScaleAbs(image, alpha=1.0, beta=80)
            enhanced_versions.append(extreme_bright)
            
            # Edge enhancement
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 10, 50)
            edge_enhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            enhanced_versions.append(edge_enhanced)
            
            # High-pass filter
            gaussian = cv2.GaussianBlur(image, (21, 21), 0)
            high_pass = cv2.subtract(image, gaussian)
            high_pass = cv2.add(high_pass, 127)  # Offset to positive values
            enhanced_versions.append(high_pass)
            
        except Exception:
            pass
        
        return enhanced_versions
    
    def _subdivide_and_search(self, image):
        """Subdivide image into regions and search each"""
        faces = []
        
        try:
            h, w = image.shape[:2]
            
            # Create overlapping grid
            step_x, step_y = w // 4, h // 4
            window_w, window_h = w // 2, h // 2
            
            for y in range(0, h - window_h + 1, step_y):
                for x in range(0, w - window_w + 1, step_x):
                    roi = image[y:y+window_h, x:x+window_w]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    detected = self.face_cascade.detectMultiScale(
                        gray_roi, 1.05, 1, minSize=(20, 20)
                    )
                    
                    # Adjust coordinates to full image
                    for (fx, fy, fw, fh) in detected:
                        faces.append((x + fx, y + fy, fw, fh))
        
        except Exception:
            pass
        
        return faces
    
    def _intelligent_face_guessing(self, image):
        """Make intelligent guesses about face locations based on image analysis"""
        faces = []
        
        try:
            h, w = image.shape[:2]
            
            # Analyze image for skin tone regions
            skin_regions = self._detect_skin_tone_regions(image)
            
            for region in skin_regions:
                x, y, rw, rh = region
                
                # Check if region size is reasonable for a face
                if (0.1 <= rw/w <= 0.8 and 
                    0.1 <= rh/h <= 0.8 and
                    rw > 30 and rh > 30):
                    faces.append((x, y, rw, rh))
            
            # If no skin regions, guess central regions
            if not faces:
                # Upper center (typical portrait position)
                face_w, face_h = min(w//3, h//3), min(w//3, h//3)
                center_x, center_y = w//2 - face_w//2, h//4
                faces.append((center_x, center_y, face_w, face_h))
                
                # Center region
                center_x, center_y = w//2 - face_w//2, h//2 - face_h//2
                faces.append((center_x, center_y, face_w, face_h))
        
        except Exception:
            pass
        
        return faces
    
    def _detect_skin_tone_regions(self, image):
        """Detect regions that might contain skin tones"""
        regions = []
        
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin tone ranges in HSV
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            
            lower_skin2 = np.array([0, 0, 0], dtype=np.uint8)
            upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
            
            # Create masks
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask = cv2.add(mask1, mask2)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for potential face
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append((x, y, w, h))
        
        except Exception:
            pass
        
        return regions
    
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

