import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """Load image from Streamlit uploaded file"""
    try:
        if uploaded_file is not None:
            # Read file as bytes
            file_bytes = uploaded_file.read()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def convert_image_to_bytes(image_array: np.ndarray, format: str = 'JPEG') -> bytes:
    """Convert numpy image array to bytes"""
    try:
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        img_bytes = img_buffer.getvalue()
        
        return img_bytes
    except Exception as e:
        st.error(f"Error converting image to bytes: {str(e)}")
        return b''

def encode_image_base64(image_array: np.ndarray) -> str:
    """Encode image array as base64 string"""
    try:
        img_bytes = convert_image_to_bytes(image_array)
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string
    except Exception as e:
        st.error(f"Error encoding image to base64: {str(e)}")
        return ""

def resize_image(image_array: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    try:
        height, width = image_array.shape[:2]
        
        # Calculate scale factor
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image_array
    except Exception as e:
        st.error(f"Error resizing image: {str(e)}")
        return image_array

def draw_bounding_boxes(image_array: np.ndarray, boxes: List[Tuple], labels: List[str] = None, 
                       colors: List[Tuple] = None) -> np.ndarray:
    """Draw bounding boxes on image"""
    try:
        result_image = image_array.copy()
        
        default_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, box in enumerate(boxes):
            # Handle different box formats
            if len(box) == 4:
                x, y, w, h = box
                x2, y2 = x + w, y + h
            elif len(box) == 2 and len(box[0]) == 2:
                (x, y), (x2, y2) = box
            else:
                continue
            
            # Get color
            color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
            
            # Draw rectangle
            cv2.rectangle(result_image, (int(x), int(y)), (int(x2), int(y2)), color, 2)
            
            # Add label if provided
            if labels and i < len(labels):
                label = labels[i]
                font_scale = 0.7
                thickness = 2
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(result_image, (int(x), int(y) - text_height - 10), 
                            (int(x) + text_width, int(y)), color, -1)
                
                # Draw text
                cv2.putText(result_image, label, (int(x), int(y) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return result_image
    except Exception as e:
        st.error(f"Error drawing bounding boxes: {str(e)}")
        return image_array

def create_image_grid(images: List[np.ndarray], labels: List[str] = None, max_cols: int = 3) -> np.ndarray:
    """Create a grid of images for display"""
    try:
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Ensure all images are the same size
        target_height, target_width = 200, 200
        resized_images = []
        
        for img in images:
            if len(img.shape) == 3:
                resized = cv2.resize(img, (target_width, target_height))
            else:
                resized = cv2.resize(img, (target_width, target_height))
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            resized_images.append(resized)
        
        # Calculate grid dimensions
        num_images = len(resized_images)
        cols = min(max_cols, num_images)
        rows = (num_images + cols - 1) // cols
        
        # Create grid
        grid_height = rows * target_height
        grid_width = cols * target_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            
            y_start = row * target_height
            y_end = y_start + target_height
            x_start = col * target_width
            x_end = x_start + target_width
            
            grid[y_start:y_end, x_start:x_end] = img
            
            # Add label if provided
            if labels and i < len(labels):
                cv2.putText(grid, labels[i], (x_start + 10, y_start + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return grid
    except Exception as e:
        st.error(f"Error creating image grid: {str(e)}")
        return np.zeros((100, 100, 3), dtype=np.uint8)

def extract_frames_from_video(video_bytes: bytes, num_frames: int = 10) -> List[np.ndarray]:
    """Extract frames from video for preview/analysis"""
    try:
        import tempfile
        import os
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            temp_path = tmp_file.name
        
        # Extract frames
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        frame_count = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        os.unlink(temp_path)
        
        return frames
    except Exception as e:
        st.error(f"Error extracting video frames: {str(e)}")
        return []

def apply_image_enhancement(image_array: np.ndarray, enhancement_type: str = "auto") -> np.ndarray:
    """Apply various image enhancements"""
    try:
        enhanced = image_array.copy()
        
        if enhancement_type == "auto":
            # Auto contrast and brightness adjustment
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            
        elif enhancement_type == "contrast":
            # Increase contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
            
        elif enhancement_type == "brightness":
            # Increase brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=30)
            
        elif enhancement_type == "sharpen":
            # Sharpen image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
        elif enhancement_type == "denoise":
            # Denoise image
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    except Exception as e:
        st.error(f"Error enhancing image: {str(e)}")
        return image_array

def get_image_histogram(image_array: np.ndarray) -> dict:
    """Calculate and return image histogram data"""
    try:
        # Calculate histograms for each channel
        histograms = {}
        
        if len(image_array.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image_array], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()
        else:
            # Grayscale
            hist = cv2.calcHist([image_array], [0], None, [256], [0, 256])
            histograms['gray'] = hist.flatten()
        
        return histograms
    except Exception as e:
        st.error(f"Error calculating histogram: {str(e)}")
        return {}

def display_image_with_info(image_array: np.ndarray, title: str = "", info: dict = None):
    """Display image with additional information using Streamlit"""
    try:
        # Display image
        st.image(image_array, caption=title, use_column_width=True)
        
        # Display image info
        if info:
            st.subheader("Image Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Width", info.get('width', image_array.shape[1]))
                st.metric("Height", info.get('height', image_array.shape[0]))
            
            with col2:
                st.metric("Channels", info.get('channels', image_array.shape[2] if len(image_array.shape) > 2 else 1))
                st.metric("Data Type", str(image_array.dtype))
            
            with col3:
                if 'file_size' in info:
                    st.metric("File Size", f"{info['file_size'] / 1024:.1f} KB")
                st.metric("Mean Intensity", f"{np.mean(image_array):.1f}")
    
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
