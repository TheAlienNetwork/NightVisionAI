import streamlit as st
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import magic
from PIL import Image, ExifTags
import cv2
import pandas as pd
from datetime import datetime
import tempfile

class FileProcessor:
    def __init__(self):
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        self.supported_document_formats = ['.pdf', '.docx', '.txt', '.csv', '.xlsx']
        self.temp_dir = tempfile.mkdtemp()
        
    def get_file_hash(self, file_data: bytes) -> str:
        """Generate SHA-256 hash of file"""
        return hashlib.sha256(file_data).hexdigest()
    
    def get_file_type(self, file_data: bytes, filename: str) -> str:
        """Determine file type using magic numbers and extension"""
        try:
            mime_type = magic.from_buffer(file_data, mime=True)
            return mime_type
        except:
            # Fallback to extension-based detection
            ext = Path(filename).suffix.lower()
            if ext in self.supported_image_formats:
                return f"image/{ext[1:]}"
            elif ext in self.supported_video_formats:
                return f"video/{ext[1:]}"
            elif ext in self.supported_document_formats:
                return f"application/{ext[1:]}"
            else:
                return "application/octet-stream"
    
    def extract_image_metadata(self, file_data: bytes) -> Dict:
        """Extract EXIF and other metadata from image"""
        metadata = {}
        
        try:
            # Save temporary file for processing
            temp_path = os.path.join(self.temp_dir, "temp_image")
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            # Extract EXIF data
            image = Image.open(temp_path)
            metadata['format'] = image.format
            metadata['mode'] = image.mode
            metadata['size'] = image.size
            metadata['has_transparency'] = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            
            # EXIF data
            if hasattr(image, '_getexif'):
                exif_data = image._getexif()
                if exif_data:
                    exif = {}
                    for tag, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        exif[tag_name] = str(value)
                    metadata['exif'] = exif
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def extract_video_metadata(self, file_data: bytes) -> Dict:
        """Extract metadata from video files"""
        metadata = {}
        
        try:
            # Save temporary file for processing
            temp_path = os.path.join(self.temp_dir, "temp_video.mp4")
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            # Use OpenCV to extract video properties
            cap = cv2.VideoCapture(temp_path)
            
            if cap.isOpened():
                metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
                metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata['duration'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
                metadata['codec'] = int(cap.get(cv2.CAP_PROP_FOURCC))
                
            cap.release()
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def extract_document_metadata(self, file_data: bytes, filename: str) -> Dict:
        """Extract metadata from documents"""
        metadata = {}
        ext = Path(filename).suffix.lower()
        
        try:
            if ext == '.csv':
                # Save temporary file and analyze CSV
                temp_path = os.path.join(self.temp_dir, "temp.csv")
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                df = pd.read_csv(temp_path)
                metadata['rows'] = len(df)
                metadata['columns'] = len(df.columns)
                metadata['column_names'] = df.columns.tolist()
                metadata['data_types'] = df.dtypes.to_dict()
                
                os.remove(temp_path)
                
            elif ext == '.xlsx':
                # Analyze Excel file
                temp_path = os.path.join(self.temp_dir, "temp.xlsx")
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                df = pd.read_excel(temp_path)
                metadata['rows'] = len(df)
                metadata['columns'] = len(df.columns)
                metadata['column_names'] = df.columns.tolist()
                metadata['data_types'] = df.dtypes.to_dict()
                
                os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def process_file(self, uploaded_file) -> Dict:
        """Process uploaded file and extract all relevant information"""
        if uploaded_file is None:
            return {}
        
        file_data = uploaded_file.read()
        filename = uploaded_file.name
        file_size = len(file_data)
        
        # Basic file information
        file_info = {
            'filename': filename,
            'size': file_size,
            'hash': self.get_file_hash(file_data),
            'mime_type': self.get_file_type(file_data, filename),
            'upload_timestamp': datetime.now().isoformat(),
        }
        
        # Extract specific metadata based on file type
        if file_info['mime_type'].startswith('image/'):
            file_info['metadata'] = self.extract_image_metadata(file_data)
            file_info['category'] = 'image'
        elif file_info['mime_type'].startswith('video/'):
            file_info['metadata'] = self.extract_video_metadata(file_data)
            file_info['category'] = 'video'
        elif filename.endswith(('.csv', '.xlsx')):
            file_info['metadata'] = self.extract_document_metadata(file_data, filename)
            file_info['category'] = 'dataset'
        else:
            file_info['metadata'] = {}
            file_info['category'] = 'document'
        
        # Store file data for processing
        file_info['file_data'] = file_data
        
        return file_info
    
    def save_file_to_disk(self, file_data: bytes, filename: str, case_id: int) -> str:
        """Save file to disk and return path"""
        # Create case directory if it doesn't exist
        case_dir = f"evidence/case_{case_id}"
        os.makedirs(case_dir, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        file_path = os.path.join(case_dir, unique_filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        return file_path
    
    def batch_process_files(self, uploaded_files: List) -> List[Dict]:
        """Process multiple files at once"""
        processed_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                file_info = self.process_file(uploaded_file)
                processed_files.append(file_info)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        return processed_files
    
    def detect_duplicates(self, processed_files: List[Dict]) -> Dict[str, List[str]]:
        """Detect duplicate files based on hash"""
        hash_to_files = {}
        duplicates = {}
        
        for file_info in processed_files:
            file_hash = file_info.get('hash')
            filename = file_info.get('filename')
            
            if file_hash in hash_to_files:
                if file_hash not in duplicates:
                    duplicates[file_hash] = [hash_to_files[file_hash], filename]
                else:
                    duplicates[file_hash].append(filename)
            else:
                hash_to_files[file_hash] = filename
        
        return duplicates
    
    def get_processing_summary(self, processed_files: List[Dict]) -> Dict:
        """Generate processing summary statistics"""
        if not processed_files:
            return {}
        
        categories = {}
        total_size = 0
        formats = {}
        
        for file_info in processed_files:
            category = file_info.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            total_size += file_info.get('size', 0)
            
            mime_type = file_info.get('mime_type', 'unknown')
            formats[mime_type] = formats.get(mime_type, 0) + 1
        
        return {
            'total_files': len(processed_files),
            'total_size': total_size,
            'categories': categories,
            'formats': formats,
            'duplicates': self.detect_duplicates(processed_files)
        }
