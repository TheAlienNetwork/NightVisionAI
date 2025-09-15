import streamlit as st
import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import magic
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
import tempfile
import zipfile
import py7zr
import rarfile
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

class DigitalForensicsToolkit:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.hash_algorithms = ['md5', 'sha1', 'sha256', 'sha512']
        self.supported_archives = ['.zip', '.7z', '.rar', '.tar', '.gz']
        
    def calculate_file_hashes(self, file_data: bytes) -> Dict[str, str]:
        """Calculate multiple hash values for file integrity verification"""
        hashes = {}
        
        for algorithm in self.hash_algorithms:
            hash_func = hashlib.new(algorithm)
            hash_func.update(file_data)
            hashes[algorithm] = hash_func.hexdigest()
        
        return hashes
    
    def extract_comprehensive_metadata(self, file_data: bytes, filename: str) -> Dict:
        """Extract comprehensive metadata from various file types"""
        metadata = {
            'filename': filename,
            'file_size': len(file_data),
            'file_type': self._detect_file_type(file_data),
            'extraction_timestamp': datetime.now().isoformat(),
            'hashes': self.calculate_file_hashes(file_data)
        }
        
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext in ['.jpg', '.jpeg', '.tiff', '.png']:
                metadata['image_metadata'] = self._extract_image_metadata(file_data)
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                metadata['video_metadata'] = self._extract_video_metadata(file_data)
            elif file_ext in ['.pdf']:
                metadata['document_metadata'] = self._extract_pdf_metadata(file_data)
            elif file_ext in self.supported_archives:
                metadata['archive_metadata'] = self._extract_archive_metadata(file_data, file_ext)
            
        except Exception as e:
            metadata['extraction_error'] = str(e)
        
        return metadata
    
    def _detect_file_type(self, file_data: bytes) -> str:
        """Detect file type using magic numbers"""
        try:
            mime_type = magic.from_buffer(file_data, mime=True)
            return mime_type
        except:
            return "application/octet-stream"
    
    def _extract_image_metadata(self, file_data: bytes) -> Dict:
        """Extract detailed image metadata including EXIF, GPS, and camera info"""
        metadata = {}
        
        try:
            # Save temporary file for EXIF extraction
            temp_path = os.path.join(self.temp_dir, "temp_image")
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            # Extract EXIF using PIL
            image = Image.open(temp_path)
            
            # Basic image info
            metadata['format'] = image.format
            metadata['mode'] = image.mode
            metadata['size'] = image.size
            metadata['has_transparency'] = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            
            # EXIF data extraction
            if hasattr(image, '_getexif'):
                exif_data = image._getexif()
                if exif_data:
                    exif_dict = {}
                    gps_data = {}
                    
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        
                        if tag_name == 'GPSInfo':
                            # Extract GPS coordinates
                            for gps_tag in value:
                                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                                gps_data[gps_tag_name] = value[gps_tag]
                            
                            # Convert GPS to decimal degrees
                            if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                                lat = self._convert_gps_to_decimal(gps_data['GPSLatitude'], gps_data.get('GPSLatitudeRef'))
                                lon = self._convert_gps_to_decimal(gps_data['GPSLongitude'], gps_data.get('GPSLongitudeRef'))
                                metadata['gps_coordinates'] = {'latitude': lat, 'longitude': lon}
                        
                        # Convert complex values to strings
                        if isinstance(value, (tuple, list)):
                            exif_dict[tag_name] = str(value)
                        else:
                            exif_dict[tag_name] = value
                    
                    metadata['exif'] = exif_dict
                    if gps_data:
                        metadata['gps_info'] = gps_data
            
            # Additional metadata using exifread
            with open(temp_path, 'rb') as f:
                tags = exifread.process_file(f)
                if tags:
                    exifread_data = {}
                    for key, value in tags.items():
                        exifread_data[key] = str(value)
                    metadata['exifread_data'] = exifread_data
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def _convert_gps_to_decimal(self, coordinate: tuple, reference: str) -> float:
        """Convert GPS coordinates from EXIF format to decimal degrees"""
        try:
            degrees = float(coordinate[0])
            minutes = float(coordinate[1])
            seconds = float(coordinate[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            if reference in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except:
            return 0.0
    
    def _extract_video_metadata(self, file_data: bytes) -> Dict:
        """Extract video file metadata"""
        metadata = {}
        
        try:
            # Save temporary file
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
                metadata['duration_seconds'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
                metadata['codec_fourcc'] = int(cap.get(cv2.CAP_PROP_FOURCC))
                metadata['bit_rate'] = cap.get(cv2.CAP_PROP_BITRATE)
                
                # Extract a few frames for analysis
                frames_info = []
                frame_positions = [0, metadata['frame_count'] // 4, metadata['frame_count'] // 2, 3 * metadata['frame_count'] // 4]
                
                for pos in frame_positions:
                    if pos < metadata['frame_count']:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ret, frame = cap.read()
                        if ret:
                            # Calculate frame statistics
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frames_info.append({
                                'frame_number': pos,
                                'mean_brightness': np.mean(gray),
                                'brightness_std': np.std(gray),
                                'timestamp': pos / metadata['fps'] if metadata['fps'] > 0 else 0
                            })
                
                metadata['sample_frames'] = frames_info
            
            cap.release()
            os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def _extract_pdf_metadata(self, file_data: bytes) -> Dict:
        """Extract PDF metadata"""
        metadata = {}
        
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_data))
            
            metadata['pages'] = len(pdf_reader.pages)
            
            if pdf_reader.metadata:
                metadata['title'] = pdf_reader.metadata.get('/Title', 'Unknown')
                metadata['author'] = pdf_reader.metadata.get('/Author', 'Unknown')
                metadata['subject'] = pdf_reader.metadata.get('/Subject', 'Unknown')
                metadata['creator'] = pdf_reader.metadata.get('/Creator', 'Unknown')
                metadata['producer'] = pdf_reader.metadata.get('/Producer', 'Unknown')
                metadata['creation_date'] = str(pdf_reader.metadata.get('/CreationDate', 'Unknown'))
                metadata['modification_date'] = str(pdf_reader.metadata.get('/ModDate', 'Unknown'))
            
            # Extract text from first few pages for content analysis
            text_content = []
            for i in range(min(3, len(pdf_reader.pages))):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    text_content.append(page_text[:500])  # First 500 characters
                except:
                    text_content.append("Unable to extract text")
            
            metadata['sample_content'] = text_content
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def _extract_archive_metadata(self, file_data: bytes, file_ext: str) -> Dict:
        """Extract metadata from archive files"""
        metadata = {}
        
        try:
            temp_path = os.path.join(self.temp_dir, f"temp_archive{file_ext}")
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            files_list = []
            
            if file_ext == '.zip':
                with zipfile.ZipFile(temp_path, 'r') as zip_file:
                    for info in zip_file.infolist():
                        files_list.append({
                            'filename': info.filename,
                            'file_size': info.file_size,
                            'compressed_size': info.compress_size,
                            'date_time': info.date_time,
                            'is_directory': info.is_dir()
                        })
            
            elif file_ext == '.7z':
                with py7zr.SevenZipFile(temp_path, mode='r') as archive:
                    file_list = archive.list()
                    for file_info in file_list:
                        files_list.append({
                            'filename': file_info.filename,
                            'file_size': file_info.uncompressed if hasattr(file_info, 'uncompressed') else 'Unknown',
                            'is_directory': file_info.is_directory if hasattr(file_info, 'is_directory') else False
                        })
            
            elif file_ext == '.rar':
                with rarfile.RarFile(temp_path) as rar_file:
                    for info in rar_file.infolist():
                        files_list.append({
                            'filename': info.filename,
                            'file_size': info.file_size,
                            'compressed_size': info.compress_size,
                            'date_time': info.date_time,
                            'is_directory': info.is_dir()
                        })
            
            metadata['total_files'] = len(files_list)
            metadata['files'] = files_list[:50]  # Limit to first 50 files
            metadata['total_uncompressed_size'] = sum(f.get('file_size', 0) for f in files_list if isinstance(f.get('file_size'), int))
            
            os.remove(temp_path)
            
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def analyze_file_timeline(self, file_metadata_list: List[Dict]) -> Dict:
        """Analyze file timeline and create chronological sequence"""
        timeline = []
        
        for file_meta in file_metadata_list:
            events = []
            
            # File creation/modification dates from metadata
            if 'image_metadata' in file_meta and 'exif' in file_meta['image_metadata']:
                exif = file_meta['image_metadata']['exif']
                
                if 'DateTime' in exif:
                    events.append({
                        'timestamp': exif['DateTime'],
                        'event_type': 'Image Captured',
                        'filename': file_meta['filename'],
                        'source': 'EXIF'
                    })
                
                if 'DateTimeOriginal' in exif:
                    events.append({
                        'timestamp': exif['DateTimeOriginal'],
                        'event_type': 'Original Date',
                        'filename': file_meta['filename'],
                        'source': 'EXIF'
                    })
            
            # PDF metadata dates
            if 'document_metadata' in file_meta:
                doc_meta = file_meta['document_metadata']
                
                if 'creation_date' in doc_meta and doc_meta['creation_date'] != 'Unknown':
                    events.append({
                        'timestamp': doc_meta['creation_date'],
                        'event_type': 'Document Created',
                        'filename': file_meta['filename'],
                        'source': 'PDF Metadata'
                    })
                
                if 'modification_date' in doc_meta and doc_meta['modification_date'] != 'Unknown':
                    events.append({
                        'timestamp': doc_meta['modification_date'],
                        'event_type': 'Document Modified',
                        'filename': file_meta['filename'],
                        'source': 'PDF Metadata'
                    })
            
            timeline.extend(events)
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return {
            'timeline': timeline,
            'total_events': len(timeline),
            'date_range': {
                'earliest': timeline[0]['timestamp'] if timeline else None,
                'latest': timeline[-1]['timestamp'] if timeline else None
            }
        }
    
    def detect_file_manipulation(self, file_data: bytes, filename: str) -> Dict:
        """Detect potential file manipulation or tampering"""
        analysis = {
            'filename': filename,
            'analysis_timestamp': datetime.now().isoformat(),
            'manipulation_indicators': [],
            'risk_level': 'Low'
        }
        
        try:
            # Check for EXIF inconsistencies
            if filename.lower().endswith(('.jpg', '.jpeg', '.tiff')):
                temp_path = os.path.join(self.temp_dir, "temp_check")
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                image = Image.open(temp_path)
                
                if hasattr(image, '_getexif'):
                    exif = image._getexif()
                    if exif:
                        # Check for suspicious software signatures
                        software = exif.get(272, '')  # Software tag
                        if any(editor in str(software).lower() for editor in ['photoshop', 'gimp', 'paint']):
                            analysis['manipulation_indicators'].append({
                                'type': 'Image Editor Signature',
                                'description': f'File shows signs of editing with {software}',
                                'severity': 'Medium'
                            })
                        
                        # Check for timestamp inconsistencies
                        datetime_original = exif.get(36867)  # DateTimeOriginal
                        datetime_digitized = exif.get(36868)  # DateTimeDigitized
                        datetime_modified = exif.get(306)     # DateTime
                        
                        if datetime_original and datetime_modified:
                            if datetime_original != datetime_modified:
                                analysis['manipulation_indicators'].append({
                                    'type': 'Timestamp Inconsistency',
                                    'description': 'Original and modified timestamps differ',
                                    'severity': 'Medium'
                                })
                
                # Analyze image for digital artifacts
                image_array = np.array(image.convert('RGB'))
                
                # Check for JPEG compression artifacts
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    # Simple compression artifact detection
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    
                    # Look for 8x8 block artifacts
                    blocks = []
                    for y in range(0, gray.shape[0] - 8, 8):
                        for x in range(0, gray.shape[1] - 8, 8):
                            block = gray[y:y+8, x:x+8]
                            block_variance = np.var(block)
                            blocks.append(block_variance)
                    
                    if blocks:
                        variance_std = np.std(blocks)
                        if variance_std < 10:  # Very uniform blocks might indicate heavy compression
                            analysis['manipulation_indicators'].append({
                                'type': 'Compression Artifacts',
                                'description': 'Image shows signs of heavy compression or re-compression',
                                'severity': 'Low'
                            })
                
                os.remove(temp_path)
            
            # Determine overall risk level
            high_severity_count = len([i for i in analysis['manipulation_indicators'] if i['severity'] == 'High'])
            medium_severity_count = len([i for i in analysis['manipulation_indicators'] if i['severity'] == 'Medium'])
            
            if high_severity_count > 0:
                analysis['risk_level'] = 'High'
            elif medium_severity_count > 1:
                analysis['risk_level'] = 'High'
            elif medium_severity_count > 0:
                analysis['risk_level'] = 'Medium'
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_forensics_report(self, files_metadata: List[Dict], timeline_analysis: Dict, 
                                manipulation_results: List[Dict]) -> Dict:
        """Generate comprehensive forensics report"""
        report = {
            'report_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            'generation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_analyzed': len(files_metadata),
                'files_with_metadata': len([f for f in files_metadata if 'image_metadata' in f or 'video_metadata' in f or 'document_metadata' in f]),
                'files_with_gps': len([f for f in files_metadata if 'image_metadata' in f and 'gps_coordinates' in f['image_metadata']]),
                'files_with_manipulation_indicators': len([m for m in manipulation_results if m['manipulation_indicators']]),
                'timeline_events': timeline_analysis.get('total_events', 0)
            },
            'detailed_analysis': {
                'file_metadata': files_metadata,
                'timeline_analysis': timeline_analysis,
                'manipulation_analysis': manipulation_results
            },
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        if report['summary']['files_with_manipulation_indicators'] > 0:
            report['recommendations'].append("Review files flagged for potential manipulation")
        
        if report['summary']['files_with_gps'] > 0:
            report['recommendations'].append("Analyze GPS coordinates for location correlation")
        
        if report['summary']['timeline_events'] > 10:
            report['recommendations'].append("Detailed timeline analysis recommended for sequence reconstruction")
        
        return report
    
    def search_similar_files(self, target_hash: str, file_database: List[Dict]) -> List[Dict]:
        """Search for files with matching hashes across different hash algorithms"""
        matches = []
        
        for file_record in file_database:
            if 'hashes' in file_record:
                for algorithm, hash_value in file_record['hashes'].items():
                    if hash_value == target_hash:
                        matches.append({
                            'filename': file_record['filename'],
                            'matching_algorithm': algorithm,
                            'hash_value': hash_value,
                            'file_record': file_record
                        })
        
        return matches
