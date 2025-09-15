# Overview

This is a comprehensive investigative platform built with Streamlit that provides law enforcement and private investigators with advanced digital analysis capabilities. The platform integrates multiple forensic tools including facial recognition, crime pattern analysis, digital forensics, evidence management, and AI-powered analysis into a unified web application. It's designed to handle multi-modal evidence (images, videos, documents, datasets) and provide real-time analysis using computer vision and machine learning techniques.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **Multi-page Application**: Organized into 6 main functional modules accessed via separate pages
- **Component-based Design**: Modular approach with dedicated modules for each major functionality
- **Real-time Processing**: Interactive widgets and real-time analysis capabilities
- **Visualization**: Plotly for interactive charts, Folium for mapping, OpenCV for image processing

## Backend Architecture
- **Database Layer**: PostgreSQL with psycopg2 for data persistence
- **Modular Design**: Six core modules handling specific domains:
  - `file_processor.py`: Multi-format file handling and metadata extraction
  - `facial_recognition.py`: OpenCV-based face detection and recognition
  - `crime_analysis.py`: Statistical analysis and pattern detection
  - `digital_forensics.py`: File integrity, hashing, and metadata extraction
  - `evidence_manager.py`: Case and evidence organization
  - `ai_analysis.py`: Computer vision and anomaly detection
- **Utility Layer**: Shared utilities for data processing and image manipulation
- **Session Management**: Streamlit session state for maintaining application state

## Data Storage
- **Primary Database**: PostgreSQL with tables for cases and evidence files
- **File Storage**: Temporary file system storage for processing
- **Session Storage**: In-memory storage for analysis results and cached data
- **Metadata Storage**: JSON fields in database for flexible evidence metadata

## Computer Vision Pipeline
- **Face Detection**: Haar cascade classifiers for real-time face detection
- **Image Analysis**: OpenCV for comprehensive image processing
- **Video Processing**: Frame extraction and analysis capabilities
- **Perceptual Hashing**: Image similarity and duplicate detection
- **Anomaly Detection**: Pattern recognition for suspicious activity identification

## Security and Forensics
- **File Integrity**: Multiple hash algorithms (MD5, SHA1, SHA256, SHA512)
- **Metadata Preservation**: EXIF data extraction and preservation
- **Chain of Custody**: Timestamp tracking and case assignment
- **Evidence Validation**: File type verification and integrity checks

# External Dependencies

## Core Framework Dependencies
- **Streamlit**: Web application framework for the entire platform
- **PostgreSQL**: Primary database for persistent data storage
- **psycopg2**: PostgreSQL adapter for Python database connectivity

## Computer Vision and AI
- **OpenCV**: Core computer vision library for image/video processing and facial recognition
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical computing for image arrays and mathematical operations
- **scikit-learn**: Machine learning algorithms for clustering and pattern analysis

## Data Processing and Visualization
- **Pandas**: Data manipulation and analysis framework
- **Plotly**: Interactive visualization library for charts and graphs
- **Folium**: Interactive mapping for geospatial crime analysis
- **streamlit-folium**: Integration between Streamlit and Folium

## File Processing and Forensics
- **python-magic**: File type detection using magic numbers
- **ExifRead**: EXIF metadata extraction from images
- **imagehash**: Perceptual hashing for image similarity detection
- **py7zr, rarfile**: Archive file processing for evidence extraction

## Geospatial and Scientific Computing
- **geopy**: Geographic calculations and distance measurements
- **DBSCAN**: Density-based clustering for crime hotspot detection
- **Matplotlib**: Additional visualization capabilities for scientific plots

## Utility Libraries
- **hashlib**: Cryptographic hashing for file integrity verification
- **tempfile**: Temporary file management for processing pipelines
- **pathlib**: Modern path handling for file operations
- **json**: Data serialization for metadata storage
- **datetime**: Timestamp management and temporal analysis