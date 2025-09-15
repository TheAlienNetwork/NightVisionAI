import os
import streamlit as st
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import sqlite3

# Database configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'database': os.getenv('PGDATABASE', 'investigative_platform'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', ''),
    'port': os.getenv('PGPORT', '5432')
}

# For development, use SQLite if PostgreSQL is not available
USE_SQLITE = os.getenv('USE_SQLITE', 'true').lower() == 'true'

def get_connection():
    """Get database connection"""
    try:
        if USE_SQLITE:
            # Use SQLite for development
            conn = sqlite3.connect('investigative_platform.db', check_same_thread=False)
            conn.row_factory = sqlite3.Row  # This allows dict-like access to rows
            return conn
        else:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def init_database():
    """Initialize database tables"""
    conn = get_connection()
    if not conn:
        raise Exception("Could not connect to database")
    
    try:
        cur = conn.cursor()
        
        # Cases table
        if USE_SQLITE:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'Active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) DEFAULT 'Active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Evidence files table
        if USE_SQLITE:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evidence_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER REFERENCES cases(id),
                    filename TEXT NOT NULL,
                    file_type TEXT,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    metadata TEXT,
                    hash_value TEXT,
                    processed BOOLEAN DEFAULT 0
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evidence_files (
                    id SERIAL PRIMARY KEY,
                    case_id INTEGER REFERENCES cases(id),
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(100),
                    file_size BIGINT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    metadata JSONB,
                    hash_value VARCHAR(64),
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
        
        # Facial recognition results table
        if USE_SQLITE:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS facial_recognition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER REFERENCES evidence_files(id),
                    face_encodings TEXT,
                    bounding_boxes TEXT,
                    confidence_scores TEXT,
                    identified_persons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Crime incidents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crime_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER REFERENCES cases(id),
                    incident_type TEXT,
                    location_lat REAL,
                    location_lng REAL,
                    address TEXT,
                    incident_date TIMESTAMP,
                    description TEXT,
                    severity INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Digital forensics results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forensics_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER REFERENCES evidence_files(id),
                    metadata_extracted TEXT,
                    file_analysis TEXT,
                    hash_matches TEXT,
                    timeline_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # AI analysis results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER REFERENCES evidence_files(id),
                    analysis_type TEXT,
                    results TEXT,
                    confidence_score REAL,
                    anomalies_detected TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Suspects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS suspects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER REFERENCES cases(id),
                    name TEXT NOT NULL,
                    description TEXT,
                    threat_level INTEGER DEFAULT 1,
                    photo_file_id INTEGER REFERENCES evidence_files(id),
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    appearance_count INTEGER DEFAULT 1,
                    confidence_score REAL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Case activity log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS case_activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER REFERENCES cases(id),
                    activity_type TEXT,
                    description TEXT,
                    user_name TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS facial_recognition (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES evidence_files(id),
                    face_encodings JSONB,
                    bounding_boxes JSONB,
                    confidence_scores JSONB,
                    identified_persons JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Crime incidents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crime_incidents (
                    id SERIAL PRIMARY KEY,
                    case_id INTEGER REFERENCES cases(id),
                    incident_type VARCHAR(100),
                    location_lat DECIMAL(10, 8),
                    location_lng DECIMAL(11, 8),
                    address TEXT,
                    incident_date TIMESTAMP,
                    description TEXT,
                    severity INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Digital forensics results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forensics_results (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES evidence_files(id),
                    metadata_extracted JSONB,
                    file_analysis JSONB,
                    hash_matches JSONB,
                    timeline_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # AI analysis results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES evidence_files(id),
                    analysis_type VARCHAR(100),
                    results JSONB,
                    confidence_score DECIMAL(3, 2),
                    anomalies_detected JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Suspects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS suspects (
                    id SERIAL PRIMARY KEY,
                    case_id INTEGER REFERENCES cases(id),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    threat_level INTEGER DEFAULT 1,
                    photo_file_id INTEGER REFERENCES evidence_files(id),
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    appearance_count INTEGER DEFAULT 1,
                    confidence_score DECIMAL(3, 2),
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Case activity log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS case_activity_log (
                    id SERIAL PRIMARY KEY,
                    case_id INTEGER REFERENCES cases(id),
                    activity_type VARCHAR(100),
                    description TEXT,
                    user_name VARCHAR(255),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        raise Exception(f"Database initialization failed: {str(e)}")

def execute_query(query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]:
    """Execute database query"""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        if USE_SQLITE:
            cur = conn.cursor()
            cur.execute(query, params or ())
            
            if fetch:
                result = [dict(row) for row in cur.fetchall()]
            else:
                conn.commit()
                result = cur.rowcount
        else:
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            
            if fetch:
                result = [dict(row) for row in cur.fetchall()]
            else:
                conn.commit()
                result = cur.rowcount
        
        cur.close()
        conn.close()
        return result
        
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        st.error(f"Database query failed: {str(e)}")
        return None

def create_case(name: str, description: str = "") -> Optional[int]:
    """Create a new case"""
    if USE_SQLITE:
        query = "INSERT INTO cases (name, description) VALUES (?, ?)"
        conn = get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(query, (name, description))
            case_id = cur.lastrowid
            conn.commit()
            cur.close()
            conn.close()
            return case_id
        except Exception as e:
            st.error(f"Error creating case: {str(e)}")
            conn.rollback()
            cur.close()
            conn.close()
            return None
    else:
        query = "INSERT INTO cases (name, description) VALUES (%s, %s) RETURNING id"
        result = execute_query(query, (name, description), fetch=True)
        return result[0]['id'] if result else None

def get_cases() -> List[Dict]:
    """Get all cases"""
    query = "SELECT * FROM cases ORDER BY created_at DESC"
    result = execute_query(query, fetch=True)
    return result or []

def add_evidence_file(case_id: int, filename: str, file_type: str, 
                     file_size: int, file_path: str, metadata: Dict = None,
                     hash_value: str = None) -> Optional[int]:
    """Add evidence file to database"""
    query = """
        INSERT INTO evidence_files 
        (case_id, filename, file_type, file_size, file_path, metadata, hash_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
    """
    params = (case_id, filename, file_type, file_size, file_path, 
              json.dumps(metadata) if metadata else None, hash_value)
    result = execute_query(query, params, fetch=True)
    return result[0]['id'] if result else None

def get_evidence_files(case_id: int = None) -> List[Dict]:
    """Get evidence files, optionally filtered by case"""
    if case_id:
        query = "SELECT * FROM evidence_files WHERE case_id = %s ORDER BY upload_date DESC"
        params = (case_id,)
    else:
        query = "SELECT * FROM evidence_files ORDER BY upload_date DESC"
        params = None
    
    result = execute_query(query, params, fetch=True)
    return result or []

def add_suspect(case_id: int, name: str, description: str = "", threat_level: int = 1, 
                photo_file_id: int = None, confidence_score: float = 0.0) -> Optional[int]:
    """Add suspect to database"""
    query = """
        INSERT INTO suspects 
        (case_id, name, description, threat_level, photo_file_id, confidence_score, first_seen, last_seen)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """
    now = datetime.now()
    params = (case_id, name, description, threat_level, photo_file_id, confidence_score, now, now)
    result = execute_query(query, params, fetch=True)
    return result[0]['id'] if result else None

def get_suspects(case_id: int = None) -> List[Dict]:
    """Get suspects, optionally filtered by case"""
    if case_id:
        query = """
            SELECT s.*, ef.filename as photo_filename 
            FROM suspects s 
            LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
            WHERE s.case_id = %s 
            ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
        """
        params = (case_id,)
    else:
        query = """
            SELECT s.*, ef.filename as photo_filename, c.name as case_name
            FROM suspects s 
            LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
            LEFT JOIN cases c ON s.case_id = c.id 
            ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
        """
        params = None
    
    result = execute_query(query, params, fetch=True)
    return result or []

def log_case_activity(case_id: int, activity_type: str, description: str, 
                     user_name: str = "System", metadata: Dict = None) -> bool:
    """Log activity for a case"""
    query = """
        INSERT INTO case_activity_log 
        (case_id, activity_type, description, user_name, metadata)
        VALUES (%s, %s, %s, %s, %s)
    """
    params = (case_id, activity_type, description, user_name, 
              json.dumps(metadata) if metadata else None)
    result = execute_query(query, params)
    return result is not None

def get_case_activity_log(case_id: int, limit: int = 50) -> List[Dict]:
    """Get activity log for a case"""
    query = """
        SELECT * FROM case_activity_log 
        WHERE case_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s
    """
    result = execute_query(query, (case_id, limit), fetch=True)
    return result or []
