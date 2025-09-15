import os
import streamlit as st
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import sqlite3
import hashlib

# Placeholder for AI analysis function
def analyze_file_with_ai(file_path: str) -> Dict:
    """
    Placeholder function for AI analysis of a file.
    In a real application, this would involve complex AI models.
    """
    # Simulate AI analysis based on file type
    if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.avi') or file_path.lower().endswith('.mov') or file_path.lower().endswith('.mkv') or file_path.lower().endswith('.wmv'):
        return {
            "analysis_type": "Video Content Analysis",
            "results": {"summary": "Video analysis summary placeholder.", "objects_detected": ["car", "person"], "actions": ["running"]},
            "confidence_score": 0.85,
            "anomalies_detected": ["suspicious activity"]
        }
    elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg') or file_path.lower().endswith('.png') or file_path.lower().endswith('.bmp') or file_path.lower().endswith('.tiff') or file_path.lower().endswith('.webp'):
        return {
            "analysis_type": "Image Content Analysis",
            "results": {"objects_detected": ["face", "building"], "attributes": ["clear sky"]},
            "confidence_score": 0.92,
            "anomalies_detected": []
        }
    elif file_path.lower().endswith('.pdf') or file_path.lower().endswith('.docx') or file_path.lower().endswith('.txt') or file_path.lower().endswith('.csv') or file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.json'):
        return {
            "analysis_type": "Document Content Analysis",
            "results": {"keywords": ["report", "investigation", "evidence"], "sentiment": "neutral"},
            "confidence_score": 0.78,
            "anomalies_detected": ["potential redaction"]
        }
    elif file_path.lower().endswith('.zip') or file_path.lower().endswith('.7z'):
        return {
            "analysis_type": "Archive Analysis",
            "results": {"file_count": 50, "total_size": "100MB"},
            "confidence_score": 0.90,
            "anomalies_detected": ["password protected file"]
        }
    else:
        return {
            "analysis_type": "General File Analysis",
            "results": {"detected_format": "unknown"},
            "confidence_score": 0.50,
            "anomalies_detected": []
        }

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
    if USE_SQLITE:
        query = """
            INSERT INTO evidence_files 
            (case_id, filename, file_type, file_size, file_path, metadata, hash_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (case_id, filename, file_type, file_size, file_path, 
                  json.dumps(metadata) if metadata else None, hash_value)
        conn = get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            file_id = cur.lastrowid
            conn.commit()
            cur.close()
            conn.close()
            return file_id
        except Exception as e:
            st.error(f"Error adding evidence file: {str(e)}")
            conn.rollback()
            cur.close()
            conn.close()
            return None
    else:
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
        if USE_SQLITE:
            query = "SELECT * FROM evidence_files WHERE case_id = ? ORDER BY upload_date DESC"
            params = (case_id,)
        else:
            query = "SELECT * FROM evidence_files WHERE case_id = %s ORDER BY upload_date DESC"
            params = (case_id,)
    else:
        if USE_SQLITE:
            query = "SELECT * FROM evidence_files ORDER BY upload_date DESC"
            params = ()
        else:
            query = "SELECT * FROM evidence_files ORDER BY upload_date DESC"
            params = ()

    result = execute_query(query, params, fetch=True)
    return result or []

def add_suspect(case_id: int, name: str, description: str = "", threat_level: int = 1, 
                photo_file_id: int = None, confidence_score: float = 0.0) -> Optional[int]:
    """Add suspect to database"""
    if USE_SQLITE:
        query = """
            INSERT INTO suspects 
            (case_id, name, description, threat_level, photo_file_id, confidence_score, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now()
        params = (case_id, name, description, threat_level, photo_file_id, confidence_score, now, now)
        conn = get_connection()
        if not conn:
            return None
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            suspect_id = cur.lastrowid
            conn.commit()
            cur.close()
            conn.close()
            return suspect_id
        except Exception as e:
            st.error(f"Error adding suspect: {str(e)}")
            conn.rollback()
            cur.close()
            conn.close()
            return None
    else:
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
        if USE_SQLITE:
            query = """
                SELECT s.*, ef.filename as photo_filename 
                FROM suspects s 
                LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
                WHERE s.case_id = ? 
                ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
            """
            params = (case_id,)
        else:
            query = """
                SELECT s.*, ef.filename as photo_filename 
                FROM suspects s 
                LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
                WHERE s.case_id = %s 
                ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
            """
            params = (case_id,)
    else:
        if USE_SQLITE:
            query = """
                SELECT s.*, ef.filename as photo_filename, c.name as case_name
                FROM suspects s 
                LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
                LEFT JOIN cases c ON s.case_id = c.id 
                ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
            """
            params = ()
        else:
            query = """
                SELECT s.*, ef.filename as photo_filename, c.name as case_name
                FROM suspects s 
                LEFT JOIN evidence_files ef ON s.photo_file_id = ef.id 
                LEFT JOIN cases c ON s.case_id = c.id 
                ORDER BY s.threat_level DESC, s.confidence_score DESC, s.last_seen DESC
            """
            params = ()

    result = execute_query(query, params, fetch=True)
    return result or []

def log_case_activity(case_id: int, activity_type: str, description: str, 
                     user_name: str = "System", metadata: Dict = None) -> bool:
    """Log activity for a case"""
    if USE_SQLITE:
        query = """
            INSERT INTO case_activity_log 
            (case_id, activity_type, description, user_name, metadata)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (case_id, activity_type, description, user_name, 
                  json.dumps(metadata) if metadata else None)
        result = execute_query(query, params)
        return result is not None
    else:
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
    if USE_SQLITE:
        query = """
            SELECT * FROM case_activity_log 
            WHERE case_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        params = (case_id, limit)
    else:
        query = """
            SELECT * FROM case_activity_log 
            WHERE case_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s
        """
        params = (case_id, limit)

    result = execute_query(query, params, fetch=True)
    return result or []

def delete_case(case_id: int) -> bool:
    """Delete a case and all its associated data"""
    if USE_SQLITE:
        conn = get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            # Delete associated evidence files
            cur.execute("DELETE FROM evidence_files WHERE case_id = ?", (case_id,))
            # Delete associated crime incidents
            cur.execute("DELETE FROM crime_incidents WHERE case_id = ?", (case_id,))
            # Delete associated suspects
            cur.execute("DELETE FROM suspects WHERE case_id = ?", (case_id,))
            # Delete associated AI analysis results
            cur.execute("""
                DELETE FROM ai_analysis 
                WHERE file_id IN (SELECT id FROM evidence_files WHERE case_id = ?)
            """, (case_id,))
            # Delete associated facial recognition results
            cur.execute("""
                DELETE FROM facial_recognition 
                WHERE file_id IN (SELECT id FROM evidence_files WHERE case_id = ?)
            """, (case_id,))
            # Delete associated forensics results
            cur.execute("""
                DELETE FROM forensics_results 
                WHERE file_id IN (SELECT id FROM evidence_files WHERE case_id = ?)
            """, (case_id,))
            # Delete associated case activity log
            cur.execute("DELETE FROM case_activity_log WHERE case_id = ?", (case_id,))
            # Delete the case itself
            cur.execute("DELETE FROM cases WHERE id = ?", (case_id,))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error deleting case: {str(e)}")
            conn.rollback()
            cur.close()
            conn.close()
            return False
    else:
        # For PostgreSQL, you'd typically use ON DELETE CASCADE in your foreign key constraints
        # or execute similar DELETE statements in order.
        # This is a simplified example assuming cascade deletion might be handled by DB design.
        # If not, you'd need to explicitly delete dependent records first.
        try:
            # Simple deletion of the case; assumes cascade delete is set up or dependent data is handled elsewhere.
            query = "DELETE FROM cases WHERE id = %s"
            result = execute_query(query, (case_id,))
            return result is not None
        except Exception as e:
            st.error(f"Error deleting case: {str(e)}")
            return False

def delete_evidence_file(file_id: int) -> bool:
    """Delete evidence file and all associated analysis results"""
    if USE_SQLITE:
        conn = get_connection()
        if not conn:
            return False
        try:
            cur = conn.cursor()
            # Delete associated AI analysis results
            cur.execute("DELETE FROM ai_analysis WHERE file_id = ?", (file_id,))
            # Delete associated facial recognition results
            cur.execute("DELETE FROM facial_recognition WHERE file_id = ?", (file_id,))
            # Delete associated forensics results
            cur.execute("DELETE FROM forensics_results WHERE file_id = ?", (file_id,))
            # Delete the evidence file itself
            cur.execute("DELETE FROM evidence_files WHERE id = ?", (file_id,))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error deleting evidence file: {str(e)}")
            conn.rollback()
            cur.close()
            conn.close()
            return False
    else:
        try:
            # Delete dependent records first
            execute_query("DELETE FROM ai_analysis WHERE file_id = %s", (file_id,))
            execute_query("DELETE FROM facial_recognition WHERE file_id = %s", (file_id,))
            execute_query("DELETE FROM forensics_results WHERE file_id = %s", (file_id,))
            # Delete the evidence file
            result = execute_query("DELETE FROM evidence_files WHERE id = %s", (file_id,))
            return result is not None
        except Exception as e:
            st.error(f"Error deleting evidence file: {str(e)}")
            return False

def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = filename.split('.')[-1].lower()
    mapping = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'bmp': 'image/bmp', 'tiff': 'image/tiff', 'tif': 'image/tiff', 'webp': 'image/webp',
        'mp4': 'video/mp4', 'avi': 'video/x-msvideo', 'mov': 'video/quicktime', 'mkv': 'video/x-matroska', 'wmv': 'video/x-ms-wmv',
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'doc': 'application/msword',
        'txt': 'text/plain',
        'csv': 'text/csv',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xls': 'application/vnd.ms-excel',
        'json': 'application/json',
        'zip': 'application/zip', '7z': 'application/x-7z-compressed',
        'mpeg4': 'video/mp4'
    }
    return mapping.get(ext, 'application/octet-stream')

def search_database(search_term: str, search_type: str = 'all') -> Dict[str, List[Dict]]:
    """Search across cases, evidence, and incidents"""
    results = {
        'cases': [],
        'evidence': [],
        'incidents': [],
        'suspects': []
    }

    if search_type in ['all', 'cases']:
        # Search in case names and descriptions
        if USE_SQLITE:
            case_query = """
                SELECT id, name, description, status, created_at 
                FROM cases 
                WHERE name LIKE ? OR description LIKE ?
            """
            params = (f'%{search_term}%', f'%{search_term}%')
        else:
            case_query = """
                SELECT id, name, description, status, created_at 
                FROM cases 
                WHERE name ILIKE %s OR description ILIKE %s
            """
            params = (f'%{search_term}%', f'%{search_term}%')
        case_results = execute_query(case_query, params, fetch=True) or []
        results['cases'] = case_results

    if search_type in ['all', 'evidence']:
        # Search in evidence files
        if USE_SQLITE:
            evidence_query = """
                SELECT ef.*, c.name as case_name 
                FROM evidence_files ef
                JOIN cases c ON ef.case_id = c.id
                WHERE ef.filename LIKE ?
            """
            params = (f'%{search_term}%',)
        else:
            evidence_query = """
                SELECT ef.*, c.name as case_name 
                FROM evidence_files ef
                JOIN cases c ON ef.case_id = c.id
                WHERE ef.filename ILIKE %s
            """
            params = (f'%{search_term}%',)
        evidence_results = execute_query(evidence_query, params, fetch=True) or []
        results['evidence'] = evidence_results

    if search_type in ['all', 'incidents']:
        # Search in incidents
        if USE_SQLITE:
            incident_query = """
                SELECT ci.*, c.name as case_name 
                FROM crime_incidents ci
                JOIN cases c ON ci.case_id = c.id
                WHERE ci.description LIKE ? OR ci.incident_type LIKE ? OR ci.address LIKE ?
            """
            params = (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%')
        else:
            incident_query = """
                SELECT ci.*, c.name as case_name 
                FROM crime_incidents ci
                JOIN cases c ON ci.case_id = c.id
                WHERE ci.description ILIKE %s OR ci.incident_type ILIKE %s OR ci.address ILIKE %s
            """
            params = (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%')
        incident_results = execute_query(incident_query, params, fetch=True) or []
        results['incidents'] = incident_results

    if search_type in ['all', 'suspects']:
        # Search in suspects
        if USE_SQLITE:
            suspect_query = """
                SELECT s.*, c.name as case_name 
                FROM suspects s
                JOIN cases c ON s.case_id = c.id
                WHERE s.name LIKE ? OR s.description LIKE ?
            """
            params = (f'%{search_term}%', f'%{search_term}%')
        else:
            suspect_query = """
                SELECT s.*, c.name as case_name 
                FROM suspects s
                JOIN cases c ON s.case_id = c.id
                WHERE s.name ILIKE %s OR s.description ILIKE %s
            """
            params = (f'%{search_term}%', f'%{search_term}%')
        suspect_results = execute_query(suspect_query, params, fetch=True) or []
        results['suspects'] = suspect_results

    return results

def display_dashboard():
    """Display the main dashboard for active cases and AI analysis"""
    st.title("Investigative Platform Dashboard")
    st.write("Overview of active cases, findings, and AI-analyzed files.")

    cases = get_cases()
    if not cases:
        st.info("No cases found. Please create a new case.")
        return

    selected_case_name = st.selectbox("Select Case", [case['name'] for case in cases])
    selected_case = next((case for case in cases if case['name'] == selected_case_name), None)

    if not selected_case:
        st.warning("Selected case not found.")
        return

    case_id = selected_case['id']
    st.subheader(f"Case: {selected_case['name']}")
    st.write(f"Description: {selected_case['description']}")
    st.write(f"Status: {selected_case['status']}")

    if st.button("Delete Case"):
        if st.warning(f"Are you sure you want to delete case '{selected_case['name']}'? This action cannot be undone."):
            if delete_case(case_id):
                st.success(f"Case '{selected_case['name']}' deleted successfully.")
                st.experimental_rerun() # Rerun to update the case list
            else:
                st.error("Failed to delete case.")

    st.markdown("---")
    st.subheader("Evidence Files")

    evidence_files = get_evidence_files(case_id=case_id)
    if not evidence_files:
        st.info("No evidence files uploaded for this case yet.")
    else:
        for file_info in evidence_files:
            with st.expander(f"{file_info['filename']} ({file_info['file_type']}) - {file_info['file_size']} bytes"):
                st.write(f"Uploaded: {file_info['upload_date']}")
                if file_info['file_path']:
                    st.write(f"File Path: {file_info['file_path']}")
                if file_info['metadata']:
                    st.json(json.loads(file_info['metadata']) if isinstance(file_info['metadata'], str) else file_info['metadata'])
                if file_info['hash_value']:
                    st.write(f"Hash: {file_info['hash_value']}")

                # Button to trigger AI analysis if not processed
                if not file_info['processed']:
                    if st.button(f"Analyze {file_info['filename']} with AI", key=f"analyze_{file_info['id']}"):
                        try:
                            analysis_results = analyze_file_with_ai(file_info['file_path'])

                            # Add forensics results (e.g., metadata extraction)
                            # This part needs to be carefully integrated with file analysis
                            # For now, let's assume metadata is extracted and we're storing it.
                            # A more robust solution would involve separate functions for each analysis type.
                            
                            # Placeholder for metadata extraction and storing in forensics_results
                            # In a real scenario, you would call specific functions here to extract metadata,
                            # perform integrity checks, timeline analysis, etc.
                            # For this example, we'll simulate saving some basic info.
                            
                            # --- Begin simulated forensics data storage ---
                            simulated_metadata_extracted = {
                                "file_name": file_info['filename'],
                                "file_size": file_info['file_size'],
                                "file_type": file_info['file_type'],
                                "detected_hashes": {
                                    "MD5": "d41d8cd98f00b204e9800998ecf8427e", # Example hash
                                    "SHA256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # Example hash
                                }
                            }
                            
                            simulated_timeline_data = [
                                {"timestamp": datetime.now().isoformat(), "event": "File Uploaded"}
                            ]

                            # Store in database
                            if USE_SQLITE:
                                forensics_query = """
                                    INSERT INTO forensics_results 
                                    (file_id, metadata_extracted, file_analysis, timeline_data, created_at)
                                    VALUES (?, ?, ?, ?, ?)
                                """
                                execute_query(forensics_query, (
                                    file_info['id'], 
                                    json.dumps(simulated_metadata_extracted),
                                    json.dumps({'analysis_type': 'metadata_extraction'}),
                                    json.dumps(simulated_timeline_data),
                                    datetime.now()
                                ))
                            else:
                                forensics_query = """
                                    INSERT INTO forensics_results 
                                    (file_id, metadata_extracted, file_analysis, timeline_data, created_at)
                                    VALUES (%s, %s, %s, %s, %s)
                                """
                                execute_query(forensics_query, (
                                    file_info['id'], 
                                    json.dumps(simulated_metadata_extracted),
                                    json.dumps({'analysis_type': 'metadata_extraction'}),
                                    json.dumps(simulated_timeline_data),
                                    datetime.now()
                                ))
                            # --- End simulated forensics data storage ---

                            # Add AI analysis results to the database
                            ai_query = """
                                INSERT INTO ai_analysis (file_id, analysis_type, results, confidence_score, anomalies_detected)
                                VALUES (?, ?, ?, ?, ?)
                            """
                            ai_params = (
                                file_info['id'],
                                analysis_results.get("analysis_type"),
                                json.dumps(analysis_results.get("results")),
                                analysis_results.get("confidence_score"),
                                json.dumps(analysis_results.get("anomalies_detected"))
                            )

                            if USE_SQLITE:
                                conn = get_connection()
                                if conn:
                                    cur = conn.cursor()
                                    cur.execute(ai_query, ai_params)
                                    conn.commit()
                                    # Mark file as processed
                                    cur.execute("UPDATE evidence_files SET processed = 1 WHERE id = ?", (file_info['id'],))
                                    conn.commit()
                                    cur.close()
                                    conn.close()
                                    st.success(f"Analysis complete for {file_info['filename']}. Results saved.")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to connect to database for saving analysis.")
                            else:
                                # For PostgreSQL, use execute_query
                                execute_query(ai_query, ai_params)
                                # Mark file as processed
                                execute_query("UPDATE evidence_files SET processed = TRUE WHERE id = %s", (file_info['id'],))
                                st.success(f"Analysis complete for {file_info['filename']}. Results saved.")
                                st.experimental_rerun() # Rerun to reflect processed status

                        except Exception as e:
                            st.error(f"Error analyzing file {file_info['filename']}: {str(e)}")
                elif file_info['processed']:
                    st.write("File has been processed by AI.")
                    # Optionally display AI analysis results here if available
                    ai_results = get_ai_analysis_for_file(file_info['id'])
                    if ai_results:
                        for res in ai_results:
                            with st.expander(f"AI Analysis Results ({res['analysis_type']})"):
                                st.write(f"Confidence: {res['confidence_score']:.2f}")
                                st.write(f"Anomalies: {', '.join(json.loads(res['anomalies_detected'])) if res['anomalies_detected'] else 'None'}")
                                st.json(json.loads(res['results']) if isinstance(res['results'], str) else res['results'])
                    # Display forensics results as well
                    forensics_results = get_forensics_results_for_file(file_info['id'])
                    if forensics_results:
                        for res in forensics_results:
                            with st.expander(f"Forensics Analysis Results ({res.get('analysis_type', 'N/A')})"):
                                st.subheader("Extracted Metadata")
                                st.json(json.loads(res['metadata_extracted']) if res['metadata_extracted'] else {})
                                st.subheader("File Analysis")
                                st.json(json.loads(res['file_analysis']) if res['file_analysis'] else {})
                                st.subheader("Timeline Data")
                                st.json(json.loads(res['timeline_data']) if res['timeline_data'] else {})
                    else:
                        st.write("No analysis results found for this file.")


    st.markdown("---")
    st.subheader("Case Activity Log")
    activity_log = get_case_activity_log(case_id)
    if not activity_log:
        st.info("No activity logged for this case.")
    else:
        for entry in activity_log:
            st.write(f"- **{entry['created_at']}**: [{entry['activity_type']}] {entry['description']} (by {entry['user_name']})")

def get_ai_analysis_for_file(file_id: int) -> List[Dict]:
    """Get AI analysis results for a specific file"""
    if USE_SQLITE:
        query = "SELECT * FROM ai_analysis WHERE file_id = ?"
        params = (file_id,)
    else:
        query = "SELECT * FROM ai_analysis WHERE file_id = %s"
        params = (file_id,)

    results = execute_query(query, params, fetch=True)
    return results or []

def get_forensics_results_for_file(file_id: int) -> List[Dict]:
    """Get forensics analysis results for a specific file"""
    if USE_SQLITE:
        query = "SELECT * FROM forensics_results WHERE file_id = ?"
        params = (file_id,)
    else:
        query = "SELECT * FROM forensics_results WHERE file_id = %s"
        params = (file_id,)

    results = execute_query(query, params, fetch=True)
    return results or []


def upload_file_section():
    """Section for uploading files"""
    st.header("Upload Evidence Files")
    st.markdown("""
        Drag and drop files here
        Limit 200MB per file • JPG, JPEG, PNG, BMP, TIFF, WEBP, MP4, AVI, MOV, MKV, WMV, PDF, DOCX, TXT, CSV, XLSX, JSON, ZIP, 7Z, MPEG4, TIF
    """)

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, label_visibility="collapsed")

    case_id = st.session_state.get("current_case_id")
    if not case_id:
        st.warning("Please select or create a case first.")
        return

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} file(s) for case ID: {case_id}...")

        uploaded_file_details = []
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            file_type = get_file_type(filename)
            file_size = uploaded_file.size

            # Define a directory to save uploaded files
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, filename)

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_hash = get_file_hash(file_path)

                metadata = {
                    "original_filename": filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "upload_timestamp": datetime.now().isoformat()
                }

                new_file_id = add_evidence_file(case_id, filename, file_type, file_size, file_path, metadata, file_hash)

                if new_file_id:
                    st.success(f"'{filename}' uploaded and saved successfully (ID: {new_file_id}).")
                    log_case_activity(case_id, "file_upload", f"Uploaded file: {filename}", metadata={"file_id": new_file_id, "filename": filename})
                    uploaded_file_details.append({"filename": filename, "status": "Success", "id": new_file_id})
                else:
                    st.error(f"Failed to save '{filename}' to the database.")
                    uploaded_file_details.append({"filename": filename, "status": "Database Error"})

            except Exception as e:
                st.error(f"Error processing file '{filename}': {str(e)}")
                uploaded_file_details.append({"filename": filename, "status": f"Error: {str(e)}"})

        # Display summary of uploads
        st.subheader("Upload Summary")
        for detail in uploaded_file_details:
            st.write(f"- {detail['filename']}: {detail['status']}")

        # After processing, potentially trigger AI analysis for newly uploaded files
        # For simplicity, we'll prompt the user to analyze from the dashboard
        st.info("You can now go to the Dashboard to analyze uploaded files with AI.")


def main():
    """Main application logic"""
    init_database() # Ensure tables are created

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Create Case", "Search", "Upload Files"])

    if page == "Dashboard":
        display_dashboard()
    elif page == "Create Case":
        st.header("Create New Case")
        case_name = st.text_input("Case Name", key="new_case_name")
        case_description = st.text_area("Case Description", key="new_case_description")
        if st.button("Create Case"):
            if case_name:
                new_case_id = create_case(case_name, case_description)
                if new_case_id:
                    st.success(f"Case '{case_name}' created successfully with ID: {new_case_id}")
                    st.session_state["current_case_id"] = new_case_id # Set current case for upload
                    log_case_activity(new_case_id, "case_create", f"Case '{case_name}' created.")
                else:
                    st.error("Failed to create case.")
            else:
                st.warning("Case name cannot be empty.")
    elif page == "Search":
        st.header("Search Database")
        search_term = st.text_input("Enter search term")
        search_type = st.selectbox("Search in", ["all", "cases", "evidence", "incidents", "suspects"])

        if st.button("Search"):
            if search_term:
                results = search_database(search_term, search_type)

                if any(results.values()):
                    st.subheader("Search Results")

                    if results['cases']:
                        st.write("**Cases:**")
                        for case in results['cases']:
                            st.write(f"- {case['name']} ({case['status']}) - {case['description'][:50]}...")
                            if st.button("View Case", key=f"view_case_{case['id']}"):
                                st.session_state["current_case_id"] = case['id']
                                st.sidebar.radio("Go to", ["Dashboard", "Create Case", "Search", "Upload Files"], index=0) # Reset to Dashboard
                                st.experimental_rerun()

                    if results['evidence']:
                        st.write("**Evidence Files:**")
                        for evidence in results['evidence']:
                            st.write(f"- {evidence['filename']} (Case: {evidence['case_name']})")
                            if st.button("View Evidence", key=f"view_evidence_{evidence['id']}"):
                                # Logic to view evidence details, maybe expander or separate page
                                st.warning(f"Details for evidence file ID: {evidence['id']} (Filename: {evidence['filename']})")

                    if results['incidents']:
                        st.write("**Incidents:**")
                        for incident in results['incidents']:
                            st.write(f"- {incident['incident_type']} at {incident['address']} (Case: {incident['case_name']})")
                            if st.button("View Incident", key=f"view_incident_{incident['id']}"):
                                st.warning(f"Details for incident ID: {incident['id']} (Type: {incident['incident_type']})")

                    if results['suspects']:
                        st.write("**Suspects:**")
                        for suspect in results['suspects']:
                            st.write(f"- {suspect['name']} (Case: {suspect['case_name']})")
                            if st.button("View Suspect", key=f"view_suspect_{suspect['id']}"):
                                st.warning(f"Details for suspect ID: {suspect['id']} (Name: {suspect['name']})")
                else:
                    st.info("No results found for your search term.")
            else:
                st.warning("Please enter a search term.")
    elif page == "Upload Files":
        upload_file_section()

if __name__ == "__main__":
    main()