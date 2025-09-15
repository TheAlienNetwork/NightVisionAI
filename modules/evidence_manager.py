import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from database import execute_query, get_cases, create_case, USE_SQLITE

class EvidenceManager:
    def __init__(self):
        self.case_statuses = ['Active', 'Closed', 'On Hold', 'Under Review']
        self.evidence_categories = ['Image', 'Video', 'Document', 'Audio', 'Dataset', 'Other']
        self.priority_levels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
    
    def create_new_case(self, case_name: str, description: str, priority: int = 2) -> Optional[int]:
        """Create a new investigation case"""
        try:
            case_id = create_case(case_name, description)
            
            if case_id:
                # Initialize case in session state
                if 'cases' not in st.session_state:
                    st.session_state.cases = {}
                
                st.session_state.cases[case_id] = {
                    'id': case_id,
                    'name': case_name,
                    'description': description,
                    'priority': priority,
                    'created_at': datetime.now(),
                    'status': 'Active',
                    'evidence_count': 0,
                    'last_updated': datetime.now()
                }
                
                return case_id
            
        except Exception as e:
            st.error(f"Error creating case: {str(e)}")
            return None
    
    def get_case_details(self, case_id: int) -> Optional[Dict]:
        """Get detailed information about a case"""
        try:
            # Get case basic info
            query = "SELECT * FROM cases WHERE id = %s"
            case_result = execute_query(query, (case_id,), fetch=True)
            
            if not case_result:
                return None
            
            case = case_result[0]
            
            # Get evidence files for this case
            evidence_query = "SELECT * FROM evidence_files WHERE case_id = %s ORDER BY upload_date DESC"
            evidence_files = execute_query(evidence_query, (case_id,), fetch=True) or []
            
            # Get crime incidents for this case
            incidents_query = "SELECT * FROM crime_incidents WHERE case_id = %s ORDER BY incident_date DESC"
            incidents = execute_query(incidents_query, (case_id,), fetch=True) or []
            
            # Get facial recognition results
            face_query = """
                SELECT fr.*, ef.filename 
                FROM facial_recognition fr 
                JOIN evidence_files ef ON fr.file_id = ef.id 
                WHERE ef.case_id = %s
            """
            face_results = execute_query(face_query, (case_id,), fetch=True) or []
            
            # Get forensics results
            forensics_query = """
                SELECT fo.*, ef.filename 
                FROM forensics_results fo 
                JOIN evidence_files ef ON fo.file_id = ef.id 
                WHERE ef.case_id = %s
            """
            forensics_results = execute_query(forensics_query, (case_id,), fetch=True) or []
            
            # Get AI analysis results
            ai_query = """
                SELECT ai.*, ef.filename 
                FROM ai_analysis ai 
                JOIN evidence_files ef ON ai.file_id = ef.id 
                WHERE ef.case_id = %s
            """
            ai_results = execute_query(ai_query, (case_id,), fetch=True) or []
            
            case_details = {
                'basic_info': case,
                'evidence_files': evidence_files,
                'crime_incidents': incidents,
                'facial_recognition': face_results,
                'forensics_results': forensics_results,
                'ai_analysis': ai_results,
                'statistics': self._calculate_case_statistics(evidence_files, incidents, face_results, forensics_results, ai_results)
            }
            
            return case_details
            
        except Exception as e:
            st.error(f"Error retrieving case details: {str(e)}")
            return None
    
    def _calculate_case_statistics(self, evidence_files: List, incidents: List, 
                                 face_results: List, forensics_results: List, ai_results: List) -> Dict:
        """Calculate statistics for a case"""
        stats = {
            'total_evidence_files': len(evidence_files),
            'total_incidents': len(incidents),
            'faces_detected': sum(len(json.loads(fr.get('face_encodings', '[]'))) for fr in face_results),
            'forensics_analyses': len(forensics_results),
            'ai_analyses': len(ai_results)
        }
        
        # Evidence by type
        evidence_types = {}
        for evidence in evidence_files:
            file_type = evidence.get('file_type', 'Unknown')
            evidence_types[file_type] = evidence_types.get(file_type, 0) + 1
        
        stats['evidence_by_type'] = evidence_types
        
        # Incidents by type
        incident_types = {}
        for incident in incidents:
            inc_type = incident.get('incident_type', 'Unknown')
            incident_types[inc_type] = incident_types.get(inc_type, 0) + 1
        
        stats['incidents_by_type'] = incident_types
        
        return stats
    
    def add_incident_to_case(self, case_id: int, incident_data: Dict) -> bool:
        """Add a crime incident to a case"""
        try:
            if USE_SQLITE:
                query = """
                    INSERT INTO crime_incidents 
                    (case_id, incident_type, location_lat, location_lng, address, 
                     incident_date, description, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
            else:
                query = """
                    INSERT INTO crime_incidents 
                    (case_id, incident_type, location_lat, location_lng, address, 
                     incident_date, description, severity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
            
            params = (
                case_id,
                incident_data.get('incident_type'),
                incident_data.get('latitude'),
                incident_data.get('longitude'),
                incident_data.get('address'),
                incident_data.get('incident_date'),
                incident_data.get('description'),
                incident_data.get('severity', 2)
            )
            
            result = execute_query(query, params)
            return result is not None
            
        except Exception as e:
            st.error(f"Error adding incident: {str(e)}")
            return False
    
    def create_case_timeline(self, case_details: Dict) -> go.Figure:
        """Create timeline visualization for a case"""
        timeline_events = []
        
        # Add evidence upload events
        for evidence in case_details.get('evidence_files', []):
            timeline_events.append({
                'date': evidence.get('upload_date'),
                'event': 'Evidence Upload',
                'description': f"Uploaded: {evidence.get('filename')}",
                'category': 'Evidence',
                'details': evidence
            })
        
        # Add incident events
        for incident in case_details.get('crime_incidents', []):
            timeline_events.append({
                'date': incident.get('incident_date'),
                'event': 'Crime Incident',
                'description': f"{incident.get('incident_type')} - {incident.get('address', 'Unknown location')}",
                'category': 'Incident',
                'details': incident
            })
        
        # Add analysis events
        for analysis in case_details.get('forensics_results', []):
            timeline_events.append({
                'date': analysis.get('created_at'),
                'event': 'Forensics Analysis',
                'description': f"Analysis of {analysis.get('filename')}",
                'category': 'Analysis',
                'details': analysis
            })
        
        for analysis in case_details.get('ai_analysis', []):
            timeline_events.append({
                'date': analysis.get('created_at'),
                'event': 'AI Analysis',
                'description': f"{analysis.get('analysis_type')} - {analysis.get('filename')}",
                'category': 'Analysis',
                'details': analysis
            })
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(timeline_events)
        
        if df.empty:
            # Create empty timeline
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline events available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create timeline chart
        fig = px.timeline(
            df, 
            x_start='date', 
            x_end='date',
            y='category',
            color='category',
            title='Case Timeline',
            hover_data=['event', 'description']
        )
        
        # Update layout for better visualization
        fig.update_layout(
            showlegend=True,
            height=400,
            xaxis_title="Date/Time",
            yaxis_title="Event Category"
        )
        
        return fig
    
    def generate_case_summary_report(self, case_details: Dict) -> Dict:
        """Generate comprehensive case summary report"""
        basic_info = case_details.get('basic_info', {})
        stats = case_details.get('statistics', {})
        
        report = {
            'case_id': basic_info.get('id'),
            'case_name': basic_info.get('name'),
            'case_description': basic_info.get('description'),
            'case_status': basic_info.get('status'),
            'created_date': basic_info.get('created_at'),
            'last_updated': basic_info.get('updated_at'),
            'report_generated': datetime.now().isoformat(),
            'statistics': stats
        }
        
        # Evidence summary
        evidence_summary = {
            'total_files': stats.get('total_evidence_files', 0),
            'file_types': stats.get('evidence_by_type', {}),
            'processed_files': len([e for e in case_details.get('evidence_files', []) if e.get('processed')])
        }
        report['evidence_summary'] = evidence_summary
        
        # Incidents summary
        incidents_summary = {
            'total_incidents': stats.get('total_incidents', 0),
            'incident_types': stats.get('incidents_by_type', {}),
            'severity_distribution': self._analyze_incident_severity(case_details.get('crime_incidents', []))
        }
        report['incidents_summary'] = incidents_summary
        
        # Analysis summary
        analysis_summary = {
            'facial_recognition_performed': stats.get('faces_detected', 0) > 0,
            'total_faces_detected': stats.get('faces_detected', 0),
            'forensics_analyses': stats.get('forensics_analyses', 0),
            'ai_analyses': stats.get('ai_analyses', 0)
        }
        report['analysis_summary'] = analysis_summary
        
        # Key findings
        key_findings = []
        
        # Check for high-priority findings
        if stats.get('faces_detected', 0) > 10:
            key_findings.append("High number of faces detected - potential for person identification")
        
        if any(inc.get('severity', 0) >= 3 for inc in case_details.get('crime_incidents', [])):
            key_findings.append("High-severity incidents present")
        
        if len(case_details.get('forensics_results', [])) > 0:
            key_findings.append("Digital forensics evidence available")
        
        gps_files = len([e for e in case_details.get('evidence_files', []) 
                        if 'metadata' in e and 'gps_coordinates' in str(e.get('metadata', {}))])
        if gps_files > 0:
            key_findings.append(f"Location data available from {gps_files} files")
        
        report['key_findings'] = key_findings
        
        return report
    
    def _analyze_incident_severity(self, incidents: List[Dict]) -> Dict:
        """Analyze severity distribution of incidents"""
        severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for incident in incidents:
            severity = incident.get('severity', 2)
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return {
            'Low': severity_counts[1],
            'Medium': severity_counts[2],
            'High': severity_counts[3],
            'Critical': severity_counts[4]
        }
    
    def create_evidence_overview_chart(self, case_details: Dict) -> go.Figure:
        """Create overview chart of evidence in the case"""
        stats = case_details.get('statistics', {})
        evidence_by_type = stats.get('evidence_by_type', {})
        
        if not evidence_by_type:
            fig = go.Figure()
            fig.add_annotation(
                text="No evidence data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(evidence_by_type.keys()),
            values=list(evidence_by_type.values()),
            title="Evidence by File Type"
        )])
        
        fig.update_layout(
            title="Evidence Distribution by File Type",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_incident_analysis_chart(self, case_details: Dict) -> go.Figure:
        """Create incident analysis visualization"""
        incidents = case_details.get('crime_incidents', [])
        
        if not incidents:
            fig = go.Figure()
            fig.add_annotation(
                text="No incident data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create incident type distribution
        incident_types = {}
        for incident in incidents:
            inc_type = incident.get('incident_type', 'Unknown')
            incident_types[inc_type] = incident_types.get(inc_type, 0) + 1
        
        fig = go.Figure(data=[go.Bar(
            x=list(incident_types.keys()),
            y=list(incident_types.values()),
            name="Incident Count"
        )])
        
        fig.update_layout(
            title="Incident Distribution by Type",
            xaxis_title="Incident Type",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def search_across_cases(self, search_term: str, search_type: str = 'all') -> List[Dict]:
        """Search across all cases for specific terms or evidence"""
        results = []
        
        try:
            if search_type in ['all', 'cases']:
                # Search in case names and descriptions
                case_query = """
                    SELECT id, name, description, status, created_at 
                    FROM cases 
                    WHERE name ILIKE %s OR description ILIKE %s
                """
                case_results = execute_query(case_query, (f'%{search_term}%', f'%{search_term}%'), fetch=True) or []
                
                for case in case_results:
                    results.append({
                        'type': 'Case',
                        'case_id': case['id'],
                        'title': case['name'],
                        'description': case['description'],
                        'relevance': 'Case name/description match',
                        'date': case['created_at']
                    })
            
            if search_type in ['all', 'evidence']:
                # Search in evidence files
                evidence_query = """
                    SELECT ef.*, c.name as case_name 
                    FROM evidence_files ef
                    JOIN cases c ON ef.case_id = c.id
                    WHERE ef.filename ILIKE %s
                """
                evidence_results = execute_query(evidence_query, (f'%{search_term}%',), fetch=True) or []
                
                for evidence in evidence_results:
                    results.append({
                        'type': 'Evidence',
                        'case_id': evidence['case_id'],
                        'case_name': evidence['case_name'],
                        'title': evidence['filename'],
                        'description': f"File type: {evidence.get('file_type', 'Unknown')}",
                        'relevance': 'Filename match',
                        'date': evidence['upload_date']
                    })
            
            if search_type in ['all', 'incidents']:
                # Search in incidents
                incident_query = """
                    SELECT ci.*, c.name as case_name 
                    FROM crime_incidents ci
                    JOIN cases c ON ci.case_id = c.id
                    WHERE ci.description ILIKE %s OR ci.incident_type ILIKE %s OR ci.address ILIKE %s
                """
                incident_results = execute_query(incident_query, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'), fetch=True) or []
                
                for incident in incident_results:
                    results.append({
                        'type': 'Incident',
                        'case_id': incident['case_id'],
                        'case_name': incident['case_name'],
                        'title': f"{incident['incident_type']} - {incident.get('address', 'Unknown location')}",
                        'description': incident.get('description', ''),
                        'relevance': 'Incident details match',
                        'date': incident['incident_date']
                    })
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        
        return results
    
    def export_case_data(self, case_id: int, export_format: str = 'json') -> Optional[str]:
        """Export case data in various formats"""
        try:
            case_details = self.get_case_details(case_id)
            
            if not case_details:
                return None
            
            if export_format == 'json':
                # Convert datetime objects to strings for JSON serialization
                def convert_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return obj
                
                export_data = {}
                for key, value in case_details.items():
                    if isinstance(value, list):
                        export_data[key] = [
                            {k: convert_datetime(v) for k, v in item.items()} if isinstance(item, dict) else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        export_data[key] = {k: convert_datetime(v) for k, v in value.items()}
                    else:
                        export_data[key] = convert_datetime(value)
                
                return json.dumps(export_data, indent=2)
            
            elif export_format == 'csv':
                # Create CSV export for evidence files
                evidence_files = case_details.get('evidence_files', [])
                if evidence_files:
                    df = pd.DataFrame(evidence_files)
                    return df.to_csv(index=False)
                return "No evidence files to export"
            
        except Exception as e:
            st.error(f"Export error: {str(e)}")
            return None
