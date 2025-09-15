import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class CrimeAnalysisSystem:
    def __init__(self):
        self.incident_types = [
            'Assault', 'Robbery', 'Burglary', 'Theft', 'Vandalism',
            'Drug-related', 'Fraud', 'Homicide', 'Sexual Assault',
            'Domestic Violence', 'Vehicle Crime', 'Other'
        ]
        self.severity_levels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
    
    def load_crime_data(self, data_source: Dict) -> pd.DataFrame:
        """Load crime data from various sources"""
        try:
            if data_source.get('type') == 'csv':
                df = pd.read_csv(io.StringIO(data_source['data'].decode('utf-8')))
            elif data_source.get('type') == 'excel':
                df = pd.read_excel(io.BytesIO(data_source['data']))
            elif data_source.get('type') == 'json':
                data = json.loads(data_source['data'].decode('utf-8'))
                df = pd.DataFrame(data)
            else:
                st.error("Unsupported data format")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading crime data: {str(e)}")
            return pd.DataFrame()
    
    def validate_crime_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate crime data format and required columns"""
        required_columns = ['incident_type', 'latitude', 'longitude', 'date']
        missing_columns = []
        
        for col in required_columns:
            if col not in df.columns:
                # Try common alternative names
                alternatives = {
                    'incident_type': ['crime_type', 'type', 'category'],
                    'latitude': ['lat', 'y', 'coord_y'],
                    'longitude': ['lng', 'lon', 'x', 'coord_x'],
                    'date': ['datetime', 'timestamp', 'incident_date']
                }
                
                found_alternative = None
                for alt in alternatives.get(col, []):
                    if alt in df.columns:
                        df.rename(columns={alt: col}, inplace=True)
                        found_alternative = alt
                        break
                
                if not found_alternative:
                    missing_columns.append(col)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns
    
    def preprocess_crime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess crime data"""
        try:
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove rows with invalid coordinates or dates
            df = df.dropna(subset=['latitude', 'longitude', 'date'])
            
            # Ensure coordinates are within reasonable bounds
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
            
            # Add derived temporal features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour
            df['day_name'] = df['date'].dt.day_name()
            df['month_name'] = df['date'].dt.month_name()
            
            # Add severity if not present
            if 'severity' not in df.columns:
                df['severity'] = 2  # Default to medium
            
            return df
            
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return df
    
    def detect_crime_hotspots(self, df: pd.DataFrame, eps: float = 0.01, min_samples: int = 5) -> pd.DataFrame:
        """Detect crime hotspots using DBSCAN clustering"""
        try:
            if len(df) < min_samples:
                st.warning("Insufficient data for hotspot analysis")
                return df
            
            # Prepare coordinates for clustering
            coordinates = df[['latitude', 'longitude']].values
            
            # Apply DBSCAN clustering
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(coordinates_scaled)
            
            # Add cluster information to dataframe
            df['cluster'] = cluster_labels
            df['is_hotspot'] = df['cluster'] != -1
            
            # Calculate hotspot statistics
            hotspot_stats = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Exclude noise points
                    cluster_data = df[df['cluster'] == cluster_id]
                    
                    center_lat = cluster_data['latitude'].mean()
                    center_lng = cluster_data['longitude'].mean()
                    incident_count = len(cluster_data)
                    
                    # Calculate radius (max distance from center)
                    distances = []
                    for _, row in cluster_data.iterrows():
                        dist = geodesic((center_lat, center_lng), (row['latitude'], row['longitude'])).meters
                        distances.append(dist)
                    
                    hotspot_stats.append({
                        'cluster_id': cluster_id,
                        'center_lat': center_lat,
                        'center_lng': center_lng,
                        'incident_count': incident_count,
                        'radius_meters': max(distances) if distances else 0,
                        'dominant_type': cluster_data['incident_type'].mode().iloc[0] if not cluster_data['incident_type'].mode().empty else 'Unknown'
                    })
            
            df.hotspot_stats = hotspot_stats
            return df
            
        except Exception as e:
            st.error(f"Error detecting hotspots: {str(e)}")
            return df
    
    def predict_risk_areas(self, df: pd.DataFrame, prediction_days: int = 30) -> List[Dict]:
        """Predict high-risk areas based on historical patterns"""
        try:
            if len(df) < 10:
                st.warning("Insufficient data for risk prediction")
                return []
            
            # Calculate incident density by grid
            grid_size = 0.01  # Approximately 1km grid
            
            # Create grid boundaries
            lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
            lng_min, lng_max = df['longitude'].min(), df['longitude'].max()
            
            # Generate grid cells
            risk_areas = []
            for lat in np.arange(lat_min, lat_max, grid_size):
                for lng in np.arange(lng_min, lng_max, grid_size):
                    # Count incidents in this grid cell
                    cell_data = df[
                        (df['latitude'] >= lat) & (df['latitude'] < lat + grid_size) &
                        (df['longitude'] >= lng) & (df['longitude'] < lng + grid_size)
                    ]
                    
                    if len(cell_data) > 0:
                        # Calculate risk factors
                        recent_incidents = len(cell_data[cell_data['date'] >= (datetime.now() - timedelta(days=90))])
                        severity_score = cell_data['severity'].mean()
                        incident_variety = cell_data['incident_type'].nunique()
                        
                        # Calculate risk score
                        risk_score = (recent_incidents * 0.4 + severity_score * 0.3 + incident_variety * 0.3)
                        
                        if risk_score > 1.0:  # Threshold for risk areas
                            risk_areas.append({
                                'center_lat': lat + grid_size/2,
                                'center_lng': lng + grid_size/2,
                                'risk_score': risk_score,
                                'recent_incidents': recent_incidents,
                                'avg_severity': severity_score,
                                'incident_types': cell_data['incident_type'].unique().tolist(),
                                'prediction_period': prediction_days
                            })
            
            # Sort by risk score
            risk_areas.sort(key=lambda x: x['risk_score'], reverse=True)
            return risk_areas[:20]  # Return top 20 risk areas
            
        except Exception as e:
            st.error(f"Error predicting risk areas: {str(e)}")
            return []
    
    def create_crime_map(self, df: pd.DataFrame, hotspots: List[Dict] = None, risk_areas: List[Dict] = None) -> folium.Map:
        """Create interactive crime map with incidents, hotspots, and risk areas"""
        try:
            if len(df) == 0:
                # Create default map
                return folium.Map(location=[40.7128, -74.0060], zoom_start=10)
            
            # Calculate map center
            center_lat = df['latitude'].mean()
            center_lng = df['longitude'].mean()
            
            # Create base map
            m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
            
            # Color mapping for incident types
            incident_colors = {
                'Assault': 'red',
                'Robbery': 'darkred',
                'Burglary': 'orange',
                'Theft': 'blue',
                'Vandalism': 'green',
                'Drug-related': 'purple',
                'Fraud': 'pink',
                'Homicide': 'black',
                'Sexual Assault': 'darkblue',
                'Domestic Violence': 'darkgreen',
                'Vehicle Crime': 'lightblue',
                'Other': 'gray'
            }
            
            # Add individual incidents
            for _, row in df.iterrows():
                color = incident_colors.get(row['incident_type'], 'gray')
                
                popup_text = f"""
                <b>Type:</b> {row['incident_type']}<br>
                <b>Date:</b> {row['date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['date']) else 'Unknown'}<br>
                <b>Severity:</b> {self.severity_levels.get(row.get('severity', 2), 'Unknown')}
                """
                
                if 'description' in row and pd.notna(row['description']):
                    popup_text += f"<br><b>Description:</b> {row['description'][:100]}..."
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
            
            # Add hotspots
            if hotspots:
                for hotspot in hotspots:
                    folium.Circle(
                        location=[hotspot['center_lat'], hotspot['center_lng']],
                        radius=hotspot['radius_meters'],
                        popup=f"Hotspot: {hotspot['incident_count']} incidents<br>Type: {hotspot['dominant_type']}",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.2
                    ).add_to(m)
            
            # Add risk areas
            if risk_areas:
                for area in risk_areas:
                    folium.Rectangle(
                        bounds=[
                            [area['center_lat'] - 0.005, area['center_lng'] - 0.005],
                            [area['center_lat'] + 0.005, area['center_lng'] + 0.005]
                        ],
                        popup=f"Risk Score: {area['risk_score']:.2f}<br>Recent Incidents: {area['recent_incidents']}",
                        color='orange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.3
                    ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating crime map: {str(e)}")
            return folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal crime patterns"""
        try:
            if len(df) == 0:
                return {}
            
            patterns = {}
            
            # Hourly patterns
            hourly_counts = df.groupby('hour').size()
            patterns['hourly'] = {
                'data': hourly_counts.to_dict(),
                'peak_hour': hourly_counts.idxmax(),
                'peak_count': hourly_counts.max()
            }
            
            # Daily patterns
            daily_counts = df.groupby('day_name').size()
            patterns['daily'] = {
                'data': daily_counts.to_dict(),
                'peak_day': daily_counts.idxmax(),
                'peak_count': daily_counts.max()
            }
            
            # Monthly patterns
            monthly_counts = df.groupby('month_name').size()
            patterns['monthly'] = {
                'data': monthly_counts.to_dict(),
                'peak_month': monthly_counts.idxmax(),
                'peak_count': monthly_counts.max()
            }
            
            # Seasonal patterns
            seasonal_mapping = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            
            seasonal_counts = {}
            for season, months in seasonal_mapping.items():
                seasonal_counts[season] = len(df[df['month'].isin(months)])
            
            patterns['seasonal'] = {
                'data': seasonal_counts,
                'peak_season': max(seasonal_counts.items(), key=lambda x: x[1])[0],
                'peak_count': max(seasonal_counts.values())
            }
            
            return patterns
            
        except Exception as e:
            st.error(f"Error analyzing temporal patterns: {str(e)}")
            return {}
    
    def generate_crime_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive crime statistics"""
        try:
            if len(df) == 0:
                return {}
            
            stats = {}
            
            # Basic statistics
            stats['total_incidents'] = len(df)
            stats['unique_types'] = df['incident_type'].nunique()
            stats['date_range'] = {
                'start': df['date'].min().strftime('%Y-%m-%d') if not df['date'].isna().all() else 'Unknown',
                'end': df['date'].max().strftime('%Y-%m-%d') if not df['date'].isna().all() else 'Unknown'
            }
            
            # Incident type distribution
            type_counts = df['incident_type'].value_counts()
            stats['incident_types'] = {
                'distribution': type_counts.to_dict(),
                'most_common': type_counts.index[0] if len(type_counts) > 0 else 'Unknown',
                'least_common': type_counts.index[-1] if len(type_counts) > 0 else 'Unknown'
            }
            
            # Severity distribution
            if 'severity' in df.columns:
                severity_counts = df['severity'].value_counts()
                stats['severity'] = {
                    'distribution': severity_counts.to_dict(),
                    'average': df['severity'].mean(),
                    'most_common_level': self.severity_levels.get(severity_counts.index[0], 'Unknown')
                }
            
            # Geographic statistics
            stats['geographic'] = {
                'area_bounds': {
                    'north': df['latitude'].max(),
                    'south': df['latitude'].min(),
                    'east': df['longitude'].max(),
                    'west': df['longitude'].min()
                },
                'center': {
                    'latitude': df['latitude'].mean(),
                    'longitude': df['longitude'].mean()
                }
            }
            
            # Trend analysis (if sufficient data)
            if len(df) >= 30:
                # Recent vs historical comparison
                recent_cutoff = datetime.now() - timedelta(days=30)
                recent_incidents = len(df[df['date'] >= recent_cutoff])
                historical_avg = len(df[df['date'] < recent_cutoff]) / max(1, (df['date'].max() - df['date'].min()).days - 30) * 30
                
                stats['trends'] = {
                    'recent_incidents_30d': recent_incidents,
                    'historical_avg_30d': historical_avg,
                    'trend_direction': 'increasing' if recent_incidents > historical_avg else 'decreasing',
                    'trend_percentage': ((recent_incidents - historical_avg) / max(1, historical_avg)) * 100
                }
            
            return stats
            
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
            return {}
    
    def create_temporal_charts(self, patterns: Dict) -> Dict:
        """Create temporal analysis charts"""
        charts = {}
        
        try:
            # Hourly pattern chart
            if 'hourly' in patterns:
                hourly_data = patterns['hourly']['data']
                fig_hourly = px.line(
                    x=list(hourly_data.keys()),
                    y=list(hourly_data.values()),
                    title="Crime Incidents by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Number of Incidents'}
                )
                fig_hourly.update_layout(showlegend=False)
                charts['hourly'] = fig_hourly
            
            # Daily pattern chart
            if 'daily' in patterns:
                daily_data = patterns['daily']['data']
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ordered_days = [day for day in day_order if day in daily_data]
                ordered_values = [daily_data[day] for day in ordered_days]
                
                fig_daily = px.bar(
                    x=ordered_days,
                    y=ordered_values,
                    title="Crime Incidents by Day of Week",
                    labels={'x': 'Day', 'y': 'Number of Incidents'}
                )
                charts['daily'] = fig_daily
            
            # Monthly pattern chart
            if 'monthly' in patterns:
                monthly_data = patterns['monthly']['data']
                fig_monthly = px.bar(
                    x=list(monthly_data.keys()),
                    y=list(monthly_data.values()),
                    title="Crime Incidents by Month",
                    labels={'x': 'Month', 'y': 'Number of Incidents'}
                )
                charts['monthly'] = fig_monthly
        
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
        
        return charts

import io  # Add this import at the top
