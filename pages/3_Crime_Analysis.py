import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from modules.crime_analysis import CrimeAnalysisSystem
from database import execute_query, get_cases
from utils.data_utils import parse_uploaded_data, format_timestamp, create_summary_cards
import json
import io

# Configure page
st.set_page_config(
    page_title="Crime Analysis - Investigative Platform",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Crime Pattern Analysis & Mapping")
st.markdown("Analyze crime patterns, detect hotspots, and predict risk areas using advanced analytics")

# Initialize crime analysis system
if 'crime_analysis_system' not in st.session_state:
    st.session_state.crime_analysis_system = CrimeAnalysisSystem()

# Initialize data storage
if 'crime_data' not in st.session_state:
    st.session_state.crime_data = pd.DataFrame()

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar for data management and settings
with st.sidebar:
    st.header("Data Management")
    
    # Data source selection
    data_source = st.selectbox(
        "Data Source",
        ["Upload File", "Database Cases", "Sample Data", "Manual Entry"]
    )
    
    # Data loading
    if data_source == "Upload File":
        st.subheader("Upload Crime Data")
        uploaded_file = st.file_uploader(
            "Choose CSV/Excel file",
            type=['csv', 'xlsx'],
            help="File should contain columns: incident_type, latitude, longitude, date"
        )
        
        if uploaded_file:
            df = parse_uploaded_data(uploaded_file)
            if df is not None:
                st.success(f"Loaded {len(df)} records")
                
                if st.button("Use This Data"):
                    # Validate and preprocess data
                    is_valid, missing_cols = st.session_state.crime_analysis_system.validate_crime_data(df)
                    
                    if is_valid:
                        processed_df = st.session_state.crime_analysis_system.preprocess_crime_data(df)
                        st.session_state.crime_data = processed_df
                        st.success("Data loaded and preprocessed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Missing required columns: {missing_cols}")
    
    elif data_source == "Database Cases":
        st.subheader("Load from Cases")
        cases = get_cases()
        
        if cases:
            case_options = {f"{case['name']} (ID: {case['id']})": case['id'] for case in cases}
            selected_case = st.selectbox("Select Case", list(case_options.keys()))
            
            if st.button("Load Case Incidents"):
                case_id = case_options[selected_case]
                query = "SELECT * FROM crime_incidents WHERE case_id = %s"
                incidents = execute_query(query, (case_id,), fetch=True)
                
                if incidents:
                    df = pd.DataFrame(incidents)
                    processed_df = st.session_state.crime_analysis_system.preprocess_crime_data(df)
                    st.session_state.crime_data = processed_df
                    st.success(f"Loaded {len(df)} incidents from case")
                    st.rerun()
                else:
                    st.warning("No incidents found in selected case")
        else:
            st.info("No cases available")
    
    elif data_source == "Sample Data":
        st.subheader("Generate Sample Data")
        num_incidents = st.slider("Number of Incidents", 10, 500, 100)
        
        if st.button("Generate Sample Data"):
            # Generate sample crime data for demonstration
            np.random.seed(42)
            
            # Sample incident types
            incident_types = ['Theft', 'Assault', 'Burglary', 'Vandalism', 'Drug-related', 'Vehicle Crime']
            
            # Generate coordinates around a central point (e.g., city center)
            center_lat, center_lng = 40.7128, -74.0060  # NYC coordinates
            
            sample_data = []
            for i in range(num_incidents):
                # Random location within ~10km radius
                lat_offset = np.random.normal(0, 0.05)
                lng_offset = np.random.normal(0, 0.05)
                
                incident = {
                    'incident_type': np.random.choice(incident_types),
                    'latitude': center_lat + lat_offset,
                    'longitude': center_lng + lng_offset,
                    'date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                    'severity': np.random.randint(1, 5),
                    'description': f"Sample incident {i+1}"
                }
                sample_data.append(incident)
            
            df = pd.DataFrame(sample_data)
            processed_df = st.session_state.crime_analysis_system.preprocess_crime_data(df)
            st.session_state.crime_data = processed_df
            st.success(f"Generated {num_incidents} sample incidents")
            st.rerun()
    
    # Analysis settings
    if not st.session_state.crime_data.empty:
        st.markdown("---")
        st.subheader("Analysis Settings")
        
        # Hotspot detection parameters
        eps = st.slider("Hotspot Clustering Distance", 0.005, 0.05, 0.01, 0.005,
                       help="Smaller values create tighter clusters")
        
        min_samples = st.slider("Minimum Incidents per Hotspot", 3, 20, 5,
                               help="Minimum incidents to form a hotspot")
        
        # Time range filter
        if 'date' in st.session_state.crime_data.columns:
            date_range = st.date_input(
                "Date Range",
                value=(st.session_state.crime_data['date'].min().date(),
                      st.session_state.crime_data['date'].max().date()),
                help="Filter incidents by date range"
            )

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Interactive Map", "📈 Pattern Analysis", "🎯 Risk Prediction"])

with tab1:
    st.header("Crime Data Overview")
    
    if st.session_state.crime_data.empty:
        st.info("No crime data loaded. Please load data using the sidebar options.")
    else:
        df = st.session_state.crime_data
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Incidents", len(df))
        
        with col2:
            if 'incident_type' in df.columns:
                st.metric("Incident Types", df['incident_type'].nunique())
        
        with col3:
            if 'date' in df.columns:
                date_range_days = (df['date'].max() - df['date'].min()).days
                st.metric("Date Range (days)", date_range_days)
        
        with col4:
            if 'severity' in df.columns:
                avg_severity = df['severity'].mean()
                st.metric("Avg Severity", f"{avg_severity:.1f}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        
        # Show sample records
        st.dataframe(df.head(10), use_container_width=True)
        
        # Incident type distribution
        if 'incident_type' in df.columns:
            st.subheader("🏷️ Incident Type Distribution")
            
            type_counts = df['incident_type'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    x=type_counts.values,
                    y=type_counts.index,
                    orientation='h',
                    title="Incidents by Type",
                    labels={'x': 'Count', 'y': 'Incident Type'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Incident Type Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Temporal distribution
        if 'date' in df.columns:
            st.subheader("📅 Temporal Distribution")
            
            # Daily trend
            daily_counts = df.groupby(df['date'].dt.date).size()
            
            fig_timeline = px.line(
                x=daily_counts.index,
                y=daily_counts.values,
                title="Incidents Over Time",
                labels={'x': 'Date', 'y': 'Number of Incidents'}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

with tab2:
    st.header("Interactive Crime Map")
    
    if st.session_state.crime_data.empty:
        st.info("No crime data available for mapping.")
    else:
        df = st.session_state.crime_data
        
        # Map controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_hotspots = st.checkbox("Show Hotspots", value=True)
        
        with col2:
            show_risk_areas = st.checkbox("Show Risk Areas", value=False)
        
        with col3:
            incident_filter = st.multiselect(
                "Filter by Type",
                options=df['incident_type'].unique() if 'incident_type' in df.columns else [],
                default=df['incident_type'].unique()[:5] if 'incident_type' in df.columns else []
            )
        
        # Filter data
        filtered_df = df.copy()
        if incident_filter and 'incident_type' in df.columns:
            filtered_df = filtered_df[filtered_df['incident_type'].isin(incident_filter)]
        
        if len(filtered_df) > 0:
            # Detect hotspots
            hotspots_df = None
            hotspot_stats = None
            
            if show_hotspots:
                with st.spinner("Detecting crime hotspots..."):
                    hotspots_df = st.session_state.crime_analysis_system.detect_crime_hotspots(
                        filtered_df, eps=eps, min_samples=min_samples
                    )
                    hotspot_stats = getattr(hotspots_df, 'hotspot_stats', [])
            
            # Predict risk areas
            risk_areas = None
            if show_risk_areas:
                with st.spinner("Predicting risk areas..."):
                    risk_areas = st.session_state.crime_analysis_system.predict_risk_areas(filtered_df)
            
            # Create map
            crime_map = st.session_state.crime_analysis_system.create_crime_map(
                filtered_df, hotspot_stats, risk_areas
            )
            
            # Display map
            map_data = st_folium(crime_map, width=700, height=500)
            
            # Map statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Incidents Shown", len(filtered_df))
            
            with col2:
                if hotspot_stats:
                    st.metric("Hotspots Detected", len(hotspot_stats))
            
            with col3:
                if risk_areas:
                    st.metric("Risk Areas", len(risk_areas))
            
            # Hotspot details
            if hotspot_stats:
                st.subheader("🔥 Detected Hotspots")
                
                for i, hotspot in enumerate(hotspot_stats[:5]):  # Show top 5
                    with st.expander(f"Hotspot {i+1} - {hotspot['dominant_type']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Location:** {hotspot['center_lat']:.4f}, {hotspot['center_lng']:.4f}")
                            st.write(f"**Incident Count:** {hotspot['incident_count']}")
                        
                        with col2:
                            st.write(f"**Radius:** {hotspot['radius_meters']:.0f} meters")
                            st.write(f"**Dominant Type:** {hotspot['dominant_type']}")
        
        else:
            st.warning("No incidents match the selected filters.")

with tab3:
    st.header("Crime Pattern Analysis")
    
    if st.session_state.crime_data.empty:
        st.info("No crime data available for pattern analysis.")
    else:
        df = st.session_state.crime_data
        
        # Generate temporal patterns
        if st.button("🔍 Analyze Patterns", type="primary"):
            with st.spinner("Analyzing crime patterns..."):
                patterns = st.session_state.crime_analysis_system.analyze_temporal_patterns(df)
                charts = st.session_state.crime_analysis_system.create_temporal_charts(patterns)
                
                st.session_state.analysis_results = {
                    'patterns': patterns,
                    'charts': charts
                }
        
        # Display patterns if available
        if st.session_state.analysis_results.get('patterns'):
            patterns = st.session_state.analysis_results['patterns']
            charts = st.session_state.analysis_results['charts']
            
            # Pattern summary
            st.subheader("📊 Pattern Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'hourly' in patterns:
                    peak_hour = patterns['hourly']['peak_hour']
                    st.metric("Peak Hour", f"{peak_hour}:00")
            
            with col2:
                if 'daily' in patterns:
                    peak_day = patterns['daily']['peak_day']
                    st.metric("Peak Day", peak_day)
            
            with col3:
                if 'monthly' in patterns:
                    peak_month = patterns['monthly']['peak_month']
                    st.metric("Peak Month", peak_month)
            
            with col4:
                if 'seasonal' in patterns:
                    peak_season = patterns['seasonal']['peak_season']
                    st.metric("Peak Season", peak_season)
            
            # Pattern charts
            st.subheader("📈 Temporal Patterns")
            
            if charts.get('hourly'):
                st.plotly_chart(charts['hourly'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if charts.get('daily'):
                    st.plotly_chart(charts['daily'], use_container_width=True)
            
            with col2:
                if charts.get('monthly'):
                    st.plotly_chart(charts['monthly'], use_container_width=True)
            
            # Detailed pattern analysis
            st.subheader("🔍 Detailed Analysis")
            
            # Hourly distribution insights
            if 'hourly' in patterns:
                hourly_data = patterns['hourly']['data']
                high_activity_hours = [hour for hour, count in hourly_data.items() if count > np.mean(list(hourly_data.values()))]
                
                st.write(f"**High Activity Hours:** {', '.join(map(str, high_activity_hours))}")
            
            # Daily distribution insights
            if 'daily' in patterns:
                daily_data = patterns['daily']['data']
                weekend_crimes = daily_data.get('Saturday', 0) + daily_data.get('Sunday', 0)
                weekday_crimes = sum(count for day, count in daily_data.items() if day not in ['Saturday', 'Sunday'])
                
                st.write(f"**Weekend vs Weekday:** {weekend_crimes} weekend crimes, {weekday_crimes} weekday crimes")

with tab4:
    st.header("Risk Prediction & Hotspot Analysis")
    
    if st.session_state.crime_data.empty:
        st.info("No crime data available for risk prediction.")
    else:
        df = st.session_state.crime_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Risk Area Prediction")
            
            prediction_days = st.slider("Prediction Period (days)", 7, 90, 30)
            
            if st.button("🎯 Predict Risk Areas"):
                with st.spinner("Analyzing risk patterns..."):
                    risk_areas = st.session_state.crime_analysis_system.predict_risk_areas(
                        df, prediction_days
                    )
                    
                    if risk_areas:
                        st.success(f"Identified {len(risk_areas)} high-risk areas")
                        
                        # Display top risk areas
                        st.subheader("🏆 Top Risk Areas")
                        
                        for i, area in enumerate(risk_areas[:5]):
                            with st.expander(f"Risk Area {i+1} - Score: {area['risk_score']:.2f}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Location:** {area['center_lat']:.4f}, {area['center_lng']:.4f}")
                                    st.write(f"**Risk Score:** {area['risk_score']:.2f}")
                                
                                with col2:
                                    st.write(f"**Recent Incidents:** {area['recent_incidents']}")
                                    st.write(f"**Avg Severity:** {area['avg_severity']:.1f}")
                                
                                st.write(f"**Incident Types:** {', '.join(area['incident_types'])}")
                    
                    else:
                        st.warning("No significant risk areas identified")
        
        with col2:
            st.subheader("🔥 Hotspot Analysis")
            
            if st.button("🔍 Detect Hotspots"):
                with st.spinner("Detecting crime hotspots..."):
                    hotspots_df = st.session_state.crime_analysis_system.detect_crime_hotspots(
                        df, eps=eps, min_samples=min_samples
                    )
                    
                    hotspot_stats = getattr(hotspots_df, 'hotspot_stats', [])
                    
                    if hotspot_stats:
                        st.success(f"Detected {len(hotspot_stats)} hotspots")
                        
                        # Hotspot summary
                        total_hotspot_incidents = sum(h['incident_count'] for h in hotspot_stats)
                        avg_incidents_per_hotspot = total_hotspot_incidents / len(hotspot_stats)
                        
                        st.metric("Total Hotspot Incidents", total_hotspot_incidents)
                        st.metric("Avg Incidents per Hotspot", f"{avg_incidents_per_hotspot:.1f}")
                        
                        # Hotspot types distribution
                        hotspot_types = {}
                        for hotspot in hotspot_stats:
                            dominant_type = hotspot['dominant_type']
                            hotspot_types[dominant_type] = hotspot_types.get(dominant_type, 0) + 1
                        
                        st.write("**Hotspot Types:**")
                        for crime_type, count in hotspot_types.items():
                            st.write(f"• {crime_type}: {count} hotspots")
                    
                    else:
                        st.warning("No hotspots detected with current settings")
        
        # Combined analysis
        st.subheader("📋 Combined Risk Assessment")
        
        if st.button("🔬 Generate Comprehensive Analysis"):
            with st.spinner("Generating comprehensive crime analysis..."):
                # Generate statistics
                stats = st.session_state.crime_analysis_system.generate_crime_statistics(df)
                
                if stats:
                    st.subheader("📊 Crime Statistics Summary")
                    
                    # Basic stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Incidents", stats['total_incidents'])
                        st.metric("Unique Types", stats['unique_types'])
                    
                    with col2:
                        if 'date_range' in stats:
                            st.write(f"**Date Range:**")
                            st.write(f"From: {stats['date_range']['start']}")
                            st.write(f"To: {stats['date_range']['end']}")
                    
                    with col3:
                        if 'trends' in stats:
                            trend = stats['trends']
                            direction = "📈" if trend['trend_direction'] == 'increasing' else "📉"
                            st.write(f"**Trend:** {direction} {trend['trend_direction'].title()}")
                            st.write(f"**Change:** {trend['trend_percentage']:.1f}%")
                    
                    # Most common incident type
                    if 'incident_types' in stats:
                        most_common = stats['incident_types']['most_common']
                        st.info(f"**Most Common Crime Type:** {most_common}")
                    
                    # Geographic coverage
                    if 'geographic' in stats:
                        geo = stats['geographic']
                        st.write(f"**Geographic Coverage:**")
                        st.write(f"Center: {geo['center']['latitude']:.4f}, {geo['center']['longitude']:.4f}")

# Data export and sharing
if not st.session_state.crime_data.empty:
    st.markdown("---")
    st.header("📤 Export & Share")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data"):
            csv = st.session_state.crime_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="crime_analysis_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export Analysis"):
            if st.session_state.analysis_results:
                analysis_json = json.dumps(st.session_state.analysis_results.get('patterns', {}), 
                                         indent=2, default=str)
                st.download_button(
                    label="Download Analysis",
                    data=analysis_json,
                    file_name="crime_pattern_analysis.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("🗺️ Export Map"):
            st.info("Map export functionality would be implemented here")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### Crime Analysis System Help
    
    **Data Requirements:**
    - **Required Columns:** incident_type, latitude, longitude, date
    - **Optional Columns:** severity, description, address
    - **Supported Formats:** CSV, Excel
    
    **Analysis Features:**
    - **Hotspot Detection:** Identifies areas with high crime concentration
    - **Risk Prediction:** Predicts future high-risk areas based on historical patterns
    - **Temporal Analysis:** Analyzes crime patterns by time (hour, day, month, season)
    - **Interactive Mapping:** Visualizes crimes and patterns on interactive maps
    
    **Tips:**
    - Ensure coordinate data is accurate for proper mapping
    - Use date filters to analyze specific time periods
    - Adjust clustering parameters for different hotspot sensitivities
    - Review predicted risk areas for patrol planning
    
    **Analysis Parameters:**
    - **Clustering Distance:** Smaller values create tighter hotspots
    - **Minimum Samples:** Higher values require more incidents to form hotspots
    - **Prediction Period:** Longer periods may be less accurate but show broader trends
    """)
