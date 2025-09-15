
import streamlit as st
import os
from database import init_database

# Configure page
st.set_page_config(
    page_title="Investigative Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }
    
    /* Headers */
    h1 {
        color: #00d4aa !important;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 212, 170, 0.3);
    }
    
    h2, h3 {
        color: #fafafa !important;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1d29 0%, #262730 100%);
        border-right: 2px solid #00d4aa;
    }
    
    /* Metric cards */
    .metric-container {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid #00d4aa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00d4aa, #00b894);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 212, 170, 0.4);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 212, 170, 0.1);
        border: 1px solid #00d4aa;
        border-radius: 8px;
    }
    
    /* Cards and expanders */
    .streamlit-expanderHeader {
        background: rgba(38, 39, 48, 0.8);
        border: 1px solid #00d4aa;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(38, 39, 48, 0.6);
        border: 1px solid #444;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00d4aa, #00b894);
        border-color: #00d4aa;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize database on first run
    if 'db_initialized' not in st.session_state:
        try:
            init_database()
            st.session_state.db_initialized = True
        except Exception as e:
            st.error(f"Database initialization failed: {str(e)}")
            return

    # Main page content
    st.title("🔍 Advanced Investigative Intelligence Platform")
    
    # Hero section with gradient background
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0,212,170,0.1) 0%, rgba(38,39,48,0.1) 100%); 
                padding: 2rem; border-radius: 15px; margin: 1rem 0; border: 1px solid #00d4aa;">
        <h2 style="margin: 0; color: #00d4aa;">Next-Generation Crime Investigation Suite</h2>
        <p style="color: #b0b3b8; font-size: 1.1rem; margin-top: 0.5rem;">
            Leverage AI, machine learning, and advanced analytics for comprehensive criminal investigations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Platform overview with modern cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3 style="color: #00d4aa; margin-top: 0;">🎯 Core Intelligence</h3>
            <ul style="color: #fafafa; list-style: none; padding-left: 0;">
                <li>• Multi-modal evidence processing</li>
                <li>• Real-time facial recognition</li>
                <li>• Criminal pattern analysis</li>
                <li>• Digital forensics toolkit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3 style="color: #00d4aa; margin-top: 0;">🤖 AI-Powered Analysis</h3>
            <ul style="color: #fafafa; list-style: none; padding-left: 0;">
                <li>• Advanced anomaly detection</li>
                <li>• Perceptual hash matching</li>
                <li>• Suspect identification</li>
                <li>• Threat level assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3 style="color: #00d4aa; margin-top: 0;">📊 Intelligence Visualization</h3>
            <ul style="color: #fafafa; list-style: none; padding-left: 0;">
                <li>• Interactive crime mapping</li>
                <li>• Timeline reconstruction</li>
                <li>• Geospatial hotspot analysis</li>
                <li>• Comprehensive case reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation instructions
    st.subheader("🚀 Investigation Modules")
    
    st.markdown("""
    <div style="background: rgba(38, 39, 48, 0.6); padding: 1.5rem; border-radius: 10px; border: 1px solid #444;">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div>
                <h4 style="color: #00d4aa; margin-top: 0;">📁 Evidence Processing</h4>
                <p style="color: #b0b3b8;">Upload and analyze multimedia evidence with AI-powered insights</p>
            </div>
            <div>
                <h4 style="color: #00d4aa; margin-top: 0;">👤 Facial Recognition</h4>
                <p style="color: #b0b3b8;">Identify and track persons of interest across evidence files</p>
            </div>
            <div>
                <h4 style="color: #00d4aa; margin-top: 0;">🗺️ Crime Analysis</h4>
                <p style="color: #b0b3b8;">Analyze patterns, predict hotspots, and assess risk areas</p>
            </div>
            <div>
                <h4 style="color: #00d4aa; margin-top: 0;">🔬 Digital Forensics</h4>
                <p style="color: #b0b3b8;">Extract metadata and verify evidence integrity</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System status with modern styling
    st.subheader("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        db_status = "🟢 Connected" if st.session_state.get('db_initialized') else "🔴 Disconnected"
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #00d4aa; margin: 0;">Database</h4>
            <p style="color: #fafafa; margin: 0.5rem 0 0 0;">{db_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        openai_key = os.getenv("OPENAI_API_KEY")
        ai_status = "🟢 Available" if openai_key else "🟡 Not Configured"
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #00d4aa; margin: 0;">AI Services</h4>
            <p style="color: #fafafa; margin: 0.5rem 0 0 0;">{ai_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        files_count = len(st.session_state.get('uploaded_files', []))
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #00d4aa; margin: 0;">Evidence Files</h4>
            <p style="color: #fafafa; margin: 0.5rem 0 0 0;">{files_count} processed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cases_count = len(st.session_state.get('active_cases', []))
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #00d4aa; margin: 0;">Active Cases</h4>
            <p style="color: #fafafa; margin: 0.5rem 0 0 0;">{cases_count} ongoing</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
