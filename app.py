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
    st.title("🔍 Comprehensive Investigative Platform")
    st.markdown("---")
    
    # Platform overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("🎯 Core Features")
        st.write("• Multi-modal file upload system")
        st.write("• Real-time facial recognition")
        st.write("• Crime pattern analysis")
        st.write("• Digital forensics toolkit")
    
    with col2:
        st.header("🤖 AI-Powered Analysis")
        st.write("• Image/video anomaly detection")
        st.write("• Perceptual hashing")
        st.write("• Advanced object recognition")
        st.write("• Suspicious activity detection")
    
    with col3:
        st.header("📊 Visualization & Mapping")
        st.write("• Interactive crime mapping")
        st.write("• Timeline visualization")
        st.write("• Geospatial analysis")
        st.write("• Evidence management")
    
    st.markdown("---")
    
    # Navigation instructions
    st.header("🚀 Getting Started")
    st.info("""
    Navigate through the different modules using the sidebar:
    
    1. **File Upload** - Upload and process images, videos, documents, and datasets
    2. **Facial Recognition** - Identify and match faces across uploaded media
    3. **Crime Analysis** - Analyze patterns, hotspots, and predict risk areas
    4. **Digital Forensics** - Extract metadata and analyze digital evidence
    5. **Evidence Manager** - Organize cases with timeline visualization
    6. **AI Analysis** - Advanced AI-powered image and video analysis
    """)
    
    # System status
    st.markdown("---")
    st.header("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database", "Connected" if st.session_state.get('db_initialized') else "Disconnected")
    
    with col2:
        openai_key = os.getenv("OPENAI_API_KEY")
        st.metric("OpenAI API", "Available" if openai_key else "Not Configured")
    
    with col3:
        st.metric("Files Processed", len(st.session_state.get('uploaded_files', [])))
    
    with col4:
        st.metric("Active Cases", len(st.session_state.get('active_cases', [])))

if __name__ == "__main__":
    main()
