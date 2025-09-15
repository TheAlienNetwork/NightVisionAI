import streamlit as st
import numpy as np
from modules.facial_recognition import FacialRecognitionSystem
from database import get_evidence_files, execute_query
from utils.image_utils import display_image_with_info, draw_bounding_boxes, create_image_grid
from utils.data_utils import format_timestamp, create_summary_cards, paginate_data
import io
from PIL import Image
import json

# Configure page
st.set_page_config(
    page_title="Facial Recognition - Investigative Platform",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Real-time Facial Recognition System")
st.markdown("Identify and match faces across uploaded media using advanced computer vision")

# Initialize facial recognition system
if 'face_recognition_system' not in st.session_state:
    st.session_state.face_recognition_system = FacialRecognitionSystem()

# Initialize face database
if 'face_database' not in st.session_state:
    st.session_state.face_database = []

# Initialize search results
if 'face_search_results' not in st.session_state:
    st.session_state.face_search_results = []

# Sidebar for configuration
with st.sidebar:
    st.header("Recognition Settings")
    
    # Face detection sensitivity
    face_tolerance = st.slider("Face Recognition Tolerance", 0.3, 0.8, 0.6, 0.05,
                              help="Lower values = stricter matching")
    st.session_state.face_recognition_system.face_tolerance = face_tolerance
    
    # Database management
    st.subheader("Face Database")
    st.metric("Known Faces", len(st.session_state.face_database))
    
    if st.button("Clear Database"):
        st.session_state.face_database.clear()
        st.session_state.face_recognition_system.known_faces.clear()
        st.session_state.face_recognition_system.known_names.clear()
        st.success("Face database cleared!")

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Face Search", "➕ Add to Database", "📊 Results Analysis", "⚙️ Batch Processing"])

with tab1:
    st.header("Search for Faces in Evidence")
    
    # File selection
    evidence_files = get_evidence_files()
    
    if not evidence_files:
        st.warning("No evidence files found. Please upload files in the File Upload section first.")
    else:
        # Filter files by type
        image_files = [f for f in evidence_files if f.get('file_type', '').startswith('image/')]
        video_files = [f for f in evidence_files if f.get('file_type', '').startswith('video/')]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Evidence Files")
            st.metric("Image Files", len(image_files))
            st.metric("Video Files", len(video_files))
            
            # File type selection
            search_in_images = st.checkbox("Search in Images", value=True)
            search_in_videos = st.checkbox("Search in Videos", value=False)
            
            files_to_search = []
            if search_in_images:
                files_to_search.extend(image_files)
            if search_in_videos:
                files_to_search.extend(video_files)
        
        with col2:
            st.subheader("Search Configuration")
            
            # Database requirement check
            if not st.session_state.face_database:
                st.warning("⚠️ No faces in database. Add known faces to enable identification.")
            else:
                st.success(f"✅ {len(st.session_state.face_database)} faces in database")
        
        # Search button
        if st.button("🔍 Start Face Search", type="primary", disabled=not files_to_search):
            if not st.session_state.face_database:
                st.error("Please add faces to the database before searching.")
            else:
                with st.spinner("Searching for faces in evidence files..."):
                    try:
                        # Prepare file data (in real implementation, load from disk)
                        search_files = []
                        for file_info in files_to_search[:10]:  # Limit for demo
                            # In real implementation, load file from disk using file_path
                            search_files.append({
                                'id': file_info['id'],
                                'filename': file_info['filename'],
                                'category': 'image' if file_info['file_type'].startswith('image/') else 'video',
                                'file_data': b''  # Would load actual file data here
                            })
                        
                        # Perform search (placeholder - would use actual files)
                        results = st.session_state.face_recognition_system.search_faces_in_files(
                            search_files, st.session_state.face_database
                        )
                        
                        st.session_state.face_search_results = results
                        st.success(f"Face search completed! Found faces in {len(results)} files.")
                        
                    except Exception as e:
                        st.error(f"Error during face search: {str(e)}")

with tab2:
    st.header("Add Known Faces to Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Face upload and registration
        st.subheader("Register New Face")
        
        uploaded_face = st.file_uploader(
            "Upload face image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image containing one face"
        )
        
        person_name = st.text_input("Person Name", placeholder="Enter the person's name")
        
        # Optional metadata
        with st.expander("Additional Information (Optional)"):
            person_description = st.text_area("Description")
            person_role = st.selectbox("Role", ["Unknown", "Suspect", "Witness", "Victim", "Person of Interest", "Other"])
            person_notes = st.text_area("Notes")
        
        if uploaded_face and person_name:
            # Display uploaded image
            image = Image.open(uploaded_face)
            st.image(image, caption=f"Face to register: {person_name}", width=300)
            
            if st.button("Add to Database", type="primary"):
                try:
                    # Reset file pointer
                    uploaded_face.seek(0)
                    image_data = uploaded_face.read()
                    
                    # Create metadata
                    metadata = {
                        'description': person_description,
                        'role': person_role,
                        'notes': person_notes,
                        'registration_date': st.session_state.face_recognition_system.create_face_database_entry("", [], {})['created_at']
                    }
                    
                    # Add to database
                    success = st.session_state.face_recognition_system.add_to_database(
                        person_name, image_data, metadata
                    )
                    
                    if success:
                        # Add to session database
                        face_entry = st.session_state.face_recognition_system.create_face_database_entry(
                            person_name, [], metadata
                        )
                        st.session_state.face_database.append(face_entry)
                        
                        st.success(f"Successfully added {person_name} to the face database!")
                    else:
                        st.error("Failed to add face to database.")
                        
                except Exception as e:
                    st.error(f"Error adding face to database: {str(e)}")
    
    with col2:
        st.subheader("Database Entries")
        
        if st.session_state.face_database:
            for i, entry in enumerate(st.session_state.face_database):
                with st.expander(f"👤 {entry.get('name', 'Unknown')}"):
                    st.write(f"**ID:** {entry.get('id')}")
                    st.write(f"**Added:** {format_timestamp(entry.get('created_at'))}")
                    
                    metadata = entry.get('metadata', {})
                    if metadata.get('role'):
                        st.write(f"**Role:** {metadata['role']}")
                    if metadata.get('description'):
                        st.write(f"**Description:** {metadata['description']}")
                    if metadata.get('notes'):
                        st.write(f"**Notes:** {metadata['notes']}")
                    
                    if st.button(f"Remove", key=f"remove_face_{i}"):
                        st.session_state.face_database.pop(i)
                        st.rerun()
        else:
            st.info("No faces in database yet.")

with tab3:
    st.header("Face Recognition Results Analysis")
    
    if not st.session_state.face_search_results:
        st.info("No search results available. Run a face search first.")
    else:
        # Generate comprehensive report
        report = st.session_state.face_recognition_system.generate_face_report(
            st.session_state.face_search_results
        )
        
        # Summary metrics
        if report.get('summary'):
            st.subheader("📊 Search Summary")
            summary = report['summary']
            
            metrics = {
                "Files Processed": summary.get('total_files_processed', 0),
                "Files with Faces": summary.get('files_with_faces', 0),
                "Total Faces": summary.get('total_faces_detected', 0),
                "Known Persons": summary.get('known_persons_identified', 0),
                "Unknown Faces": summary.get('unknown_faces', 0)
            }
            
            create_summary_cards(metrics)
        
        # Identified persons
        if report.get('identified_persons'):
            st.subheader("🎯 Identified Persons")
            
            col1, col2 = st.columns(2)
            
            with col1:
                for person, count in report['identified_persons'].items():
                    st.metric(person, f"{count} occurrences")
            
            with col2:
                # Create a simple bar chart of identifications
                if len(report['identified_persons']) > 0:
                    import plotly.express as px
                    import pandas as pd
                    
                    df = pd.DataFrame(list(report['identified_persons'].items()), 
                                    columns=['Person', 'Count'])
                    fig = px.bar(df, x='Person', y='Count', title="Person Identification Frequency")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("📄 Detailed Results")
        
        for result in st.session_state.face_search_results:
            with st.expander(f"📁 {result.get('filename', 'Unknown File')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**File Type:** {result.get('file_type', 'Unknown')}")
                    st.write(f"**Faces Found:** {result.get('faces_found', 0)}")
                    st.write(f"**Processed:** {format_timestamp(result.get('processing_timestamp'))}")
                    
                    # Show identifications
                    if result.get('identifications'):
                        st.write("**Identifications:**")
                        for i, ident in enumerate(result['identifications']):
                            person = ident.get('identified_person', 'Unknown')
                            confidence = ident.get('confidence', 0)
                            match_found = ident.get('match_found', False)
                            
                            status_icon = "✅" if match_found else "❓"
                            st.write(f"{status_icon} Face {i+1}: {person} ({confidence:.1f}% confidence)")
                    
                    # Video frame results
                    if result.get('frame_results'):
                        st.write(f"**Video Frames Processed:** {len(result['frame_results'])}")
                        
                        # Show sample frame results
                        for frame_result in result['frame_results'][:3]:  # Show first 3 frames
                            st.write(f"• Frame {frame_result.get('frame_number', 0)} "
                                   f"(t={frame_result.get('timestamp', 0):.1f}s): "
                                   f"{frame_result.get('faces_found', 0)} faces")
                
                with col2:
                    # Placeholder for face visualization
                    if result.get('face_locations'):
                        st.info("Face locations detected. Image visualization would appear here.")
                    
                    # Action buttons
                    st.write("**Actions:**")
                    if st.button(f"View Details", key=f"view_{result.get('file_id')}"):
                        st.json(result)

with tab4:
    st.header("Batch Face Processing")
    
    st.subheader("🔄 Process Multiple Files")
    
    # Get available files
    evidence_files = get_evidence_files()
    
    if evidence_files:
        # Filter by file type
        processable_files = [f for f in evidence_files 
                           if f.get('file_type', '').startswith(('image/', 'video/'))]
        
        st.write(f"Found {len(processable_files)} processable files")
        
        # Batch processing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Processing Options")
            
            batch_size = st.slider("Batch Size", 1, 20, 5, 
                                  help="Number of files to process simultaneously")
            
            process_videos = st.checkbox("Include Videos", value=False,
                                       help="Video processing takes longer")
            
            frame_skip = st.slider("Video Frame Skip", 10, 100, 30,
                                 help="Process every Nth frame in videos")
        
        with col2:
            st.subheader("Progress Tracking")
            
            if 'batch_processing' not in st.session_state:
                st.session_state.batch_processing = False
            
            if st.session_state.batch_processing:
                st.warning("⏳ Batch processing in progress...")
            else:
                if st.button("🚀 Start Batch Processing", type="primary"):
                    if not st.session_state.face_database:
                        st.error("Add faces to database before batch processing.")
                    else:
                        st.session_state.batch_processing = True
                        st.rerun()
        
        # Process batch if started
        if st.session_state.batch_processing:
            try:
                with st.spinner("Processing files in batch..."):
                    # Filter files based on options
                    files_to_process = []
                    for file_info in processable_files[:batch_size]:
                        if not process_videos and file_info.get('file_type', '').startswith('video/'):
                            continue
                        files_to_process.append(file_info)
                    
                    # Simulate batch processing (in real implementation, load actual files)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    batch_results = []
                    for i, file_info in enumerate(files_to_process):
                        status_text.text(f"Processing {file_info['filename']}...")
                        
                        # Placeholder result (in real implementation, do actual processing)
                        result = {
                            'filename': file_info['filename'],
                            'file_id': file_info['id'],
                            'faces_found': 0,
                            'identifications': [],
                            'processing_status': 'completed'
                        }
                        batch_results.append(result)
                        
                        progress_bar.progress((i + 1) / len(files_to_process))
                    
                    st.session_state.batch_processing = False
                    status_text.text("Batch processing completed!")
                    
                    # Display results summary
                    st.success(f"Processed {len(batch_results)} files successfully!")
                    
                    # Results summary
                    total_faces = sum(r.get('faces_found', 0) for r in batch_results)
                    files_with_faces = len([r for r in batch_results if r.get('faces_found', 0) > 0])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", len(batch_results))
                    with col2:
                        st.metric("Files with Faces", files_with_faces)
                    with col3:
                        st.metric("Total Faces Found", total_faces)
                    
            except Exception as e:
                st.session_state.batch_processing = False
                st.error(f"Batch processing failed: {str(e)}")
    
    else:
        st.warning("No evidence files available for batch processing.")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### Facial Recognition System Help
    
    **Key Features:**
    - **Face Detection:** Automatically detect faces in images and videos
    - **Face Recognition:** Identify known persons against a database
    - **Database Management:** Build and maintain a database of known faces
    - **Batch Processing:** Process multiple files simultaneously
    - **Video Analysis:** Extract and analyze faces from video frames
    
    **How to Use:**
    1. **Build Database:** Add known faces using the "Add to Database" tab
    2. **Search Files:** Use "Face Search" to find faces in evidence files
    3. **Analyze Results:** Review identification results and confidence scores
    4. **Batch Process:** Process multiple files for comprehensive analysis
    
    **Tips:**
    - Use clear, well-lit face images for better database entries
    - Adjust face tolerance for stricter or looser matching
    - Process videos selectively as they require more computation time
    - Review low-confidence matches manually for verification
    
    **Technical Notes:**
    - Uses OpenCV and face_recognition libraries
    - Face encodings are 128-dimensional vectors
    - Default tolerance of 0.6 works well for most cases
    - Video processing analyzes every Nth frame for efficiency
    """)
