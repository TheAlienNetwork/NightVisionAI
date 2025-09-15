import streamlit as st
import os
from typing import List, Dict
from modules.file_processor import FileProcessor
from database import add_evidence_file, get_cases, create_case
from utils.data_utils import format_file_size, format_timestamp, create_summary_cards
from utils.image_utils import display_image_with_info, extract_frames_from_video
import io

# Configure page
st.set_page_config(
    page_title="File Upload - Investigative Platform",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Multi-Modal File Upload System")
st.markdown("Upload and process images, videos, documents, and datasets for investigation")

# Initialize file processor
if 'file_processor' not in st.session_state:
    st.session_state.file_processor = FileProcessor()

# Initialize session state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar for case selection and settings
with st.sidebar:
    st.header("Upload Settings")
    
    # Case selection
    st.subheader("Case Assignment")
    cases = get_cases()
    
    # Option to create new case
    create_new_case = st.checkbox("Create New Case")
    
    if create_new_case:
        new_case_name = st.text_input("Case Name")
        new_case_description = st.text_area("Case Description")
        
        if st.button("Create Case") and new_case_name:
            case_id = create_case(new_case_name, new_case_description)
            if case_id:
                st.success(f"Case '{new_case_name}' created successfully!")
                st.rerun()
    
    # Select existing case
    if cases:
        case_options = {f"🔍 {case['name']} ({case['status']})": case['id'] for case in cases}
        case_options = {"🚫 No case selected": None, **case_options}
        
        selected_case_display = st.selectbox("🎯 **Select Investigation Case**", list(case_options.keys()))
        selected_case_id = case_options[selected_case_display]
        
        if selected_case_id:
            # Show case info
            selected_case = next(c for c in cases if c['id'] == selected_case_id)
            st.markdown(f"""
            <div style="background: rgba(0, 212, 170, 0.1); border: 1px solid #00d4aa; 
                        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="color: #00d4aa; margin: 0;">📁 {selected_case['name']}</h4>
                <p style="color: #b0b3b8; margin: 0.5rem 0 0 0;">Status: {selected_case['status']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No cases available. Create a case in Evidence Manager first.")
        selected_case_id = None
    
    # Upload settings
    st.subheader("Processing Options")
    auto_process = st.checkbox("Auto-process files", value=True)
    extract_metadata = st.checkbox("Extract metadata", value=True)
    calculate_hashes = st.checkbox("Calculate file hashes", value=True)
    detect_duplicates = st.checkbox("Detect duplicates", value=True)

# Main upload interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("File Upload")
    
    # File uploader with multiple file support
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov', 'mkv', 'wmv', 
              'pdf', 'docx', 'txt', 'csv', 'xlsx', 'json', 'zip', '7z'],
        help="Supported formats: Images (JPG, PNG, TIFF, etc.), Videos (MP4, AVI, MOV, etc.), Documents (PDF, DOCX, TXT), Datasets (CSV, XLSX, JSON), Archives (ZIP, 7Z)"
    )
    
    # Process uploaded files
    if uploaded_files and st.button("Process Files", type="primary"):
        if not selected_case_id:
            st.error("Please select or create a case before uploading files.")
        else:
            with st.spinner("Processing uploaded files..."):
                try:
                    # Process files
                    processed_files = st.session_state.file_processor.batch_process_files(uploaded_files)
                    
                    # Save files to database and disk
                    saved_files = []
                    for processed_file in processed_files:
                        # Save file to disk
                        file_path = st.session_state.file_processor.save_file_to_disk(
                            processed_file['file_data'],
                            processed_file['filename'],
                            selected_case_id
                        )
                        
                        # Add to database
                        file_id = add_evidence_file(
                            case_id=selected_case_id,
                            filename=processed_file['filename'],
                            file_type=processed_file['mime_type'],
                            file_size=processed_file['size'],
                            file_path=file_path,
                            metadata=processed_file.get('metadata', {}),
                            hash_value=processed_file['hash']
                        )
                        
                        if file_id:
                            processed_file['id'] = file_id
                            processed_file['file_path'] = file_path
                            saved_files.append(processed_file)
                            
                            # Log activity
                            from database import log_case_activity
                            log_case_activity(
                                selected_case_id, 
                                "evidence_uploaded", 
                                f"Evidence file uploaded: {processed_file['filename']} ({processed_file['mime_type']})",
                                metadata={
                                    "file_size": processed_file['size'],
                                    "file_type": processed_file['mime_type'],
                                    "hash": processed_file['hash']
                                }
                            )
                    
                    # Update session state
                    st.session_state.uploaded_files.extend(saved_files)
                    
                    st.success(f"Successfully processed and saved {len(saved_files)} files!")
                    
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

with col2:
    st.header("Upload Summary")
    
    if st.session_state.uploaded_files:
        # Processing summary
        summary = st.session_state.file_processor.get_processing_summary(st.session_state.uploaded_files)
        
        # Summary metrics
        metrics = {
            "Total Files": summary.get('total_files', 0),
            "Total Size": format_file_size(summary.get('total_size', 0)),
            "Categories": len(summary.get('categories', {})),
            "Duplicates": len(summary.get('duplicates', {}))
        }
        
        create_summary_cards(metrics, columns=2)
        
        # File categories chart
        if summary.get('categories'):
            st.subheader("Files by Category")
            categories = summary['categories']
            
            for category, count in categories.items():
                st.metric(category.title(), count)
        
        # Duplicate detection
        if summary.get('duplicates'):
            st.subheader("⚠️ Duplicates Detected")
            for hash_value, filenames in summary['duplicates'].items():
                with st.expander(f"Duplicate set ({len(filenames)} files)"):
                    for filename in filenames:
                        st.write(f"• {filename}")
    
    else:
        st.info("No files uploaded yet.")

# File management section
st.markdown("---")
st.header("📋 Uploaded Files Management")

if st.session_state.uploaded_files:
    # Filter and search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Category filter
        categories = list(set(f.get('category', 'Unknown') for f in st.session_state.uploaded_files))
        selected_category = st.selectbox("Filter by Category", ["All"] + categories)
    
    with col2:
        # File type filter
        file_types = list(set(f.get('mime_type', 'Unknown') for f in st.session_state.uploaded_files))
        selected_type = st.selectbox("Filter by Type", ["All"] + file_types)
    
    with col3:
        # Search
        search_term = st.text_input("Search files", placeholder="Enter filename or keywords")
    
    # Filter files
    filtered_files = st.session_state.uploaded_files.copy()
    
    if selected_category != "All":
        filtered_files = [f for f in filtered_files if f.get('category') == selected_category]
    
    if selected_type != "All":
        filtered_files = [f for f in filtered_files if f.get('mime_type') == selected_type]
    
    if search_term:
        filtered_files = [f for f in filtered_files if search_term.lower() in f.get('filename', '').lower()]
    
    # Display files
    st.subheader(f"Files ({len(filtered_files)})")
    
    for i, file_info in enumerate(filtered_files):
        with st.expander(f"📄 {file_info.get('filename', 'Unknown')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**File Type:** {file_info.get('mime_type', 'Unknown')}")
                st.write(f"**Size:** {format_file_size(file_info.get('size', 0))}")
                st.write(f"**Category:** {file_info.get('category', 'Unknown').title()}")
                st.write(f"**Hash:** `{file_info.get('hash', 'Unknown')[:16]}...`")
                st.write(f"**Uploaded:** {format_timestamp(file_info.get('upload_timestamp', ''))}")
                
                # Display metadata if available
                metadata = file_info.get('metadata', {})
                if metadata:
                    st.write("**Metadata:**")
                    for key, value in metadata.items():
                        if key not in ['error'] and value:
                            st.write(f"• {key}: {value}")
            
            with col2:
                # Preview for images
                if file_info.get('category') == 'image' and 'file_data' in file_info:
                    try:
                        image_data = file_info['file_data']
                        st.image(image_data, width=200, caption="Preview")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                
                # Preview for videos
                elif file_info.get('category') == 'video' and 'file_data' in file_info:
                    try:
                        frames = extract_frames_from_video(file_info['file_data'], num_frames=1)
                        if frames:
                            st.image(frames[0], width=200, caption="Video Frame")
                    except Exception as e:
                        st.error(f"Error extracting video frame: {str(e)}")
                
                # Actions
                st.write("**Actions:**")
                if st.button(f"View Details", key=f"details_{i}"):
                    st.session_state[f"show_details_{i}"] = True
                
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.uploaded_files.remove(file_info)
                    st.rerun()
        
        # Show detailed view if requested
        if st.session_state.get(f"show_details_{i}"):
            st.subheader(f"Detailed View: {file_info.get('filename')}")
            
            # Technical details
            st.json(file_info)
            
            if st.button(f"Close Details", key=f"close_{i}"):
                st.session_state[f"show_details_{i}"] = False
                st.rerun()

else:
    st.info("No files uploaded yet. Use the upload section above to add files.")

# Bulk actions
if st.session_state.uploaded_files:
    st.markdown("---")
    st.header("🔧 Bulk Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear All Files"):
            st.session_state.uploaded_files.clear()
            st.success("All files cleared!")
            st.rerun()
    
    with col2:
        if st.button("Recalculate Hashes"):
            with st.spinner("Recalculating hashes..."):
                for file_info in st.session_state.uploaded_files:
                    if 'file_data' in file_info:
                        file_info['hash'] = st.session_state.file_processor.get_file_hash(file_info['file_data'])
            st.success("Hashes recalculated!")
    
    with col3:
        if st.button("Export File List"):
            import pandas as pd
            
            # Create export data
            export_data = []
            for file_info in st.session_state.uploaded_files:
                export_data.append({
                    'filename': file_info.get('filename'),
                    'category': file_info.get('category'),
                    'mime_type': file_info.get('mime_type'),
                    'size': file_info.get('size'),
                    'hash': file_info.get('hash'),
                    'upload_timestamp': file_info.get('upload_timestamp')
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="uploaded_files.csv",
                mime="text/csv"
            )

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### File Upload System Help
    
    **Supported File Types:**
    - **Images:** JPG, JPEG, PNG, BMP, TIFF, WebP
    - **Videos:** MP4, AVI, MOV, MKV, WMV
    - **Documents:** PDF, DOCX, TXT
    - **Datasets:** CSV, XLSX, JSON
    - **Archives:** ZIP, 7Z
    
    **Features:**
    - **Metadata Extraction:** Automatically extracts EXIF data, file properties, and other metadata
    - **Hash Calculation:** Generates SHA-256 hashes for file integrity verification
    - **Duplicate Detection:** Identifies duplicate files based on hash values
    - **Case Management:** Organize files by investigation case
    - **Batch Processing:** Upload and process multiple files simultaneously
    
    **Tips:**
    - Create or select a case before uploading files for better organization
    - Use descriptive filenames for easier identification
    - Review the duplicate detection results to avoid redundant evidence
    - Check the metadata extraction results for potential forensic value
    """)
