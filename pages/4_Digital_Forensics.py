import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.digital_forensics import DigitalForensicsToolkit
from database import get_evidence_files, execute_query
from utils.data_utils import format_file_size, format_timestamp, create_summary_cards
from utils.image_utils import display_image_with_info
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Digital Forensics - Investigative Platform",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Digital Forensics Toolkit")
st.markdown("Extract metadata, analyze file integrity, and detect tampering using advanced forensic techniques")

# Initialize forensics toolkit
if 'forensics_toolkit' not in st.session_state:
    st.session_state.forensics_toolkit = DigitalForensicsToolkit()

# Initialize analysis results
if 'forensics_results' not in st.session_state:
    st.session_state.forensics_results = []

if 'timeline_analysis' not in st.session_state:
    st.session_state.timeline_analysis = {}

if 'manipulation_results' not in st.session_state:
    st.session_state.manipulation_results = []

# Sidebar for settings and file selection
with st.sidebar:
    st.header("Forensics Settings")
    
    # Hash algorithms selection
    st.subheader("Hash Algorithms")
    selected_algorithms = st.multiselect(
        "Select algorithms",
        ['md5', 'sha1', 'sha256', 'sha512'],
        default=['sha256', 'md5']
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    extract_metadata = st.checkbox("Extract Metadata", value=True)
    detect_manipulation = st.checkbox("Detect Manipulation", value=True)
    timeline_analysis = st.checkbox("Timeline Analysis", value=True)
    
    # File filters
    st.subheader("File Filters")
    
    # Get available evidence files
    evidence_files = get_evidence_files()
    
    if evidence_files:
        file_types = list(set(f.get('file_type', 'Unknown') for f in evidence_files))
        selected_file_types = st.multiselect(
            "Filter by file type",
            file_types,
            default=file_types
        )
        
        # Filter files
        filtered_files = [f for f in evidence_files if f.get('file_type') in selected_file_types]
        
        st.metric("Available Files", len(filtered_files))
    else:
        st.warning("No evidence files found")
        filtered_files = []

# Main interface tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 File Analysis", "📅 Timeline Analysis", "🛡️ Integrity Check", "📊 Forensics Report"])

with tab1:
    st.header("Comprehensive File Analysis")
    
    if not filtered_files:
        st.warning("No evidence files available. Please upload files in the File Upload section first.")
    else:
        # File selection for analysis
        st.subheader("Select Files for Analysis")
        
        # Display files with checkboxes
        selected_files = []
        
        # Pagination for large file lists
        files_per_page = 10
        total_pages = (len(filtered_files) + files_per_page - 1) // files_per_page
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            start_idx = page * files_per_page
            end_idx = min(start_idx + files_per_page, len(filtered_files))
            page_files = filtered_files[start_idx:end_idx]
        else:
            page_files = filtered_files
        
        # File selection interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for i, file_info in enumerate(page_files):
                file_key = f"file_{file_info['id']}"
                
                if st.checkbox(
                    f"📄 {file_info['filename']}",
                    key=file_key,
                    help=f"Type: {file_info.get('file_type', 'Unknown')} | Size: {format_file_size(file_info.get('file_size', 0))}"
                ):
                    selected_files.append(file_info)
        
        with col2:
            st.subheader("Quick Actions")
            
            if st.button("Select All on Page"):
                for file_info in page_files:
                    st.session_state[f"file_{file_info['id']}"] = True
                st.rerun()
            
            if st.button("Deselect All"):
                for file_info in page_files:
                    st.session_state[f"file_{file_info['id']}"] = False
                st.rerun()
        
        # Analysis execution
        if selected_files:
            st.subheader(f"🔬 Analyze {len(selected_files)} Selected Files")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                analyze_metadata = st.checkbox("Extract Metadata", value=True)
            
            with col2:
                analyze_manipulation = st.checkbox("Check for Manipulation", value=True)
            
            with col3:
                calculate_hashes = st.checkbox("Calculate Hashes", value=True)
            
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Performing forensic analysis..."):
                    try:
                        analysis_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file_info in enumerate(selected_files):
                            status_text.text(f"Analyzing {file_info['filename']}...")
                            
                            # In a real implementation, load file data from disk
                            # For now, simulate analysis
                            file_data = b''  # Would load actual file data here
                            
                            if analyze_metadata:
                                # Extract metadata
                                metadata = st.session_state.forensics_toolkit.extract_comprehensive_metadata(
                                    file_data, file_info['filename']
                                )
                                
                                # Store in database
                                forensics_query = """
                                    INSERT INTO forensics_results 
                                    (file_id, metadata_extracted, file_analysis, created_at)
                                    VALUES (%s, %s, %s, %s)
                                """
                                execute_query(forensics_query, (
                                    file_info['id'], 
                                    json.dumps(metadata),
                                    json.dumps({'analysis_type': 'metadata_extraction'}),
                                    datetime.now()
                                ))
                                
                                analysis_results.append({
                                    'file_id': file_info['id'],
                                    'filename': file_info['filename'],
                                    'analysis_type': 'metadata',
                                    'results': metadata
                                })
                            
                            if analyze_manipulation:
                                # Check for manipulation
                                manipulation_result = st.session_state.forensics_toolkit.detect_file_manipulation(
                                    file_data, file_info['filename']
                                )
                                
                                st.session_state.manipulation_results.append(manipulation_result)
                                
                                analysis_results.append({
                                    'file_id': file_info['id'],
                                    'filename': file_info['filename'],
                                    'analysis_type': 'manipulation_detection',
                                    'results': manipulation_result
                                })
                            
                            progress_bar.progress((i + 1) / len(selected_files))
                        
                        st.session_state.forensics_results.extend(analysis_results)
                        status_text.text("Analysis completed!")
                        st.success(f"Successfully analyzed {len(selected_files)} files!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
        # Display recent analysis results
        if st.session_state.forensics_results:
            st.subheader("📋 Recent Analysis Results")
            
            for result in st.session_state.forensics_results[-5:]:  # Show last 5 results
                with st.expander(f"📄 {result['filename']} - {result['analysis_type'].title()}"):
                    
                    if result['analysis_type'] == 'metadata':
                        metadata = result['results']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**File Type:** {metadata.get('file_type', 'Unknown')}")
                            st.write(f"**File Size:** {format_file_size(metadata.get('file_size', 0))}")
                            
                            # Hash values
                            if 'hashes' in metadata:
                                st.write("**File Hashes:**")
                                for alg, hash_val in metadata['hashes'].items():
                                    st.code(f"{alg.upper()}: {hash_val}")
                        
                        with col2:
                            # Image metadata
                            if 'image_metadata' in metadata:
                                img_meta = metadata['image_metadata']
                                st.write("**Image Properties:**")
                                
                                if 'size' in img_meta:
                                    st.write(f"• Dimensions: {img_meta['size'][0]}x{img_meta['size'][1]}")
                                
                                if 'format' in img_meta:
                                    st.write(f"• Format: {img_meta['format']}")
                                
                                # GPS coordinates
                                if 'gps_coordinates' in img_meta:
                                    gps = img_meta['gps_coordinates']
                                    st.write(f"• GPS: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
                        
                        # EXIF data
                        if 'image_metadata' in metadata and 'exif' in metadata['image_metadata']:
                            with st.expander("📷 EXIF Data"):
                                exif_data = metadata['image_metadata']['exif']
                                for key, value in list(exif_data.items())[:10]:  # Show first 10
                                    st.write(f"**{key}:** {value}")
                    
                    elif result['analysis_type'] == 'manipulation_detection':
                        manipulation = result['results']
                        
                        # Risk level indicator
                        risk_level = manipulation.get('overall_risk_level', 'Unknown')
                        risk_colors = {
                            'High': '🔴',
                            'Medium': '🟡',
                            'Low': '🟢',
                            'Minimal': '⚪'
                        }
                        
                        st.write(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
                        
                        # Manipulation indicators
                        indicators = manipulation.get('manipulation_indicators', [])
                        if indicators:
                            st.write("**Potential Issues Found:**")
                            for indicator in indicators:
                                severity = indicator.get('severity', 'Unknown')
                                description = indicator.get('description', 'No description')
                                st.write(f"• {severity}: {description}")
                        else:
                            st.write("✅ No manipulation indicators found")

with tab2:
    st.header("Timeline Analysis")
    
    if st.session_state.forensics_results:
        if st.button("🕒 Generate Timeline Analysis"):
            with st.spinner("Analyzing file timeline..."):
                # Extract metadata for timeline analysis
                metadata_results = [r for r in st.session_state.forensics_results 
                                 if r['analysis_type'] == 'metadata']
                
                if metadata_results:
                    timeline_analysis = st.session_state.forensics_toolkit.analyze_file_timeline(
                        [r['results'] for r in metadata_results]
                    )
                    
                    st.session_state.timeline_analysis = timeline_analysis
                    st.success("Timeline analysis completed!")
                else:
                    st.warning("No metadata available for timeline analysis")
        
        # Display timeline if available
        if st.session_state.timeline_analysis:
            timeline = st.session_state.timeline_analysis
            
            # Timeline summary
            st.subheader("📊 Timeline Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Events", timeline.get('total_events', 0))
            
            with col2:
                date_range = timeline.get('date_range', {})
                if date_range.get('earliest'):
                    st.write(f"**Earliest:** {date_range['earliest']}")
            
            with col3:
                if date_range.get('latest'):
                    st.write(f"**Latest:** {date_range['latest']}")
            
            # Timeline events
            events = timeline.get('timeline', [])
            if events:
                st.subheader("🕒 Timeline Events")
                
                # Create timeline visualization
                df_events = pd.DataFrame(events)
                
                if len(df_events) > 0:
                    # Convert timestamps to datetime
                    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], errors='coerce')
                    df_events = df_events.dropna(subset=['timestamp'])
                    
                    if len(df_events) > 0:
                        # Timeline chart
                        fig = px.scatter(
                            df_events,
                            x='timestamp',
                            y='event_type',
                            color='source',
                            hover_data=['filename'],
                            title="File Timeline Analysis"
                        )
                        
                        fig.update_layout(
                            xaxis_title="Date/Time",
                            yaxis_title="Event Type",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Event details
                st.subheader("📋 Event Details")
                
                for event in events[:20]:  # Show first 20 events
                    with st.expander(f"{event['event_type']} - {event['filename']}"):
                        st.write(f"**Timestamp:** {event['timestamp']}")
                        st.write(f"**Source:** {event['source']}")
                        st.write(f"**Filename:** {event['filename']}")
    
    else:
        st.info("No forensics results available. Please run file analysis first.")

with tab3:
    st.header("File Integrity & Hash Verification")
    
    st.subheader("🔐 Hash Database")
    
    # Display hash information for analyzed files
    if st.session_state.forensics_results:
        hash_data = []
        
        for result in st.session_state.forensics_results:
            if result['analysis_type'] == 'metadata':
                metadata = result['results']
                hashes = metadata.get('hashes', {})
                
                for algorithm, hash_value in hashes.items():
                    hash_data.append({
                        'filename': result['filename'],
                        'algorithm': algorithm.upper(),
                        'hash_value': hash_value,
                        'file_size': metadata.get('file_size', 0)
                    })
        
        if hash_data:
            df_hashes = pd.DataFrame(hash_data)
            
            # Hash comparison tools
            st.subheader("🔍 Hash Verification Tools")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Search for Hash:**")
                search_hash = st.text_input("Enter hash value", placeholder="SHA256, MD5, etc.")
                
                if search_hash and st.button("Search"):
                    matches = df_hashes[df_hashes['hash_value'].str.contains(search_hash, case=False, na=False)]
                    
                    if not matches.empty:
                        st.success(f"Found {len(matches)} matches!")
                        st.dataframe(matches, use_container_width=True)
                    else:
                        st.warning("No matches found")
            
            with col2:
                st.write("**Duplicate Detection:**")
                
                if st.button("Find Duplicates"):
                    # Group by hash values to find duplicates
                    duplicates_found = False
                    
                    for algorithm in ['sha256', 'md5', 'sha1']:
                        alg_hashes = df_hashes[df_hashes['algorithm'] == algorithm.upper()]
                        
                        if not alg_hashes.empty:
                            duplicate_groups = alg_hashes.groupby('hash_value').filter(lambda x: len(x) > 1)
                            
                            if not duplicate_groups.empty:
                                st.write(f"**{algorithm.upper()} Duplicates:**")
                                
                                for hash_val, group in duplicate_groups.groupby('hash_value'):
                                    st.write(f"Hash: `{hash_val[:16]}...`")
                                    for _, row in group.iterrows():
                                        st.write(f"  • {row['filename']}")
                                
                                duplicates_found = True
                    
                    if not duplicates_found:
                        st.success("No duplicates found!")
            
            # Hash database table
            st.subheader("📊 Hash Database")
            st.dataframe(df_hashes, use_container_width=True)
        
        else:
            st.info("No hash data available. Run file analysis with hash calculation enabled.")
    
    else:
        st.info("No forensics results available.")
    
    # File integrity verification
    st.subheader("✅ Integrity Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Verify Against Known Hashes:**")
        known_hashes_text = st.text_area(
            "Enter known hash values (one per line)",
            placeholder="sha256:abc123...\nmd5:def456...",
            height=100
        )
        
        if known_hashes_text and st.button("Verify Integrity"):
            known_hashes = {}
            for line in known_hashes_text.strip().split('\n'):
                if ':' in line:
                    alg, hash_val = line.split(':', 1)
                    known_hashes[alg.lower()] = hash_val.strip()
            
            if known_hashes:
                st.write("**Verification Results:**")
                # Compare with stored hashes
                verification_results = []
                
                for result in st.session_state.forensics_results:
                    if result['analysis_type'] == 'metadata':
                        metadata = result['results']
                        file_hashes = metadata.get('hashes', {})
                        
                        for alg, expected_hash in known_hashes.items():
                            if alg in file_hashes:
                                actual_hash = file_hashes[alg]
                                matches = actual_hash == expected_hash
                                
                                verification_results.append({
                                    'filename': result['filename'],
                                    'algorithm': alg.upper(),
                                    'status': '✅ Match' if matches else '❌ Mismatch',
                                    'expected': expected_hash[:16] + '...',
                                    'actual': actual_hash[:16] + '...'
                                })
                
                if verification_results:
                    df_verification = pd.DataFrame(verification_results)
                    st.dataframe(df_verification, use_container_width=True)
    
    with col2:
        st.write("**Chain of Custody Tracking:**")
        
        # Simple chain of custody log
        if 'custody_log' not in st.session_state:
            st.session_state.custody_log = []
        
        custody_action = st.selectbox("Action", ["Access", "Modify", "Transfer", "Archive"])
        custody_person = st.text_input("Person", placeholder="Investigator name")
        custody_notes = st.text_area("Notes", placeholder="Action details", height=60)
        
        if st.button("Add to Custody Log") and custody_person:
            st.session_state.custody_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': custody_action,
                'person': custody_person,
                'notes': custody_notes
            })
            st.success("Added to custody log")
        
        # Display custody log
        if st.session_state.custody_log:
            st.write("**Recent Custody Events:**")
            for i, entry in enumerate(st.session_state.custody_log[-5:]):
                st.write(f"{i+1}. {entry['action']} by {entry['person']} "
                        f"({format_timestamp(entry['timestamp'])})")

with tab4:
    st.header("Comprehensive Forensics Report")
    
    if st.session_state.forensics_results or st.session_state.manipulation_results:
        if st.button("📊 Generate Forensics Report", type="primary"):
            with st.spinner("Generating comprehensive forensics report..."):
                # Generate report
                report = st.session_state.forensics_toolkit.generate_forensics_report(
                    [r['results'] for r in st.session_state.forensics_results if r['analysis_type'] == 'metadata'],
                    st.session_state.timeline_analysis,
                    st.session_state.manipulation_results
                )
                
                st.session_state.forensics_report = report
                st.success("Forensics report generated!")
        
        # Display report if available
        if hasattr(st.session_state, 'forensics_report'):
            report = st.session_state.forensics_report
            
            # Report header
            st.subheader("📋 Executive Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            summary = report.get('summary', {})
            
            with col1:
                st.metric("Files Analyzed", summary.get('total_files_analyzed', 0))
            
            with col2:
                st.metric("Files with Metadata", summary.get('files_with_metadata', 0))
            
            with col3:
                st.metric("Files with GPS", summary.get('files_with_gps', 0))
            
            with col4:
                st.metric("Manipulation Flags", summary.get('files_with_manipulation_indicators', 0))
            
            # Key findings
            if report.get('recommendations'):
                st.subheader("🎯 Key Recommendations")
                for recommendation in report['recommendations']:
                    st.write(f"• {recommendation}")
            
            # Detailed analysis sections
            st.subheader("📊 Detailed Analysis")
            
            # Timeline analysis
            if 'timeline_analysis' in report['detailed_analysis']:
                with st.expander("🕒 Timeline Analysis"):
                    timeline = report['detailed_analysis']['timeline_analysis']
                    st.write(f"**Total Events:** {timeline.get('total_events', 0)}")
                    
                    if timeline.get('date_range'):
                        date_range = timeline['date_range']
                        st.write(f"**Date Range:** {date_range.get('earliest')} to {date_range.get('latest')}")
            
            # Manipulation analysis
            if 'manipulation_analysis' in report['detailed_analysis']:
                with st.expander("🛡️ Manipulation Analysis"):
                    manipulation_results = report['detailed_analysis']['manipulation_analysis']
                    
                    high_risk_files = [r for r in manipulation_results 
                                     if r.get('risk_level') == 'High']
                    
                    if high_risk_files:
                        st.warning(f"⚠️ {len(high_risk_files)} files flagged as high risk")
                        
                        for file_result in high_risk_files:
                            st.write(f"**{file_result['filename']}**")
                            indicators = file_result.get('manipulation_indicators', [])
                            for indicator in indicators:
                                st.write(f"  • {indicator.get('type', 'Unknown')}: {indicator.get('description', 'No description')}")
                    else:
                        st.success("✅ No high-risk files identified")
            
            # Export report
            st.subheader("📤 Export Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download JSON Report"):
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="Download Report",
                        data=report_json,
                        file_name=f"forensics_report_{report['report_id']}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📊 Download Summary CSV"):
                    # Create summary CSV
                    summary_data = []
                    for result in st.session_state.forensics_results:
                        if result['analysis_type'] == 'metadata':
                            metadata = result['results']
                            summary_data.append({
                                'filename': result['filename'],
                                'file_type': metadata.get('file_type', 'Unknown'),
                                'file_size': metadata.get('file_size', 0),
                                'sha256': metadata.get('hashes', {}).get('sha256', ''),
                                'has_gps': 'gps_coordinates' in str(metadata),
                                'has_exif': 'exif' in str(metadata)
                            })
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        csv = df_summary.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"forensics_summary_{report['report_id']}.csv",
                            mime="text/csv"
                        )
    
    else:
        st.info("No forensics analysis results available. Please run file analysis first.")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### Digital Forensics Toolkit Help
    
    **Key Features:**
    - **Metadata Extraction:** Extract EXIF, GPS, and technical metadata from files
    - **Hash Calculation:** Generate MD5, SHA1, SHA256, and SHA512 hashes
    - **Manipulation Detection:** Identify potential file tampering or editing
    - **Timeline Analysis:** Reconstruct chronological sequence of file events
    - **Integrity Verification:** Verify file integrity using hash comparison
    
    **Supported File Types:**
    - **Images:** JPG, PNG, TIFF (EXIF data, GPS coordinates)
    - **Videos:** MP4, AVI, MOV (technical metadata, frame analysis)
    - **Documents:** PDF (metadata, creation info)
    - **Archives:** ZIP, 7Z, RAR (content analysis)
    
    **Analysis Types:**
    - **Comprehensive Metadata:** Full technical and descriptive metadata
    - **Manipulation Detection:** Digital artifact analysis and tampering indicators
    - **Timeline Reconstruction:** Chronological analysis of file events
    - **Hash Verification:** File integrity and duplicate detection
    
    **Best Practices:**
    - Always calculate multiple hash types for verification
    - Document chain of custody for legal proceedings
    - Review manipulation indicators carefully - false positives are possible
    - Use timeline analysis to establish sequence of events
    
    **Technical Notes:**
    - Hash algorithms provide different levels of security and speed
    - EXIF data can contain sensitive location and device information
    - Manipulation detection uses multiple techniques but requires expert interpretation
    - Timeline analysis relies on embedded timestamps which can be modified
    """)
