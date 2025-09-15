import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.evidence_manager import EvidenceManager
from database import get_cases, get_evidence_files, execute_query
from utils.data_utils import format_file_size, format_timestamp, create_summary_cards, paginate_data, search_in_data
from utils.image_utils import display_image_with_info, extract_frames_from_video
import json
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Evidence Manager - Investigative Platform",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Evidence Management Dashboard")
st.markdown("Organize cases with timeline visualization and comprehensive evidence tracking")

# Initialize evidence manager
if 'evidence_manager' not in st.session_state:
    st.session_state.evidence_manager = EvidenceManager()

# Initialize session state
if 'selected_case_id' not in st.session_state:
    st.session_state.selected_case_id = None

if 'case_details' not in st.session_state:
    st.session_state.case_details = None

# Sidebar for case selection and quick actions
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0; text-align: center;">🕵️ Case Operations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Case selection
    cases = get_cases()
    
    if cases:
        case_options = {f"🔍 {case['name']} ({case['status']})": case['id'] for case in cases}
        case_options = {"➕ Select a case...": None, **case_options}
        
        selected_case_display = st.selectbox("🎯 **Active Investigation**", list(case_options.keys()))
        selected_case_id = case_options[selected_case_display]
        
        if selected_case_id != st.session_state.selected_case_id:
            st.session_state.selected_case_id = selected_case_id
            st.session_state.case_details = None
            if selected_case_id:
                # Log case selection activity
                from database import log_case_activity
                log_case_activity(selected_case_id, "case_opened", f"Case opened for investigation")
                st.rerun()
    else:
        st.info("🚀 No cases available - Create your first case below")
        selected_case_id = None
    
    # Quick case creation
    st.markdown("""
    <div style="background: rgba(0, 212, 170, 0.1); border: 1px solid #00d4aa; 
                border-radius: 10px; padding: 1rem; margin: 1rem 0;">
        <h3 style="color: #00d4aa; margin-top: 0;">⚡ Quick Actions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("➕ **Create New Investigation**", expanded=not cases):
        new_case_name = st.text_input("🏷️ Case Name", placeholder="Enter investigation case name...")
        new_case_description = st.text_area("📝 Description", height=80, 
                                           placeholder="Brief description of the investigation...")
        
        col1, col2 = st.columns(2)
        with col1:
            new_case_priority = st.selectbox("🔥 Priority Level", [1, 2, 3, 4], index=1, 
                                            format_func=lambda x: st.session_state.evidence_manager.priority_levels[x])
        with col2:
            case_type = st.selectbox("📁 Case Type", 
                                   ["Criminal Investigation", "Missing Person", "Fraud Investigation", 
                                    "Cybercrime", "Drug Investigation", "Other"])
        
        if st.button("🚀 **Create Investigation Case**", type="primary") and new_case_name:
            from database import log_case_activity
            case_id = st.session_state.evidence_manager.create_new_case(
                new_case_name, new_case_description, new_case_priority
            )
            if case_id:
                # Log case creation
                log_case_activity(case_id, "case_created", 
                                f"New {case_type} case created: {new_case_name}", 
                                metadata={"priority": new_case_priority, "case_type": case_type})
                st.success(f"✅ Investigation '{new_case_name}' created successfully!")
                st.balloons()
                st.rerun()
    
    # Current suspects list
    if selected_case_id:
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(220, 20, 60, 0.1); border: 1px solid #dc143c; 
                    border-radius: 10px; padding: 1rem;">
            <h3 style="color: #dc143c; margin-top: 0;">🎯 Current Suspects</h3>
        </div>
        """, unsafe_allow_html=True)
        
        from database import get_suspects
        suspects = get_suspects(selected_case_id)
        
        if suspects:
            for suspect in suspects[:5]:  # Show top 5 suspects
                threat_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
                threat_level = suspect.get('threat_level', 1)
                threat_icon = threat_colors.get(threat_level, "⚪")
                
                with st.expander(f"{threat_icon} **{suspect['name']}** (Threat: {threat_level})"):
                    st.write(f"**👁️ Appearances:** {suspect.get('appearance_count', 1)}")
                    st.write(f"**📊 Confidence:** {suspect.get('confidence_score', 0):.2f}")
                    st.write(f"**🕒 Last Seen:** {format_timestamp(suspect.get('last_seen'))}")
                    if suspect.get('description'):
                        st.write(f"**📝 Notes:** {suspect['description']}")
        else:
            st.info("🔍 No suspects identified yet")
            st.write("Suspects will appear here as facial recognition analysis identifies persons of interest.")
    
    # Case statistics
    if selected_case_id:
        st.subheader("Case Statistics")
        
        # Load case details if not already loaded
        if st.session_state.case_details is None:
            st.session_state.case_details = st.session_state.evidence_manager.get_case_details(selected_case_id)
        
        if st.session_state.case_details:
            stats = st.session_state.case_details.get('statistics', {})
            
            st.metric("Evidence Files", stats.get('total_evidence_files', 0))
            st.metric("Crime Incidents", stats.get('total_incidents', 0))
            st.metric("Faces Detected", stats.get('faces_detected', 0))
            st.metric("Analyses Performed", stats.get('forensics_analyses', 0) + stats.get('ai_analyses', 0))

# Main interface tabs with modern styling
st.markdown("""
<style>
    .tab-content {
        background: rgba(38, 39, 48, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Case Dashboard", "🗂️ Evidence Files", "🕒 Timeline", "👥 Suspects", "📈 Analytics", "📋 Activity Log"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if not selected_case_id:
        # Modern system overview
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #00d4aa;">🏛️ Investigation Command Center</h1>
            <p style="color: #b0b3b8; font-size: 1.2rem;">Select a case from the sidebar to begin investigation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System statistics with modern cards
        total_cases = len(cases) if cases else 0
        total_evidence = len(get_evidence_files()) if get_evidence_files() else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Total Cases", total_cases, "🗂️"),
            ("Evidence Files", total_evidence, "📁"),
            ("Active Cases", len([c for c in cases if c.get('status') == 'Active']) if cases else 0, "🔥"),
            ("Avg Evidence/Case", f"{total_evidence/max(1, total_cases):.1f}", "📊")
        ]
        
        for i, (label, value, icon) in enumerate(metrics):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0,212,170,0.1) 0%, rgba(0,184,148,0.1) 100%);
                            border: 2px solid #00d4aa; border-radius: 15px; padding: 1.5rem; text-align: center;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="color: #00d4aa; font-size: 1.8rem; font-weight: bold;">{value}</div>
                    <div style="color: #fafafa; font-size: 0.9rem; margin-top: 0.5rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent activity
        if cases:
            st.subheader("📅 Recent Cases")
            
            recent_cases = sorted(cases, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
            
            for case in recent_cases:
                with st.expander(f"📁 {case['name']} - {case.get('status', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Created:** {format_timestamp(case.get('created_at'))}")
                        st.write(f"**Status:** {case.get('status', 'Unknown')}")
                    
                    with col2:
                        if st.button(f"Open Case", key=f"open_{case['id']}"):
                            st.session_state.selected_case_id = case['id']
                            st.session_state.case_details = None
                            st.rerun()
                    
                    if case.get('description'):
                        st.write(f"**Description:** {case['description']}")
    
    else:
        # Display case details
        if st.session_state.case_details:
            case_info = st.session_state.case_details['basic_info']
            stats = st.session_state.case_details['statistics']
            
            # Case header
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"📁 {case_info['name']}")
                st.write(f"**Status:** {case_info.get('status', 'Unknown')}")
                if case_info.get('description'):
                    st.write(f"**Description:** {case_info['description']}")
            
            with col2:
                st.write(f"**Created:** {format_timestamp(case_info.get('created_at'))}")
                st.write(f"**Updated:** {format_timestamp(case_info.get('updated_at'))}")
            
            with col3:
                # Case actions
                if st.button("🔄 Refresh Case Data"):
                    st.session_state.case_details = st.session_state.evidence_manager.get_case_details(selected_case_id)
                    st.rerun()
                
                new_status = st.selectbox("Update Status", 
                                        st.session_state.evidence_manager.case_statuses,
                                        index=st.session_state.evidence_manager.case_statuses.index(case_info.get('status', 'Active')))
                
                if st.button("Update Status") and new_status != case_info.get('status'):
                    from database import USE_SQLITE
                    if USE_SQLITE:
                        query = "UPDATE cases SET status = ?, updated_at = ? WHERE id = ?"
                    else:
                        query = "UPDATE cases SET status = %s, updated_at = %s WHERE id = %s"
                    execute_query(query, (new_status, datetime.now(), selected_case_id))
                    st.success(f"Status updated to {new_status}")
                    st.rerun()
                
                # Case deletion
                st.markdown("---")
                st.subheader("⚠️ Danger Zone")
                
                if st.button("🗑️ Delete Case", type="secondary"):
                    st.session_state.show_delete_confirmation = True
                
                if st.session_state.get('show_delete_confirmation'):
                    st.warning("⚠️ **This action cannot be undone!** All evidence, analysis results, and case data will be permanently deleted.")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Confirm Delete", type="primary"):
                            from database import delete_case
                            success = delete_case(selected_case_id)
                            if success:
                                st.success("Case deleted successfully!")
                                st.session_state.selected_case_id = None
                                st.session_state.case_details = None
                                st.session_state.show_delete_confirmation = False
                                st.rerun()
                            else:
                                st.error("Failed to delete case")
                    
                    with col2:
                        if st.button("❌ Cancel"):
                            st.session_state.show_delete_confirmation = False
                            st.rerun()
            
            # AI-Powered Case Overview Dashboard
            st.subheader("🤖 AI-Powered Case Overview")
            
            # Smart case insights
            insights = []
            evidence_files = st.session_state.case_details.get('evidence_files', [])
            
            # Analyze file types and suggest actions
            video_files = [f for f in evidence_files if f.get('file_type', '').startswith('video/')]
            image_files = [f for f in evidence_files if f.get('file_type', '').startswith('image/')]
            
            if video_files:
                insights.append(f"🎥 {len(video_files)} video files detected - Consider facial recognition analysis")
            if image_files:
                insights.append(f"📷 {len(image_files)} image files detected - Consider perceptual hash analysis")
            
            # Analyze temporal patterns
            if evidence_files:
                recent_uploads = len([f for f in evidence_files 
                                   if f.get('upload_date') and 
                                   (datetime.now() - pd.to_datetime(f['upload_date'])).days <= 7])
                if recent_uploads > 0:
                    insights.append(f"📈 {recent_uploads} files uploaded in the last 7 days - Active investigation")
            
            # Analyze metadata for GPS data
            gps_files = len([f for f in evidence_files 
                           if f.get('metadata') and 'gps' in str(f.get('metadata', '')).lower()])
            if gps_files > 0:
                insights.append(f"🗺️ {gps_files} files contain location data - Consider geographic analysis")
            
            # Display insights
            if insights:
                st.markdown("**🔍 AI Insights:**")
                for insight in insights:
                    st.write(f"• {insight}")
            else:
                st.info("🤖 Upload evidence files for AI-powered insights")
            
            # Case statistics cards
            st.subheader("📊 Case Statistics")
            
            metrics = {
                "Evidence Files": stats.get('total_evidence_files', 0),
                "Crime Incidents": stats.get('total_incidents', 0),
                "Faces Detected": stats.get('faces_detected', 0),
                "Forensics Analyses": stats.get('forensics_analyses', 0),
                "AI Analyses": stats.get('ai_analyses', 0)
            }
            
            create_summary_cards(metrics)
            
            # Smart file organization
            if evidence_files:
                st.subheader("🗂️ AI File Organization")
                
                # Organize files by analysis status
                unprocessed_files = [f for f in evidence_files if not f.get('processed')]
                recent_files = sorted([f for f in evidence_files if f.get('upload_date')], 
                                    key=lambda x: x.get('upload_date'), reverse=True)[:5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if unprocessed_files:
                        st.markdown("**⏳ Pending Analysis:**")
                        for file_info in unprocessed_files[:3]:
                            st.write(f"• {file_info.get('filename')} ({file_info.get('file_type', 'Unknown')})")
                        if len(unprocessed_files) > 3:
                            st.write(f"... and {len(unprocessed_files) - 3} more files")
                    else:
                        st.success("✅ All files processed")
                
                with col2:
                    if recent_files:
                        st.markdown("**📅 Recently Added:**")
                        for file_info in recent_files:
                            upload_time = format_timestamp(file_info.get('upload_date'))
                            st.write(f"• {file_info.get('filename')} - {upload_time}")
            
            # Risk assessment
            st.subheader("⚠️ Risk Assessment")
            
            risk_level = "Low"
            risk_factors = []
            
            # Assess based on incidents
            incidents = st.session_state.case_details.get('crime_incidents', [])
            high_severity_incidents = len([i for i in incidents if i.get('severity', 1) >= 3])
            
            if high_severity_incidents > 0:
                risk_level = "High"
                risk_factors.append(f"{high_severity_incidents} high-severity incidents")
            
            # Assess based on evidence volume
            if len(evidence_files) > 20:
                risk_level = "Medium" if risk_level == "Low" else risk_level
                risk_factors.append(f"High volume of evidence ({len(evidence_files)} files)")
            
            # Assess based on suspects
            from database import get_suspects
            suspects = get_suspects(selected_case_id)
            high_threat_suspects = len([s for s in suspects if s.get('threat_level', 1) >= 3])
            
            if high_threat_suspects > 0:
                risk_level = "High"
                risk_factors.append(f"{high_threat_suspects} high-threat suspects")
            
            # Display risk assessment
            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
            st.markdown(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
            
            if risk_factors:
                st.write("**Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors identified")
            
            # Evidence type distribution
            if stats.get('evidence_by_type'):
                st.subheader("📁 Evidence Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Evidence by type chart
                    evidence_chart = st.session_state.evidence_manager.create_evidence_overview_chart(
                        st.session_state.case_details
                    )
                    st.plotly_chart(evidence_chart, use_container_width=True)
                
                with col2:
                    # Incident analysis if available
                    if stats.get('incidents_by_type'):
                        incident_chart = st.session_state.evidence_manager.create_incident_analysis_chart(
                            st.session_state.case_details
                        )
                        st.plotly_chart(incident_chart, use_container_width=True)
        
        else:
            st.error("Failed to load case details.")

with tab2:
    st.header("Evidence Files Management")
    
    if not selected_case_id:
        st.info("Please select a case to view evidence files.")
    else:
        # Get evidence files for the case
        evidence_files = get_evidence_files(selected_case_id)
        
        if not evidence_files:
            st.info("No evidence files found for this case.")
            
            # Quick add evidence option
            st.subheader("➕ Add Evidence to Case")
            st.info("Use the File Upload page to add evidence files to this case.")
        
        else:
            # Evidence files interface
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # File type filter
                file_types = list(set(f.get('file_type', 'Unknown') for f in evidence_files))
                selected_types = st.multiselect("Filter by Type", file_types, default=file_types)
            
            with col2:
                # Processing status filter
                processing_statuses = ["All", "Processed", "Unprocessed"]
                processing_filter = st.selectbox("Processing Status", processing_statuses)
            
            with col3:
                # Search
                search_term = st.text_input("Search files", placeholder="Filename, hash, etc.")
            
            # Filter evidence files
            filtered_files = evidence_files
            
            if selected_types:
                filtered_files = [f for f in filtered_files if f.get('file_type') in selected_types]
            
            if processing_filter == "Processed":
                filtered_files = [f for f in filtered_files if f.get('processed')]
            elif processing_filter == "Unprocessed":
                filtered_files = [f for f in filtered_files if not f.get('processed')]
            
            if search_term:
                filtered_files = [f for f in filtered_files 
                                if search_term.lower() in f.get('filename', '').lower() 
                                or search_term.lower() in f.get('hash_value', '').lower()]
            
            # Display results
            st.subheader(f"📁 Evidence Files ({len(filtered_files)})")
            
            # Pagination
            files_per_page = 10
            paginated_files, pagination_info = paginate_data(filtered_files, files_per_page, 1)
            
            if pagination_info['total_pages'] > 1:
                page_number = st.selectbox("Page", range(1, pagination_info['total_pages'] + 1))
                paginated_files, _ = paginate_data(filtered_files, files_per_page, page_number)
            
            # Display files
            for i, file_info in enumerate(paginated_files):
                with st.expander(f"📄 {file_info['filename']} ({format_file_size(file_info.get('file_size', 0))})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**File Type:** {file_info.get('file_type', 'Unknown')}")
                        st.write(f"**Upload Date:** {format_timestamp(file_info.get('upload_date'))}")
                        st.write(f"**Hash:** `{file_info.get('hash_value', 'Unknown')[:32]}...`")
                        
                        # Processing status
                        status_icon = "✅" if file_info.get('processed') else "⏳"
                        st.write(f"**Processing Status:** {status_icon} {'Processed' if file_info.get('processed') else 'Pending'}")
                        
                        # Metadata preview
                        if file_info.get('metadata'):
                            metadata = file_info['metadata']
                            if isinstance(metadata, str):
                                try:
                                    metadata = json.loads(metadata)
                                except:
                                    pass
                            
                            if isinstance(metadata, dict) and metadata:
                                st.write("**Metadata Preview:**")
                                for key, value in list(metadata.items())[:3]:
                                    st.write(f"• {key}: {str(value)[:50]}...")
                    
                    with col2:
                        # File preview
                        if file_info.get('file_type', '').startswith('image/'):
                            st.write("📷 Image File")
                            st.info("Preview available in detailed view")
                        elif file_info.get('file_type', '').startswith('video/'):
                            st.write("🎥 Video File")
                            st.info("Frame extraction available")
                        elif file_info.get('file_type', '').startswith('application/'):
                            st.write("📄 Document")
                        else:
                            st.write("📁 File")
                    
                    with col3:
                        # Actions
                        if st.button(f"View Details", key=f"details_{file_info['id']}"):
                            st.session_state[f"show_file_details_{file_info['id']}"] = True
                            st.rerun()
                        
                        if st.button(f"Analysis History", key=f"history_{file_info['id']}"):
                            # Show analysis history
                            from database import USE_SQLITE
                            if USE_SQLITE:
                                query = """
                                    SELECT 'Facial Recognition' as analysis_type, created_at 
                                    FROM facial_recognition WHERE file_id = ?
                                    UNION ALL
                                    SELECT 'Digital Forensics' as analysis_type, created_at 
                                    FROM forensics_results WHERE file_id = ?
                                    UNION ALL
                                    SELECT analysis_type, created_at 
                                    FROM ai_analysis WHERE file_id = ?
                                    ORDER BY created_at DESC
                                """
                            else:
                                query = """
                                    SELECT 'Facial Recognition' as analysis_type, created_at 
                                    FROM facial_recognition WHERE file_id = %s
                                    UNION ALL
                                    SELECT 'Digital Forensics' as analysis_type, created_at 
                                    FROM forensics_results WHERE file_id = %s
                                    UNION ALL
                                    SELECT analysis_type, created_at 
                                    FROM ai_analysis WHERE file_id = %s
                                    ORDER BY created_at DESC
                                """
                            analyses = execute_query(query, (file_info['id'], file_info['id'], file_info['id']), fetch=True)
                            
                            if analyses:
                                st.write("**Analysis History:**")
                                for analysis in analyses:
                                    st.write(f"• {analysis['analysis_type']} - {format_timestamp(analysis['created_at'])}")
                            else:
                                st.write("No analyses performed yet")
                        
                        # Add delete button
                        if st.button(f"🗑️ Delete", key=f"delete_{file_info['id']}", type="secondary"):
                            st.session_state[f"confirm_delete_{file_info['id']}"] = True
                            st.rerun()
                
                # Show delete confirmation if requested
                if st.session_state.get(f"confirm_delete_{file_info['id']}"):
                    st.warning(f"⚠️ Delete {file_info['filename']}? This will also delete all analysis results.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Confirm", key=f"confirm_{file_info['id']}"):
                            from database import delete_evidence_file
                            success = delete_evidence_file(file_info['id'])
                            if success:
                                st.success("File deleted successfully!")
                                st.session_state[f"confirm_delete_{file_info['id']}"] = False
                                st.rerun()
                            else:
                                st.error("Failed to delete file")
                    
                    with col2:
                        if st.button(f"❌ Cancel", key=f"cancel_{file_info['id']}"):
                            st.session_state[f"confirm_delete_{file_info['id']}"] = False
                            st.rerun()
                
                # Show detailed file view if requested
                if st.session_state.get(f"show_file_details_{file_info['id']}"):
                    st.subheader(f"📄 Detailed View: {file_info['filename']}")
                    
                    # Technical details
                    st.json(file_info)
                    
                    if st.button(f"Close Details", key=f"close_details_{file_info['id']}"):
                        st.session_state[f"show_file_details_{file_info['id']}"] = False
                        st.rerun()

with tab3:
    st.header("Case Timeline")
    
    if not selected_case_id:
        st.info("Please select a case to view timeline.")
    elif st.session_state.case_details:
        # Generate timeline visualization
        timeline_fig = st.session_state.evidence_manager.create_case_timeline(st.session_state.case_details)
        
        st.subheader("📅 Case Timeline Visualization")
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Timeline events table
        st.subheader("📋 Timeline Events")
        
        # Collect all events
        timeline_events = []
        
        # Evidence upload events
        for evidence in st.session_state.case_details.get('evidence_files', []):
            timeline_events.append({
                'date': evidence.get('upload_date'),
                'event': 'Evidence Upload',
                'description': f"Uploaded: {evidence.get('filename')}",
                'category': 'Evidence',
                'item_id': evidence.get('id')
            })
        
        # Crime incident events
        for incident in st.session_state.case_details.get('crime_incidents', []):
            timeline_events.append({
                'date': incident.get('incident_date'),
                'event': 'Crime Incident',
                'description': f"{incident.get('incident_type')} at {incident.get('address', 'Unknown location')}",
                'category': 'Incident',
                'item_id': incident.get('id')
            })
        
        # Analysis events
        for analysis in st.session_state.case_details.get('forensics_results', []):
            timeline_events.append({
                'date': analysis.get('created_at'),
                'event': 'Forensics Analysis',
                'description': f"Analysis of {analysis.get('filename')}",
                'category': 'Analysis',
                'item_id': analysis.get('id')
            })
        
        # Sort events by date
        timeline_events = sorted(timeline_events, key=lambda x: x.get('date') or '', reverse=True)
        
        # Display events
        if timeline_events:
            for event in timeline_events[:20]:  # Show recent 20 events
                with st.expander(f"{event['category']}: {event['event']} - {format_timestamp(event.get('date'))}"):
                    st.write(f"**Description:** {event['description']}")
                    st.write(f"**Category:** {event['category']}")
                    st.write(f"**Date/Time:** {format_timestamp(event.get('date'))}")
        else:
            st.info("No timeline events available for this case.")
        
        # Add new incident to case
        st.subheader("➕ Add Crime Incident")
        
        with st.expander("Add New Incident"):
            col1, col2 = st.columns(2)
            
            with col1:
                incident_type = st.selectbox("Incident Type", st.session_state.evidence_manager.incident_types)
                incident_date = st.datetime_input("Incident Date/Time", value=datetime.now())
                severity = st.selectbox("Severity", [1, 2, 3, 4], 
                                      format_func=lambda x: st.session_state.evidence_manager.priority_levels[x])
            
            with col2:
                address = st.text_input("Address/Location")
                latitude = st.number_input("Latitude", value=0.0, format="%.6f")
                longitude = st.number_input("Longitude", value=0.0, format="%.6f")
            
            description = st.text_area("Incident Description")
            
            if st.button("Add Incident") and incident_type:
                incident_data = {
                    'incident_type': incident_type,
                    'incident_date': incident_date,
                    'latitude': latitude,
                    'longitude': longitude,
                    'address': address,
                    'description': description,
                    'severity': severity
                }
                
                success = st.session_state.evidence_manager.add_incident_to_case(selected_case_id, incident_data)
                
                if success:
                    st.success("Incident added successfully!")
                    # Refresh case details
                    st.session_state.case_details = st.session_state.evidence_manager.get_case_details(selected_case_id)
                    st.rerun()

with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #dc143c;">👥 Suspects Management</h2>
        <p style="color: #b0b3b8;">Track and manage persons of interest in this investigation</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not selected_case_id:
        st.warning("⚠️ Please select a case to manage suspects")
    else:
        from database import get_suspects, add_suspect
        
        # Add new suspect
        with st.expander("➕ **Add New Suspect**"):
            col1, col2 = st.columns(2)
            
            with col1:
                suspect_name = st.text_input("🏷️ Suspect Name")
                threat_level = st.selectbox("⚠️ Threat Level", [1, 2, 3, 4], 
                                          format_func=lambda x: {1: "🟢 Low", 2: "🟡 Medium", 3: "🟠 High", 4: "🔴 Critical"}[x])
                confidence = st.slider("📊 Confidence Score", 0.0, 1.0, 0.5, 0.1)
            
            with col2:
                suspect_desc = st.text_area("📝 Description", height=100)
                suspect_status = st.selectbox("Status", ["active", "detained", "cleared", "unknown"])
            
            if st.button("🎯 **Add Suspect**", type="primary") and suspect_name:
                suspect_id = add_suspect(selected_case_id, suspect_name, suspect_desc, threat_level, 
                                       confidence_score=confidence)
                if suspect_id:
                    from database import log_case_activity
                    log_case_activity(selected_case_id, "suspect_added", 
                                    f"New suspect added: {suspect_name}", 
                                    metadata={"threat_level": threat_level, "confidence": confidence})
                    st.success(f"✅ Suspect '{suspect_name}' added to investigation")
                    st.rerun()
        
        # Display current suspects
        suspects = get_suspects(selected_case_id)
        
        if suspects:
            st.subheader(f"🎯 Current Suspects ({len(suspects)})")
            
            # Sort suspects by threat level and confidence
            suspects.sort(key=lambda x: (x.get('threat_level', 1), x.get('confidence_score', 0)), reverse=True)
            
            for suspect in suspects:
                threat_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
                threat_labels = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
                threat_level = suspect.get('threat_level', 1)
                threat_icon = threat_colors.get(threat_level, "⚪")
                threat_label = threat_labels.get(threat_level, "Unknown")
                
                with st.expander(f"{threat_icon} **{suspect['name']}** - {threat_label} Threat"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("👁️ Appearances", suspect.get('appearance_count', 1))
                        st.metric("📊 Confidence", f"{suspect.get('confidence_score', 0):.2f}")
                    
                    with col2:
                        st.write(f"**🕒 First Seen:** {format_timestamp(suspect.get('first_seen'))}")
                        st.write(f"**🕒 Last Seen:** {format_timestamp(suspect.get('last_seen'))}")
                        st.write(f"**📋 Status:** {suspect.get('status', 'unknown').title()}")
                    
                    with col3:
                        if suspect.get('photo_filename'):
                            st.write(f"**📷 Photo:** {suspect['photo_filename']}")
                        
                        # Action buttons
                        if st.button(f"🔍 Investigate {suspect['name']}", key=f"investigate_{suspect['id']}"):
                            st.info(f"Investigation view for {suspect['name']} would open here")
                    
                    if suspect.get('description'):
                        st.write(f"**📝 Description:** {suspect['description']}")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: rgba(38, 39, 48, 0.3); 
                        border-radius: 10px; border: 2px dashed #444;">
                <h3 style="color: #888;">🔍 No Suspects Identified</h3>
                <p style="color: #666;">Suspects will appear here as facial recognition and analysis identify persons of interest</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("📈 Case Analytics")
    
    if not selected_case_id:
        st.info("Please select a case to view analytics.")
    elif st.session_state.case_details:
        # Analysis overview
        stats = st.session_state.case_details.get('statistics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Analysis Summary")
            
            analysis_metrics = {
                "Facial Recognition": len(st.session_state.case_details.get('facial_recognition', [])),
                "Digital Forensics": len(st.session_state.case_details.get('forensics_results', [])),
                "AI Analysis": len(st.session_state.case_details.get('ai_analysis', []))
            }
            
            for analysis_type, count in analysis_metrics.items():
                st.metric(analysis_type, count)
        
        with col2:
            st.subheader("🎯 Key Findings")
            
            # Generate key findings based on available data
            findings = []
            
            if stats.get('faces_detected', 0) > 0:
                findings.append(f"🔍 {stats['faces_detected']} faces detected across evidence")
            
            if stats.get('total_incidents', 0) > 0:
                findings.append(f"📍 {stats['total_incidents']} crime incidents recorded")
            
            if stats.get('evidence_by_type'):
                most_common_type = max(stats['evidence_by_type'].items(), key=lambda x: x[1])
                findings.append(f"📁 Most common evidence type: {most_common_type[0]} ({most_common_type[1]} files)")
            
            evidence_files = st.session_state.case_details.get('evidence_files', [])
            gps_files = len([f for f in evidence_files if 'gps' in str(f.get('metadata', '')).lower()])
            if gps_files > 0:
                findings.append(f"📍 {gps_files} files contain GPS location data")
            
            if findings:
                for finding in findings:
                    st.write(finding)
            else:
                st.info("Perform analysis on evidence files to generate findings.")
        
        # Evidence analysis charts
        if stats.get('evidence_by_type'):
            st.subheader("📈 Evidence Analysis")
            
            # Create analysis charts
            evidence_chart = st.session_state.evidence_manager.create_evidence_overview_chart(
                st.session_state.case_details
            )
            st.plotly_chart(evidence_chart, use_container_width=True)
        
        # Incident analysis
        if stats.get('incidents_by_type'):
            incident_chart = st.session_state.evidence_manager.create_incident_analysis_chart(
                st.session_state.case_details
            )
            st.plotly_chart(incident_chart, use_container_width=True)
        
        # Activity timeline
        st.subheader("📅 Activity Over Time")
        
        # Create activity timeline chart
        evidence_files = st.session_state.case_details.get('evidence_files', [])
        if evidence_files:
            # Group evidence by upload date
            activity_data = {}
            for evidence in evidence_files:
                upload_date = evidence.get('upload_date')
                if upload_date:
                    try:
                        date_key = pd.to_datetime(upload_date).date()
                        activity_data[date_key] = activity_data.get(date_key, 0) + 1
                    except:
                        pass
            
            if activity_data:
                dates = list(activity_data.keys())
                counts = list(activity_data.values())
                
                fig = px.line(
                    x=dates,
                    y=counts,
                    title="Evidence Upload Activity",
                    labels={'x': 'Date', 'y': 'Files Uploaded'}
                )
                st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #00d4aa;">📋 Case Activity Log</h2>
        <p style="color: #b0b3b8;">Complete chronological record of all investigation activities</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not selected_case_id:
        st.warning("⚠️ Please select a case to view activity log")
    else:
        from database import get_case_activity_log
        
        # Get activity log
        activities = get_case_activity_log(selected_case_id, limit=100)
        
        if activities:
            # Activity summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📝 Total Activities", len(activities))
            
            with col2:
                today_activities = len([a for a in activities 
                                     if a.get('created_at') and 
                                     a['created_at'].date() == datetime.now().date()])
                st.metric("📅 Today's Activities", today_activities)
            
            with col3:
                activity_types = len(set(a.get('activity_type') for a in activities))
                st.metric("🔄 Activity Types", activity_types)
            
            # Activity timeline
            st.subheader("🕒 Activity Timeline")
            
            # Group activities by date
            from collections import defaultdict
            activities_by_date = defaultdict(list)
            
            for activity in activities:
                if activity.get('created_at'):
                    date_key = activity['created_at'].date()
                    activities_by_date[date_key].append(activity)
            
            # Display activities by date (most recent first)
            for date, day_activities in sorted(activities_by_date.items(), reverse=True):
                st.markdown(f"""
                <div style="background: rgba(0, 212, 170, 0.1); border-left: 4px solid #00d4aa; 
                            padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">
                    <h4 style="color: #00d4aa; margin: 0;">{date.strftime('%B %d, %Y')} ({len(day_activities)} activities)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for activity in sorted(day_activities, key=lambda x: x.get('created_at', ''), reverse=True):
                    activity_icons = {
                        'case_created': '🆕',
                        'case_opened': '📂',
                        'evidence_uploaded': '📁',
                        'analysis_completed': '🔍',
                        'suspect_added': '👤',
                        'report_generated': '📊',
                        'case_updated': '✏️',
                        'default': '📝'
                    }
                    
                    activity_type = activity.get('activity_type', 'unknown')
                    icon = activity_icons.get(activity_type, activity_icons['default'])
                    timestamp = activity.get('created_at')
                    user_name = activity.get('user_name', 'System')
                    description = activity.get('description', 'No description')
                    
                    time_str = timestamp.strftime('%H:%M:%S') if timestamp else 'Unknown time'
                    
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 1rem; background: rgba(38, 39, 48, 0.5); 
                                border-radius: 8px; border-left: 3px solid #444;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                            <strong style="color: #fafafa;">{time_str}</strong>
                            <span style="color: #888; margin-left: auto;">{user_name}</span>
                        </div>
                        <p style="color: #b0b3b8; margin: 0;">{description}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: rgba(38, 39, 48, 0.3); 
                        border-radius: 10px; border: 2px dashed #444;">
                <h3 style="color: #888;">📝 No Activity Recorded</h3>
                <p style="color: #666;">Case activities will appear here as investigation progresses</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# Rest of tabs...
with st.expander("📊 **Case Reports**"):
    st.header("Case Reports")
    
    if not selected_case_id:
        st.info("Please select a case to generate reports.")
    elif st.session_state.case_details:
        # Report generation
        st.subheader("📊 Generate Case Summary Report")
        
        if st.button("📋 Generate Report", type="primary"):
            with st.spinner("Generating comprehensive case report..."):
                report = st.session_state.evidence_manager.generate_case_summary_report(
                    st.session_state.case_details
                )
                
                st.session_state.case_report = report
                st.success("Report generated successfully!")
        
        # Display report if available
        if hasattr(st.session_state, 'case_report'):
            report = st.session_state.case_report
            
            # Report header
            st.subheader(f"📋 Case Report: {report['case_name']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Case ID:** {report['case_id']}")
                st.write(f"**Status:** {report['case_status']}")
            
            with col2:
                st.write(f"**Created:** {format_timestamp(report['created_date'])}")
                st.write(f"**Last Updated:** {format_timestamp(report['last_updated'])}")
            
            with col3:
                st.write(f"**Report Generated:** {format_timestamp(report['report_generated'])}")
            
            # Case description
            if report.get('case_description'):
                st.write(f"**Description:** {report['case_description']}")
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            
            evidence_summary = report.get('evidence_summary', {})
            incidents_summary = report.get('incidents_summary', {})
            analysis_summary = report.get('analysis_summary', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Evidence Files:**")
                st.metric("Total Files", evidence_summary.get('total_files', 0))
                st.metric("Processed Files", evidence_summary.get('processed_files', 0))
                
                if evidence_summary.get('file_types'):
                    st.write("**File Types:**")
                    for file_type, count in evidence_summary['file_types'].items():
                        st.write(f"• {file_type}: {count}")
            
            with col2:
                st.write("**Crime Incidents:**")
                st.metric("Total Incidents", incidents_summary.get('total_incidents', 0))
                
                if incidents_summary.get('incident_types'):
                    st.write("**Incident Types:**")
                    for incident_type, count in incidents_summary['incident_types'].items():
                        st.write(f"• {incident_type}: {count}")
                
                if incidents_summary.get('severity_distribution'):
                    st.write("**Severity Distribution:**")
                    for severity, count in incidents_summary['severity_distribution'].items():
                        st.write(f"• {severity}: {count}")
            
            with col3:
                st.write("**Analysis Results:**")
                st.metric("Faces Detected", analysis_summary.get('total_faces_detected', 0))
                st.metric("Forensics Analyses", analysis_summary.get('forensics_analyses', 0))
                st.metric("AI Analyses", analysis_summary.get('ai_analyses', 0))
                
                if analysis_summary.get('facial_recognition_performed'):
                    st.write("✅ Facial recognition performed")
                if analysis_summary.get('forensics_analyses', 0) > 0:
                    st.write("✅ Digital forensics analysis performed")
            
            # Key findings
            if report.get('key_findings'):
                st.subheader("🎯 Key Findings")
                for finding in report['key_findings']:
                    st.write(f"• {finding}")
            
            # Export options
            st.subheader("📤 Export Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON export
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=report_json,
                    file_name=f"case_report_{report['case_id']}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export of evidence
                if st.session_state.case_details.get('evidence_files'):
                    csv_data = st.session_state.evidence_manager.export_case_data(
                        selected_case_id, 'csv'
                    )
                    if csv_data:
                        st.download_button(
                            label="📊 Download CSV",
                            data=csv_data,
                            file_name=f"case_evidence_{report['case_id']}.csv",
                            mime="text/csv"
                        )
            
            with col3:
                # Text summary
                summary_text = f"""
Case Report: {report['case_name']}
Case ID: {report['case_id']}
Status: {report['case_status']}
Generated: {format_timestamp(report['report_generated'])}

Evidence Summary:
- Total Files: {evidence_summary.get('total_files', 0)}
- Processed Files: {evidence_summary.get('processed_files', 0)}

Incidents Summary:
- Total Incidents: {incidents_summary.get('total_incidents', 0)}

Analysis Summary:
- Faces Detected: {analysis_summary.get('total_faces_detected', 0)}
- Forensics Analyses: {analysis_summary.get('forensics_analyses', 0)}
- AI Analyses: {analysis_summary.get('ai_analyses', 0)}

Key Findings:
{chr(10).join(f"- {finding}" for finding in report.get('key_findings', []))}
                """
                
                st.download_button(
                    label="📝 Download Summary",
                    data=summary_text,
                    file_name=f"case_summary_{report['case_id']}.txt",
                    mime="text/plain"
                )
        
        # Search across cases
        st.subheader("🔍 Cross-Case Search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("Search term", placeholder="Enter keywords to search across all cases")
            search_type = st.selectbox("Search in", ["all", "cases", "evidence", "incidents"])
        
        with col2:
            if st.button("🔍 Search") and search_term:
                with st.spinner("Searching across cases..."):
                    search_results = st.session_state.evidence_manager.search_across_cases(
                        search_term, search_type
                    )
                    
                    if search_results:
                        st.subheader(f"🔍 Search Results ({len(search_results)})")
                        
                        for result in search_results:
                            with st.expander(f"{result['type']}: {result['title']}"):
                                st.write(f"**Case:** {result.get('case_name', 'Unknown')}")
                                st.write(f"**Relevance:** {result['relevance']}")
                                st.write(f"**Date:** {format_timestamp(result.get('date'))}")
                                
                                if result.get('description'):
                                    st.write(f"**Description:** {result['description']}")
                                
                                if st.button(f"Open Case", key=f"search_open_{result.get('case_id')}"):
                                    st.session_state.selected_case_id = result['case_id']
                                    st.session_state.case_details = None
                                    st.rerun()
                    else:
                        st.info("No results found.")

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### Evidence Manager Help
    
    **Key Features:**
    - **Case Management:** Create, organize, and track investigation cases
    - **Evidence Organization:** Manage files, documents, and digital evidence
    - **Timeline Visualization:** View chronological sequence of events and evidence
    - **Analytics Dashboard:** Analyze patterns and generate insights
    - **Comprehensive Reporting:** Generate detailed case reports
    - **Cross-Case Search:** Search across all cases for connections
    
    **Case Overview:**
    - View case statistics and evidence distribution
    - Track processing status of evidence files
    - Monitor analysis progress and results
    
    **Evidence Management:**
    - Filter and search evidence files
    - View file details and metadata
    - Track analysis history for each file
    - Add crime incidents to cases
    
    **Timeline Analysis:**
    - Visualize case timeline with all events
    - Add new incidents and evidence
    - Track chronological sequence of events
    
    **Analytics:**
    - Evidence type distribution analysis
    - Crime incident pattern analysis
    - Activity timeline tracking
    - Key findings identification
    
    **Reporting:**
    - Generate comprehensive case reports
    - Export data in multiple formats (JSON, CSV, TXT)
    - Search across multiple cases
    - Track case progress and status
    
    **Tips:**
    - Use consistent naming conventions for better organization
    - Regularly update case status and add relevant incidents
    - Review timeline for sequence reconstruction
    - Export reports for external sharing and documentation
    """)
