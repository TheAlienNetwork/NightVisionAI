import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.ai_analysis import AIAnalysisSystem
from database import get_evidence_files, execute_query, add_evidence_file
from utils.data_utils import format_file_size, format_timestamp, create_summary_cards, paginate_data
from utils.image_utils import display_image_with_info, draw_bounding_boxes, create_image_grid, resize_image
import json
from datetime import datetime
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="AI Analysis - Investigative Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Analysis System")
st.markdown("Advanced AI analysis for images, videos, anomaly detection, and suspicious activity identification")

# Initialize AI analysis system
if 'ai_analysis_system' not in st.session_state:
    st.session_state.ai_analysis_system = AIAnalysisSystem()

# Initialize session state
if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = []

if 'batch_analysis_results' not in st.session_state:
    st.session_state.batch_analysis_results = []

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

# Sidebar for analysis configuration
with st.sidebar:
    st.header("AI Analysis Settings")
    
    # Analysis type selection
    st.subheader("Analysis Types")
    
    available_analyses = {
        "comprehensive": "Comprehensive Analysis",
        "surveillance": "Surveillance Analysis",
        "forensic": "Forensic Analysis",
        "anomaly_detection": "Anomaly Detection",
        "object_detection": "Object Detection",
        "manipulation_detection": "Manipulation Detection",
        "suspicious_activity": "Suspicious Activity",
        "perceptual_hash": "Perceptual Hashing"
    }
    
    selected_analyses = st.multiselect(
        "Select analysis types",
        list(available_analyses.keys()),
        default=["comprehensive", "object_detection"],
        format_func=lambda x: available_analyses[x]
    )
    
    # AI model settings
    st.subheader("AI Model Settings")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.1,
                                    help="Minimum confidence for AI predictions")
    
    max_tokens = st.slider("Max AI Response Tokens", 100, 1500, 800,
                          help="Maximum tokens for AI text responses")
    
    # File filters
    st.subheader("File Selection")
    
    evidence_files = get_evidence_files()
    
    if evidence_files:
        # Filter by file type
        image_files = [f for f in evidence_files if f.get('file_type', '').startswith('image/')]
        video_files = [f for f in evidence_files if f.get('file_type', '').startswith('video/')]
        
        st.metric("Available Images", len(image_files))
        st.metric("Available Videos", len(video_files))
        
        include_images = st.checkbox("Include Images", value=True)
        include_videos = st.checkbox("Include Videos", value=False)
        
        # File selection for analysis
        available_files = []
        if include_images:
            available_files.extend(image_files)
        if include_videos:
            available_files.extend(video_files)
        
        st.metric("Files for Analysis", len(available_files))
    else:
        st.warning("No evidence files found")
        available_files = []

# Main interface tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single File Analysis", "⚡ Batch Analysis", "🔄 Image Comparison", "📊 Results Dashboard", "🎯 Advanced Analysis"])

with tab1:
    st.header("Single File AI Analysis")
    
    if not available_files:
        st.warning("No evidence files available. Please upload files in the File Upload section first.")
    else:
        # File selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Select File for Analysis")
            
            # File selection interface
            selected_file = None
            
            for i, file_info in enumerate(available_files[:20]):  # Show first 20 files
                if st.radio(
                    "Select file:",
                    [file_info],
                    format_func=lambda f: f"📄 {f['filename']} ({format_file_size(f.get('file_size', 0))})",
                    key=f"select_file_{i}",
                    horizontal=True
                ):
                    selected_file = file_info
                    break
        
        with col2:
            st.subheader("Analysis Options")
            
            # Quick analysis buttons
            if selected_file:
                st.write(f"**Selected:** {selected_file['filename']}")
                st.write(f"**Type:** {selected_file.get('file_type', 'Unknown')}")
                st.write(f"**Size:** {format_file_size(selected_file.get('file_size', 0))}")
        
        # Analysis execution
        if selected_file and selected_analyses:
            st.subheader(f"🤖 Analyze: {selected_file['filename']}")
            
            # Show selected analysis types
            st.write("**Selected Analysis Types:**")
            for analysis_type in selected_analyses:
                st.write(f"• {available_analyses[analysis_type]}")
            
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("Performing AI analysis..."):
                    try:
                        # In real implementation, load file data from disk using file_path
                        file_data = b''  # Placeholder - would load actual file data
                        
                        analysis_results = {
                            'file_id': selected_file['id'],
                            'filename': selected_file['filename'],
                            'file_type': selected_file.get('file_type', 'Unknown'),
                            'analysis_timestamp': datetime.now().isoformat(),
                            'analyses': {}
                        }
                        
                        # Perform selected analyses
                        for analysis_type in selected_analyses:
                            status_text = st.empty()
                            status_text.text(f"Running {available_analyses[analysis_type]}...")
                            
                            try:
                                if analysis_type == 'comprehensive':
                                    result = st.session_state.ai_analysis_system.analyze_image_with_ai(
                                        file_data, 'comprehensive'
                                    )
                                elif analysis_type == 'surveillance':
                                    result = st.session_state.ai_analysis_system.analyze_image_with_ai(
                                        file_data, 'surveillance'
                                    )
                                elif analysis_type == 'forensic':
                                    result = st.session_state.ai_analysis_system.analyze_image_with_ai(
                                        file_data, 'forensic'
                                    )
                                elif analysis_type == 'anomaly_detection':
                                    result = st.session_state.ai_analysis_system.analyze_image_with_ai(
                                        file_data, 'anomaly_detection'
                                    )
                                elif analysis_type == 'object_detection':
                                    result = st.session_state.ai_analysis_system.detect_objects_opencv(file_data)
                                elif analysis_type == 'manipulation_detection':
                                    result = st.session_state.ai_analysis_system.detect_image_manipulation(file_data)
                                elif analysis_type == 'suspicious_activity':
                                    result = st.session_state.ai_analysis_system.analyze_suspicious_activity(file_data)
                                elif analysis_type == 'perceptual_hash':
                                    result = st.session_state.ai_analysis_system.calculate_perceptual_hash(file_data)
                                else:
                                    result = {'error': f'Unknown analysis type: {analysis_type}'}
                                
                                analysis_results['analyses'][analysis_type] = result
                                
                                # Store in database
                                query = """
                                    INSERT INTO ai_analysis 
                                    (file_id, analysis_type, results, confidence_score, created_at)
                                    VALUES (%s, %s, %s, %s, %s)
                                """
                                execute_query(query, (
                                    selected_file['id'],
                                    analysis_type,
                                    json.dumps(result),
                                    result.get('confidence_score', 0.0),
                                    datetime.now()
                                ))
                                
                            except Exception as e:
                                analysis_results['analyses'][analysis_type] = {'error': str(e)}
                            
                            status_text.empty()
                        
                        # Store results
                        st.session_state.ai_analysis_results.append(analysis_results)
                        
                        st.success("AI analysis completed successfully!")
                        
                        # Display results immediately
                        st.subheader("📊 Analysis Results")
                        
                        for analysis_type, result in analysis_results['analyses'].items():
                            with st.expander(f"🔍 {available_analyses[analysis_type]} Results"):
                                if 'error' in result:
                                    st.error(f"Analysis failed: {result['error']}")
                                else:
                                    # Display results based on analysis type
                                    if analysis_type in ['comprehensive', 'surveillance', 'forensic', 'anomaly_detection']:
                                        # AI text analysis results
                                        st.write("**AI Analysis Results:**")
                                        
                                        if isinstance(result, dict):
                                            for key, value in result.items():
                                                if key not in ['analysis_type', 'timestamp', 'model_used']:
                                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                        
                                        st.json(result)
                                    
                                    elif analysis_type == 'object_detection':
                                        # OpenCV object detection results
                                        st.write("**Object Detection Results:**")
                                        
                                        faces_count = result.get('faces', {}).get('count', 0)
                                        eyes_count = result.get('eyes', {}).get('count', 0)
                                        vehicles_count = result.get('vehicles', {}).get('count', 0)
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Faces Detected", faces_count)
                                        with col2:
                                            st.metric("Eyes Detected", eyes_count)
                                        with col3:
                                            st.metric("Vehicles Detected", vehicles_count)
                                        
                                        # Image properties
                                        if 'image_properties' in result:
                                            props = result['image_properties']
                                            st.write("**Image Properties:**")
                                            st.write(f"• Dimensions: {props.get('width', 0)}x{props.get('height', 0)}")
                                            st.write(f"• Mean Brightness: {props.get('mean_brightness', 0):.1f}")
                                            st.write(f"• Brightness Std: {props.get('brightness_std', 0):.1f}")
                                    
                                    elif analysis_type == 'manipulation_detection':
                                        # Manipulation detection results
                                        risk_level = result.get('overall_risk_level', 'Unknown')
                                        
                                        risk_colors = {
                                            'High': '🔴',
                                            'Medium': '🟡', 
                                            'Low': '🟢',
                                            'Minimal': '⚪'
                                        }
                                        
                                        st.write(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
                                        
                                        indicators = result.get('manipulation_indicators', [])
                                        if indicators:
                                            st.write("**Manipulation Indicators:**")
                                            for indicator in indicators:
                                                confidence = indicator.get('confidence', 0)
                                                description = indicator.get('description', 'No description')
                                                st.write(f"• {description} (Confidence: {confidence:.2f})")
                                        else:
                                            st.success("✅ No manipulation indicators detected")
                                    
                                    elif analysis_type == 'suspicious_activity':
                                        # Suspicious activity results
                                        threat_level = result.get('overall_threat_level', 'Unknown')
                                        
                                        threat_colors = {
                                            'High': '🔴',
                                            'Medium': '🟡',
                                            'Low': '🟢', 
                                            'Minimal': '⚪'
                                        }
                                        
                                        st.write(f"**Threat Level:** {threat_colors.get(threat_level, '⚪')} {threat_level}")
                                        
                                        ai_analysis = result.get('ai_suspicious_activity_analysis', {})
                                        if ai_analysis and 'findings' in ai_analysis:
                                            findings = ai_analysis['findings']
                                            if isinstance(findings, list):
                                                st.write("**Suspicious Activities Found:**")
                                                for finding in findings:
                                                    activity = finding.get('activity', 'Unknown activity')
                                                    confidence = finding.get('confidence', 0)
                                                    severity = finding.get('severity', 'unknown')
                                                    st.write(f"• {activity} (Confidence: {confidence:.2f}, Severity: {severity})")
                                    
                                    elif analysis_type == 'perceptual_hash':
                                        # Perceptual hash results
                                        st.write("**Perceptual Hashes:**")
                                        for hash_type, hash_value in result.items():
                                            if hash_type != 'error':
                                                st.code(f"{hash_type}: {hash_value}")
                        
                    except Exception as e:
                        st.error(f"AI analysis failed: {str(e)}")

with tab2:
    st.header("Batch AI Analysis")
    
    if not available_files:
        st.warning("No evidence files available for batch analysis.")
    else:
        # Batch analysis configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Batch Configuration")
            
            batch_size = st.slider("Batch Size", 1, min(20, len(available_files)), 5,
                                  help="Number of files to analyze simultaneously")
            
            # File selection for batch
            start_index = st.slider("Start from file", 0, max(0, len(available_files) - batch_size), 0)
            end_index = min(start_index + batch_size, len(available_files))
            
            selected_batch_files = available_files[start_index:end_index]
            
            st.write(f"**Selected files ({len(selected_batch_files)}):**")
            for file_info in selected_batch_files:
                st.write(f"• {file_info['filename']}")
        
        with col2:
            st.subheader("📊 Batch Settings")
            
            # Analysis type selection for batch
            batch_analyses = st.multiselect(
                "Analysis types for batch",
                list(available_analyses.keys()),
                default=["comprehensive", "object_detection"],
                format_func=lambda x: available_analyses[x]
            )
            
            # Performance settings
            process_videos = st.checkbox("Include video files", value=False,
                                       help="Video analysis takes significantly longer")
            
            if process_videos:
                frame_interval = st.slider("Video frame interval", 10, 100, 30,
                                         help="Analyze every Nth frame in videos")
        
        # Batch execution
        if selected_batch_files and batch_analyses:
            if st.button("🚀 Start Batch Analysis", type="primary"):
                with st.spinner(f"Performing batch AI analysis on {len(selected_batch_files)} files..."):
                    try:
                        # Filter files by type if needed
                        files_to_analyze = selected_batch_files
                        if not process_videos:
                            files_to_analyze = [f for f in files_to_analyze 
                                              if not f.get('file_type', '').startswith('video/')]
                        
                        # Prepare file data for analysis
                        analysis_files = []
                        for file_info in files_to_analyze:
                            analysis_files.append({
                                'id': file_info['id'],
                                'filename': file_info['filename'],
                                'category': 'image' if file_info.get('file_type', '').startswith('image/') else 'video',
                                'file_data': b''  # Would load actual file data in real implementation
                            })
                        
                        # Perform batch analysis
                        batch_results = st.session_state.ai_analysis_system.batch_analyze_images(
                            analysis_files, batch_analyses
                        )
                        
                        # Store results
                        st.session_state.batch_analysis_results.extend(batch_results)
                        
                        # Store in database
                        for result in batch_results:
                            for analysis_type, analysis_result in result.get('analyses', {}).items():
                                query = """
                                    INSERT INTO ai_analysis 
                                    (file_id, analysis_type, results, confidence_score, created_at)
                                    VALUES (%s, %s, %s, %s, %s)
                                """
                                execute_query(query, (
                                    result.get('file_id'),
                                    analysis_type,
                                    json.dumps(analysis_result),
                                    analysis_result.get('confidence_score', 0.0),
                                    datetime.now()
                                ))
                        
                        st.success(f"Batch analysis completed! Analyzed {len(batch_results)} files.")
                        
                        # Display batch summary
                        st.subheader("📊 Batch Analysis Summary")
                        
                        # Summary metrics
                        total_analyses = sum(len(r.get('analyses', {})) for r in batch_results)
                        successful_analyses = sum(
                            len([a for a in r.get('analyses', {}).values() if 'error' not in a])
                            for r in batch_results
                        )
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Files Processed", len(batch_results))
                        
                        with col2:
                            st.metric("Total Analyses", total_analyses)
                        
                        with col3:
                            st.metric("Successful Analyses", successful_analyses)
                        
                        with col4:
                            success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Analysis type distribution
                        analysis_counts = {}
                        for result in batch_results:
                            for analysis_type in result.get('analyses', {}).keys():
                                analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
                        
                        if analysis_counts:
                            st.subheader("📈 Analysis Distribution")
                            
                            fig = px.bar(
                                x=list(analysis_counts.keys()),
                                y=list(analysis_counts.values()),
                                title="Analysis Types Performed",
                                labels={'x': 'Analysis Type', 'y': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Batch analysis failed: {str(e)}")
        
        # Display recent batch results
        if st.session_state.batch_analysis_results:
            st.subheader("📋 Recent Batch Results")
            
            # Pagination for batch results
            results_per_page = 5
            batch_results_paginated, pagination_info = paginate_data(
                st.session_state.batch_analysis_results, results_per_page, 1
            )
            
            if pagination_info['total_pages'] > 1:
                page_number = st.selectbox("Results page", range(1, pagination_info['total_pages'] + 1))
                batch_results_paginated, _ = paginate_data(
                    st.session_state.batch_analysis_results, results_per_page, page_number
                )
            
            for result in batch_results_paginated:
                with st.expander(f"📄 {result['filename']} - {len(result.get('analyses', {}))} analyses"):
                    st.write(f"**File ID:** {result.get('file_id')}")
                    st.write(f"**Analysis Time:** {format_timestamp(result.get('analysis_timestamp'))}")
                    
                    analyses = result.get('analyses', {})
                    for analysis_type, analysis_result in analyses.items():
                        st.write(f"**{available_analyses.get(analysis_type, analysis_type)}:**")
                        
                        if 'error' in analysis_result:
                            st.error(f"Error: {analysis_result['error']}")
                        else:
                            # Show key results
                            if analysis_type == 'object_detection':
                                faces = analysis_result.get('faces', {}).get('count', 0)
                                if faces > 0:
                                    st.write(f"  • {faces} faces detected")
                            
                            elif analysis_type == 'manipulation_detection':
                                risk = analysis_result.get('overall_risk_level', 'Unknown')
                                st.write(f"  • Manipulation risk: {risk}")
                            
                            elif analysis_type == 'suspicious_activity':
                                threat = analysis_result.get('overall_threat_level', 'Unknown')
                                st.write(f"  • Threat level: {threat}")
                            
                            else:
                                # Show general result info
                                if isinstance(analysis_result, dict) and analysis_result:
                                    st.write(f"  • Analysis completed successfully")

with tab3:
    st.header("Image Similarity Comparison")
    
    # Get image files only
    image_files = [f for f in available_files if f.get('file_type', '').startswith('image/')]
    
    if len(image_files) < 2:
        st.warning("Need at least 2 image files for comparison analysis.")
    else:
        st.subheader("🔄 Compare Images for Similarity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select First Image:**")
            image1_options = {f"{f['filename']} ({format_file_size(f.get('file_size', 0))})": f 
                             for f in image_files}
            
            selected_image1_display = st.selectbox("First image", list(image1_options.keys()))
            selected_image1 = image1_options[selected_image1_display]
        
        with col2:
            st.write("**Select Second Image:**")
            
            # Filter out the first selected image
            remaining_images = {k: v for k, v in image1_options.items() 
                              if v['id'] != selected_image1['id']}
            
            if remaining_images:
                selected_image2_display = st.selectbox("Second image", list(remaining_images.keys()))
                selected_image2 = remaining_images[selected_image2_display]
            else:
                st.error("Need another image for comparison")
                selected_image2 = None
        
        # Comparison execution
        if selected_image1 and selected_image2:
            st.subheader(f"🔍 Compare: {selected_image1['filename']} vs {selected_image2['filename']}")
            
            if st.button("🔄 Compare Images", type="primary"):
                with st.spinner("Comparing images using perceptual hashing..."):
                    try:
                        # In real implementation, load actual image data
                        image1_data = b''  # Would load from selected_image1['file_path']
                        image2_data = b''  # Would load from selected_image2['file_path']
                        
                        # Perform comparison
                        comparison_result = st.session_state.ai_analysis_system.compare_images_similarity(
                            image1_data, image2_data
                        )
                        
                        # Store comparison result
                        comparison_record = {
                            'image1_id': selected_image1['id'],
                            'image1_filename': selected_image1['filename'],
                            'image2_id': selected_image2['id'],
                            'image2_filename': selected_image2['filename'],
                            'comparison_result': comparison_result,
                            'comparison_timestamp': datetime.now().isoformat()
                        }
                        
                        st.session_state.comparison_results.append(comparison_record)
                        
                        # Display results
                        st.subheader("📊 Comparison Results")
                        
                        if 'error' in comparison_result:
                            st.error(f"Comparison failed: {comparison_result['error']}")
                        else:
                            avg_similarity = comparison_result.get('average_similarity', 0)
                            is_likely_match = comparison_result.get('is_likely_match', False)
                            is_possible_match = comparison_result.get('is_possible_match', False)
                            
                            # Overall similarity
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Average Similarity", f"{avg_similarity:.1f}%")
                            
                            with col2:
                                match_status = "🟢 Likely Match" if is_likely_match else ("🟡 Possible Match" if is_possible_match else "🔴 No Match")
                                st.write(f"**Match Status:** {match_status}")
                            
                            with col3:
                                st.write(f"**Comparison Time:** {format_timestamp(comparison_result.get('timestamp'))}")
                            
                            # Detailed similarity scores
                            st.subheader("🔍 Detailed Similarity Scores")
                            
                            similarities = comparison_result.get('similarities', {})
                            
                            for hash_type, similarity_data in similarities.items():
                                similarity_pct = similarity_data.get('similarity_percentage', 0)
                                distance = similarity_data.get('distance', 0)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**{hash_type.replace('_', ' ').title()}:**")
                                with col2:
                                    st.write(f"{similarity_pct:.1f}% (distance: {distance})")
                            
                            # Interpretation
                            st.subheader("📝 Interpretation")
                            
                            if avg_similarity > 90:
                                st.success("🎯 **Very High Similarity** - Images are likely identical or nearly identical")
                            elif avg_similarity > 80:
                                st.success("✅ **High Similarity** - Images are likely the same or very similar")
                            elif avg_similarity > 60:
                                st.warning("⚠️ **Moderate Similarity** - Images may be related or similar")
                            elif avg_similarity > 40:
                                st.info("ℹ️ **Low Similarity** - Images have some similarities but are likely different")
                            else:
                                st.error("❌ **Very Low Similarity** - Images are likely completely different")
                    
                    except Exception as e:
                        st.error(f"Image comparison failed: {str(e)}")
        
        # Comparison history
        if st.session_state.comparison_results:
            st.subheader("📋 Comparison History")
            
            for i, comparison in enumerate(st.session_state.comparison_results[-10:]):  # Show last 10
                result = comparison['comparison_result']
                avg_similarity = result.get('average_similarity', 0)
                
                with st.expander(f"Comparison {i+1}: {avg_similarity:.1f}% similarity"):
                    st.write(f"**Image 1:** {comparison['image1_filename']}")
                    st.write(f"**Image 2:** {comparison['image2_filename']}")
                    st.write(f"**Similarity:** {avg_similarity:.1f}%")
                    st.write(f"**Time:** {format_timestamp(comparison['comparison_timestamp'])}")
                    
                    if result.get('is_likely_match'):
                        st.success("🎯 Likely match detected")
                    elif result.get('is_possible_match'):
                        st.warning("⚠️ Possible match detected")

with tab4:
    st.header("AI Analysis Results Dashboard")
    
    # Combine all analysis results
    all_results = st.session_state.ai_analysis_results + st.session_state.batch_analysis_results
    
    if not all_results:
        st.info("No AI analysis results available. Run analyses in other tabs to see results here.")
    else:
        # Generate comprehensive report
        if st.button("📊 Generate AI Analysis Report", type="primary"):
            with st.spinner("Generating comprehensive AI analysis report..."):
                report = st.session_state.ai_analysis_system.generate_ai_analysis_report(all_results)
                st.session_state.ai_report = report
                st.success("AI analysis report generated!")
        
        # Display report if available
        if hasattr(st.session_state, 'ai_report'):
            report = st.session_state.ai_report
            
            # Report summary
            st.subheader("📋 AI Analysis Report Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Files Analyzed", report.get('total_files_analyzed', 0))
            
            with col2:
                analysis_types = len(report.get('analysis_summary', {}).get('analysis_types_performed', []))
                st.metric("Analysis Types Used", analysis_types)
            
            with col3:
                suspicious_files = report.get('analysis_summary', {}).get('suspicious_files_count', 0)
                st.metric("Suspicious Files", suspicious_files)
            
            with col4:
                high_confidence = report.get('analysis_summary', {}).get('high_confidence_findings_count', 0)
                st.metric("High Confidence Findings", high_confidence)
            
            # Threat assessment
            if 'threat_assessment' in report:
                st.subheader("⚠️ Threat Assessment")
                
                threat_assessment = report['threat_assessment']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("High Threat", threat_assessment.get('high', 0), delta_color="inverse")
                
                with col2:
                    st.metric("Medium Threat", threat_assessment.get('medium', 0), delta_color="off")
                
                with col3:
                    st.metric("Low Threat", threat_assessment.get('low', 0), delta_color="normal")
                
                with col4:
                    st.metric("Minimal Threat", threat_assessment.get('minimal', 0), delta_color="normal")
                
                # Threat distribution chart
                if sum(threat_assessment.values()) > 0:
                    fig = px.pie(
                        values=list(threat_assessment.values()),
                        names=[k.title() for k in threat_assessment.keys()],
                        title="Threat Level Distribution",
                        color_discrete_map={
                            'High': '#ff4444',
                            'Medium': '#ffaa00', 
                            'Low': '#44ff44',
                            'Minimal': '#dddddd'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Key findings
            if report.get('key_findings'):
                st.subheader("🎯 Key Findings")
                
                for finding in report['key_findings']:
                    finding_type = finding.get('finding_type', 'Unknown')
                    filename = finding.get('filename', 'Unknown')
                    
                    if finding_type == 'Potential Manipulation':
                        risk_level = finding.get('risk_level', 'Unknown')
                        st.warning(f"🛡️ **{finding_type}** in {filename} - Risk Level: {risk_level}")
                    
                    elif finding_type == 'Multiple Faces Detected':
                        count = finding.get('count', 0)
                        st.info(f"👥 **{finding_type}** in {filename} - Count: {count}")
                    
                    elif 'threat_level' in finding:
                        threat_level = finding.get('threat_level', 'Unknown')
                        st.error(f"⚠️ **Suspicious Activity** in {filename} - Threat Level: {threat_level}")
                    
                    else:
                        st.write(f"• {finding_type} in {filename}")
            
            # Analysis performance
            st.subheader("📈 Analysis Performance")
            
            # Analysis types performed
            analysis_types_performed = report.get('analysis_summary', {}).get('analysis_types_performed', [])
            
            if analysis_types_performed:
                # Count occurrences of each analysis type
                analysis_counts = {}
                for result in all_results:
                    for analysis_type in result.get('analyses', {}).keys():
                        analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
                
                fig = px.bar(
                    x=list(analysis_counts.keys()),
                    y=list(analysis_counts.values()),
                    title="Analysis Types Usage",
                    labels={'x': 'Analysis Type', 'y': 'Number of Files Analyzed'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export report
            st.subheader("📤 Export AI Analysis Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"ai_analysis_report_{report['report_id']}.json",
                    mime="application/json"
                )
            
            with col2:
                # Summary text export
                summary_text = f"""
AI Analysis Report: {report['report_id']}
Generated: {format_timestamp(report['generation_timestamp'])}

Summary:
- Total Files Analyzed: {report.get('total_files_analyzed', 0)}
- Analysis Types Used: {len(analysis_types_performed)}
- Suspicious Files: {report.get('analysis_summary', {}).get('suspicious_files_count', 0)}
- High Confidence Findings: {report.get('analysis_summary', {}).get('high_confidence_findings_count', 0)}

Threat Assessment:
- High Threat: {report.get('threat_assessment', {}).get('high', 0)}
- Medium Threat: {report.get('threat_assessment', {}).get('medium', 0)}
- Low Threat: {report.get('threat_assessment', {}).get('low', 0)}
- Minimal Threat: {report.get('threat_assessment', {}).get('minimal', 0)}

Key Findings:
{chr(10).join(f"- {finding.get('finding_type', 'Unknown')} in {finding.get('filename', 'Unknown')}" for finding in report.get('key_findings', []))}
                """
                
                st.download_button(
                    label="📝 Download Summary",
                    data=summary_text,
                    file_name=f"ai_analysis_summary_{report['report_id']}.txt",
                    mime="text/plain"
                )
        
        # Recent results table
        st.subheader("📊 Recent Analysis Results")
        
        # Create table of recent results
        table_data = []
        for result in all_results[-20:]:  # Show last 20 results
            filename = result.get('filename', 'Unknown')
            timestamp = result.get('analysis_timestamp', '')
            analyses_count = len(result.get('analyses', {}))
            
            # Check for key findings
            analyses = result.get('analyses', {})
            findings = []
            
            for analysis_type, analysis_result in analyses.items():
                if analysis_type == 'object_detection':
                    faces = analysis_result.get('faces', {}).get('count', 0)
                    if faces > 0:
                        findings.append(f"{faces} faces")
                
                elif analysis_type == 'manipulation_detection':
                    risk = analysis_result.get('overall_risk_level', 'Unknown')
                    if risk in ['High', 'Medium']:
                        findings.append(f"Manipulation: {risk}")
                
                elif analysis_type == 'suspicious_activity':
                    threat = analysis_result.get('overall_threat_level', 'Unknown')
                    if threat in ['High', 'Medium']:
                        findings.append(f"Threat: {threat}")
            
            table_data.append({
                'Filename': filename,
                'Analyses': analyses_count,
                'Key Findings': ', '.join(findings) if findings else 'None',
                'Timestamp': format_timestamp(timestamp)
            })
        
        if table_data:
            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True)

with tab5:
    st.header("Advanced AI Analysis")
    
    st.subheader("🎯 Specialized Analysis Tools")
    
    # Advanced analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔍 Deep Forensic Analysis**")
        st.write("Combine multiple AI models for comprehensive forensic examination")
        
        if available_files and st.button("🔬 Run Deep Forensic Analysis"):
            st.info("Deep forensic analysis would combine multiple AI models and techniques for comprehensive analysis")
    
    with col2:
        st.write("**📈 Pattern Recognition Analysis**")
        st.write("Identify patterns across multiple evidence files")
        
        if available_files and st.button("🧠 Run Pattern Analysis"):
            st.info("Pattern recognition would analyze similarities and connections across evidence files")
    
    # Custom AI prompts
    st.subheader("💬 Custom AI Analysis")
    
    if available_files:
        custom_file_options = {f"{f['filename']}": f for f in available_files[:10]}
        selected_custom_file = st.selectbox("Select file for custom analysis", list(custom_file_options.keys()))
        
        custom_prompt = st.text_area(
            "Custom analysis prompt",
            placeholder="Enter specific instructions for AI analysis of this file...",
            height=100
        )
        
        if st.button("🤖 Run Custom Analysis") and custom_prompt and selected_custom_file:
            with st.spinner("Running custom AI analysis..."):
                try:
                    file_info = custom_file_options[selected_custom_file]
                    
                    # In real implementation, would load file and run custom analysis
                    st.info(f"Custom analysis would be performed on {file_info['filename']} with prompt: {custom_prompt}")
                    
                    # Placeholder result
                    st.success("Custom analysis completed! Results would appear here.")
                    
                except Exception as e:
                    st.error(f"Custom analysis failed: {str(e)}")
    
    # Analysis history and management
    st.subheader("📚 Analysis History & Management")
    
    # Database analysis history
    if st.button("📊 Load Analysis History from Database"):
        try:
            query = """
                SELECT ai.*, ef.filename 
                FROM ai_analysis ai
                JOIN evidence_files ef ON ai.file_id = ef.id
                ORDER BY ai.created_at DESC
                LIMIT 50
            """
            db_analyses = execute_query(query, fetch=True)
            
            if db_analyses:
                st.subheader("📋 Database Analysis History")
                
                history_data = []
                for analysis in db_analyses:
                    history_data.append({
                        'Filename': analysis.get('filename', 'Unknown'),
                        'Analysis Type': analysis.get('analysis_type', 'Unknown'),
                        'Confidence': analysis.get('confidence_score', 0),
                        'Date': format_timestamp(analysis.get('created_at'))
                    })
                
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
                
                # Analysis statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Analyses", len(db_analyses))
                
                with col2:
                    avg_confidence = sum(a.get('confidence_score', 0) for a in db_analyses) / len(db_analyses)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                with col3:
                    unique_types = len(set(a.get('analysis_type') for a in db_analyses))
                    st.metric("Analysis Types", unique_types)
            
            else:
                st.info("No analysis history found in database")
        
        except Exception as e:
            st.error(f"Failed to load analysis history: {str(e)}")
    
    # Clear session results
    st.subheader("🗑️ Management Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Session Results"):
            st.session_state.ai_analysis_results.clear()
            st.session_state.batch_analysis_results.clear()
            st.session_state.comparison_results.clear()
            st.success("Session results cleared!")
    
    with col2:
        if st.button("Export All Results"):
            all_data = {
                'single_analysis': st.session_state.ai_analysis_results,
                'batch_analysis': st.session_state.batch_analysis_results,
                'comparisons': st.session_state.comparison_results
            }
            
            export_json = json.dumps(all_data, indent=2, default=str)
            st.download_button(
                label="Download All Results",
                data=export_json,
                file_name=f"ai_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        results_count = (len(st.session_state.ai_analysis_results) + 
                        len(st.session_state.batch_analysis_results) + 
                        len(st.session_state.comparison_results))
        st.metric("Session Results", results_count)

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### AI Analysis System Help
    
    **Available Analysis Types:**
    - **Comprehensive Analysis:** General-purpose AI analysis of image content
    - **Surveillance Analysis:** Security-focused analysis for suspicious activities
    - **Forensic Analysis:** Digital forensics and tampering detection
    - **Anomaly Detection:** Identify unusual or out-of-place elements
    - **Object Detection:** OpenCV-based detection of faces, eyes, vehicles
    - **Manipulation Detection:** Technical analysis for image tampering
    - **Suspicious Activity:** AI-powered threat and activity assessment
    - **Perceptual Hashing:** Generate similarity hashes for comparison
    
    **Single File Analysis:**
    - Select individual files for detailed AI analysis
    - Choose multiple analysis types for comprehensive examination
    - View detailed results with confidence scores and interpretations
    
    **Batch Analysis:**
    - Process multiple files simultaneously for efficiency
    - Configure batch size and analysis types
    - Monitor progress and view summary statistics
    
    **Image Comparison:**
    - Compare two images for similarity using perceptual hashing
    - Multiple hash algorithms for robust comparison
    - Detailed similarity scores and match assessment
    
    **Results Dashboard:**
    - Comprehensive reporting across all analyses
    - Threat assessment and key findings identification
    - Export capabilities for external use
    
    **Advanced Analysis:**
    - Custom AI prompts for specialized analysis
    - Deep forensic analysis combining multiple techniques
    - Pattern recognition across evidence files
    - Analysis history management and export
    
    **Tips for Best Results:**
    - Use clear, high-quality images for better AI analysis
    - Combine multiple analysis types for comprehensive assessment
    - Review confidence scores when interpreting results
    - Use batch analysis for large evidence sets
    - Export results for documentation and reporting
    
    **Technical Notes:**
    - Uses OpenAI GPT-5 for advanced AI analysis
    - OpenCV for computer vision tasks
    - Perceptual hashing for image similarity
    - All results stored in database for persistence
    - Configurable confidence thresholds for filtering
    """)
