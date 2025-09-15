import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import io
import base64

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """Safely load JSON string with fallback"""
    try:
        if json_string and isinstance(json_string, str):
            return json.loads(json_string)
        return default
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely dump data to JSON string"""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    try:
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    except:
        return "Unknown"

def format_timestamp(timestamp: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp to string"""
    try:
        if isinstance(timestamp, str):
            # Try to parse common timestamp formats
            for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    timestamp = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                return str(timestamp)
        
        if isinstance(timestamp, datetime):
            return timestamp.strftime(format_str)
        
        return str(timestamp)
    except:
        return "Unknown"

def calculate_time_ago(timestamp: Union[str, datetime]) -> str:
    """Calculate time ago string from timestamp"""
    try:
        if isinstance(timestamp, str):
            for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    timestamp = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                return "Unknown"
        
        if not isinstance(timestamp, datetime):
            return "Unknown"
        
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown"

def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate GPS coordinates"""
    try:
        return -90 <= lat <= 90 and -180 <= lng <= 180
    except:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system storage"""
    try:
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
        
        return filename
    except:
        return "file"

def create_download_link(data: Union[str, bytes], filename: str, mime_type: str = "application/octet-stream") -> str:
    """Create download link for data"""
    try:
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        b64_data = base64.b64encode(data).decode()
        href = f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">Download {filename}</a>'
        return href
    except:
        return ""

def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe"""
    try:
        filtered_df = df.copy()
        
        for column, filter_value in filters.items():
            if column not in df.columns:
                continue
            
            if isinstance(filter_value, dict):
                # Range filter
                if 'min' in filter_value and filter_value['min'] is not None:
                    filtered_df = filtered_df[filtered_df[column] >= filter_value['min']]
                if 'max' in filter_value and filter_value['max'] is not None:
                    filtered_df = filtered_df[filtered_df[column] <= filter_value['max']]
            
            elif isinstance(filter_value, list):
                # Multiple selection filter
                if filter_value:
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            
            elif isinstance(filter_value, str):
                # Text search filter
                if filter_value:
                    filtered_df = filtered_df[filtered_df[column].str.contains(filter_value, case=False, na=False)]
            
            elif filter_value is not None:
                # Exact match filter
                filtered_df = filtered_df[filtered_df[column] == filter_value]
        
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering dataframe: {str(e)}")
        return df

def export_dataframe(df: pd.DataFrame, format_type: str = "csv", filename: str = "data") -> bytes:
    """Export dataframe to various formats"""
    try:
        if format_type.lower() == "csv":
            return df.to_csv(index=False).encode('utf-8')
        
        elif format_type.lower() == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        
        elif format_type.lower() == "json":
            return df.to_json(orient='records', indent=2).encode('utf-8')
        
        else:
            return df.to_csv(index=False).encode('utf-8')
    
    except Exception as e:
        st.error(f"Error exporting dataframe: {str(e)}")
        return b""

def parse_uploaded_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse uploaded data file into DataFrame"""
    try:
        if uploaded_file is None:
            return None
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                st.error("Invalid JSON format")
                return None
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing uploaded data: {str(e)}")
        return None

def calculate_statistics(data: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate basic statistics for numerical data"""
    try:
        if isinstance(data, pd.Series):
            data = data.dropna()
        elif isinstance(data, list):
            data = [x for x in data if x is not None and not pd.isna(x)]
        
        if len(data) == 0:
            return {}
        
        data_array = np.array(data)
        
        stats = {
            'count': len(data_array),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75))
        }
        
        return stats
    
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return {}

def create_summary_cards(metrics: Dict[str, Union[int, float, str]], columns: int = 4):
    """Create summary metric cards using Streamlit"""
    try:
        cols = st.columns(columns)
        
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        st.metric(label, f"{value:.2f}")
                    else:
                        st.metric(label, f"{value:,}")
                else:
                    st.metric(label, str(value))
    
    except Exception as e:
        st.error(f"Error creating summary cards: {str(e)}")

def paginate_data(data: List[Any], page_size: int = 10, page_number: int = 1) -> Tuple[List[Any], Dict[str, int]]:
    """Paginate data list"""
    try:
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page_number - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        paginated_data = data[start_idx:end_idx]
        
        pagination_info = {
            'current_page': page_number,
            'total_pages': total_pages,
            'total_items': total_items,
            'items_per_page': page_size,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        
        return paginated_data, pagination_info
    
    except Exception as e:
        st.error(f"Error paginating data: {str(e)}")
        return data, {'current_page': 1, 'total_pages': 1, 'total_items': len(data)}

def search_in_data(data: List[Dict], search_term: str, search_fields: List[str] = None) -> List[Dict]:
    """Search for term in list of dictionaries"""
    try:
        if not search_term:
            return data
        
        search_term = search_term.lower()
        results = []
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            found = False
            search_in_fields = search_fields or item.keys()
            
            for field in search_in_fields:
                if field in item:
                    value = str(item[field]).lower()
                    if search_term in value:
                        found = True
                        break
            
            if found:
                results.append(item)
        
        return results
    
    except Exception as e:
        st.error(f"Error searching data: {str(e)}")
        return data

def validate_required_fields(data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate that required fields are present in data"""
    try:
        missing_fields = []
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                missing_fields.append(field)
        
        is_valid = len(missing_fields) == 0
        return is_valid, missing_fields
    
    except Exception as e:
        st.error(f"Error validating fields: {str(e)}")
        return False, required_fields

def merge_dictionaries(dict1: Dict, dict2: Dict, prefer_dict2: bool = True) -> Dict:
    """Merge two dictionaries"""
    try:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and not prefer_dict2:
                continue
            result[key] = value
        
        return result
    
    except Exception as e:
        st.error(f"Error merging dictionaries: {str(e)}")
        return dict1

def create_progress_tracker(total_steps: int, current_step: int, description: str = ""):
    """Create progress tracker UI"""
    try:
        progress_percentage = current_step / total_steps if total_steps > 0 else 0
        
        st.progress(progress_percentage)
        
        progress_text = f"Step {current_step} of {total_steps}"
        if description:
            progress_text += f": {description}"
        
        st.text(progress_text)
        
        return progress_percentage
    
    except Exception as e:
        st.error(f"Error creating progress tracker: {str(e)}")
        return 0.0
