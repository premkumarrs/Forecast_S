"""
Layout helper functions for Streamlit pages.
"""

import streamlit as st
from typing import List, Optional, Dict


def get_country_flag_emoji(country: str) -> str:
    """
    Get flag emoji for country.
    
    Args:
        country: Country name
        
    Returns:
        Flag emoji or default globe emoji
    """
    flag_map = {
        'United States': '🇺🇸', 'USA': '🇺🇸', 'US': '🇺🇸',
        'United Kingdom': '🇬🇧', 'UK': '🇬🇧', 'Britain': '🇬🇧',
        'Germany': '🇩🇪', 'China': '🇨🇳', 'Japan': '🇯🇵', 
        'India': '🇮🇳', 'France': '🇫🇷', 'Italy': '🇮🇹',
        'Canada': '🇨🇦', 'Australia': '🇦🇺', 'Brazil': '🇧🇷',
        'Mexico': '🇲🇽', 'South Korea': '🇰🇷', 'Spain': '🇪🇸',
        'Netherlands': '🇳🇱', 'Switzerland': '🇨🇭', 'Sweden': '🇸🇪',
        'Singapore': '🇸🇬', 'South Africa': '🇿🇦', 'Russia': '🇷🇺',
        'Ukraine': '🇺🇦', 'Argentina': '🇦🇷', 'Belgium': '🇧🇪',
        'Austria': '🇦🇹', 'Poland': '🇵🇱', 'Turkey': '🇹🇷',
        'Saudi Arabia': '🇸🇦', 'UAE': '🇦🇪', 'Israel': '🇮🇱',
        'Egypt': '🇪🇬', 'Nigeria': '🇳🇬', 'Kenya': '🇰🇪',
        'Indonesia': '🇮🇩', 'Thailand': '🇹🇭', 'Vietnam': '🇻🇳',
        'Philippines': '🇵🇭', 'Malaysia': '🇲🇾', 'New Zealand': '🇳🇿',
        'Norway': '🇳🇴', 'Denmark': '🇩🇰', 'Finland': '🇫🇮',
        'Ireland': '🇮🇪', 'Portugal': '🇵🇹', 'Greece': '🇬🇷',
        'Czech Republic': '🇨🇿', 'Romania': '🇷🇴', 'Hungary': '🇭🇺',
        'Chile': '🇨🇱', 'Colombia': '🇨🇴', 'Peru': '🇵🇪',
        'Pakistan': '🇵🇰', 'Bangladesh': '🇧🇩', 'Sri Lanka': '🇱🇰',
        'Morocco': '🇲🇦', 'Tunisia': '🇹🇳', 'Ghana': '🇬🇭',
        'Ethiopia': '🇪🇹', 'Tanzania': '🇹🇿', 'Uganda': '🇺🇬'
    }
    return flag_map.get(country, '🌐')


def render_country_pills(countries: List[str], selected_country: str) -> str:
    """
    Render country selection pills.
    
    Args:
        countries: List of country names
        selected_country: Currently selected country
        
    Returns:
        Selected country name
    """
    if not countries:
        st.warning("No countries available")
        return ""
    
    # Create columns for pills (max 5 per row)
    pills_per_row = 5
    rows_needed = (len(countries) + pills_per_row - 1) // pills_per_row
    
    selected = selected_country
    
    for row in range(rows_needed):
        cols = st.columns(min(pills_per_row, len(countries) - row * pills_per_row))
        for i, col in enumerate(cols):
            country_idx = row * pills_per_row + i
            if country_idx < len(countries):
                country = countries[country_idx]
                flag = get_country_flag_emoji(country)
                
                with col:
                    if st.button(
                        f"{flag} {country}",
                        key=f"country_pill_{country}",
                        use_container_width=True,
                        type="primary" if country == selected_country else "secondary"
                    ):
                        selected = country
    
    return selected


def create_sidebar_header(title: str, emoji: str = "📊") -> None:
    """
    Create a consistent sidebar header.
    
    Args:
        title: Header title
        emoji: Emoji to display
    """
    st.sidebar.markdown(f"## {emoji} {title}")
    st.sidebar.markdown("---")


def create_page_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Create a consistent page header.
    
    Args:
        title: Page title
        subtitle: Optional subtitle
    """
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def create_metric_card(label: str, value: str, delta: Optional[str] = None,
                      delta_color: str = "normal") -> None:
    """
    Create a metric card display.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Color for delta ("normal", "inverse", "off")
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def create_info_box(content: str, type: str = "info") -> None:
    """
    Create an information box.
    
    Args:
        content: Box content
        type: Box type ("info", "warning", "error", "success")
    """
    if type == "info":
        st.info(content)
    elif type == "warning":
        st.warning(content)
    elif type == "error":
        st.error(content)
    elif type == "success":
        st.success(content)
    else:
        st.write(content)


def create_expandable_section(title: str, content_func, expanded: bool = False) -> None:
    """
    Create an expandable section.
    
    Args:
        title: Section title
        content_func: Function to call for content
        expanded: Whether to start expanded
    """
    with st.expander(title, expanded=expanded):
        content_func()


def create_tabs(tab_names: List[str]) -> List:
    """
    Create tabs and return tab objects.
    
    Args:
        tab_names: List of tab names
        
    Returns:
        List of tab objects
    """
    return st.tabs(tab_names)


def create_columns_with_gap(num_columns: int, gap: str = "medium") -> List:
    """
    Create columns with specified gap.
    
    Args:
        num_columns: Number of columns
        gap: Gap size ("small", "medium", "large")
        
    Returns:
        List of column objects
    """
    return st.columns(num_columns, gap=gap)


def display_dataframe_with_formatting(df, 
                                     height: Optional[int] = None,
                                     use_container_width: bool = True) -> None:
    """
    Display a DataFrame with consistent formatting.
    
    Args:
        df: DataFrame to display
        height: Optional fixed height
        use_container_width: Whether to use full container width
    """
    st.dataframe(
        df,
        height=height,
        use_container_width=use_container_width,
        hide_index=True
    )


def create_download_button(data, filename: str, label: str = "Download",
                         mime_type: str = "text/csv") -> None:
    """
    Create a download button.
    
    Args:
        data: Data to download
        filename: Download filename
        label: Button label
        mime_type: MIME type of the data
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )


def create_progress_bar(value: float, label: Optional[str] = None) -> None:
    """
    Create a progress bar.
    
    Args:
        value: Progress value (0.0 to 1.0)
        label: Optional label
    """
    if label:
        st.write(label)
    st.progress(value)


def create_status_indicator(status: str) -> None:
    """
    Create a status indicator with appropriate styling.
    
    Args:
        status: Status text
    """
    status_lower = status.lower()
    
    if "complete" in status_lower or "success" in status_lower:
        st.success(f"✅ {status}")
    elif "error" in status_lower or "fail" in status_lower:
        st.error(f"❌ {status}")
    elif "warning" in status_lower or "pending" in status_lower:
        st.warning(f"⚠️ {status}")
    elif "progress" in status_lower or "running" in status_lower:
        st.info(f"🔄 {status}")
    else:
        st.write(f"ℹ️ {status}")


def create_divider(style: str = "default") -> None:
    """
    Create a divider line.
    
    Args:
        style: Divider style ("default", "rainbow", "thin")
    """
    if style == "rainbow":
        st.markdown(
            """<hr style="height:2px;border:none;
            background:linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);">""",
            unsafe_allow_html=True
        )
    elif style == "thin":
        st.markdown(
            """<hr style="height:1px;border:none;background-color:#e0e0e0;">""",
            unsafe_allow_html=True
        )
    else:
        st.divider()


def create_empty_state(message: str, emoji: str = "📭") -> None:
    """
    Create an empty state display.
    
    Args:
        message: Empty state message
        emoji: Emoji to display
    """
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <p style="color: #666; margin-top: 1rem;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )