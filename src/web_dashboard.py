"""
🚦 PROJECT NETRA - Web Dashboard
Interactive web interface for traffic management and analytics
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="NETRA Traffic Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_traffic_data():
    """Load all traffic CSV files"""
    csv_files = glob.glob("data/traffic_logs/Traffic_Data_*.csv")
    
    if not csv_files:
        return None
    
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        date_str = filename.replace("Traffic_Data_", "").replace(".csv", "")
        df['Date'] = date_str[:8]
        dataframes.append(df)
    
    data = pd.concat(dataframes, ignore_index=True)
    
    # Backward compatibility: add tracking columns if missing (old CSV format)
    for col, default in [('Lane1_Unique', 0), ('Lane2_Unique', 0),
                         ('Avg_Speed_L1', 0.0), ('Avg_Speed_L2', 0.0)]:
        if col not in data.columns:
            data[col] = default
    
    # Parse timestamps
    try:
        data['DateTime'] = pd.to_datetime(
            data['Date'] + ' ' + data['Timestamp'], 
            format='%Y%m%d %H:%M:%S'
        )
        data['Hour'] = data['DateTime'].dt.hour
        data['Minute'] = data['DateTime'].dt.minute
    except:
        pass
    
    return data


def get_latest_reports():
    """Get the most recent analytics reports"""
    reports = {
        'analysis': None,
        'heatmap': None,
        'summary': None
    }
    
    # Find latest analysis
    analysis_files = glob.glob("reports/analytics_output/Traffic_Analysis_*.png")
    if analysis_files:
        reports['analysis'] = max(analysis_files, key=os.path.getctime)
    
    # Find latest heatmap
    heatmap_files = glob.glob("reports/analytics_output/Correlation_Heatmap_*.png")
    if heatmap_files:
        reports['heatmap'] = max(heatmap_files, key=os.path.getctime)
    
    # Find latest summary
    summary_files = glob.glob("reports/analytics_output/Traffic_Summary_*.txt")
    if summary_files:
        reports['summary'] = max(summary_files, key=os.path.getctime)
    
    return reports


def calculate_kpis(data):
    """Calculate Key Performance Indicators"""
    kpis = {}
    
    if data is None or len(data) == 0:
        return kpis
    
    kpis['total_observations'] = len(data)
    kpis['avg_lane1'] = data['Lane1_Count'].mean()
    kpis['avg_lane2'] = data['Lane2_Count'].mean()
    kpis['max_lane1'] = data['Lane1_Count'].max()
    kpis['max_lane2'] = data['Lane2_Count'].max()
    kpis['ambulance_count'] = data['Ambulance_Detected'].sum()
    kpis['avg_green_time_l1'] = data['Green_Time_L1'].mean()
    kpis['avg_green_time_l2'] = data['Green_Time_L2'].mean()
    
    # Tracking metrics
    kpis['max_unique_l1'] = int(data['Lane1_Unique'].max())
    kpis['max_unique_l2'] = int(data['Lane2_Unique'].max())
    kpis['avg_speed_l1'] = data['Avg_Speed_L1'].mean()
    kpis['avg_speed_l2'] = data['Avg_Speed_L2'].mean()
    
    # Calculate efficiency
    total_vehicles = data['Lane1_Count'].sum() + data['Lane2_Count'].sum()
    if total_vehicles > 0:
        kpis['lane1_utilization'] = (data['Lane1_Count'].sum() / total_vehicles * 100)
        kpis['lane2_utilization'] = (data['Lane2_Count'].sum() / total_vehicles * 100)
    
    # Peak hour
    if 'Hour' in data.columns:
        hourly_traffic = data.groupby('Hour').agg({
            'Lane1_Count': 'mean',
            'Lane2_Count': 'mean'
        })
        total_traffic = hourly_traffic['Lane1_Count'] + hourly_traffic['Lane2_Count']
        kpis['peak_hour'] = total_traffic.idxmax()
    
    return kpis


# ==================== PAGE: HOME ====================

def page_home():
    """Main dashboard page"""
    st.markdown('<h1 class="main-header">🚦 NETRA Traffic Management System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Network Enabled Traffic Regulation & Analysis
    
    **Real-time traffic monitoring with AI-powered ambulance detection and adaptive signal timing**
    """)
    
    # Load data
    data = load_traffic_data()
    
    if data is None:
        st.warning("⚠️ No traffic data found. Run the main system first to generate data.")
        st.code("python main.py", language="bash")
        return
    
    # Calculate KPIs
    kpis = calculate_kpis(data)
    
    # Display KPIs in columns
    st.markdown("---")
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Observations",
            value=f"{kpis['total_observations']:,}",
            delta="Live Data"
        )
    
    with col2:
        st.metric(
            label="Ambulances Detected",
            value=kpis['ambulance_count'],
            delta="Emergency Responses"
        )
    
    with col3:
        st.metric(
            label="Avg Lane 1 Traffic",
            value=f"{kpis['avg_lane1']:.1f}",
            delta=f"{kpis['lane1_utilization']:.1f}% utilization"
        )
    
    with col4:
        st.metric(
            label="Avg Lane 2 Traffic",
            value=f"{kpis['avg_lane2']:.1f}",
            delta=f"{kpis['lane2_utilization']:.1f}% utilization"
        )
    
    # Vehicle Tracking Stats
    st.markdown("---")
    st.subheader("🔍 Vehicle Tracking (ByteTrack)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Unique Vehicles (L1)",
            value=kpis.get('max_unique_l1', 0),
            delta="Session Max"
        )
    
    with col2:
        st.metric(
            label="Unique Vehicles (L2)",
            value=kpis.get('max_unique_l2', 0),
            delta="Session Max"
        )
    
    with col3:
        st.metric(
            label="Avg Speed L1",
            value=f"{kpis.get('avg_speed_l1', 0):.1f} px/s"
        )
    
    with col4:
        st.metric(
            label="Avg Speed L2",
            value=f"{kpis.get('avg_speed_l2', 0):.1f} px/s"
        )
    
    # Lane comparison
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛣️ Lane Utilization")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        lanes = ['Lane 1', 'Lane 2']
        utilization = [kpis['lane1_utilization'], kpis['lane2_utilization']]
        colors = ['#ff4444', '#4444ff']
        
        bars = ax.bar(lanes, utilization, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Utilization (%)', fontsize=12)
        ax.set_title('Lane Traffic Distribution', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("⏱️ Average Green Time")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        lanes = ['Lane 1', 'Lane 2']
        green_times = [kpis['avg_green_time_l1'], kpis['avg_green_time_l2']]
        colors = ['#ff4444', '#4444ff']
        
        bars = ax.bar(lanes, green_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Signal Timing Efficiency', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    # Peak hour info
    if 'peak_hour' in kpis:
        st.markdown("---")
        st.info(f"🕐 **Peak Traffic Hour:** {kpis['peak_hour']}:00 - {kpis['peak_hour']+1}:00")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📋 Recent Activity")
    
    recent_data = data.tail(10).sort_values('Timestamp', ascending=False)
    display_cols = ['Timestamp', 'Lane1_Count', 'Lane2_Count', 'Ambulance_Detected', 
                    'Green_Time_L1', 'Green_Time_L2']
    
    st.dataframe(
        recent_data[display_cols].style.background_gradient(cmap='RdYlGn_r', subset=['Lane1_Count', 'Lane2_Count']),
        use_container_width=True
    )


# ==================== PAGE: ANALYTICS ====================

def page_analytics():
    """Analytics and visualizations page"""
    st.markdown('<h1 class="main-header">📊 Traffic Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    data = load_traffic_data()
    
    if data is None:
        st.warning("⚠️ No traffic data available.")
        return
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trends", "🔥 Heatmap", "📊 Statistics", "📁 Reports"
    ])
    
    with tab1:
        st.subheader("Traffic Trends Analysis")
        
        # Time series plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(data)), data['Lane1_Count'], 
               label='Lane 1', color='red', alpha=0.7, linewidth=2)
        ax.plot(range(len(data)), data['Lane2_Count'], 
               label='Lane 2', color='blue', alpha=0.7, linewidth=2)
        ax.set_xlabel('Observation Number', fontsize=12)
        ax.set_ylabel('Number of Vehicles', fontsize=12)
        ax.set_title('Vehicle Count Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Speed Trends (only if tracking data exists)
        if data['Avg_Speed_L1'].sum() > 0 or data['Avg_Speed_L2'].sum() > 0:
            st.subheader("Vehicle Speed Trends")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(range(len(data)), data['Avg_Speed_L1'],
                   label='Lane 1 Speed', color='red', alpha=0.7, linewidth=2)
            ax.plot(range(len(data)), data['Avg_Speed_L2'],
                   label='Lane 2 Speed', color='blue', alpha=0.7, linewidth=2)
            ax.set_xlabel('Observation Number', fontsize=12)
            ax.set_ylabel('Speed (px/s)', fontsize=12)
            ax.set_title('Average Vehicle Speed Over Time', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Hourly pattern
        if 'Hour' in data.columns:
            st.subheader("Hourly Traffic Pattern")
            
            hourly_data = data.groupby('Hour').agg({
                'Lane1_Count': 'mean',
                'Lane2_Count': 'mean'
            }).reset_index()
            
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(hourly_data))
            width = 0.35
            
            ax.bar(x - width/2, hourly_data['Lane1_Count'], width,
                  label='Lane 1', color='red', alpha=0.7)
            ax.bar(x + width/2, hourly_data['Lane2_Count'], width,
                  label='Lane 2', color='blue', alpha=0.7)
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Average Vehicle Count', fontsize=12)
            ax.set_title('Average Traffic by Hour', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(hourly_data['Hour'])
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        numeric_cols = ['Lane1_Count', 'Lane2_Count', 'Green_Time_L1', 'Green_Time_L2']
        corr_data = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Traffic Data Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Interpretation:**
        - Values close to **+1** indicate strong positive correlation
        - Values close to **-1** indicate strong negative correlation
        - Values close to **0** indicate no correlation
        """)
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        kpis = calculate_kpis(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚗 Lane 1 Metrics")
            st.write(f"**Average Vehicles:** {kpis['avg_lane1']:.2f}")
            st.write(f"**Maximum Vehicles:** {kpis['max_lane1']}")
            st.write(f"**Average Green Time:** {kpis['avg_green_time_l1']:.1f}s")
            st.write(f"**Utilization:** {kpis['lane1_utilization']:.1f}%")
            st.write(f"**Max Unique Vehicles:** {kpis.get('max_unique_l1', 'N/A')}")
            st.write(f"**Average Speed:** {kpis.get('avg_speed_l1', 0):.1f} px/s")
        
        with col2:
            st.markdown("### 🚙 Lane 2 Metrics")
            st.write(f"**Average Vehicles:** {kpis['avg_lane2']:.2f}")
            st.write(f"**Maximum Vehicles:** {kpis['max_lane2']}")
            st.write(f"**Average Green Time:** {kpis['avg_green_time_l2']:.1f}s")
            st.write(f"**Utilization:** {kpis['lane2_utilization']:.1f}%")
            st.write(f"**Max Unique Vehicles:** {kpis.get('max_unique_l2', 'N/A')}")
            st.write(f"**Average Speed:** {kpis.get('avg_speed_l2', 0):.1f} px/s")
        
        st.markdown("---")
        st.markdown("### 🚑 Emergency Alerts")
        st.write(f"**Total Ambulance Detections:** {kpis['ambulance_count']}")
        
        if kpis['ambulance_count'] > 0:
            override_rate = (kpis['ambulance_count'] / kpis['total_observations'] * 100)
            st.write(f"**Emergency Override Rate:** {override_rate:.2f}%")
    
    with tab4:
        st.subheader("Generated Reports")
        
        reports = get_latest_reports()
        
        if reports['analysis']:
            st.markdown("#### 📊 Traffic Analysis Dashboard")
            image = Image.open(reports['analysis'])
            st.image(image, use_container_width=True)
            st.caption(f"Generated: {datetime.fromtimestamp(os.path.getctime(reports['analysis'])).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if reports['heatmap']:
            st.markdown("#### 🔥 Correlation Heatmap")
            image = Image.open(reports['heatmap'])
            st.image(image, use_container_width=True)
        
        if reports['summary']:
            st.markdown("#### 📄 Summary Report")
            with open(reports['summary'], 'r') as f:
                st.text(f.read())


# ==================== PAGE: DATA EXPLORER ====================

def page_data_explorer():
    """Data exploration and filtering page"""
    st.markdown('<h1 class="main-header">🔍 Data Explorer</h1>', 
                unsafe_allow_html=True)
    
    data = load_traffic_data()
    
    if data is None:
        st.warning("⚠️ No traffic data available.")
        return
    
    st.subheader("Filter and Explore Traffic Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_lane1 = st.slider("Min Lane 1 Vehicles", 0, int(data['Lane1_Count'].max()), 0)
    
    with col2:
        min_lane2 = st.slider("Min Lane 2 Vehicles", 0, int(data['Lane2_Count'].max()), 0)
    
    with col3:
        ambulance_filter = st.selectbox("Ambulance Status", 
                                        ["All", "Detected Only", "Not Detected"])
    
    # Apply filters
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data['Lane1_Count'] >= min_lane1]
    filtered_data = filtered_data[filtered_data['Lane2_Count'] >= min_lane2]
    
    if ambulance_filter == "Detected Only":
        filtered_data = filtered_data[filtered_data['Ambulance_Detected'] == True]
    elif ambulance_filter == "Not Detected":
        filtered_data = filtered_data[filtered_data['Ambulance_Detected'] == False]
    
    st.write(f"**Showing {len(filtered_data)} of {len(data)} records**")
    
    # Display filtered data
    display_cols = ['Timestamp', 'Lane1_Count', 'Lane2_Count',
                    'Lane1_Unique', 'Lane2_Unique',
                    'Avg_Speed_L1', 'Avg_Speed_L2',
                    'Ambulance_Detected', 'Green_Time_L1', 'Green_Time_L2']
    st.dataframe(
        filtered_data[display_cols],
        use_container_width=True
    )
    
    # Download filtered data
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"filtered_traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# ==================== PAGE: SYSTEM INFO ====================

def page_system_info():
    """System information and configuration"""
    st.markdown('<h1 class="main-header">⚙️ System Information</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Project Structure")
        st.code("""
PROJECT-NETRA/
├── main.py                  # Main system
├── src/
│   ├── analytics_report.py  # Analytics
│   ├── web_dashboard.py     # This dashboard
│   └── utils/               # Utilities
├── models/                  # AI models
├── data/traffic_logs/       # CSV data
├── reports/                 # Reports
└── videos/                  # Videos
        """, language="text")
    
    with col2:
        st.subheader("🤖 AI Models")
        
        # Check model files
        models_exist = {
            'yolov8m.pt': os.path.exists('models/yolov8m.pt'),
            'best.pt': os.path.exists('models/best.pt')
        }
        
        for model, exists in models_exist.items():
            if exists:
                size = os.path.getsize(f'models/{model}') / (1024*1024)
                st.success(f"✅ {model} ({size:.1f} MB)")
            else:
                st.error(f"❌ {model} (Missing)")
    
    st.markdown("---")
    
    st.subheader("📊 Data Summary")
    
    # Count files
    csv_files = glob.glob("data/traffic_logs/*.csv")
    report_files = glob.glob("reports/analytics_output/*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CSV Files", len(csv_files))
    
    with col2:
        st.metric("Generated Reports", len(report_files))
    
    with col3:
        data = load_traffic_data()
        total_records = len(data) if data is not None else 0
        st.metric("Total Records", total_records)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("python main.py", language="bash")
        st.caption("Run traffic detection system")
    
    with col2:
        st.code("python src/analytics_report.py", language="bash")
        st.caption("Generate analytics reports")
    
    st.markdown("---")
    
    st.subheader("📖 About")
    st.markdown("""
    **NETRA** (*Network Enabled Traffic Regulation & Analysis*) is an intelligent 
    traffic management system that uses:
    
    - 🧠 **YOLOv8** for vehicle detection
    - 🚑 **Custom model** for ambulance detection
    - 🔍 **ByteTrack** for multi-object vehicle tracking
    - ⏱️ **Adaptive timing** based on traffic density
    - 📊 **Real-time analytics** for traffic insights
    
    **Version:** 2.0 (with Vehicle Tracking)  
    **Developer:** Faiz Ahmad Khan  
    **Institution:** 3rd Year Project
    """)


# ==================== MAIN APP ====================

def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.title("🚦 NETRA Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Analytics", "🔍 Data Explorer", "⚙️ System Info"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    data = load_traffic_data()
    if data is not None:
        st.sidebar.subheader("Quick Stats")
        st.sidebar.metric("Total Records", len(data))
        st.sidebar.metric("Ambulances", int(data['Ambulance_Detected'].sum()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("🔄 Data refreshes every 60 seconds")
    
    # Route to pages
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Analytics":
        page_analytics()
    elif page == "🔍 Data Explorer":
        page_data_explorer()
    elif page == "⚙️ System Info":
        page_system_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "PROJECT NETRA © 2026 | Powered by Streamlit & YOLOv8"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
