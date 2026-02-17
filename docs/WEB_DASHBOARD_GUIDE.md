# 🌐 NETRA Web Dashboard - User Guide

## 🚀 Quick Start

### Installation

1. **Install required packages:**
   ```bash
   pip install streamlit pillow
   ```
   
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the dashboard:**
   ```bash
   streamlit run src/web_dashboard.py
   ```
   
   Or use the launcher script:
   ```bash
   bash start_dashboard.sh
   ```

3. **Access the dashboard:**
   - The browser will open automatically at: `http://localhost:8501`
   - If not, manually visit that URL

---

## 📊 Dashboard Features

### 🏠 Home Page

**What you see:**
- **Key Performance Indicators (KPIs)**
  - Total observations
  - Ambulance detections
  - Average traffic per lane
  - Lane utilization percentages

- **Visual Charts:**
  - Lane utilization bar chart
  - Average green time comparison
  - Peak traffic hour identification

- **Recent Activity Table:**
  - Last 10 traffic observations
  - Color-coded by traffic density

**Use for:**
- Quick system overview
- Real-time status monitoring
- Presentation demos

---

### 📊 Analytics Page

**4 Tabs Available:**

#### 1. 📈 Trends Tab
- **Vehicle Count Over Time** - Line graph showing traffic patterns
- **Hourly Traffic Pattern** - Bar chart showing average traffic by hour
- **Use for:** Identifying traffic trends and patterns

#### 2. 🔥 Heatmap Tab
- **Correlation Matrix** - Shows relationships between variables
- Values range from -1 to +1
- **Use for:** Understanding how variables relate (e.g., vehicle count vs green time)

#### 3. 📊 Statistics Tab
- Detailed metrics for each lane
- Average vehicles, max vehicles, green times
- Emergency alert statistics
- **Use for:** Detailed analysis and project reports

#### 4. 📁 Reports Tab
- Displays previously generated analytics reports
- Shows PNG visualizations and text summaries
- **Use for:** Viewing historical analysis

---

### 🔍 Data Explorer Page

**Features:**
- **Interactive Filtering:**
  - Filter by minimum lane vehicles
  - Filter by ambulance detection status
  
- **Data Table:**
  - View filtered results
  - Sortable columns
  
- **Download Option:**
  - Export filtered data as CSV
  - Timestamped filenames

**Use for:**
- Custom data analysis
- Exporting specific data subsets
- Finding specific events (e.g., all ambulance detections)

---

### ⚙️ System Info Page

**What it shows:**
- **Project Structure** - File organization
- **AI Models Status** - Check if models are loaded
- **Data Summary** - Count of files and records
- **Quick Actions** - Command references
- **About Section** - Project information

**Use for:**
- Troubleshooting
- System health check
- Documentation reference

---

## 🎯 Common Use Cases

### For Project Demo/Presentation

1. **Start with Home page** - Show live KPIs
2. **Navigate to Analytics** → Trends - Show traffic patterns
3. **Show Heatmap** - Explain correlation analysis
4. **Demo Data Explorer** - Show filtering capabilities
5. **End with System Info** - Show project structure

### For Data Analysis

1. Go to **Data Explorer**
2. Apply filters you need
3. Download filtered CSV
4. Or go to **Analytics** → Statistics for quick insights

### For Monitoring

1. Keep **Home page** open
2. Data auto-refreshes every 60 seconds
3. Watch KPIs update in real-time

---

## 💡 Pro Tips

### Making it Impressive

1. **Run the main system first** to generate fresh data:
   ```bash
   python main.py
   ```

2. **Generate analytics** before demo:
   ```bash
   python src/analytics_report.py
   ```

3. **Use full-screen mode** in browser (F11)

4. **Dark mode:** Click settings (top-right) → Choose theme → Dark

### Customization

**Change colors:** Edit the CSS in `web_dashboard.py` lines 25-50

**Add more metrics:** Modify the `calculate_kpis()` function

**Add new pages:** Create new `page_xxx()` functions and add to navigation

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Install streamlit
pip install streamlit

# Or reinstall all requirements
pip install -r requirements.txt
```

### "No traffic data found" error
```bash
# Run the main system first to generate data
python main.py

# Let it run for a minute, then press 'q'
```

### Charts not showing
- Make sure you have run `python src/analytics_report.py` at least once
- Check that `reports/analytics_output/` folder has PNG files

### Port already in use
```bash
# Use a different port
streamlit run src/web_dashboard.py --server.port 8502
```

---

## 📱 Keyboard Shortcuts

While in the dashboard:

- `R` - Rerun the app (refresh data)
- `C` - Clear cache
- `?` - Show keyboard shortcuts
- `Ctrl + C` (in terminal) - Stop the server

---

## 🎨 Features Highlights

### Auto-Refresh
- Data cached for 60 seconds
- Automatic updates without manual refresh

### Responsive Design
- Works on desktop, tablet, mobile
- Adaptive layout

### Interactive Charts
- Hover for details
- Zoom capabilities
- Download as PNG

### Color Coding
- Red = Lane 1
- Blue = Lane 2
- Color intensity = Traffic density

---

## 📊 What Makes This Impressive for Your Project

1. **Professional UI** - Looks like a commercial product
2. **Real-time Updates** - Shows live data
3. **Interactive** - Not just static reports
4. **Multiple Views** - Different analysis perspectives
5. **Export Capability** - Download filtered data
6. **Modern Tech Stack** - Uses Streamlit (industry-standard)

---

## 🚀 Advanced Features

### Running on Network

To access from other devices on your network:

```bash
streamlit run src/web_dashboard.py --server.address 0.0.0.0
```

Then access from other devices at: `http://YOUR_IP:8501`

### Deploy Online (Optional)

Can be deployed to:
- **Streamlit Cloud** (free, easiest)
- **Heroku**
- **AWS/Azure**

---

## 📝 For Your Project Report

Include these points:

> "The web interface was built using **Streamlit**, a Python framework for creating data applications. It provides:
> - Real-time traffic monitoring dashboard
> - Interactive data visualization with Matplotlib and Seaborn
> - Client-server architecture using HTTP
> - Responsive design for cross-device compatibility
> - Data caching for optimized performance
> - Export functionality for further analysis"

---

## 🎓 Technical Details (For Viva/Presentation)

- **Framework:** Streamlit (Python)
- **Visualization:** Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy
- **Architecture:** Client-server (web-based)
- **Updates:** Auto-refresh with 60s cache
- **Deployment:** Local host or cloud-ready

---

**Need help?** Check the code comments in `src/web_dashboard.py` or refer to [Streamlit Documentation](https://docs.streamlit.io)
