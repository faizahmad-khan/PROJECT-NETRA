# 📊 NETRA Analytics Dashboard - Quick Start Guide

## Overview
The analytics dashboard analyzes all your traffic CSV files and generates comprehensive visualizations and reports.

## Installation

Install required packages:
```bash
pip install pandas matplotlib seaborn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Headless Mode (Recommended)
Generates all reports and saves them without showing GUI windows:
```bash
python3 analytics_report.py
```

### Option 2: Interactive Mode
Shows interactive plots (requires display):
```bash
python3 analytics.py
```

## What Gets Generated

### 1. **Traffic Analysis Dashboard** (PNG)
   - 📈 Vehicle count trends over time
   - 📊 Green time distribution histogram  
   - ⏰ Hourly traffic patterns
   - 🔄 Lane comparison scatter plot

### 2. **Correlation Heatmap** (PNG)
   - Shows relationships between variables
   - Useful for understanding traffic patterns

### 3. **Summary Report** (TXT)
   - Overall statistics
   - Lane-by-lane metrics
   - Peak hour identification
   - Emergency vehicle analytics

## Key Features

✅ **Traffic Pattern Analysis**
   - Hourly and daily trends
   - Time-series visualization

✅ **Peak Hour Identification**
   - Automatically detects busiest hours
   - Shows average vehicle counts

✅ **Lane Utilization Comparison**
   - Percentage distribution across lanes
   - Warns about imbalanced usage

✅ **Ambulance Frequency Analytics**
   - Counts emergency overrides
   - Calculates override rates

✅ **Average Wait Time Calculations**
   - Green time statistics per lane
   - Distribution analysis

## Sample Output

```
============================================================
📊 NETRA TRAFFIC ANALYTICS REPORT
============================================================

📈 OVERALL STATISTICS
  • Total Observations: 100
  • Time Period: 20260131 to 20260131

🚗 LANE 1 (Left Lane)
  • Average Vehicles: 13.00
  • Maximum Vehicles: 21
  • Average Green Time: 31.0s

🚙 LANE 2 (Right Lane)
  • Average Vehicles: 15.63
  • Maximum Vehicles: 20
  • Average Green Time: 36.3s

⏰ PEAK TRAFFIC HOUR
  • Peak Hour: 22:00 - 23:00
  • Average Vehicles: 28.63

🛣️  LANE UTILIZATION
  • Lane 1: 45.4%
  • Lane 2: 54.6%
============================================================
```

## Tips for Your Project Presentation

1. **Run analytics after collecting data** from main.py
2. **Include the PNG visualizations** in your report/presentation
3. **Use the text summary** for quick statistics
4. **Compare different time periods** by running multiple times
5. **Highlight the peak hour findings** - shows real-world applicability

## Troubleshooting

**Issue**: Module not found errors  
**Solution**: Make sure packages are installed: `pip3 install pandas matplotlib seaborn`

**Issue**: Display errors on headless systems  
**Solution**: Use `analytics_report.py` instead of `analytics.py`

## For Academic Report

Include these analytics sections in your project report:

1. **Data Collection**: Show total observations
2. **Traffic Patterns**: Use the hourly graph
3. **System Performance**: Green time distributions
4. **Lane Balancing**: Utilization percentages
5. **Emergency Response**: Ambulance detection stats

---

**Pro Tip**: Run the analytics weekly during your project demo period to show real data trends!
