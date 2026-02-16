"""
🚦 PROJECT NETRA - Traffic Analytics Dashboard
Analyzes traffic data from CSV logs and generates insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class TrafficAnalytics:
    """Main analytics engine for NETRA traffic data"""
    
    def __init__(self, data_folder="data/traffic_logs"):
        """Initialize analytics engine
        
        Args:
            data_folder: Folder containing Traffic_Data_*.csv files
        """
        self.data_folder = data_folder
        self.data = None
        self.stats = {}
        
    def load_data(self):
        """Load all CSV files and combine them"""
        csv_files = glob.glob(os.path.join(self.data_folder, "Traffic_Data_*.csv"))
        
        if not csv_files:
            print("❌ No traffic data files found!")
            return False
        
        print(f"📂 Found {len(csv_files)} traffic data files")
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Extract date from filename
                filename = os.path.basename(file)
                date_str = filename.replace("Traffic_Data_", "").replace(".csv", "")
                df['Date'] = date_str[:8]  # YYYYMMDD
                dataframes.append(df)
                print(f"  ✅ Loaded: {filename}")
            except Exception as e:
                print(f"  ⚠️  Error loading {file}: {e}")
        
        if not dataframes:
            print("❌ No valid data loaded!")
            return False
        
        # Combine all data
        self.data = pd.concat(dataframes, ignore_index=True)
        
        # Convert timestamp to datetime
        try:
            self.data['DateTime'] = pd.to_datetime(
                self.data['Date'] + ' ' + self.data['Timestamp'], 
                format='%Y%m%d %H:%M:%S'
            )
            self.data['Hour'] = self.data['DateTime'].dt.hour
            self.data['Minute'] = self.data['DateTime'].dt.minute
        except:
            print("⚠️  Could not parse timestamps")
        
        print(f"\n✅ Total records loaded: {len(self.data)}")
        return True
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        if self.data is None or len(self.data) == 0:
            print("❌ No data available!")
            return
        
        print("\n" + "="*60)
        print("📊 NETRA TRAFFIC ANALYTICS REPORT")
        print("="*60)
        
        # Basic Statistics
        self.stats['total_records'] = len(self.data)
        self.stats['avg_lane1'] = self.data['Lane1_Count'].mean()
        self.stats['avg_lane2'] = self.data['Lane2_Count'].mean()
        self.stats['max_lane1'] = self.data['Lane1_Count'].max()
        self.stats['max_lane2'] = self.data['Lane2_Count'].max()
        self.stats['ambulance_count'] = self.data['Ambulance_Detected'].sum()
        self.stats['avg_green_time_l1'] = self.data['Green_Time_L1'].mean()
        self.stats['avg_green_time_l2'] = self.data['Green_Time_L2'].mean()
        
        print(f"\n📈 OVERALL STATISTICS")
        print(f"  • Total Observations: {self.stats['total_records']}")
        print(f"  • Time Period: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        print(f"\n🚗 LANE 1 (Left Lane)")
        print(f"  • Average Vehicles: {self.stats['avg_lane1']:.2f}")
        print(f"  • Maximum Vehicles: {self.stats['max_lane1']}")
        print(f"  • Average Green Time: {self.stats['avg_green_time_l1']:.1f}s")
        
        print(f"\n🚙 LANE 2 (Right Lane)")
        print(f"  • Average Vehicles: {self.stats['avg_lane2']:.2f}")
        print(f"  • Maximum Vehicles: {self.stats['max_lane2']}")
        print(f"  • Average Green Time: {self.stats['avg_green_time_l2']:.1f}s")
        
        print(f"\n🚑 EMERGENCY VEHICLES")
        print(f"  • Ambulance Detections: {self.stats['ambulance_count']}")
        if self.stats['ambulance_count'] > 0:
            print(f"  • Emergency Override Rate: {(self.stats['ambulance_count']/self.stats['total_records']*100):.2f}%")
        
        # Peak Hour Analysis
        if 'Hour' in self.data.columns:
            hourly_traffic = self.data.groupby('Hour').agg({
                'Lane1_Count': 'mean',
                'Lane2_Count': 'mean'
            })
            total_traffic = hourly_traffic['Lane1_Count'] + hourly_traffic['Lane2_Count']
            peak_hour = total_traffic.idxmax()
            
            self.stats['peak_hour'] = peak_hour
            print(f"\n⏰ PEAK TRAFFIC HOUR")
            print(f"  • Peak Hour: {peak_hour}:00 - {peak_hour+1}:00")
            print(f"  • Average Vehicles: {total_traffic.max():.2f}")
        
        # Lane Utilization
        total_vehicles = self.data['Lane1_Count'].sum() + self.data['Lane2_Count'].sum()
        if total_vehicles > 0:
            lane1_utilization = (self.data['Lane1_Count'].sum() / total_vehicles * 100)
            lane2_utilization = (self.data['Lane2_Count'].sum() / total_vehicles * 100)
            
            print(f"\n🛣️  LANE UTILIZATION")
            print(f"  • Lane 1: {lane1_utilization:.1f}%")
            print(f"  • Lane 2: {lane2_utilization:.1f}%")
            
            if abs(lane1_utilization - lane2_utilization) > 20:
                print(f"  ⚠️  Warning: Unbalanced lane usage detected!")
        
        print("\n" + "="*60)
    
    def plot_traffic_trends(self):
        """Generate time-series plots of traffic patterns"""
        if self.data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('🚦 NETRA Traffic Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Vehicle Count Over Time
        ax1 = axes[0, 0]
        ax1.plot(range(len(self.data)), self.data['Lane1_Count'], 
                label='Lane 1', color='red', alpha=0.7, linewidth=2)
        ax1.plot(range(len(self.data)), self.data['Lane2_Count'], 
                label='Lane 2', color='blue', alpha=0.7, linewidth=2)
        ax1.set_title('Vehicle Count Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Observation Number')
        ax1.set_ylabel('Number of Vehicles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Green Time Distribution
        ax2 = axes[0, 1]
        ax2.hist(self.data['Green_Time_L1'], bins=20, alpha=0.6, 
                label='Lane 1', color='red', edgecolor='black')
        ax2.hist(self.data['Green_Time_L2'], bins=20, alpha=0.6, 
                label='Lane 2', color='blue', edgecolor='black')
        ax2.set_title('Green Time Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Green Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hourly Traffic Pattern
        ax3 = axes[1, 0]
        if 'Hour' in self.data.columns:
            hourly_data = self.data.groupby('Hour').agg({
                'Lane1_Count': 'mean',
                'Lane2_Count': 'mean'
            }).reset_index()
            
            x = np.arange(len(hourly_data))
            width = 0.35
            
            ax3.bar(x - width/2, hourly_data['Lane1_Count'], width, 
                   label='Lane 1', color='red', alpha=0.7)
            ax3.bar(x + width/2, hourly_data['Lane2_Count'], width, 
                   label='Lane 2', color='blue', alpha=0.7)
            ax3.set_title('Average Traffic by Hour', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Average Vehicle Count')
            ax3.set_xticks(x)
            ax3.set_xticklabels(hourly_data['Hour'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Hourly data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Lane Comparison Scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(self.data['Lane1_Count'], self.data['Lane2_Count'], 
                             c=self.data['Ambulance_Detected'], cmap='RdYlGn_r', 
                             alpha=0.6, edgecolors='black', s=50)
        ax4.plot([0, max(self.data['Lane1_Count'].max(), self.data['Lane2_Count'].max())],
                [0, max(self.data['Lane1_Count'].max(), self.data['Lane2_Count'].max())],
                'k--', alpha=0.3, label='Equal Traffic')
        ax4.set_title('Lane 1 vs Lane 2 Comparison', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Lane 1 Count')
        ax4.set_ylabel('Lane 2 Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax4, label='Ambulance Detected')
        plt.tight_layout()
        
        # Save figure
        filename = f"reports/analytics_output/Traffic_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {filename}")
        
        plt.show()
    
    def plot_heatmap(self):
        """Generate correlation heatmap"""
        if self.data is None:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns
        numeric_cols = ['Lane1_Count', 'Lane2_Count', 'Green_Time_L1', 'Green_Time_L2']
        corr_data = self.data[numeric_cols].corr()
        
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Traffic Data Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = f"reports/analytics_output/Correlation_Heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"🔥 Heatmap saved: {filename}")
        
        plt.show()
    
    def export_summary_report(self):
        """Export a text summary report"""
        if not self.stats:
            self.calculate_statistics()
        
        filename = f"reports/analytics_output/Traffic_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("🚦 PROJECT NETRA - TRAFFIC ANALYTICS SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Observations: {self.stats.get('total_records', 'N/A')}\n")
            f.write(f"Ambulance Detections: {self.stats.get('ambulance_count', 'N/A')}\n\n")
            
            f.write("LANE 1 METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average Vehicles: {self.stats.get('avg_lane1', 0):.2f}\n")
            f.write(f"Maximum Vehicles: {self.stats.get('max_lane1', 'N/A')}\n")
            f.write(f"Average Green Time: {self.stats.get('avg_green_time_l1', 0):.1f}s\n\n")
            
            f.write("LANE 2 METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average Vehicles: {self.stats.get('avg_lane2', 0):.2f}\n")
            f.write(f"Maximum Vehicles: {self.stats.get('max_lane2', 'N/A')}\n")
            f.write(f"Average Green Time: {self.stats.get('avg_green_time_l2', 0):.1f}s\n\n")
            
            if 'peak_hour' in self.stats:
                f.write("PEAK TRAFFIC INSIGHTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Peak Hour: {self.stats['peak_hour']}:00\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"📄 Summary report saved: {filename}")
    
    def generate_full_report(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Traffic Analytics...\n")
        
        if not self.load_data():
            return
        
        self.calculate_statistics()
        self.plot_traffic_trends()
        self.plot_heatmap()
        self.export_summary_report()
        
        print("\n✅ Analytics complete! Check the generated files.")


def main():
    """Main execution function"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        🚦 NETRA TRAFFIC ANALYTICS DASHBOARD 🚦            ║
    ║     Network Enabled Traffic Regulation & Analysis         ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize analytics
    analyzer = TrafficAnalytics(data_folder="data/traffic_logs")
    
    # Generate full report
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()
