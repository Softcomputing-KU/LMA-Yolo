import pandas as pd
import matplotlib.pyplot as plt

def smooth_curve(values, window_size):
    """Smooths the curve using a moving average."""
    smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
    return smoothed

# Set the file path
file_path = '/root/autodl-tmp/ultralytics-main/result/results-mult.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Sort the data by recall values to ensure a smooth curve
data = data.sort_values(by='      metrics/recall(B)')

# Apply smoothing
window_size = 5  # Adjust the window size as needed
smoothed_precision = smooth_curve(data['   metrics/precision(B)'], window_size)

# Plot Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(data['      metrics/recall(B)'], smoothed_precision, label='Smoothed Precision-Recall Curve')

# Configure plot
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid()

# Save the plot to the current directory
plt.savefig('precision_recall_curve_smoothed.png', dpi=300, bbox_inches='tight')

# Optionally, close the plot to free memory
plt.close()
