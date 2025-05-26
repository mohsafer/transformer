# quick_plot_simple.py
import numpy as np
import matplotlib.pyplot as plt
from data_factory.data_loader import SMDSegLoader

def simple_plot():
    # Just load the data directly without creating a Solver
    data_path = 'SMD'
    
    # Load the data loader to get TS
    loader = SMDSegLoader('dataset/'+data_path, win_size=100, step=10)
    TS = loader.TS
    
    # Create dummy data for demonstration
    test_energy = np.random.random(50000)
    gt = np.random.choice([0, 1], size=50000, p=[0.9, 0.1])
    thresh = 0.2
    
    # Extract segments
    start_idx = 0
    segment_length = 25000
    
    def extract_segment(data, length, start):
        return data[start:start+length]
    
    as_segment = extract_segment(test_energy, segment_length, start_idx)
    gt_segment = extract_segment(gt, segment_length, start_idx)
    TS_segment = extract_segment(TS, segment_length, start_idx)
    
    # Plot
    def smooth(y, box_pts=10):
        box = np.ones(box_pts)/box_pts
        return np.convolve(y, box, mode='same')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(smooth(TS_segment), label="Time Series Data", color='black', linewidth=0.3)
    plt.title("Time Series Plot")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(smooth(as_segment), label='Anomaly Scores', color='blue', linewidth=0.1)
    plt.axhline(y=thresh, color='red', linestyle='--', label='Threshold', linewidth=0.5)
    plt.fill_between(range(len(as_segment)), 0, 1, where=(gt_segment == 1), 
                     color='blue', alpha=0.2, label='Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title(f'Anomaly Scores Over Time (Area{start_idx})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('simple_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved to simple_plot.png")

if __name__ == "__main__":
    simple_plot()