import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os
import torch
import numpy as np
from data_factory.data_loader import SMDSegLoader

# Set IEEE style plotting
plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

class AnomalyPlotter:
    def __init__(self, solver):
        """
        Initialize plotter with solver instance to access model and data
        """
        self.solver = solver
        self.device = solver.device
        self.model = solver.model
        self.data_path = solver.data_path
        self.win_size = solver.win_size
        self.divergence = solver.divergence
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        os.makedirs(f'plots/{self.data_path}_{self.divergence}', exist_ok=True)
    
    def smooth(self, y, box_pts=1):
        """Smooth function matching your codebase"""
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def extract_model_outputs(self, segment_length=1000, start_idx=None):
        """
        Extract y_true, y_pred, anomaly_scores, and labels from your model
        """
        self.model.eval()
        temperature = 50
        
        # Get time series data
        loader = SMDSegLoader(f'dataset/{self.data_path}', win_size=self.win_size, step=10)
        TS = loader.TS
        
        # Extract test data and compute anomaly scores
        test_labels = []
        anomaly_scores = []
        reconstructions = []
        true_values = []
        
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.solver.test_loader):
                if i >= 10:  # Limit to first 10 batches for visualization
                    break
                    
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                # Compute anomaly scores using your divergence function
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = self.solver.divergence_fn(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.solver.win_size)).detach()) * temperature
                        prior_loss = self.solver.divergence_fn(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.solver.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += self.solver.divergence_fn(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.solver.win_size)).detach()) * temperature
                        prior_loss += self.solver.divergence_fn(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.solver.win_size)),
                            series[u].detach()) * temperature
                
                # Compute anomaly score
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                anomaly_score = metric.detach().cpu().numpy()
                
                # Store data
                anomaly_scores.append(anomaly_score)
                test_labels.append(labels.numpy())
                
                # For reconstruction, we'll use the series output as "predicted" values
                # and input data as "true" values
                reconstructions.append(series[0].detach().cpu().numpy())  # Use first scale
                true_values.append(input_data.numpy())
        
        # Concatenate all batches
        anomaly_scores = np.concatenate(anomaly_scores, axis=0).flatten()
        test_labels = np.concatenate(test_labels, axis=0).flatten()
        reconstructions = np.concatenate(reconstructions, axis=0)
        true_values = np.concatenate(true_values, axis=0)
        
        # Extract segment if specified
        if start_idx is None:
            start_idx = 0
        
        end_idx = min(start_idx + segment_length, len(anomaly_scores))
        
        return {
            'y_true': true_values[start_idx:end_idx],
            'y_pred': reconstructions[start_idx:end_idx], 
            'ascore': anomaly_scores[start_idx:end_idx],
            'labels': test_labels[start_idx:end_idx],
            'TS': TS[start_idx:end_idx] if len(TS) > end_idx else TS[:segment_length]
        }
    
    def plotter(self, name=None, segment_length=1000, start_idx=None):
        """
        Create plots matching the original plotter function but using your model data
        """
        if name is None:
            name = f"{self.data_path}_{self.divergence}"
        
        # Extract data from your model
        data = self.extract_model_outputs(segment_length, start_idx)
        y_true = data['y_true']
        y_pred = data['y_pred'] 
        ascore = data['ascore']
        labels = data['labels']
        
        # Create output directory
        plot_dir = f'plots/{name}'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create PDF for multi-page output
        pdf = PdfPages(f'{plot_dir}/output.pdf')
        
        # Plot each dimension/channel
        num_dims = min(y_true.shape[-1], 5)  # Limit to 5 dimensions for readability
        
        for dim in range(num_dims):
            if len(y_true.shape) == 3:  # [batch, seq_len, features]
                y_t = y_true[:, :, dim].flatten()
                y_p = y_pred[:, :, dim].flatten()
            else:  # [batch, features] 
                y_t = y_true[:, dim]
                y_p = y_pred[:, dim]
            
            # Ensure labels and ascore match the length
            min_len = min(len(y_t), len(y_p), len(labels), len(ascore))
            y_t = y_t[:min_len]
            y_p = y_p[:min_len]
            l = labels[:min_len] 
            a_s = ascore[:min_len]
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            
            # Top subplot: True vs Predicted values
            ax1.set_ylabel('Value')
            ax1.set_title(f'{name} - Dimension = {dim}')
            
            # Plot smoothed true and predicted values
            ax1.plot(self.smooth(y_t, box_pts=3), linewidth=0.8, label='True', color='blue')
            ax1.plot(self.smooth(y_p, box_pts=3), '-', alpha=0.7, linewidth=0.8, 
                    label='Predicted', color='orange')
            
            # Add ground truth anomalies on secondary y-axis
            ax3 = ax1.twinx()
            ax3.plot(l, '--', linewidth=0.5, alpha=0.6, color='red')
            ax3.fill_between(np.arange(len(l)), l, color='red', alpha=0.2, label='Ground Truth')
            ax3.set_ylabel('Anomaly Label')
            
            if dim == 0: 
                ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
                ax3.legend(bbox_to_anchor=(1.0, 1.02))
            
            # Bottom subplot: Anomaly scores
            ax2.plot(self.smooth(a_s, box_pts=3), linewidth=0.8, color='green', label='Anomaly Score')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Anomaly Score')
            ax2.legend()
            
            # Add threshold line if available
            if hasattr(self.solver, 'threshold'):
                ax2.axhline(y=self.solver.threshold, color='red', linestyle='--', 
                           alpha=0.7, label='Threshold')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Save individual dimension data for analysis
            if dim == 0:
                np.save(f'{plot_dir}/true_dim{dim}.npy', y_t)
                np.save(f'{plot_dir}/pred_dim{dim}.npy', y_p) 
                np.save(f'{plot_dir}/ascore_dim{dim}.npy', a_s)
                np.save(f'{plot_dir}/labels_dim{dim}.npy', l)
        
        pdf.close()
        print(f"Plots saved to {plot_dir}/output.pdf")
        
        return {
            'plot_dir': plot_dir,
            'num_dimensions': num_dims,
            'data_length': min_len
        }
    
    def create_summary_plot(self, segment_length=1000, start_idx=None):
        """
        Create a summary plot similar to your existing test() method visualization
        """
        data = self.extract_model_outputs(segment_length, start_idx)
        
        # Use time series data if available
        if 'TS' in data and len(data['TS']) > 0:
            ts_data = data['TS']
        else:
            ts_data = data['y_true'][:, 0] if len(data['y_true'].shape) > 1 else data['y_true']
        
        ascore = data['ascore']
        labels = data['labels']
        
        # Calculate threshold
        thresh = np.percentile(ascore, 100 - self.solver.anormly_ratio)
        
        plt.figure(figsize=(12, 8))
        
        # Top subplot: Time Series
        plt.subplot(2, 1, 1)
        plt.plot(self.smooth(ts_data, box_pts=10), label="Time Series Data", color='lightblue')
        plt.title("Time Series Plot")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        
        # Bottom subplot: Anomaly Scores
        plt.subplot(2, 1, 2)
        plt.plot(self.smooth(ascore, box_pts=3), label='Anomaly Scores', color='green')
        plt.axhline(y=thresh, color='red', linestyle='--', label='Threshold')
        plt.fill_between(range(len(ascore)), 0, 1, where=(labels == 1), 
                        color='red', alpha=0.2, label='Ground Truth')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.title(f'Anomaly Scores Over Time ({self.divergence} divergence)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'plots/{self.data_path}_{self.divergence}/summary_plot.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to {plot_filename}")
        plt.show()
        
        return {
            'threshold': thresh,
            'plot_filename': plot_filename
        }

# Integration with your Solver class
def add_plotter_to_solver(solver_instance):
    """
    Add plotting functionality to your existing Solver instance
    """
    # Create plotter instance
    plotter = AnomalyPlotter(solver_instance)
    
    # Add methods to solver
    solver_instance.create_anomaly_plots = plotter.plotter
    solver_instance.create_summary_plot = plotter.create_summary_plot
    solver_instance.plotter_instance = plotter
    
    return solver_instance

# Usage example:
def create_plots_from_solver(solver):
    """
    Example usage with your existing solver
    """
    # Add plotting functionality
    enhanced_solver = add_plotter_to_solver(solver)
    
    # Create detailed multi-dimensional plots
    plot_results = enhanced_solver.create_anomaly_plots(
        name=f"{solver.data_path}_{solver.divergence}",
        segment_length=2000,
        start_idx=0
    )
    
    # Create summary plot
    summary_results = enhanced_solver.create_summary_plot(
        segment_length=2000,
        start_idx=0
    )
    
    return plot_results, summary_results

# Add this method to your Solver class:
def enhanced_test_with_plots(self):
    """
    Enhanced version of your test method that includes the plotting functionality
    """
    # Run your original test method
    accuracy, precision, recall, f_score = self.test()
    
    # Create enhanced plots
    plotter = AnomalyPlotter(self)
    
    # Generate comprehensive plots
    plot_results = plotter.plotter(
        name=f"{self.data_path}_{self.divergence}_detailed",
        segment_length=5000,
        start_idx=0
    )
    
    # Generate summary plot
    summary_results = plotter.create_summary_plot(
        segment_length=5000,
        start_idx=0
    )
    
    print(f"Enhanced plots created in: {plot_results['plot_dir']}")
    
    return accuracy, precision, recall, f_score, plot_results, summary_results
