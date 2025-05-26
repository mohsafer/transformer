import numpy as np
import matplotlib.pyplot as plt
import torch
from solver import Solver, get_divergence_fn
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl

class BifurcationAnalysis:
    def __init__(self, base_config):
        """
        Initialize bifurcation analysis with base configuration
        """
        self.base_config = base_config
        self.results = {}
        
    def analyze_parameter_bifurcation(self, param_name, param_range, divergence_types=['tsallis', 'kl']):
        """
        Analyze bifurcation behavior by varying a specific parameter
        
        Args:
            param_name: Parameter to vary (e.g., 'q_param', 'lr', 'temperature')
            param_range: Range of parameter values to test
            divergence_types: List of divergence types to analyze
        """
        
        for divergence in divergence_types:
            print(f"\n=== Analyzing {divergence.upper()} Divergence ===")
            
            # Store results for this divergence
            self.results[divergence] = {
                'param_values': [],
                'final_losses': [],
                'convergence_rates': [],
                'stability_metrics': [],
                'anomaly_scores': []
            }
            
            for param_value in tqdm(param_range, desc=f"{divergence} bifurcation"):
                # Create modified config
                config = self.base_config.copy()
                config['divergence'] = divergence
                config[param_name] = param_value
                
                # Run analysis for this parameter value
                metrics = self._analyze_single_point(config)
                
                # Store results
                self.results[divergence]['param_values'].append(param_value)
                self.results[divergence]['final_losses'].append(metrics['final_loss'])
                self.results[divergence]['convergence_rates'].append(metrics['convergence_rate'])
                self.results[divergence]['stability_metrics'].append(metrics['stability'])
                self.results[divergence]['anomaly_scores'].append(metrics['anomaly_score_std'])
    
    def _analyze_single_point(self, config):
        """
        Analyze a single parameter point
        """
        try:
            # Initialize solver with current config
            solver = Solver(config)
            
            # Run short training to get dynamics
            train_losses = self._run_short_training(solver)
            
            # Calculate metrics
            final_loss = train_losses[-1] if train_losses else float('inf')
            convergence_rate = self._calculate_convergence_rate(train_losses)
            stability = self._calculate_stability(train_losses)
            
            # Get anomaly score variability (if possible)
            anomaly_score_std = self._get_anomaly_score_variability(solver)
            
            return {
                'final_loss': final_loss,
                'convergence_rate': convergence_rate,
                'stability': stability,
                'anomaly_score_std': anomaly_score_std
            }
            
        except Exception as e:
            print(f"Error at parameter value {config.get('q_param', 'unknown')}: {e}")
            return {
                'final_loss': float('inf'),
                'convergence_rate': 0,
                'stability': float('inf'),
                'anomaly_score_std': 0
            }
    
    def _run_short_training(self, solver, max_epochs=10):
        """
        Run abbreviated training to observe dynamics
        """
        solver.model.train()
        train_losses = []
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Limit to first few batches for speed
            for i, (input_data, labels) in enumerate(solver.train_loader):
                if i >= 5:  # Only first 5 batches
                    break
                    
                solver.optimizer.zero_grad()
                input = input_data.float().to(solver.device)
                series, prior = solver.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    series_loss += (torch.mean(solver.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   solver.win_size)).detach())) + torch.mean(
                        solver.divergence_fn((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           solver.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(solver.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            solver.win_size)),
                        series[u].detach())) + torch.mean(
                        solver.divergence_fn(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   solver.win_size)))))
                
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                loss = prior_loss - series_loss
                
                loss.backward()
                solver.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                train_losses.append(epoch_loss / batch_count)
        
        return train_losses
    
    def _calculate_convergence_rate(self, losses):
        """Calculate convergence rate from loss trajectory"""
        if len(losses) < 3:
            return 0
        
        # Calculate rate of change in final epochs
        final_losses = losses[-3:]
        if len(final_losses) > 1:
            return abs(final_losses[-1] - final_losses[0]) / len(final_losses)
        return 0
    
    def _calculate_stability(self, losses):
        """Calculate stability metric (variance in final losses)"""
        if len(losses) < 3:
            return float('inf')
        
        final_losses = losses[-3:]
        return np.var(final_losses)
    
    def _get_anomaly_score_variability(self, solver):
        """Get variability in anomaly scores"""
        try:
            solver.model.eval()
            anomaly_scores = []
            
            # Get scores from a few test batches
            for i, (input_data, labels) in enumerate(solver.test_loader):
                if i >= 3:  # Only first 3 batches
                    break
                    
                input = input_data.float().to(solver.device)
                series, prior = solver.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    series_loss += solver.divergence_fn(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   solver.win_size)).detach()) * 50
                    prior_loss += solver.divergence_fn(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                solver.win_size)),
                        series[u].detach()) * 50
                
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                anomaly_scores.extend(metric.detach().cpu().numpy())
            
            # Return standard deviation of scores as variability measure
            return np.std(anomaly_scores)
        
        except Exception as e:
            print(f"Error calculating anomaly score variability: {e}")
            return 0
    
    def plot_bifurcation_diagrams(self, param_name, save_path='bifurcation_diagrams.png'):
        """
        Plot IEEE-style bifurcation diagrams for all analyzed divergences
        """
        # Set IEEE style
        set_ieee_style()
        
        # Create figure with proper aspect ratio for IEEE papers
        fig = plt.figure(figsize=(7.5, 6))  # IEEE double column width: ~7.5 inches
        
        # Create subplots with proper spacing
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, 
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
        # IEEE-style colors (colorblind friendly)
        colors = {
            'tsallis': '#1f77b4',  # Blue
            'kl': '#d62728'        # Red
        }
        
        # IEEE-style markers
        markers = {
            'tsallis': 'o',  # Circle
            'kl': 's'        # Square
        }
        
        # Plot 1: Final Loss vs Parameter
        ax1 = fig.add_subplot(gs[0, 0])
        for divergence in self.results:
            params = self.results[divergence]['param_values']
            losses = self.results[divergence]['final_losses']
            ax1.plot(params, losses, marker=markers[divergence], color=colors[divergence], 
                    label=f'{divergence.capitalize()} divergence', linewidth=1.5, 
                    markersize=4, markerfacecolor='white', markeredgewidth=1.2,
                    markeredgecolor=colors[divergence])
    
        ax1.set_xlabel(f'Parameter {param_name}')
        ax1.set_ylabel('Final loss')
        ax1.set_title('(a) Final loss vs. parameter', loc='left', fontweight='normal')
        ax1.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.tick_params(direction='in', which='both')
        
        # Plot 2: Convergence Rate vs Parameter
        ax2 = fig.add_subplot(gs[0, 1])
        for divergence in self.results:
            params = self.results[divergence]['param_values']
            conv_rates = self.results[divergence]['convergence_rates']
            ax2.plot(params, conv_rates, marker=markers[divergence], color=colors[divergence], 
                    label=f'{divergence.capitalize()} divergence', linewidth=1.5,
                    markersize=4, markerfacecolor='white', markeredgewidth=1.2,
                    markeredgecolor=colors[divergence])
    
        ax2.set_xlabel(f'Parameter {param_name}')
        ax2.set_ylabel('Convergence rate')
        ax2.set_title('(b) Convergence rate vs. parameter', loc='left', fontweight='normal')
        ax2.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.tick_params(direction='in', which='both')
        
        # Plot 3: Stability vs Parameter
        ax3 = fig.add_subplot(gs[1, 0])
        for divergence in self.results:
            params = self.results[divergence]['param_values']
            stability = self.results[divergence]['stability_metrics']
            ax3.semilogy(params, stability, marker=markers[divergence], color=colors[divergence], 
                        label=f'{divergence.capitalize()} divergence', linewidth=1.5,
                        markersize=4, markerfacecolor='white', markeredgewidth=1.2,
                        markeredgecolor=colors[divergence])
    
        ax3.set_xlabel(f'Parameter {param_name}')
        ax3.set_ylabel('Stability (loss variance)')
        ax3.set_title('(c) Stability vs. parameter', loc='left', fontweight='normal')
        ax3.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
        ax3.tick_params(direction='in', which='both')
        
        # Plot 4: Anomaly Score Variability vs Parameter
        ax4 = fig.add_subplot(gs[1, 1])
        for divergence in self.results:
            params = self.results[divergence]['param_values']
            anomaly_std = self.results[divergence]['anomaly_scores']
            ax4.plot(params, anomaly_std, marker=markers[divergence], color=colors[divergence], 
                    label=f'{divergence.capitalize()} divergence', linewidth=1.5,
                    markersize=4, markerfacecolor='white', markeredgewidth=1.2,
                    markeredgecolor=colors[divergence])
    
        ax4.set_xlabel(f'Parameter {param_name}')
        ax4.set_ylabel('Anomaly score std. dev.')
        ax4.set_title('(d) Anomaly score variability vs. parameter', loc='left', fontweight='normal')
        ax4.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax4.tick_params(direction='in', which='both')
        
        # Save in multiple formats for IEEE submission
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        print(f"IEEE-style bifurcation diagrams saved to:")
        print(f"  - {save_path.replace('.png', '.pdf')} (PDF)")
        print(f"  - {save_path.replace('.png', '.eps')} (EPS)")
        print(f"  - {save_path} (PNG)")
        
        plt.show()
    
    def plot_phase_portraits(self, param_name, save_path='phase_portraits.png'):
        """
        Plot IEEE-style phase portraits showing system dynamics
        """
        # Set IEEE style
        set_ieee_style()
        
        # Create figure with IEEE double column width
        fig = plt.figure(figsize=(7.5, 3.5))
        
        # Create subplots with proper spacing
        gs = fig.add_gridspec(1, 2, hspace=0.25, wspace=0.3, 
                         left=0.08, right=0.92, top=0.88, bottom=0.15)
    
        # IEEE colormap
        cmap = plt.cm.viridis
        
        divergence_list = list(self.results.keys())
        
        for i, divergence in enumerate(divergence_list):
            ax = fig.add_subplot(gs[0, i])
            
            params = np.array(self.results[divergence]['param_values'])
            losses = np.array(self.results[divergence]['final_losses'])
            conv_rates = np.array(self.results[divergence]['convergence_rates'])
            
            # Create phase portrait with proper normalization
            scatter = ax.scatter(losses, conv_rates, c=params, cmap=cmap, 
                               s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Final loss')
            ax.set_ylabel('Convergence rate')
            ax.set_title(f'({chr(97+i)}) {divergence.capitalize()} divergence phase portrait', 
                        loc='left', fontweight='normal')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(direction='in', which='both')
            
            # Add colorbar with proper formatting
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(f'Parameter {param_name}', rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=8, direction='in')
        
        # Save in multiple formats
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        print(f"IEEE-style phase portraits saved to:")
        print(f"  - {save_path.replace('.png', '.pdf')} (PDF)")
        print(f"  - {save_path.replace('.png', '.eps')} (EPS)")
        print(f"  - {save_path} (PNG)")
        
        plt.show()
    
    def plot_comparative_analysis(self, param_name, save_path='comparative_analysis.png'):
        """
        Plot additional IEEE-style comparative analysis
        """
        set_ieee_style()
        
        # Create single column figure for detailed comparison
        fig = plt.figure(figsize=(3.5, 8))  # IEEE single column width
        
        gs = fig.add_gridspec(4, 1, hspace=0.4, left=0.15, right=0.95, top=0.95, bottom=0.08)
        
        colors = {'tsallis': '#1f77b4', 'kl': '#d62728'}
        linestyles = {'tsallis': '-', 'kl': '--'}
        
        metrics = ['final_losses', 'convergence_rates', 'stability_metrics', 'anomaly_scores']
        titles = ['(a) Final loss', '(b) Convergence rate', '(c) Stability', '(d) Anomaly score std.']
        ylabels = ['Final loss', 'Convergence rate', 'Loss variance', 'Anomaly score std.']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = fig.add_subplot(gs[i, 0])
            
            for divergence in self.results:
                params = self.results[divergence]['param_values']
                values = self.results[divergence][metric]
                
                ax.plot(params, values, color=colors[divergence], 
                       linestyle=linestyles[divergence], linewidth=1.5,
                       label=f'{divergence.capitalize()}', marker='o', markersize=3)
            
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc='left', fontweight='normal')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(direction='in', which='both')
            
            if i == 0:  # Add legend only to first subplot
                ax.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9)
            
            if i == 2:  # Use log scale for stability
                ax.set_yscale('log')
            
            if i == len(metrics) - 1:  # Add xlabel only to bottom plot
                ax.set_xlabel(f'Parameter {param_name}')
        
        # Save in multiple formats
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight', 
                    pad_inches=0.1, dpi=300)
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        
        print(f"IEEE-style comparative analysis saved to:")
        print(f"  - {save_path.replace('.png', '.pdf')} (PDF)")
        print(f"  - {save_path.replace('.png', '.eps')} (EPS)")
        print(f"  - {save_path} (PNG)")
        
        plt.show()
    
    def create_publication_table(self):
        """
        Create IEEE-style table summarizing bifurcation analysis results
        """
        import pandas as pd
        
        # Calculate summary statistics
        summary_data = []
        
        for divergence in self.results:
            params = np.array(self.results[divergence]['param_values'])
            losses = np.array(self.results[divergence]['final_losses'])
            conv_rates = np.array(self.results[divergence]['convergence_rates'])
            stability = np.array(self.results[divergence]['stability_metrics'])
            
            # Find critical points (minimum loss, maximum convergence rate)
            min_loss_idx = np.argmin(losses)
            max_conv_idx = np.argmax(conv_rates)
            min_stability_idx = np.argmin(stability)
            
            summary_data.append({
                'Divergence': divergence.capitalize(),
                'Min Loss': f"{losses[min_loss_idx]:.4f}",
                'Param at Min Loss': f"{params[min_loss_idx]:.3f}",
                'Max Conv. Rate': f"{conv_rates[max_conv_idx]:.4f}",
                'Param at Max Conv.': f"{params[max_conv_idx]:.3f}",
                'Min Stability': f"{stability[min_stability_idx]:.2e}",
                'Param at Min Stab.': f"{params[min_stability_idx]:.3f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save as LaTeX table for IEEE papers
        latex_table = df.to_latex(index=False, float_format="%.4f", 
                                 caption="Summary of bifurcation analysis results",
                                 label="tab:bifurcation_summary")
        
        with open('bifurcation_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Also save as CSV
        df.to_csv('bifurcation_summary.csv', index=False)
        
        print("Publication table saved to:")
        print("  - bifurcation_table.tex (LaTeX)")
        print("  - bifurcation_summary.csv (CSV)")
        
        return df

# Set IEEE-style plot parameters
def set_ieee_style():
    """Set matplotlib parameters for IEEE-style academic plots"""
    # Use Times New Roman font (similar to IEEE)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'mathtext.fontset': 'stix'
    })

def run_bifurcation_analysis():
    """
    Run complete bifurcation analysis
    """
    # Base configuration (modify according to your setup)
    base_config = {
        'data_path': 'SMD',  # Replace with your dataset
        'dataset': 'SMD',
        'batch_size': 256,
        'win_size': 99,  # CHANGED: Use 99 instead of 100 (divisible by 3)
        'input_c': 38,  # Replace with your input channels
        'output_c': 38,
        'lr': 0.0001,
        'num_epochs': 5,  # REDUCED: Shorter for faster analysis
        'n_heads': 1,
        'd_model': 256,
        'e_layers': 3,
        'patch_size': [3, 9, 33],  # CHANGED: All divisors of 99
        'model_save_path': './checkpoints',
        'anormly_ratio': 1,
        'loss_fuc': 'MSE',
        'index': 0,
        'q_param': 1.5  # Will be varied
    }
    
    # Initialize bifurcation analysis
    bifurcation = BifurcationAnalysis(base_config)
    

    q_param_range = np.linspace(0.5, 2.5, 10)  # REDUCED: Smaller range, fewer points
    bifurcation.analyze_parameter_bifurcation('q_param', q_param_range, 
                                             ['tsallis', 'kl'])
    

    bifurcation.plot_bifurcation_diagrams('q_param', 'q_param_bifurcation.png')
    bifurcation.plot_phase_portraits('q_param', 'q_param_phase_portraits.png')
    
    return bifurcation

if __name__ == "__main__":

    analysis = run_bifurcation_analysis()


    analysis.plot_bifurcation_diagrams('q_param', 'ieee_bifurcation.png')
    analysis.plot_phase_portraits('q_param', 'ieee_phase_portraits.png')
    analysis.plot_comparative_analysis('q_param', 'ieee_comparative.png')

    
    summary_table = analysis.create_publication_table()