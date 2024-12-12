import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from PCG import pcg_solver
from RPCholesky import rpcholesky
from HelperFunctions.gallery import smile, robspiral
from tqdm import tqdm

def run_experiment(matrix_name: str, A: Any, ranks: List[int], 
                  num_instances: int, threshold: float, mu: float) -> Dict:
    """
    Run experiment comparing Nyström PCG with RPCholesky PCG
    """
    n = A.shape[0]
    results = {
        'nystrom': {r: {'iters': []} for r in ranks},
        'rpcholesky': {r: {'iters': []} for r in ranks}
    }
    
    # Set random seed for reproducibility
    np.random.seed(101)
    
    for rank in ranks: 
        print(f"Processing rank {rank} for {matrix_name}")
        
        for instance in tqdm(range(num_instances), desc=f"Rank {rank} Instances", position=1, leave=False):
            # Generate random right-hand side vector
            b = np.random.randn(n)
            x0 = np.zeros(n)
            
            # Run Nyström PCG
            x_nys, nys_iters, nys_error = pcg_solver(A, b, x0, mu, rank, 'nystrom', threshold)
            
            # Run RPCholesky PCG
            x_rp, rp_iters, rp_error = pcg_solver(A, b, x0, mu, rank, 'rpcholesky', threshold)
            
            # Store results
            results['nystrom'][rank]['iters'].append(nys_iters)
            results['rpcholesky'][rank]['iters'].append(rp_iters)

    return results

def plot_boxplot(results: Dict, ranks: List[int], title: str, output_dir: str = 'plots'):
    """
    Create boxplot comparing the iteration distributions and save as PNG
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Prepare data for boxplot
    iters_data_nystrom = [results['nystrom'][rank]['iters'] for rank in ranks]
    iters_data_rpcholesky = [results['rpcholesky'][rank]['iters'] for rank in ranks]
    
    # Position for boxes
    positions_nystrom = np.arange(len(ranks)) * 2.5
    positions_rpcholesky = positions_nystrom + 0.8
    
    # Plot iterations boxplot
    plt.boxplot(iters_data_nystrom, positions=positions_nystrom, widths=0.6,
               patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red'), labels=[f'l={r}' for r in ranks])
    plt.boxplot(iters_data_rpcholesky, positions=positions_rpcholesky, widths=0.6,
               patch_artist=True, boxprops=dict(facecolor='lightgreen', color='black'),
               medianprops=dict(color='red'), labels=[f'l={r}' for r in ranks])
    
    # Customize plot
    plt.xlabel('Rank')
    plt.ylabel('Number of Iterations')
    plt.title(f'Iterations Distribution - {title}')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Nyström'),
        Patch(facecolor='lightgreen', edgecolor='black', label='RPCholesky')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot as PNG with high DPI
    filename = os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_boxplot.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {filename}")

def print_summary_statistics(matrix_name: str, results: Dict, ranks: List[int]):
    """Print summary statistics for a single experiment"""
    print(f"\n{matrix_name} Matrix:")
    for method in ['nystrom', 'rpcholesky']:
        print(f"\n{method.capitalize()} method:")
        for rank in ranks:
            avg_iters = np.mean(results[method][rank]['iters'])
            print(f"Rank {rank}: Avg iterations = {avg_iters:.1f}")

def plot_histogram(results: Dict, ranks: List[int], title: str, output_dir: str = 'plots'):
    """
    Create histogram comparing average iterations between methods and save as PNG
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Calculate averages for each rank
    nystrom_avgs = [np.mean(results['nystrom'][rank]['iters']) for rank in ranks]
    rpcholesky_avgs = [np.mean(results['rpcholesky'][rank]['iters']) for rank in ranks]
    
    # Set up bar positions
    x = np.arange(len(ranks))
    width = 0.35  # Width of the bars
    
    # Create bars
    plt.bar(x - width/2, nystrom_avgs, width, label='Nyström', color='lightblue', edgecolor='black')
    plt.bar(x + width/2, rpcholesky_avgs, width, label='RPCholesky', color='lightgreen', edgecolor='black')
    
    # Customize plot
    plt.xlabel('Rank')
    plt.ylabel('Average Number of Iterations')
    plt.title(f'Average Iterations Comparison - {title}')
    plt.xticks(x, [f'l={r}' for r in ranks])
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(nystrom_avgs):
        plt.text(i - width/2, v + 1, f'{v:.1f}', ha='center', va='bottom')
    for i, v in enumerate(rpcholesky_avgs):
        plt.text(i + width/2, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot as PNG
    filename = os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_histogram.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved as {filename}")

def run_smile_experiment():
    """Run experiment for Smile matrix"""
    # Parameters
    n = 100
    ranks = [50, 100] # [50, 100, 150, 200, 250]
    num_instances = 50
    threshold = 0.0001
    mu = 100
    
    print("Generating Smile matrix...")
    A_smile = smile(n, bandwidth=2.0)
    
    print("\nRunning experiments for Smile matrix...")
    results_smile = run_experiment('Smile', A_smile, ranks, num_instances, threshold, mu)
    
    print("\nPlotting results...")
    plot_boxplot(results_smile, ranks, 'Smile Matrix') # Add boxplot
    plot_histogram(results_smile, ranks, 'Smile Matrix')  # Add histogram
    
    print("\nSummary Statistics:")
    print_summary_statistics("Smile", results_smile, ranks)
    
    return results_smile

def run_spiral_experiment():
    """Run experiment for Spiral matrix"""
    # Parameters
    n = 100
    ranks = [50, 100, 150]
    num_instances = 50
    threshold = 0.0001
    mu = 10
    
    print("Generating Spiral matrix...")
    A_spiral = robspiral(n)
    
    print("\nRunning experiments for Spiral matrix...")
    results_spiral = run_experiment('Spiral', A_spiral, ranks, num_instances, threshold, mu)
    
    print("\nPlotting results...")
    plot_boxplot(results_spiral, ranks, 'Spiral Matrix')
    plot_histogram(results_spiral, ranks, 'Spiral Matrix')  # Add histogram
    
    print("\nSummary Statistics:")
    print_summary_statistics("Spiral", results_spiral, ranks)
    
    return results_spiral

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2 or sys.argv[1] not in ['smile', 'spiral', 'both']:
        print("Usage: python script.py [smile|spiral|both]")
        sys.exit(1)
        
    if sys.argv[1] == 'smile' or sys.argv[1] == 'both':
        results_smile = run_smile_experiment()
        
    if sys.argv[1] == 'spiral' or sys.argv[1] == 'both':
        results_spiral = run_spiral_experiment()