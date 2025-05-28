import pandapower as pp
import pandapower.networks as pn
import numpy as np
import torch
import itertools
import time
import datetime
import csv
import os
import copy
import sys
from tqdm import tqdm
import math

from PowerGridEnv import PowerGridEnv
from Gnn import GNN
from Train import obs_to_data
from vulnerability_finder import VulnerabilityFinder, find_model_top_vulnerabilities

def get_available_cases():
    """Return a list of available pandapower case names, ordered by size"""
    # Standard cases
    standard_cases = ["case14", "case30", "case39", "case57", "case89pegase", "case118", "case300", "case1354pegase", "case1888rte", "case2848rte"]
    
    # Filter to only include cases that are available
    available_cases = []
    for case in standard_cases:
        try:
            net = getattr(pn, case)()
            n_lines = len(net.line)
            print(f"Found {case} with {n_lines} lines")
            available_cases.append(case)
        except:
            print(f"Case {case} not available, skipping")
    
    return available_cases

def run_benchmark(cases, model_path, k=3, top_n=10):
    """
    Run vulnerability analysis benchmark across multiple cases
    
    Args:
        cases: List of case names to test
        model_path: Path to trained model
        k: Maximum number of lines to remove
        top_n: Number of top vulnerabilities to log
    """
    # Create timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results/vulnerability_benchmark_{timestamp}.csv"
    log_file = f"results/vulnerability_benchmark_{timestamp}.log"
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    # Create CSV file with headers
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Case', 'Lines', 'Method', 'Duration', 
            'Rank', 'Score', 'Status', 'Removed_Lines'
        ])
    
    # Create log file
    with open(log_file, 'w') as f:
        f.write(f"Power Grid Vulnerability Benchmark\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Max lines removed (k): {k}\n\n")
    
    # Process each case
    for case_name in cases:
        print(f"\n{'='*50}")
        print(f"Processing {case_name}")
        print(f"{'='*50}")
        
        try:
            # Load case to get info
            net = getattr(pn, case_name)()
            n_lines = len(net.line)
            n_buses = len(net.bus)
            
            log_case_info(case_name, n_lines, n_buses, log_file)
            
            # Run both methods and log results
            results = {}
            
            # Run exhaustive search for all cases (no limits)
            results["exhaustive"] = run_exhaustive(case_name, k, log_file, csv_file)
            
            # Always run model-based search
            results["model"] = run_model_based(model_path, case_name, k, log_file, csv_file)
            
            # Compare results if both methods were run
            if results["exhaustive"] and results["model"]:
                compare_results(results["exhaustive"], results["model"], case_name, k, log_file)
            
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"\nERROR processing {case_name}: {str(e)}\n")
                f.write(f"Traceback: {sys.exc_info()}\n\n")
    
    print(f"\nBenchmark complete.")
    print(f"Results saved to {csv_file}")
    print(f"Detailed log saved to {log_file}")
    
    return csv_file, log_file

def log_case_info(case_name, n_lines, n_buses, log_file):
    """Log information about the case to the log file"""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"CASE: {case_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Network information:\n")
        f.write(f"  Buses: {n_buses}\n")
        f.write(f"  Lines: {n_lines}\n")
        
        # Estimate combinations
        # Use scipy.special.comb or standard math formula
        total_combos = 0
        for i in range(1, 4):
            # Calculate binomial coefficient n choose k
            comb = math.comb(n_lines, i) if hasattr(math, 'comb') else math.factorial(n_lines) // (math.factorial(i) * math.factorial(n_lines - i))
            total_combos += comb
            
        f.write(f"  Possible combinations (k=3): {total_combos:,}\n\n")

def run_exhaustive(case_name, k, log_file, csv_file):
    """Run exhaustive search and log results"""
    print(f"Running exhaustive search for {case_name}...")
    
    try:
        with open(log_file, 'a') as f:
            f.write("EXHAUSTIVE SEARCH:\n")
        
        # Run search with timing
        start_time = time.time()
        finder = VulnerabilityFinder(case_name, k=k)
        vulnerabilities = find_unique_vulnerabilities(finder)  # Use the new optimized function
        duration = time.time() - start_time
        
        # Log results to log file
        with open(log_file, 'a') as f:
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Found {len(vulnerabilities):,} total vulnerabilities\n\n")
            
            # Log top vulnerabilities
            f.write(f"Top {min(10, len(vulnerabilities))} vulnerabilities:\n")
            f.write("-" * 80 + "\n")
            
            for i, v in enumerate(vulnerabilities[:10]):
                status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                lines_str = ", ".join(map(str, v['removed_lines']))
                
                f.write(f"{i+1}. Score: {v['score']:.2f} - Status: {status}\n")
                f.write(f"   Lines removed: {lines_str}\n")
                f.write("-" * 80 + "\n")
        
        # Log results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Log duration in one row
            writer.writerow([
                case_name,
                len(finder.lines),
                'exhaustive',
                f"{duration:.2f}",
                '', '', '', ''
            ])
            
            # Log each vulnerability
            for i, v in enumerate(vulnerabilities[:10]):
                status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                lines_str = ",".join(map(str, v['removed_lines']))
                
                writer.writerow([
                    case_name,
                    len(finder.lines),
                    'exhaustive',
                    '',
                    i+1,
                    f"{v['score']:.2f}",
                    status,
                    lines_str
                ])
        
        print(f"Exhaustive search completed in {duration:.2f} seconds")
        return vulnerabilities[:10]
        
    except Exception as e:
        print(f"Error in exhaustive search for {case_name}: {e}")
        with open(log_file, 'a') as f:
            f.write(f"ERROR in exhaustive search: {str(e)}\n\n")
        return []

def find_unique_vulnerabilities(finder):
    """
    Find vulnerabilities without redundant subsets that already cause collapse
    """
    results = []
    minimal_collapse_sets = []
    
    # Generate all combinations of line removals up to k
    all_combinations = []
    for i in range(1, finder.k + 1):
        combinations_i = list(itertools.combinations(finder.lines, i))
        all_combinations.extend(combinations_i)
        print(f"Generated {len(combinations_i):,} combinations of {i} lines")
    
    # Sort combinations by length (evaluate smaller ones first)
    all_combinations.sort(key=len)
    total = len(all_combinations)
    print(f"Testing {total:,} combinations of line removals for {finder.case_name}")
    
    # Process combinations in order of size
    for combination in tqdm(all_combinations):
        # Convert to set for easier subset checking
        combo_set = frozenset(combination)
        
        # Skip if this is a superset of an already identified collapse set
        if any(collapse_set.issubset(combo_set) for collapse_set in minimal_collapse_sets):
            continue
            
        # Evaluate this combination
        result = finder.evaluate_combination(list(combination))
        
        # If it causes collapse or islanding, remember this as a minimal set
        if result['collapsed'] or result['islanded']:
            minimal_collapse_sets.append(combo_set)
            
        # Add to results if score is non-zero
        if result['score'] > 0.01:
            results.append(result)
    
    # Sort by vulnerability score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(results):,} unique vulnerabilities after filtering")
    
    return results

def run_model_based(model_path, case_name, k, log_file, csv_file):
    """Run model-guided search and log results"""
    print(f"Running model-guided search for {case_name}...")
    
    try:
        with open(log_file, 'a') as f:
            f.write("\nMODEL-GUIDED SEARCH:\n")
        
        # Run search with timing
        start_time = time.time()
        model_vulnerabilities = find_model_top_vulnerabilities(
            model_path, case_name, k, num_vulnerabilities=10
        )
        duration = time.time() - start_time
        
        # Log results to log file
        with open(log_file, 'a') as f:
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Found {len(model_vulnerabilities)} vulnerabilities\n\n")
            
            # Log top vulnerabilities
            f.write(f"Top vulnerabilities:\n")
            f.write("-" * 80 + "\n")
            
            for i, v in enumerate(model_vulnerabilities):
                status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                lines_str = ", ".join(map(str, v['removed_lines']))
                
                f.write(f"{i+1}. Score: {v['score']:.2f} - Status: {status}\n")
                f.write(f"   Lines removed: {lines_str}\n")
                f.write("-" * 80 + "\n")
        
        # Log results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Get number of lines from the first vulnerability result
            net = getattr(pn, case_name)()
            n_lines = len(net.line)
            
            # Log duration in one row
            writer.writerow([
                case_name,
                n_lines,
                'model',
                f"{duration:.2f}",
                '', '', '', ''
            ])
            
            # Log each vulnerability
            for i, v in enumerate(model_vulnerabilities):
                status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                lines_str = ",".join(map(str, v['removed_lines']))
                
                writer.writerow([
                    case_name,
                    n_lines,
                    'model',
                    '',
                    i+1,
                    f"{v['score']:.2f}",
                    status,
                    lines_str
                ])
        
        print(f"Model-based search completed in {duration:.2f} seconds")
        return model_vulnerabilities
        
    except Exception as e:
        print(f"Error in model-based search for {case_name}: {e}")
        with open(log_file, 'a') as f:
            f.write(f"ERROR in model-based search: {str(e)}\n\n")
        return []

def compare_results(exhaustive_vulns, model_vulns, case_name, k, log_file):
    """Compare exhaustive and model results and log findings"""
    try:
        if not exhaustive_vulns or not model_vulns:
            return
            
        # Create sets of line combinations for comparison
        exhaustive_sets = [frozenset(v['removed_lines']) for v in exhaustive_vulns]
        model_sets = [frozenset(v['removed_lines']) for v in model_vulns]
        
        # Find matches
        matches = []
        for i, model_set in enumerate(model_sets):
            for j, exh_set in enumerate(exhaustive_sets):
                if model_set == exh_set:
                    matches.append((i, j, model_vulns[i], exhaustive_vulns[j]))
                    break
        
        # Log comparison results
        with open(log_file, 'a') as f:
            f.write("\nCOMPARISON RESULTS:\n")
            f.write("-" * 80 + "\n")
            
            if matches:
                f.write(f"Model found {len(matches)} of the top {len(exhaustive_vulns)} vulnerabilities:\n\n")
                
                for model_idx, exh_idx, model_v, exh_v in matches:
                    f.write(f"Vulnerability #{exh_idx+1} from exhaustive search (rank #{model_idx+1} in model results)\n")
                    f.write(f"Lines: {model_v['removed_lines']}\n")
                    f.write(f"Exhaustive score: {exh_v['score']:.2f}, Model score: {model_v['score']:.2f}\n")
                    f.write("-" * 40 + "\n")
            else:
                f.write("Model did not find any of the same top vulnerabilities as exhaustive search.\n")
                
            f.write("\n")
    except Exception as e:
        print(f"Error in result comparison for {case_name}: {e}")
        with open(log_file, 'a') as f:
            f.write(f"ERROR in result comparison: {str(e)}\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark vulnerability detection methods across multiple power grid cases")
    parser.add_argument("--model", type=str, default="best_power_grid_model.pt", help="Path to trained model")
    parser.add_argument("--k", type=int, default=3, help="Maximum number of lines to remove")
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated list of cases or 'all'")
    args = parser.parse_args()
    
    # Get cases to test
    if args.cases.lower() == "all":
        cases = get_available_cases()
    else:
        cases = [c.strip() for c in args.cases.split(",")]
    
    # Run benchmark
    run_benchmark(
        cases=cases,
        model_path=args.model,
        k=args.k
    )