import pandapower as pp
import pandapower.networks as pn
import numpy as np
import torch
import itertools
import time
import datetime
import os
import copy
import math
from tqdm import tqdm

from PowerGridEnv import PowerGridEnv
from Gnn import GNN
from Train import obs_to_data

class VulnerabilityFinder:
    def __init__(self, case_name, k=3, vmin=0.95, vmax=1.05, max_loading=1.0):
        self.case_name = case_name
        self.k = k
        self.vmin = vmin
        self.vmax = vmax
        self.max_loading = max_loading
        
        self.orig_net = getattr(pn, case_name)()
        pp.runpp(self.orig_net)
        
        self.lines = list(self.orig_net.line.index)
        self.num_lines = len(self.lines)
        self.num_buses = len(self.orig_net.bus)
        
    def evaluate_combination(self, removed_lines):
        net = copy.deepcopy(self.orig_net)
        
        for line in removed_lines:
            net.line.at[line, 'in_service'] = False
        
        collapsed = False
        islanded = False
        
        try:
            pp.runpp(net)
            if np.isnan(net.res_bus.vm_pu.values).any():
                islanded = True
        except pp.LoadflowNotConverged:
            collapsed = True
            
        bus_v = net.res_bus.vm_pu.values if not collapsed else np.zeros(self.num_buses)
        
        v_deviations = np.zeros_like(bus_v)
        for i, v in enumerate(bus_v):
            if not np.isnan(v):
                if v < self.vmin:
                    v_deviations[i] = (self.vmin - v) / self.vmin
                elif v > self.vmax:
                    v_deviations[i] = (v - self.vmax) / self.vmax
        
        bus_violation = np.sum(v_deviations ** 2) / len(bus_v)
        
        if not collapsed and not islanded:
            loadings = net.res_line.loading_percent.values / 100.0
            l_deviations = np.maximum(0, loadings - self.max_loading)
            line_violation = np.sum(l_deviations ** 2) / len(loadings)
        else:
            line_violation = 0
        
        if collapsed:
            score = 50.0
        elif islanded:
            score = 25.0
        else:
            score = (bus_violation + 1.5 * line_violation) * 10.0
        
        score *= (0.90 ** (len(removed_lines)-1))
        
        return {
            'removed_lines': removed_lines.copy(),
            'score': score,
            'collapsed': collapsed,
            'islanded': islanded,
            'bus_violation': bus_violation,
            'line_violation': line_violation
        }

def find_exhaustive_vulnerabilities(finder):
    results = []
    minimal_collapse_sets = []
    
    all_combinations = []
    for i in range(1, finder.k + 1):
        combinations_i = list(itertools.combinations(finder.lines, i))
        all_combinations.extend(combinations_i)
    
    all_combinations.sort(key=len)
    
    for combination in tqdm(all_combinations):
        combo_set = frozenset(combination)
        
        if any(collapse_set.issubset(combo_set) for collapse_set in minimal_collapse_sets):
            continue
            
        result = finder.evaluate_combination(list(combination))
        
        if result['collapsed'] or result['islanded']:
            minimal_collapse_sets.append(combo_set)
            
        if result['score'] > 0.01:
            results.append(result)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def find_model_vulnerabilities(model_path, case_name, k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = PowerGridEnv(case_list=[case_name], k=k)
    
    model = GNN(node_feat_dim=4, edge_feat_dim=5, hidden_dim=64, n_layers=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    vulnerability_candidates = []
    selected_line_sets = set()
    minimal_collapse_sets = []
    
    initial_obs, _ = env.reset()
    
    def explore_all_paths(obs, lines_removed=None, depth=0):
        if lines_removed is None:
            lines_removed = []
        
        lines_set = frozenset(lines_removed)
        if any(collapse_set.issubset(lines_set) for collapse_set in minimal_collapse_sets):
            return
            
        if depth == k:
            lines_tuple = tuple(sorted(lines_removed))
            
            if lines_tuple in selected_line_sets:
                return
                
            selected_line_sets.add(lines_tuple)
            
            finder = VulnerabilityFinder(case_name, k=k)
            result = finder.evaluate_combination(list(lines_removed))
            
            if result["collapsed"] or result["islanded"]:
                minimal_collapse_sets.append(frozenset(lines_removed))
                
            if result['score'] > 0.01:
                vulnerability_candidates.append(result)
            
            return
        
        if lines_removed:
            lines_tuple = tuple(sorted(lines_removed))
            
            if lines_tuple not in selected_line_sets:
                selected_line_sets.add(lines_tuple)
                
                finder = VulnerabilityFinder(case_name, k=len(lines_removed))
                current_result = finder.evaluate_combination(list(lines_removed))
                
                if current_result["collapsed"] or current_result["islanded"]:
                    minimal_collapse_sets.append(frozenset(lines_removed))
                    
                    if current_result['score'] > 0.01:
                        vulnerability_candidates.append(current_result)
                    
                    return
                
                elif current_result['score'] > 0.01:
                    vulnerability_candidates.append(current_result)
        
        data = obs_to_data(obs, device=device)
        with torch.no_grad():
            logits, _ = model(data)
            mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=device)
            masked_logits = logits.masked_fill(~mask, -1e8)
            
            n_actions = min(5, sum(mask).item()) 
            if n_actions == 0:
                return
                
            top_values, top_indices = torch.topk(masked_logits, n_actions)
            
        for i, action in enumerate(top_indices):
            action = action.item()
            line_id = env.all_line_ids[action]
            
            if line_id in lines_removed:
                continue
                
            try:
                env_copy = copy.deepcopy(env)
                env_copy.load_state(obs)
            except Exception:
                continue
            
            new_lines = lines_removed + [line_id]
            new_lines_set = frozenset(new_lines)
            
            if any(collapse_set.issubset(new_lines_set) for collapse_set in minimal_collapse_sets):
                continue
                
            if tuple(sorted(new_lines)) in selected_line_sets:
                continue
            
            try:
                new_obs, reward, done, _, info = env_copy.step(action)
            except Exception:
                continue
            
            explore_all_paths(new_obs, new_lines, depth + 1)
    
    try:
        explore_all_paths(initial_obs)
    except Exception:
        return []
    
    vulnerability_candidates.sort(key=lambda x: x['score'], reverse=True)
    return vulnerability_candidates[:5]

def run_benchmark(cases=["case14", "case30", "case39", "case57"], model_path="best_power_grid_model.pt", k=3):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"results/vulnerability_benchmark_{timestamp}.log"
    
    os.makedirs("results", exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"Power Grid Vulnerability Benchmark\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Max lines removed (k): {k}\n\n")
    
    for case_name in cases:
        try:
            net = getattr(pn, case_name)()
            n_lines = len(net.line)
            n_buses = len(net.bus)
            
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"CASE: {case_name}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Network information:\n")
                f.write(f"  Buses: {n_buses}\n")
                f.write(f"  Lines: {n_lines}\n")
                
                total_combos = 0
                for i in range(1, k+1):
                    comb = math.comb(n_lines, i)
                    total_combos += comb
                    
                f.write(f"  Possible combinations (k={k}): {total_combos:,}\n\n")
            
            # Exhaustive search
            start_time = time.time()
            finder = VulnerabilityFinder(case_name, k=k)
            exhaustive_vulns = find_exhaustive_vulnerabilities(finder)
            exhaustive_time = time.time() - start_time
            
            with open(log_file, 'a') as f:
                f.write("EXHAUSTIVE SEARCH:\n")
                f.write(f"Duration: {exhaustive_time:.2f} seconds\n")
                f.write(f"Found {len(exhaustive_vulns):,} total vulnerabilities\n\n")
                
                f.write(f"Top {min(5, len(exhaustive_vulns))} vulnerabilities:\n")
                f.write("-" * 80 + "\n")
                
                for i, v in enumerate(exhaustive_vulns[:5]):
                    status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                    lines_str = ", ".join(map(str, v['removed_lines']))
                    
                    f.write(f"{i+1}. Score: {v['score']:.2f} - Status: {status}\n")
                    f.write(f"   Lines removed: {lines_str}\n")
                    f.write("-" * 80 + "\n")
            
            # Model-guided search
            start_time = time.time()
            model_vulns = find_model_vulnerabilities(model_path, case_name, k)
            model_time = time.time() - start_time
            
            with open(log_file, 'a') as f:
                f.write("\nMODEL-GUIDED SEARCH:\n")
                f.write(f"Duration: {model_time:.2f} seconds\n")
                f.write(f"Found {len(model_vulns)} vulnerabilities\n\n")
                
                f.write(f"Top vulnerabilities:\n")
                f.write("-" * 80 + "\n")
                
                for i, v in enumerate(model_vulns):
                    status = "COLLAPSED" if v['collapsed'] else "ISLANDED" if v['islanded'] else "STRESSED"
                    lines_str = ", ".join(map(str, v['removed_lines']))
                    
                    f.write(f"{i+1}. Score: {v['score']:.2f} - Status: {status}\n")
                    f.write(f"   Lines removed: {lines_str}\n")
                    f.write("-" * 80 + "\n")
            
            # Compare results
            if exhaustive_vulns and model_vulns:
                exhaustive_sets = [frozenset(v['removed_lines']) for v in exhaustive_vulns[:5]]
                model_sets = [frozenset(v['removed_lines']) for v in model_vulns]
                
                matches = []
                for i, model_set in enumerate(model_sets):
                    for j, exh_set in enumerate(exhaustive_sets):
                        if model_set == exh_set:
                            matches.append((i, j, model_vulns[i], exhaustive_vulns[j]))
                            break
                
                with open(log_file, 'a') as f:
                    f.write("\nCOMPARISON RESULTS:\n")
                    f.write("-" * 80 + "\n")
                    
                    if matches:
                        f.write(f"Model found {len(matches)} of the top 5 vulnerabilities:\n\n")
                        
                        for model_idx, exh_idx, model_v, exh_v in matches:
                            f.write(f"Vulnerability #{exh_idx+1} from exhaustive search (rank #{model_idx+1} in model results)\n")
                            f.write(f"Lines: {model_v['removed_lines']}\n")
                            f.write(f"Exhaustive score: {exh_v['score']:.2f}, Model score: {model_v['score']:.2f}\n")
                            f.write("-" * 40 + "\n")
                    else:
                        f.write("Model did not find any of the top 5 vulnerabilities.\n")
        except Exception:
            continue
    
    return log_file

if __name__ == "__main__":
    run_benchmark(["case14", "case30", "case39", "case57", "case89pegase"], "best_power_grid_model.pt", 3)