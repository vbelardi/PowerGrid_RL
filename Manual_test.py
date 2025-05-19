import pandapower as pp
import pandapower.networks as pn
import numpy as np
import argparse
from tabulate import tabulate

def test_line_removals(case_name, lines_to_remove, vmin=0.95, vmax=1.05, load_factor=1.0):
    """Test specific line removals on a given power grid case."""
    # Get the network
    try:
        net = getattr(pn, case_name)()
    except AttributeError:
        print(f"Error: Case '{case_name}' not found in pandapower.networks")
        available_cases = [name for name in dir(pn) if name.startswith('case') and callable(getattr(pn, name))]
        print(f"Available cases: {', '.join(available_cases)}")
        return
    
    # Optional: Scale load
    if load_factor != 1.0:
        net.load.p_mw *= load_factor
        net.load.q_mvar *= load_factor
    
    # Initial power flow
    try:
        pp.runpp(net)
        print(f"Initial power flow successful for {case_name}")
        print(f"Network summary: {len(net.bus)} buses, {len(net.line)} lines, {len(net.gen) + len(net.ext_grid)} generators")
    except pp.LoadflowNotConverged:
        print("Initial power flow failed to converge. The network may have issues before any lines are removed.")
        return
    
    # Get initial state metrics
    initial_violations = get_violations(net, vmin, vmax)
    if initial_violations["has_violations"]:
        print("WARNING: Initial state already has violations:")
        if len(initial_violations["voltage_violations"]) > 0:
            print(f"  - {len(initial_violations['voltage_violations'])} voltage violations")
        if len(initial_violations["loading_violations"]) > 0:
            print(f"  - {len(initial_violations['loading_violations'])} loading violations")
    
    # Make a copy to restore later
    original_net = net.deepcopy()
    
    # Remove lines one by one and check results
    results = []
    removed_so_far = []
    
    for line_idx in lines_to_remove:
        if line_idx not in net.line.index:
            print(f"Warning: Line {line_idx} does not exist in {case_name}. Skipping...")
            continue
            
        # Remove the line
        removed_so_far.append(line_idx)
        net.line.at[line_idx, 'in_service'] = False
        
        # Try to run power flow
        try:
            pp.runpp(net)
            # Power flow converged
            violations = get_violations(net, vmin, vmax)
            
            # Check for NaN values which indicate isolated buses
            has_isolated = np.isnan(net.res_bus.vm_pu.values).any()
            
            result = {
                'lines_removed': removed_so_far.copy(),
                'converged': True,
                'has_isolated': has_isolated,
                'voltage_violations': violations["voltage_violations"],
                'loading_violations': violations["loading_violations"]
            }
        except pp.LoadflowNotConverged:
            # Power flow didn't converge, likely grid collapse
            result = {
                'lines_removed': removed_so_far.copy(),
                'converged': False,
                'has_isolated': False,  # Can't determine
                'voltage_violations': [],
                'loading_violations': []
            }
        
        results.append(result)
        print(f"Removed line {line_idx} - {'Grid collapsed' if not result['converged'] else 'Grid operational'}")
    
    # Print detailed results
    print("\n=== RESULTS ===")
    
    table_data = []
    for r in results:
        removed = ", ".join(map(str, r['lines_removed']))
        if not r['converged']:
            status = "COLLAPSED"
        elif r['has_isolated']:
            status = "ISLANDED"
        elif len(r['voltage_violations']) > 0 or len(r['loading_violations']) > 0:
            status = "VIOLATIONS"
        else:
            status = "STABLE"
            
        v_viol = len(r['voltage_violations'])
        l_viol = len(r['loading_violations'])
        
        table_data.append([removed, status, v_viol, l_viol])
    
    headers = ["Lines Removed", "Status", "Voltage Violations", "Loading Violations"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print details of violations for the final state
    if results and (results[-1]['converged'] and 
                   (results[-1]['voltage_violations'] or results[-1]['loading_violations'])):
        print("\n=== VIOLATION DETAILS ===")
        
        if results[-1]['voltage_violations']:
            print("\nVoltage Violations:")
            volt_data = [(bus, val, "HIGH" if val > vmax else "LOW") 
                         for bus, val in results[-1]['voltage_violations']]
            print(tabulate(volt_data, headers=["Bus", "Voltage (p.u.)", "Type"], tablefmt="simple"))
            
        if results[-1]['loading_violations']:
            print("\nLoading Violations:")
            load_data = [(line, val) for line, val in results[-1]['loading_violations']]
            print(tabulate(load_data, headers=["Line", "Loading (%)"], tablefmt="simple"))
            
    return results

def get_violations(net, vmin, vmax):
    """Get voltage and loading violations from network."""
    voltage_violations = []
    for i, vm in enumerate(net.res_bus.vm_pu.values):
        if np.isnan(vm):
            continue
        if vm < vmin or vm > vmax:
            voltage_violations.append((i, vm))
    
    loading_violations = []
    for i, loading in enumerate(net.res_line.loading_percent.values):
        if loading > 100:
            line_id = net.line.index[i]
            loading_violations.append((line_id, loading))
    
    has_violations = len(voltage_violations) > 0 or len(loading_violations) > 0
    
    return {
        "has_violations": has_violations,
        "voltage_violations": voltage_violations,
        "loading_violations": loading_violations
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test line removals in power grid cases")
    parser.add_argument("--case", type=str, default="case14", 
                        help="Power grid case name (e.g., case14, case30)")
    parser.add_argument("--lines", type=int, nargs="+", required=True,
                        help="Line indices to remove (space separated)")
    parser.add_argument("--vmin", type=float, default=0.95,
                        help="Minimum acceptable voltage in p.u.")
    parser.add_argument("--vmax", type=float, default=1.05,
                        help="Maximum acceptable voltage in p.u.")
    parser.add_argument("--load", type=float, default=1.0,
                        help="Load scaling factor (default=1.0)")
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided
    if not args.lines:
        case_name = input("Enter case name (e.g., case14, case30): ")
        lines_input = input("Enter line indices to remove (comma-separated): ")
        lines_to_remove = [int(x.strip()) for x in lines_input.split(",")]
        vmin = float(input("Enter minimum voltage (default 0.95): ") or "0.95")
        vmax = float(input("Enter maximum voltage (default 1.05): ") or "1.05")
        load_factor = float(input("Enter load scaling factor (default 1.0): ") or "1.0")
    else:
        case_name = args.case
        lines_to_remove = args.lines
        vmin = args.vmin
        vmax = args.vmax
        load_factor = args.load
    
    test_line_removals(case_name, lines_to_remove, vmin, vmax, load_factor)