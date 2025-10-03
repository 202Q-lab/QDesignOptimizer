import json
import re
import random
from typing import Dict, Union, Optional

def parse_value_with_unit(value_str: str) -> tuple:
    """Parse a value string like '400um' into (400.0, 'um')"""
    match = re.match(r'^([+-]?\d*\.?\d+)([a-zA-Z]*)$', value_str.strip())
    if match:
        number = float(match.group(1))
        unit = match.group(2)
        return number, unit
    else:
        raise ValueError(f"Cannot parse value: {value_str}")

def format_value_with_unit(number: float, unit: str) -> str:
    """Format a number with unit back to string"""
    if unit == '':
        return str(number)
    else:
        # Handle integer values for cleaner output
        if number == int(number):
            return f"{int(number)}{unit}"
        else:
            return f"{number:.1f}{unit}"

def scramble_parameters(
    json_file_path: str,
    output_file_path: str,
    parameter_bounds: Dict[str, Dict[str, str]],
    seed: Optional[int] = None
) -> Dict[str, str]:
    """
    Scramble specific parameters from a JSON file with explicit bounds.
    
    Parameters:
    -----------
    json_file_path : str
        Path to input JSON file
    output_file_path : str
        Path to save scrambled JSON file
    parameter_bounds : dict
        Dictionary specifying which parameters to scramble and their bounds.
        Format:
        {
            "design_var_width_qubit_1": {"min": "300um", "max": "500um"},
            "design_var_lj_qubit_2": {"min": "8nH", "max": "15nH"},
            "design_var_cl_pos_x_qubit_1": {"min": "-200um", "max": "200um"}
        }
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with scrambled parameters
        
    Example:
    --------
    bounds = {
        "design_var_width_qubit_1": {"min": "350um", "max": "450um"},
        "design_var_width_qubit_2": {"min": "380um", "max": "420um"},
        "design_var_lj_qubit_1": {"min": "10nH", "max": "14nH"},
        "design_var_lj_qubit_2": {"min": "8nH", "max": "12nH"},
        "design_var_cl_pos_x_qubit_1": {"min": "-150um", "max": "150um"},
        "design_var_cl_pos_y_qubit_1": {"min": "-350um", "max": "-250um"}
    }
    
    scrambled = scramble_parameters("input.json", "output.json", bounds)
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Read original parameters
    with open(json_file_path, 'r') as f:
        original_params = json.load(f)
    
    # Copy original parameters
    scrambled_params = original_params.copy()
    
    # Track what was scrambled
    scrambled_count = 0
    
    # Process each parameter specified in bounds
    for param_name, bounds in parameter_bounds.items():
        
        # Check if parameter exists in original file
        if param_name not in original_params:
            print(f"Warning: Parameter '{param_name}' not found in JSON file")
            continue
        
        original_value = original_params[param_name]
        
        # Parse original value to get unit
        try:
            orig_number, orig_unit = parse_value_with_unit(original_value)
        except ValueError:
            print(f"Warning: Could not parse {param_name}: {original_value}")
            continue
        
        # Parse min and max bounds
        try:
            min_number, min_unit = parse_value_with_unit(bounds["min"])
            max_number, max_unit = parse_value_with_unit(bounds["max"])
        except (ValueError, KeyError) as e:
            print(f"Warning: Invalid bounds for {param_name}: {e}")
            continue
        
        # Check unit consistency
        if min_unit != max_unit:
            print(f"Warning: Unit mismatch in bounds for {param_name}: {min_unit} vs {max_unit}")
            continue
        
        if min_unit != orig_unit:
            print(f"Warning: Bound units ({min_unit}) don't match original unit ({orig_unit}) for {param_name}")
            # Use original unit but convert bounds if possible
            target_unit = orig_unit
        else:
            target_unit = orig_unit
        
        # Ensure min < max
        if min_number >= max_number:
            print(f"Warning: Invalid range for {param_name}: min ({min_number}) >= max ({max_number})")
            continue
        
        # Generate random value
        new_number = random.uniform(min_number, max_number)
        new_value = format_value_with_unit(new_number, target_unit)
        
        # Update parameter
        scrambled_params[param_name] = new_value
        scrambled_count += 1
        print(f"Scrambled {param_name}: {original_value} → {new_value}")
    
    # Save scrambled parameters
    with open(output_file_path, 'w') as f:
        json.dump(scrambled_params, f, indent=4)
    
    print(f"\nScrambled {scrambled_count} parameters")
    print(f"Scrambled parameters saved to: {output_file_path}")
    return scrambled_params


def compute_plus_minus_50_percent(final_values: dict) -> dict:
    """Compute the plus and minus 50 percent variations of the final design variables."""
    variations = {}
    for key, value in final_values.items():
        # Extract the numerical value and unit
        num_value = float(value.split(" ")[0])
        unit = value.split(" ")[1]
        # Compute the variations
        variations[f"{key}_plus_50_percent"] = f"{num_value * 1.5} {unit}"
        variations[f"{key}_minus_50_percent"] = f"{num_value * 0.5} {unit}"
    return variations

if __name__ == "__main__":

    final_values_after_optimization = {
        "design_var_lj_qubit_1": "14.400163475217242 nH",
        "design_var_width_qubit_1": "822.9178863519965 um",
        "design_var_length_resonator_1": "8084.239839659384 um",
        "design_var_coupl_length_resonator_1_tee": "1086.6975079693889 um",
        "design_var_coupl_length_qubit_1_resonator_1": "439.04984991847476 um"
    }

    compute_plus_minus_50_percent(final_values_after_optimization)

    # to be executed with main_eigenmode_single_qubit_resonator.ipynb
    for i in range(10):
        scramble_parameters(
        json_file_path='design_variables.json',
        output_file_path=f'design_variables_v{i}.json',
        parameter_bounds={
            # wide initial conditions (small value, max constraint by geometry)
        "design_var_width_qubit_1": {"min": "100um", "max": "1100um"},
        "design_var_lj_qubit_1": {"min": "5nH", "max": "25nH"},
        "design_var_length_resonator_1": {"min": "4000um", "max": "12000um"},
        "design_var_coupl_length_qubit_1_resonator_1": {"min": "100um", "max": "1100um"},
        "design_var_coupl_length_resonator_1_tee": {"min": "100um", "max": "1400um"},
            },
        )
