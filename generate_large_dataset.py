"""
Generate a Large, Physically Consistent Engineering Materials Dataset
Based on seed data from materials.csv, this script generates 900+ materials
maintaining realistic physical correlations between properties.
"""

import pandas as pd
import numpy as np
import os

def generate_dataset(seed_path='materials.csv', output_path='materials_large.csv', num_variants=20, random_seed=42):
    """
    Generate an expanded materials database by applying physically consistent perturbations
    to a seed dataset.
    """
    np.random.seed(random_seed)
    
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed dataset '{seed_path}' not found.")
        
    df = pd.read_csv(seed_path)
    print(f"Loaded seed dataset with {len(df)} materials.")
    
    expanded_rows = []
    
    for idx, row in df.iterrows():
        # First, add the original material itself as the baseline variant
        base_row = row.copy()
        base_row['Material'] = f"{row['Material']}"
        expanded_rows.append(base_row)
        
        # Calculate yield-to-tensile ratio and hardness-to-tensile ratio
        yt_ratio = row['Yield_Strength_MPa'] / max(row['Tensile_Strength_MPa'], 1.0)
        ht_ratio = row['Hardness_HB'] / max(row['Tensile_Strength_MPa'], 1.0)
        
        # Generate variants
        for i in range(1, num_variants):
            new_row = row.copy()
            new_row['Material'] = f"{row['Material']} v{i}"
            
            # 1. Density: extremely stable for alloy matrices. Add very minor noise (std dev = 0.5%)
            density_noise = np.random.normal(1.0, 0.005)
            new_row['Density_g_cm3'] = round(max(0.1, row['Density_g_cm3'] * density_noise), 2)
            
            # 2. Tensile Strength: fluctuates based on heat treatment/cold work. std dev = 6%
            tensile_scale = np.random.normal(1.0, 0.06)
            # Ensure it is positive and has a reasonable value
            new_tensile = max(10.0, row['Tensile_Strength_MPa'] * tensile_scale)
            new_row['Tensile_Strength_MPa'] = round(new_tensile, 1)
            
            # 3. Yield Strength: highly correlated with Tensile Strength. Keep ratio stable with tiny fluctuation
            new_yt_ratio = yt_ratio * np.random.normal(1.0, 0.015)
            # Yield strength must not exceed tensile strength (physically impossible)
            new_yt_ratio = min(0.98, max(0.2, new_yt_ratio))
            new_row['Yield_Strength_MPa'] = round(new_tensile * new_yt_ratio, 1)
            
            # 4. Hardness: highly correlated with Tensile Strength
            new_ht_ratio = ht_ratio * np.random.normal(1.0, 0.02)
            new_hardness = new_tensile * new_ht_ratio
            new_row['Hardness_HB'] = round(max(5.0, new_hardness), 1)
            
            # 5. Corrosion Resistance: integer rating from 0 to 10
            corr_offset = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            new_row['Corrosion_Resistance_0_10'] = max(0, min(10, row['Corrosion_Resistance_0_10'] + corr_offset))
            
            # 6. Thermal Conductivity: characteristic of alloy matrix, minor variation (std dev = 3%)
            thermal_noise = np.random.normal(1.0, 0.03)
            new_row['Thermal_Conductivity_W_mK'] = round(max(0.1, row['Thermal_Conductivity_W_mK'] * thermal_noise), 2)
            
            # 7. Cost Index: integer rating from 1 to 10
            cost_offset = np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
            new_row['Cost_Index_1_10'] = max(1, min(10, row['Cost_Index_1_10'] + cost_offset))
            
            # 8. Melting Point: minor temperature variations (std dev = 15 degrees C)
            temp_noise = np.random.normal(0, 15)
            new_row['Melting_Point_C'] = round(max(100.0, row['Melting_Point_C'] + temp_noise), 1)
            
            # 9. Machinability: integer rating from 1 to 10
            mach_offset = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            new_row['Machinability'] = max(1, min(10, row['Machinability'] + mach_offset))
            
            expanded_rows.append(new_row)
            
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.to_csv(output_path, index=False)
    print(f"Successfully generated expanded dataset with {len(expanded_df)} materials saved to '{output_path}'.")
    return expanded_df

if __name__ == '__main__':
    generate_dataset()
