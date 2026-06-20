"""
Integrate Kaggle Archive Data with Original Materials Database
Cleans, maps, and blends the 1500+ materials from the Kaggle dataset with the
original 44 high-fidelity materials, maintaining physical consistency via nearest-neighbor property mapping.
"""

import pandas as pd
import numpy as np
import os
import shutil

def classify_alloy_type(name, density):
    name = str(name).lower()
    
    # 1. Classify by name keywords
    if any(k in name for k in ['aluminum', 'al ', '7075', '6061', '2024', '3003', '2014', '5083', '7178']):
        return 'Aluminum'
    if any(k in name for k in ['titanium', 'ti-', 'grade 2', 'ti ']):
        return 'Titanium'
    if any(k in name for k in ['inconel', 'hastelloy', 'rene', 'monel', 'nichrome', 'alloy 263']):
        return 'Nickel Alloy'
    if any(k in name for k in ['magnesium', 'az31', 'az91', 'ze63']):
        return 'Magnesium'
    if any(k in name for k in ['tungsten', 'molybdenum', 'tantalum', 'niobium', 'rhenium']):
        return 'Refractory'
    if any(k in name for k in ['cobalt', 'co-cr', 'mp35n', 'l-605']):
        return 'Cobalt Alloy'
    if any(k in name for k in ['stainless', 'duplex', 'z12cn', 'z15cn', '304', '316', '430', '410', '420', '440c', '630', '2205', '2507']):
        if 'duplex' in name or '2205' in name or '2507' in name:
            return 'Duplex Stainless'
        return 'Stainless Steel'
    if any(k in name for k in ['copper', 'brass', 'bronze', 'cu ']):
        if any(pk in name for pk in ['pure copper', 'c110', 'c11000']):
            return 'Copper'
        return 'Copper Alloy'
    if any(k in name for k in ['tool steel', 'h13', 'd2', 'o1', 'm2']):
        return 'Tool Steel'
    if any(k in name for k in ['steel', 'sae', 'gost', 'aisi', 'structural', 'ck']):
        return 'Carbon Steel'
    if any(k in name for k in ['iron', 'cast iron', 'grey iron', 'ductile', 'nodular', 'malleable']):
        return 'Carbon Steel'
        
    # 2. Classify by density if name is ambiguous
    if 2.5 <= density <= 2.9:
        return 'Aluminum'
    if 4.0 <= density <= 5.0:
        return 'Titanium'
    if 1.6 <= density <= 2.0:
        return 'Magnesium'
    if 9.5 <= density:
        return 'Refractory'
    if 8.1 <= density <= 9.0:
        if 'nickel' in name or 'ni ' in name:
            return 'Nickel'
        if 'cobalt' in name or 'co ' in name:
            return 'Cobalt Alloy'
        return 'Copper Alloy'
    if 7.0 <= density <= 8.1:
        if 'csn 17' in name or 'z12' in name or 'z15' in name:
            return 'Stainless Steel'
        return 'Carbon Steel'
        
    return 'Carbon Steel'  # Fallback

def main():
    original_path = 'materials.csv'
    kaggle_path = './archive/Data.csv'
    output_path = 'materials_large.csv'
    
    if not os.path.exists(original_path):
        print(f"Error: Original database '{original_path}' not found.")
        return
        
    if not os.path.exists(kaggle_path):
        print(f"Error: Kaggle database '{kaggle_path}' not found.")
        return
        
    # Load datasets
    df_orig = pd.read_csv(original_path)
    df_kagg = pd.read_csv(kaggle_path)
    
    print(f"Original dataset: {len(df_orig)} materials.")
    print(f"Kaggle dataset: {len(df_kagg)} materials.")
    
    # 1. Clean Kaggle Data
    df_kagg = df_kagg.dropna(subset=['Material', 'Ro', 'Su'])
    
    # Convert density (Ro in kg/m3 -> Density_g_cm3 in g/cm3)
    df_kagg['Density_g_cm3'] = df_kagg['Ro'] / 1000.0
    
    # Convert tensile strength (Su -> Tensile_Strength_MPa)
    df_kagg['Tensile_Strength_MPa'] = df_kagg['Su'].astype(float)
    
    # Classify Alloy_Type
    df_kagg['Alloy_Type'] = df_kagg.apply(lambda r: classify_alloy_type(r['Material'], r['Density_g_cm3']), axis=1)
    
    # Clean yield strength (Sy -> Yield_Strength_MPa)
    df_kagg['Yield_Strength_MPa'] = pd.to_numeric(df_kagg['Sy'], errors='coerce')
    # Estimate null yield strengths using average ratio of Yield/Tensile (0.75)
    df_kagg['Yield_Strength_MPa'] = df_kagg['Yield_Strength_MPa'].fillna(df_kagg['Tensile_Strength_MPa'] * 0.75)
    # Ensure yield strength doesn't exceed tensile strength
    df_kagg['Yield_Strength_MPa'] = df_kagg.apply(lambda r: min(r['Yield_Strength_MPa'], r['Tensile_Strength_MPa'] * 0.98), axis=1)
    
    # Clean hardness (Bhn -> Hardness_HB)
    df_kagg['Hardness_HB'] = pd.to_numeric(df_kagg['Bhn'], errors='coerce')
    # Estimate null hardness based on tensile strength ratios
    def estimate_hardness(row):
        alloy = row['Alloy_Type']
        su = row['Tensile_Strength_MPa']
        if pd.notna(row['Hardness_HB']):
            return row['Hardness_HB']
        if 'Steel' in alloy:
            return su / 3.4
        elif alloy == 'Aluminum':
            return su / 3.5
        elif 'Copper' in alloy or alloy == 'Copper':
            return su / 4.0
        elif alloy == 'Magnesium':
            return su / 3.2
        return su / 3.4
        
    df_kagg['Hardness_HB'] = df_kagg.apply(estimate_hardness, axis=1)
    
    # Unique Material Names by adding Heat Treatment if present
    def format_material_name(row):
        mat = str(row['Material']).strip()
        ht = str(row['Heat treatment']).strip()
        if ht and ht != 'nan' and ht != '':
            return f"{mat} ({ht})"
        return mat
        
    df_kagg['Material'] = df_kagg.apply(format_material_name, axis=1)
    
    # 2. Nearest-Neighbor Mapping for Missing Columns
    # Columns to copy: Corrosion_Resistance_0_10, Thermal_Conductivity_W_mK, Cost_Index_1_10, Melting_Point_C, Machinability, Primary_Application
    mapped_rows = []
    
    for idx, row in df_kagg.iterrows():
        alloy = row['Alloy_Type']
        
        # Filter original materials of the same alloy type
        ref_candidates = df_orig[df_orig['Alloy_Type'] == alloy]
        if len(ref_candidates) == 0:
            # Fallbacks if no exact alloy match in original
            if 'Steel' in alloy:
                ref_candidates = df_orig[df_orig['Alloy_Type'].str.contains('Steel', na=False)]
            elif 'Copper' in alloy:
                ref_candidates = df_orig[df_orig['Alloy_Type'].str.contains('Copper', na=False)]
            else:
                ref_candidates = df_orig
                
        # Find nearest neighbor by Density and Tensile Strength
        # Normalize features for fair distance calculation
        ref_densities = ref_candidates['Density_g_cm3'].values
        ref_strengths = ref_candidates['Tensile_Strength_MPa'].values
        
        # Distance calculation
        d_diff = ref_densities - row['Density_g_cm3']
        s_diff = (ref_strengths - row['Tensile_Strength_MPa']) / 100.0  # Scale strength difference to match density scale
        distances = np.sqrt(d_diff**2 + s_diff**2)
        
        nearest_idx = np.argmin(distances)
        nearest_ref = ref_candidates.iloc[nearest_idx]
        
        # Copy properties and add minor physical noise for diversity
        corr = max(0, min(10, nearest_ref['Corrosion_Resistance_0_10'] + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])))
        thermal = max(0.1, nearest_ref['Thermal_Conductivity_W_mK'] * np.random.normal(1.0, 0.05))
        cost = max(1, min(10, nearest_ref['Cost_Index_1_10'] + np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])))
        melting = max(100.0, nearest_ref['Melting_Point_C'] + np.random.normal(0, 15))
        mach = max(1, min(10, nearest_ref['Machinability'] + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])))
        
        mapped_row = {
            'Material': row['Material'],
            'Alloy_Type': alloy,
            'Density_g_cm3': round(row['Density_g_cm3'], 2),
            'Yield_Strength_MPa': round(row['Yield_Strength_MPa'], 1),
            'Tensile_Strength_MPa': round(row['Tensile_Strength_MPa'], 1),
            'Hardness_HB': round(row['Hardness_HB'], 1),
            'Corrosion_Resistance_0_10': int(corr),
            'Thermal_Conductivity_W_mK': round(thermal, 2),
            'Cost_Index_1_10': int(cost),
            'Melting_Point_C': round(melting, 1),
            'Machinability': int(mach),
            'Primary_Application': nearest_ref['Primary_Application']
        }
        mapped_rows.append(mapped_row)
        
    df_kagg_mapped = pd.DataFrame(mapped_rows)
    
    # 3. Mix with original dataset
    df_combined = pd.concat([df_orig, df_kagg_mapped], ignore_index=True)
    
    # Drop duplicates on Material name, keeping the first occurrence (original has priority)
    df_combined = df_combined.drop_duplicates(subset=['Material'], keep='first')
    
    # Save mixed database
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully generated blended dataset '{output_path}' with {len(df_combined)} materials.")
    
    # Backup original and deploy new database to materials.csv
    backup_path = 'materials_original.csv'
    if not os.path.exists(backup_path):
        shutil.copy(original_path, backup_path)
        print(f"✓ Backed up original database to '{backup_path}'")
        
    shutil.copy(output_path, original_path)
    print(f"✓ Copied new blended database to '{original_path}'")

if __name__ == '__main__':
    main()
