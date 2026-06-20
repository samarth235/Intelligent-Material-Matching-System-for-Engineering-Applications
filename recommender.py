"""
Recommendation Engine for Material Selection
Implements weighted scoring algorithm and ML-based recommendations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class MaterialRecommender:
    """
    Material recommendation system using weighted scoring and ML
    """
    
    def __init__(self, csv_path='materials.csv'):
        """
        Initialize recommender with material database
        
        Args:
            csv_path: Path to materials CSV file
        """
        self.materials_df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.normalized_df = None
        self._normalize_properties()
        
    def _normalize_properties(self):
        """
        Normalize material properties to 0-10 scale for fair comparison
        """
        self.normalized_df = self.materials_df.copy()
        
        # Properties to normalize
        properties_to_normalize = [
            'Density_g_cm3',
            'Yield_Strength_MPa',
            'Tensile_Strength_MPa',
            'Hardness_HB',
            'Corrosion_Resistance_0_10',
            'Thermal_Conductivity_W_mK',
            'Cost_Index_1_10',
            'Machinability'
        ]
        
        for prop in properties_to_normalize:
            if prop in self.normalized_df.columns:
                min_val = self.normalized_df[prop].min()
                max_val = self.normalized_df[prop].max()
                
                if max_val - min_val != 0:
                    # Normalize to 0-10 scale
                    self.normalized_df[prop + '_norm'] = (
                        (self.normalized_df[prop] - min_val) / (max_val - min_val) * 10
                    )
                else:
                    self.normalized_df[prop + '_norm'] = 5
        
        return self.normalized_df
    
    def get_weighted_recommendations(self, 
                                    strength_weight=5,
                                    weight_weight=5,
                                    corrosion_weight=5,
                                    cost_weight=5,
                                    thermal_weight=3,
                                    top_n=3,
                                    application_filter=None):
        """
        Get material recommendations using weighted scoring
        
        Args:
            strength_weight: Weight for strength (1-10)
            weight_weight: Weight for low density (1-10)
            corrosion_weight: Weight for corrosion resistance (1-10)
            cost_weight: Weight for low cost (1-10)
            thermal_weight: Weight for thermal conductivity (1-10)
            top_n: Number of top recommendations (default 3)
            application_filter: Filter by primary application
            
        Returns:
            DataFrame with top recommendations and scores
        """
        
        df = self.normalized_df.copy()
        
        # Filter by application if specified
        if application_filter:
            df = df[df['Primary_Application'].str.contains(
                application_filter, case=False, na=False
            )]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Calculate composite score
        # Higher is better for: strength, corrosion, thermal
        # Lower is better for: density, cost (so we negate)
        score = (
            strength_weight * (df['Tensile_Strength_MPa_norm'] + df['Yield_Strength_MPa_norm']) / 2 +
            corrosion_weight * df['Corrosion_Resistance_0_10_norm'] +
            thermal_weight * df['Thermal_Conductivity_W_mK_norm'] +
            (10 - weight_weight) * (10 - df['Density_g_cm3_norm']) / (10 - weight_weight + 0.001) +
            (10 - cost_weight) * (10 - df['Cost_Index_1_10_norm']) / (10 - cost_weight + 0.001)
        )
        
        df['Recommendation_Score'] = score
        df = df.sort_values('Recommendation_Score', ascending=False)
        
        return df.head(top_n)
    
    def get_similarity_recommendations(self, 
                                     strength_weight=5,
                                     weight_weight=5,
                                     corrosion_weight=5,
                                     cost_weight=5,
                                     thermal_weight=3,
                                     top_n=3,
                                     reference_material=None,
                                     application_filter=None):
        """
        Get recommendations using cosine similarity to ideal material
        
        Args:
            strength_weight: Weight for strength preference
            weight_weight: Weight for light weight preference
            corrosion_weight: Weight for corrosion resistance
            cost_weight: Weight for cost consideration
            thermal_weight: Weight for thermal conductivity
            top_n: Number of recommendations
            reference_material: Material to compare similarity with
            application_filter: Filter by application
            
        Returns:
            DataFrame with similarity-based recommendations
        """
        
        df = self.normalized_df.copy()
        
        if application_filter:
            df = df[df['Primary_Application'].str.contains(
                application_filter, case=False, na=False
            )]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Create ideal material profile based on weights
        ideal_profile = np.array([
            (strength_weight / 30) * 10,  # Max strength
            (weight_weight / 30) * 0,      # Min density
            (corrosion_weight / 30) * 10,  # Max corrosion resistance
            (cost_weight / 30) * 0,        # Min cost
            (thermal_weight / 30) * 10,    # Max thermal
        ])
        
        # Extract relevant normalized features
        features = df[[
            'Tensile_Strength_MPa_norm',
            'Density_g_cm3_norm',
            'Corrosion_Resistance_0_10_norm',
            'Cost_Index_1_10_norm',
            'Thermal_Conductivity_W_mK_norm'
        ]].values
        
        # Calculate similarity
        similarities = cosine_similarity(
            [ideal_profile],
            features
        )[0]
        
        df['Similarity_Score'] = similarities
        df = df.sort_values('Similarity_Score', ascending=False)
        
        return df.head(top_n)
    
    def get_material_explanation(self, material_name):
        """
        Generate explanation for why a material is recommended
        
        Args:
            material_name: Name of the material
            
        Returns:
            String explanation
        """
        
        material = self.materials_df[
            self.materials_df['Material'] == material_name
        ]
        
        if len(material) == 0:
            return f"Material {material_name} not found in database."
        
        material = material.iloc[0]
        
        explanation = f"\n📊 **{material_name}** Analysis\n"
        explanation += f"{'='*50}\n\n"
        
        explanation += f"**Physical Properties:**\n"
        explanation += f"  • Density: {material['Density_g_cm3']:.2f} g/cm³\n"
        explanation += f"  • Yield Strength: {material['Yield_Strength_MPa']:.0f} MPa\n"
        explanation += f"  • Tensile Strength: {material['Tensile_Strength_MPa']:.0f} MPa\n"
        explanation += f"  • Hardness: {material['Hardness_HB']:.0f} HB\n\n"
        
        explanation += f"**Environmental Performance:**\n"
        explanation += f"  • Corrosion Resistance: {material['Corrosion_Resistance_0_10']:.1f}/10\n"
        explanation += f"  • Thermal Conductivity: {material['Thermal_Conductivity_W_mK']:.1f} W/mK\n"
        explanation += f"  • Melting Point: {material['Melting_Point_C']:.0f}°C\n\n"
        
        explanation += f"**Cost & Machinability:**\n"
        explanation += f"  • Cost Index: {material['Cost_Index_1_10']:.1f}/10\n"
        explanation += f"  • Machinability: {material['Machinability']:.0f}/10\n\n"
        
        explanation += f"**Primary Application:** {material['Primary_Application']}\n"
        explanation += f"**Alloy Type:** {material['Alloy_Type']}\n\n"
        
        # Add strengths
        explanation += f"**Why This Material?**\n"
        strengths = []
        
        if material['Tensile_Strength_MPa'] > 600:
            strengths.append("✓ Excellent strength")
        elif material['Tensile_Strength_MPa'] > 400:
            strengths.append("✓ Good strength")
            
        if material['Density_g_cm3'] < 5:
            strengths.append("✓ Lightweight")
            
        if material['Corrosion_Resistance_0_10'] >= 9:
            strengths.append("✓ Excellent corrosion resistance")
        elif material['Corrosion_Resistance_0_10'] >= 7:
            strengths.append("✓ Good corrosion resistance")
            
        if material['Cost_Index_1_10'] <= 4:
            strengths.append("✓ Cost-effective")
            
        if material['Thermal_Conductivity_W_mK'] > 100:
            strengths.append("✓ Excellent thermal conductivity")
            
        if material['Machinability'] >= 8:
            strengths.append("✓ Easy to machine")
        
        if strengths:
            for strength in strengths:
                explanation += f"  {strength}\n"
        else:
            explanation += f"  ✓ Specialized material with unique properties\n"
        
        return explanation
    
    def compare_materials(self, material_names):
        """
        Compare multiple materials
        
        Args:
            material_names: List of material names
            
        Returns:
            DataFrame with material comparison
        """
        
        comparison = self.materials_df[
            self.materials_df['Material'].isin(material_names)
        ].copy()
        
        return comparison
    
    def get_materials_by_application(self, application):
        """
        Get all materials suitable for an application
        
        Args:
            application: Application type
            
        Returns:
            DataFrame of materials for the application
        """
        
        return self.materials_df[
            self.materials_df['Primary_Application'].str.contains(
                application, case=False, na=False
            )
        ]
    
    def get_available_applications(self):
        """
        Get list of available applications
        
        Returns:
            List of unique applications
        """
        
        return sorted(self.materials_df['Primary_Application'].unique().tolist())
    
    def get_material_names(self):
        """
        Get all material names
        
        Returns:
            List of material names
        """
        
        return self.materials_df['Material'].tolist()


# Test the recommender
if __name__ == "__main__":
    recommender = MaterialRecommender('materials.csv')
    
    print("Available Applications:")
    print(recommender.get_available_applications())
    print("\n" + "="*50 + "\n")
    
    # Test 1: Weighted recommendation
    print("TEST 1: Aerospace Recommendation")
    print("="*50)
    results = recommender.get_weighted_recommendations(
        strength_weight=10,
        weight_weight=10,
        corrosion_weight=8,
        cost_weight=3,
        thermal_weight=8,
        application_filter="Aerospace"
    )
    print(results[['Material', 'Recommendation_Score']].head(3))
    print("\n")
    
    # Test 2: Explanation
    print("TEST 2: Material Explanation")
    print("="*50)
    print(recommender.get_material_explanation("Ti-6Al-4V"))
    print("\n")
    
    # Test 3: Similarity recommendation
    print("TEST 3: Similarity-based Recommendation (Marine)")
    print("="*50)
    results = recommender.get_similarity_recommendations(
        strength_weight=6,
        weight_weight=8,
        corrosion_weight=10,
        cost_weight=5,
        thermal_weight=3,
        application_filter="Marine"
    )
    print(results[['Material', 'Similarity_Score']].head(3))
