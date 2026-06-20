"""
Material Database Analysis Script
Analyzes material properties and generates insights
"""

import pandas as pd
import numpy as np
import json


class MaterialAnalyzer:
    """
    Analyze material database and generate insights
    """
    
    def __init__(self, csv_path='materials.csv'):
        """Initialize analyzer with material database"""
        self.df = pd.read_csv(csv_path)
        self.report = {}
        
    def get_database_summary(self):
        """Get summary statistics of the database"""
        summary = {
            'total_materials': len(self.df),
            'alloy_types': self.df['Alloy_Type'].nunique(),
            'applications': self.df['Primary_Application'].nunique(),
            'properties_tracked': len([col for col in self.df.columns if col != 'Material'])
        }
        
        self.report['summary'] = summary
        return summary
    
    def get_alloy_type_distribution(self):
        """Analyze distribution of alloy types"""
        distribution = self.df['Alloy_Type'].value_counts().to_dict()
        
        self.report['alloy_distribution'] = distribution
        return distribution
    
    def get_application_distribution(self):
        """Analyze distribution of applications"""
        distribution = self.df['Primary_Application'].value_counts().to_dict()
        
        self.report['application_distribution'] = distribution
        return distribution
    
    def get_property_statistics(self):
        """Get statistics for all numerical properties"""
        numeric_cols = [
            'Density_g_cm3', 'Yield_Strength_MPa', 'Tensile_Strength_MPa',
            'Hardness_HB', 'Corrosion_Resistance_0_10', 'Thermal_Conductivity_W_mK',
            'Cost_Index_1_10', 'Melting_Point_C', 'Machinability'
        ]
        
        statistics = {}
        
        for col in numeric_cols:
            statistics[col] = {
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'q25': float(self.df[col].quantile(0.25)),
                'q75': float(self.df[col].quantile(0.75))
            }
        
        self.report['property_statistics'] = statistics
        return statistics
    
    def get_strength_to_weight_ratio(self):
        """Calculate strength-to-weight ratio (ideal for aerospace)"""
        self.df['Strength_to_Weight_Ratio'] = (
            self.df['Tensile_Strength_MPa'] / self.df['Density_g_cm3']
        )
        
        top_materials = self.df.nlargest(10, 'Strength_to_Weight_Ratio')[
            ['Material', 'Tensile_Strength_MPa', 'Density_g_cm3', 'Strength_to_Weight_Ratio']
        ]
        
        self.report['strength_to_weight'] = top_materials.to_dict('records')
        return top_materials
    
    def get_cost_to_performance_ratio(self):
        """Calculate cost-effectiveness (strength per unit cost)"""
        self.df['Performance_per_Cost'] = (
            self.df['Tensile_Strength_MPa'] / self.df['Cost_Index_1_10']
        )
        
        top_materials = self.df.nlargest(10, 'Performance_per_Cost')[
            ['Material', 'Tensile_Strength_MPa', 'Cost_Index_1_10', 'Performance_per_Cost']
        ]
        
        self.report['cost_effectiveness'] = top_materials.to_dict('records')
        return top_materials
    
    def get_corrosion_resistant_materials(self):
        """Find best corrosion-resistant materials"""
        corrosion_leaders = self.df[self.df['Corrosion_Resistance_0_10'] >= 9][
            ['Material', 'Alloy_Type', 'Corrosion_Resistance_0_10', 'Cost_Index_1_10']
        ].sort_values('Corrosion_Resistance_0_10', ascending=False)
        
        self.report['corrosion_resistant'] = corrosion_leaders.to_dict('records')
        return corrosion_leaders
    
    def get_thermal_conductors(self):
        """Find best thermal conductors"""
        thermal_leaders = self.df[self.df['Thermal_Conductivity_W_mK'] > 100][
            ['Material', 'Alloy_Type', 'Thermal_Conductivity_W_mK', 'Density_g_cm3']
        ].sort_values('Thermal_Conductivity_W_mK', ascending=False)
        
        self.report['thermal_conductors'] = thermal_leaders.to_dict('records')
        return thermal_leaders
    
    def get_material_density_range(self):
        """Categorize materials by density"""
        categories = {
            'Ultra-Light (<3.0 g/cm³)': [],
            'Light (3.0-5.0 g/cm³)': [],
            'Medium (5.0-10.0 g/cm³)': [],
            'Heavy (>10.0 g/cm³)': []
        }
        
        for _, material in self.df.iterrows():
            density = material['Density_g_cm3']
            if density < 3.0:
                categories['Ultra-Light (<3.0 g/cm³)'].append(material['Material'])
            elif density < 5.0:
                categories['Light (3.0-5.0 g/cm³)'].append(material['Material'])
            elif density < 10.0:
                categories['Medium (5.0-10.0 g/cm³)'].append(material['Material'])
            else:
                categories['Heavy (>10.0 g/cm³)'].append(material['Material'])
        
        self.report['density_categories'] = categories
        return categories
    
    def get_strength_ranking(self):
        """Rank materials by tensile strength"""
        ranking = self.df[
            ['Material', 'Tensile_Strength_MPa', 'Density_g_cm3', 'Cost_Index_1_10']
        ].sort_values('Tensile_Strength_MPa', ascending=False).head(15)
        
        self.report['strength_ranking'] = ranking.to_dict('records')
        return ranking
    
    def get_material_recommendations_by_use_case(self):
        """Get best materials for common use cases"""
        recommendations = {}
        
        # Aerospace: High strength, low weight, corrosion resistant
        aerospace = self.df[
            (self.df['Tensile_Strength_MPa'] > 500) &
            (self.df['Density_g_cm3'] < 6) &
            (self.df['Corrosion_Resistance_0_10'] >= 8)
        ].nlargest(3, 'Tensile_Strength_MPa')[['Material', 'Tensile_Strength_MPa', 'Density_g_cm3']]
        
        recommendations['Aerospace'] = aerospace.to_dict('records')
        
        # Marine: High corrosion resistance, decent strength
        marine = self.df[
            self.df['Corrosion_Resistance_0_10'] >= 8
        ].nlargest(3, 'Tensile_Strength_MPa')[['Material', 'Corrosion_Resistance_0_10', 'Cost_Index_1_10']]
        
        recommendations['Marine'] = marine.to_dict('records')
        
        # Automotive: Good strength-to-weight, cost-effective
        automotive = self.df[
            (self.df['Density_g_cm3'] < 3.5) |
            (self.df['Cost_Index_1_10'] <= 3)
        ].nlargest(3, 'Tensile_Strength_MPa')[['Material', 'Tensile_Strength_MPa', 'Cost_Index_1_10']]
        
        recommendations['Automotive'] = automotive.to_dict('records')
        
        # Electronics: High thermal conductivity
        electronics = self.df[
            self.df['Thermal_Conductivity_W_mK'] > 50
        ].nlargest(3, 'Thermal_Conductivity_W_mK')[['Material', 'Thermal_Conductivity_W_mK']]
        
        recommendations['Electronics'] = electronics.to_dict('records')
        
        self.report['use_case_recommendations'] = recommendations
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("MATERIAL DATABASE COMPREHENSIVE ANALYSIS REPORT")
        print("="*70 + "\n")
        
        # Summary
        summary = self.get_database_summary()
        print("📊 DATABASE SUMMARY")
        print("-" * 70)
        print(f"  Total Materials: {summary['total_materials']}")
        print(f"  Alloy Types: {summary['alloy_types']}")
        print(f"  Application Categories: {summary['applications']}")
        print(f"  Properties Tracked: {summary['properties_tracked']}")
        print()
        
        # Alloy distribution
        print("🔧 ALLOY TYPE DISTRIBUTION")
        print("-" * 70)
        alloy_dist = self.get_alloy_type_distribution()
        for alloy_type, count in sorted(alloy_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {alloy_type:.<40} {count:>3} materials")
        print()
        
        # Application distribution
        print("📍 APPLICATION DISTRIBUTION")
        print("-" * 70)
        app_dist = self.get_application_distribution()
        for app, count in sorted(app_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {app:.<40} {count:>3} materials")
        print()
        
        # Property statistics
        print("📈 PROPERTY STATISTICS")
        print("-" * 70)
        stats = self.get_property_statistics()
        for prop, values in stats.items():
            print(f"\n  {prop}")
            print(f"    Min: {values['min']:.2f} | Max: {values['max']:.2f} | Mean: {values['mean']:.2f}")
        print()
        
        # Strength-to-weight ratio
        print("🚀 TOP 10 STRENGTH-TO-WEIGHT RATIO (Aerospace)")
        print("-" * 70)
        sw_ratio = self.get_strength_to_weight_ratio()
        for idx, (_, mat) in enumerate(sw_ratio.iterrows(), 1):
            print(f"  {idx:2d}. {mat['Material']:.<30} {mat['Strength_to_Weight_Ratio']:>8.1f}")
        print()
        
        # Cost-to-performance
        print("💰 TOP 10 COST-EFFECTIVE MATERIALS (Performance/Cost)")
        print("-" * 70)
        cost_perf = self.get_cost_to_performance_ratio()
        for idx, (_, mat) in enumerate(cost_perf.iterrows(), 1):
            print(f"  {idx:2d}. {mat['Material']:.<30} {mat['Performance_per_Cost']:>8.1f}")
        print()
        
        # Corrosion resistant
        print("🛡️  CORROSION-RESISTANT MATERIALS (Rating ≥ 9/10)")
        print("-" * 70)
        corr_resistant = self.get_corrosion_resistant_materials()
        for idx, (_, mat) in enumerate(corr_resistant.iterrows(), 1):
            print(f"  {idx:2d}. {mat['Material']:.<30} Rating: {mat['Corrosion_Resistance_0_10']:.1f}/10")
        print()
        
        # Thermal conductors
        print("🔥 BEST THERMAL CONDUCTORS (>100 W/mK)")
        print("-" * 70)
        thermal = self.get_thermal_conductors()
        for idx, (_, mat) in enumerate(thermal.iterrows(), 1):
            print(f"  {idx:2d}. {mat['Material']:.<30} {mat['Thermal_Conductivity_W_mK']:>8.1f} W/mK")
        print()
        
        # Strength ranking
        print("💪 TOP 15 STRONGEST MATERIALS (Tensile Strength)")
        print("-" * 70)
        strength = self.get_strength_ranking()
        for idx, (_, mat) in enumerate(strength.iterrows(), 1):
            print(f"  {idx:2d}. {mat['Material']:.<25} {mat['Tensile_Strength_MPa']:>8.0f} MPa")
        print()
        
        # Density categories
        print("⚖️  MATERIALS BY DENSITY CATEGORY")
        print("-" * 70)
        density = self.get_material_density_range()
        for category, materials in density.items():
            print(f"  {category}")
            for mat in materials:
                print(f"    • {mat}")
        print()
        
        # Use case recommendations
        print("🎯 RECOMMENDED MATERIALS BY USE CASE")
        print("-" * 70)
        use_cases = self.get_material_recommendations_by_use_case()
        for use_case, materials in use_cases.items():
            print(f"  {use_case}:")
            for mat in materials:
                print(f"    • {mat['Material']}")
        print()
        
        print("="*70)
        print("END OF REPORT")
        print("="*70 + "\n")
    
    def save_report(self, filename='material_analysis_report.json'):
        """Save analysis report to JSON"""
        # Ensure all data is JSON serializable
        json_report = {}
        for key, value in self.report.items():
            if isinstance(value, dict):
                json_report[key] = value
            elif isinstance(value, (list, pd.DataFrame)):
                if isinstance(value, pd.DataFrame):
                    json_report[key] = value.to_dict('records')
                else:
                    json_report[key] = value
            else:
                json_report[key] = str(value)
        
        with open(filename, 'w') as f:
            json.dump(json_report, f, indent=4)
        
        print(f"✓ Report saved to: {filename}")


def main():
    """Main analysis script"""
    analyzer = MaterialAnalyzer('materials.csv')
    analyzer.generate_report()
    analyzer.save_report()


if __name__ == "__main__":
    main()
