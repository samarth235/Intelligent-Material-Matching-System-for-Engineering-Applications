"""
Intelligent Material Selection & Alloy Recommendation System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender import MaterialRecommender
import numpy as np
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Material Recommender",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    db_path = 'materials_large.csv' if os.path.exists('materials_large.csv') else 'materials.csv'
    st.session_state.recommender = MaterialRecommender(db_path)

recommender = st.session_state.recommender

# Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <h1>⚙️ Intelligent Material Selection System</h1>
    <p style='text-align: center; color: gray;'>
    Intelligent Recommendation Platform for Engineering Materials
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("🔧 Configuration")
    
    mode = st.radio(
        "Select Mode:",
        ["Quick Recommendation", "Advanced Search", "Material Database", "Comparison"],
        help="Choose how you want to find materials"
    )
    
    st.markdown("---")
    
    if mode == "Quick Recommendation":
        st.markdown("### 📋 Set Priorities (1-10)")
        
        strength = st.slider(
            "Strength Priority",
            1, 10, 7,
            help="How important is material strength?"
        )
        
        weight = st.slider(
            "Weight Priority (Light = Important)",
            1, 10, 7,
            help="How important is low weight/density?"
        )
        
        corrosion = st.slider(
            "Corrosion Resistance Priority",
            1, 10, 5,
            help="How important is corrosion resistance?"
        )
        
        cost = st.slider(
            "Cost Priority (Low Cost = Important)",
            1, 10, 3,
            help="How important is low cost?"
        )
        
        thermal = st.slider(
            "Thermal Conductivity Priority",
            1, 10, 3,
            help="How important is thermal performance?"
        )
        
        application = st.selectbox(
            "Application Domain:",
            ["All"] + recommender.get_available_applications()
        )
        
        recommend_button = st.button("🚀 Get Recommendations", use_container_width=True)

# Main content
if mode == "Quick Recommendation":
    st.header("📊 Material Recommendations")
    
    if recommend_button:
        with st.spinner("Analyzing materials..."):
            app_filter = None if application == "All" else application
            
            # Get recommendations
            recommendations = recommender.get_weighted_recommendations(
                strength_weight=strength,
                weight_weight=weight,
                corrosion_weight=corrosion,
                cost_weight=cost,
                thermal_weight=thermal,
                top_n=5,
                application_filter=app_filter
            )
        
        if len(recommendations) == 0:
            st.error(f"❌ No materials found for {application} application.")
        else:
            # Display top 3 recommendations
            st.subheader("🏆 Top 3 Recommendations")
            
            cols = st.columns(3)
            
            for idx, (i, row) in enumerate(recommendations.head(3).iterrows()):
                with cols[idx]:
                    st.markdown(f"### #{idx+1}: {row['Material']}")
                    st.metric("Score", f"{row['Recommendation_Score']:.2f}", "✓ Recommended")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Strength", f"{row['Tensile_Strength_MPa']:.0f} MPa")
                        st.metric("Density", f"{row['Density_g_cm3']:.2f} g/cm³")
                    
                    with col_b:
                        st.metric("Corrosion", f"{row['Corrosion_Resistance_0_10']:.1f}/10")
                        st.metric("Cost", f"{row['Cost_Index_1_10']:.1f}/10")
            
            st.markdown("---")
            
            # Detailed analysis
            st.subheader("📈 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Comparison Chart", "Material Details", "Properties Table", "Selection Reasoning"]
            )
            
            with tab1:
                st.markdown("### Radar Chart - Property Comparison")
                
                # Prepare data for radar chart
                top_3 = recommendations.head(3)
                
                fig = go.Figure()
                
                categories = [
                    'Strength (norm)',
                    'Density (inv norm)',
                    'Corrosion (norm)',
                    'Cost (inv norm)',
                    'Thermal (norm)',
                    'Hardness (norm)'
                ]
                
                for _, material in top_3.iterrows():
                    values = [
                        material['Tensile_Strength_MPa_norm'],
                        10 - material['Density_g_cm3_norm'],
                        material['Corrosion_Resistance_0_10_norm'],
                        10 - material['Cost_Index_1_10_norm'],
                        material['Thermal_Conductivity_W_mK_norm'],
                        material['Hardness_HB_norm']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=material['Material']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_strength = px.bar(
                        top_3,
                        x='Material',
                        y='Tensile_Strength_MPa',
                        title='Tensile Strength Comparison',
                        color='Tensile_Strength_MPa',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_strength, use_container_width=True)
                
                with col2:
                    fig_density = px.bar(
                        top_3,
                        x='Material',
                        y='Density_g_cm3',
                        title='Density Comparison (Lower is Better)',
                        color='Density_g_cm3',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_density, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    fig_corrosion = px.bar(
                        top_3,
                        x='Material',
                        y='Corrosion_Resistance_0_10',
                        title='Corrosion Resistance',
                        color='Corrosion_Resistance_0_10',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_corrosion, use_container_width=True)
                
                with col4:
                    fig_cost = px.bar(
                        top_3,
                        x='Material',
                        y='Cost_Index_1_10',
                        title='Cost Index (Lower is Better)',
                        color='Cost_Index_1_10',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
            
            with tab2:
                st.markdown("### Material Details & Properties")
                
                selected_material = st.selectbox(
                    "Select Material for Details:",
                    top_3['Material'].tolist()
                )
                
                explanation = recommender.get_material_explanation(selected_material)
                st.markdown(explanation)
            
            with tab3:
                st.markdown("### Detailed Properties Table")
                
                # Show all properties
                display_columns = [
                    'Material', 'Alloy_Type', 'Density_g_cm3',
                    'Yield_Strength_MPa', 'Tensile_Strength_MPa',
                    'Hardness_HB', 'Corrosion_Resistance_0_10',
                    'Thermal_Conductivity_W_mK', 'Cost_Index_1_10',
                    'Melting_Point_C', 'Machinability'
                ]
                
                st.dataframe(
                    top_3[display_columns],
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab4:
                st.markdown("### Why These Materials?")
                
                for idx, (_, material) in enumerate(top_3.head(3).iterrows()):
                    st.markdown(f"#### #{idx+1} - {material['Material']}")
                    
                    reasons = []
                    
                    if strength >= 7 and material['Tensile_Strength_MPa'] > 600:
                        reasons.append("✓ High strength meets your requirements")
                    elif strength >= 5 and material['Tensile_Strength_MPa'] > 400:
                        reasons.append("✓ Good strength for the application")
                    
                    if weight >= 7 and material['Density_g_cm3'] < 5:
                        reasons.append("✓ Lightweight - excellent for weight-critical applications")
                    
                    if corrosion >= 7 and material['Corrosion_Resistance_0_10'] >= 9:
                        reasons.append("✓ Excellent corrosion resistance")
                    elif corrosion >= 5 and material['Corrosion_Resistance_0_10'] >= 7:
                        reasons.append("✓ Good corrosion resistance")
                    
                    if cost <= 3 and material['Cost_Index_1_10'] <= 4:
                        reasons.append("✓ Cost-effective option")
                    
                    if thermal >= 7 and material['Thermal_Conductivity_W_mK'] > 100:
                        reasons.append("✓ Excellent thermal conductivity")
                    elif thermal >= 5 and material['Thermal_Conductivity_W_mK'] > 30:
                        reasons.append("✓ Good thermal performance")
                    
                    if material['Machinability'] >= 8:
                        reasons.append("✓ Easy to machine and process")
                    
                    for reason in reasons:
                        st.write(reason)
                    
                    st.markdown("---")

elif mode == "Advanced Search":
    st.header("🔍 Advanced Material Search")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.radio(
            "Search By:",
            ["Strength", "Weight", "Corrosion", "Cost", "Thermal"]
        )
    
    with col2:
        min_val = st.number_input("Minimum Value:", value=0.0)
    
    with col3:
        max_val = st.number_input("Maximum Value:", value=1000.0)
    
    # Map search types to columns
    search_column_map = {
        "Strength": "Tensile_Strength_MPa",
        "Weight": "Density_g_cm3",
        "Corrosion": "Corrosion_Resistance_0_10",
        "Cost": "Cost_Index_1_10",
        "Thermal": "Thermal_Conductivity_W_mK"
    }
    
    search_column = search_column_map[search_type]
    
    filtered_materials = recommender.materials_df[
        (recommender.materials_df[search_column] >= min_val) &
        (recommender.materials_df[search_column] <= max_val)
    ].sort_values(search_column, ascending=False)
    
    st.markdown(f"### Found {len(filtered_materials)} materials")
    
    st.dataframe(
        filtered_materials[[
            'Material', 'Alloy_Type', search_column, 'Primary_Application'
        ]],
        use_container_width=True,
        hide_index=True
    )

elif mode == "Material Database":
    st.header("📚 Complete Material Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("Search materials by name...")
    
    with col2:
        application_filter = st.selectbox(
            "Filter by Application:",
            ["All"] + recommender.get_available_applications()
        )
    
    # Filter database
    filtered_db = recommender.materials_df.copy()
    
    if search_term:
        filtered_db = filtered_db[
            filtered_db['Material'].str.contains(search_term, case=False, na=False)
        ]
    
    if application_filter != "All":
        filtered_db = filtered_db[
            filtered_db['Primary_Application'].str.contains(
                application_filter, case=False, na=False
            )
        ]
    
    st.markdown(f"### Total Materials: {len(filtered_db)}")
    
    st.dataframe(filtered_db, use_container_width=True, hide_index=True)
    
    # Download button
    csv = filtered_db.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"materials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

elif mode == "Comparison":
    st.header("⚖️ Material Comparison")
    
    st.markdown("### Select materials to compare:")
    
    all_materials = recommender.get_material_names()
    
    col1, col2 = st.columns(2)
    
    with col1:
        material1 = st.selectbox("Material 1:", all_materials, key="mat1")
    
    with col2:
        material2 = st.selectbox("Material 2:", all_materials, index=1, key="mat2")
    
    material3_enabled = st.checkbox("Add Material 3?")
    if material3_enabled:
        material3 = st.selectbox("Material 3:", all_materials, index=2, key="mat3")
        materials_to_compare = [material1, material2, material3]
    else:
        materials_to_compare = [material1, material2]
    
    st.markdown("---")
    
    comparison = recommender.compare_materials(materials_to_compare)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            comparison,
            x='Material',
            y='Tensile_Strength_MPa',
            title='Tensile Strength',
            color='Material'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            comparison,
            x='Material',
            y='Density_g_cm3',
            title='Density (Lower is Better)',
            color='Material'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig = px.bar(
            comparison,
            x='Material',
            y='Corrosion_Resistance_0_10',
            title='Corrosion Resistance',
            color='Material'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = px.bar(
            comparison,
            x='Material',
            y='Cost_Index_1_10',
            title='Cost Index (Lower is Better)',
            color='Material'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Comparison Table")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 2rem;'>
    <p>🔬 Intelligent Material Selection & Alloy Recommendation System</p>
    <p>Developed for Engineering Material Selection</p>
    <p style='font-size: 0.8rem;'>Data sourced from material databases and engineering handbooks</p>
</div>
""", unsafe_allow_html=True)
