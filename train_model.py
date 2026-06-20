"""
Machine Learning Model Training for Material Recommendations
Uses Random Forest and Gradient Boosting with Hyperparameter Tuning,
Class Balancing, Outlier Removal, and Feature Engineering.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class MaterialMLModel:
    """
    Machine Learning models for material recommendation
    """
    
    def __init__(self, csv_path=None):
        """
        Initialize ML model trainer
        
        Args:
            csv_path: Path to materials CSV (defaults to materials_large.csv if it exists)
        """
        if csv_path is None:
            csv_path = 'materials_large.csv' if os.path.exists('materials_large.csv') else 'materials.csv'
            
        print(f"Loading database from: {csv_path}")
        self.materials_df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.best_metrics = {}
        
    def prepare_data(self, target_property='Primary_Application'):
        """
        Prepare data for model training with cleaning and feature engineering
        
        Args:
            target_property: Target variable for prediction
            
        Returns:
            X_train, X_test, y_train, y_test, feature_columns
        """
        df = self.materials_df.copy()
        
        # 1. Data Quality Checks
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Check/handle missing values
        df = df.dropna()
        
        # Remove outliers group-wise by Alloy_Type using IQR to preserve minority classes
        numeric_cols = [
            'Density_g_cm3', 'Yield_Strength_MPa', 'Tensile_Strength_MPa',
            'Hardness_HB', 'Corrosion_Resistance_0_10', 'Thermal_Conductivity_W_mK',
            'Cost_Index_1_10', 'Melting_Point_C', 'Machinability'
        ]
        
        df_cleaned = []
        for name, group in df.groupby('Alloy_Type'):
            mask = pd.Series(True, index=group.index)
            for col in numeric_cols:
                Q1 = group[col].quantile(0.25)
                Q3 = group[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    col_mask = (group[col] >= (Q1 - 1.5 * IQR)) & (group[col] <= (Q3 + 1.5 * IQR))
                    mask = mask & col_mask
            df_cleaned.append(group[mask])
            
        df = pd.concat(df_cleaned)
        print(f"Data cleaned. Rows remaining: {len(df)} (out of {len(self.materials_df)})")
        
        # Select raw features
        X = df[numeric_cols].copy()
        y = df[target_property].copy()
        
        # 2. Feature Engineering
        X['strength_weight'] = X['Tensile_Strength_MPa'] / X['Density_g_cm3']
        X['cost_perf'] = X['Tensile_Strength_MPa'] / X['Cost_Index_1_10']
        X['thermal_strength_ratio'] = X['Thermal_Conductivity_W_mK'] / X['Tensile_Strength_MPa']
        
        feature_columns = numeric_cols + ['strength_weight', 'cost_perf', 'thermal_strength_ratio']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[target_property] = le
        
        # Split data (80% train, 20% test)
        # Note: Stratify by y if possible to keep classes represented
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # Fallback if some classes have only 1 member
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_random_forest(self, target_property='Primary_Application'):
        """
        Train Random Forest model with class balancing and GridSearchCV hyperparameter tuning
        
        Args:
            target_property: Target variable
            
        Returns:
            Dictionary with model metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Random Forest (Tuned) for: {target_property}")
        print(f"{'='*60}")
        
        X_train, X_test, y_train, y_test, features = self.prepare_data(target_property)
        
        # Hyperparameter tuning
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_rf = grid.best_estimator_
        
        # Evaluate using 5-fold cross-validation on train data
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
        
        # Predictions
        y_pred = best_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Store model & importance
        self.models['random_forest'] = best_rf
        self.feature_importance['random_forest'] = dict(
            zip(features, best_rf.feature_importances_)
        )
        
        print(f"\nBest Parameters: {grid.best_params_}")
        print(f"CV Train Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        classes_in_test = np.unique(y_test)
        target_names = self.label_encoders[target_property].inverse_transform(classes_in_test)
        target_names = [str(name) for name in target_names]
        print(classification_report(y_test, y_pred, labels=classes_in_test, target_names=target_names))
        
        return {
            'model_type': 'Random Forest',
            'accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'best_params': grid.best_params_,
            'features': features,
            'feature_importance': self.feature_importance['random_forest'],
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def train_gradient_boosting(self, target_property='Primary_Application'):
        """
        Train Gradient Boosting model with GridSearchCV hyperparameter tuning
        
        Args:
            target_property: Target variable
            
        Returns:
            Dictionary with model metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Gradient Boosting (Tuned) for: {target_property}")
        print(f"{'='*60}")
        
        X_train, X_test, y_train, y_test, features = self.prepare_data(target_property)
        
        # Hyperparameter tuning
        params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid = GridSearchCV(gb, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_gb = grid.best_estimator_
        
        # Evaluate using 5-fold cross-validation on train data
        cv_scores = cross_val_score(best_gb, X_train, y_train, cv=5)
        
        # Predictions
        y_pred = best_gb.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Store model & importance
        self.models['gradient_boosting'] = best_gb
        self.feature_importance['gradient_boosting'] = dict(
            zip(features, best_gb.feature_importances_)
        )
        
        print(f"\nBest Parameters: {grid.best_params_}")
        print(f"CV Train Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        
        return {
            'model_type': 'Gradient Boosting',
            'accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'best_params': grid.best_params_,
            'features': features,
            'feature_importance': self.feature_importance['gradient_boosting'],
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def train_all_models(self):
        """
        Train all models, compare, save the best to metrics file
        
        Returns:
            Dictionary with all model metrics
        """
        results = {}
        
        print("\n" + "="*60)
        print("MATERIAL RECOMMENDATION ML MODEL TRAINING")
        print("="*60)
        
        # Train Random Forest
        results['random_forest'] = self.train_random_forest()
        
        # Train Gradient Boosting
        results['gradient_boosting'] = self.train_gradient_boosting()
        
        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_result = results[best_model_name]
        
        # Save metrics for tracking
        metrics = {
            'train_size': int(best_result['train_size']),
            'test_size': int(best_result['test_size']),
            'accuracy': float(best_result['accuracy']),
            'best_model': best_result['model_type'],
            'best_params': best_result['best_params'],
            'timestamp': str(datetime.now())
        }
        
        with open('model_performance.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✓ Saved model performance metrics to model_performance.json")
        
        return results
    
    def save_models(self, output_dir='models'):
        """
        Save trained models to disk
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved: {scaler_path}")
        
        # Save label encoders
        encoder_path = os.path.join(output_dir, 'label_encoders.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"✓ Saved: {encoder_path}")
        
        # Save feature importance
        importance_path = os.path.join(output_dir, 'feature_importance.json')
        with open(importance_path, 'w') as f:
            importance_json = {}
            for model_name, importance in self.feature_importance.items():
                importance_json[model_name] = {
                    k: float(v) for k, v in importance.items()
                }
            json.dump(importance_json, f, indent=4)
        print(f"✓ Saved: {importance_path}")
    
    def print_feature_importance(self):
        """
        Print feature importance for all trained models
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        for model_name, importance in self.feature_importance.items():
            print(f"\n{model_name.upper()}")
            print("-" * 40)
            
            sorted_importance = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for feature, importance_value in sorted_importance:
                print(f"  {feature:.<35} {importance_value:.4f}")
    
    def predict_recommendation(self, 
                              density,
                              yield_strength,
                              tensile_strength,
                              hardness,
                              corrosion,
                              thermal_conductivity,
                              cost,
                              melting_point,
                              machinability,
                              model_name='random_forest'):
        """
        Make a recommendation using trained model
        
        Args:
            All material properties (as numeric values)
            model_name: Which model to use
            
        Returns:
            Predicted application
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
            
        # Construct DataFrame to apply feature engineering
        feature_columns = [
            'Density_g_cm3',
            'Yield_Strength_MPa',
            'Tensile_Strength_MPa',
            'Hardness_HB',
            'Corrosion_Resistance_0_10',
            'Thermal_Conductivity_W_mK',
            'Cost_Index_1_10',
            'Melting_Point_C',
            'Machinability'
        ]
        
        input_df = pd.DataFrame([[
            density, yield_strength, tensile_strength,
            hardness, corrosion, thermal_conductivity,
            cost, melting_point, machinability
        ]], columns=feature_columns)
        
        # Feature engineering
        input_df['strength_weight'] = input_df['Tensile_Strength_MPa'] / input_df['Density_g_cm3']
        input_df['cost_perf'] = input_df['Tensile_Strength_MPa'] / input_df['Cost_Index_1_10']
        input_df['thermal_strength_ratio'] = input_df['Thermal_Conductivity_W_mK'] / input_df['Tensile_Strength_MPa']
        
        # Scale
        features_scaled = self.scaler.transform(input_df)
        
        # Predict
        model = self.models[model_name]
        prediction_encoded = model.predict(features_scaled)[0]
        
        # Decode
        le = self.label_encoders['Primary_Application']
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        return prediction


def main():
    """Main training script"""
    trainer = MaterialMLModel()
    
    # Train all models
    results = trainer.train_all_models()
    
    # Print feature importance
    trainer.print_feature_importance()
    
    # Save models
    trainer.save_models('models')
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    
    # Test predictions
    print("\nTEST PREDICTIONS:")
    print("-" * 60)
    
    test_cases = [
        {
            'name': 'Aerospace Material (Ti-like)',
            'props': (4.5, 880, 950, 334, 10, 7, 9, 1600, 5)
        },
        {
            'name': 'Marine Material (Steel-like)',
            'props': (8.0, 170, 485, 217, 9, 16, 5, 1425, 5)
        },
        {
            'name': 'General Purpose (Al-like)',
            'props': (2.7, 276, 310, 95, 7, 167, 3, 660, 8)
        }
    ]
    
    for test_case in test_cases:
        prediction = trainer.predict_recommendation(*test_case['props'])
        print(f"  {test_case['name']}")
        print(f"  → Recommended for: {prediction}\n")


if __name__ == "__main__":
    main()
