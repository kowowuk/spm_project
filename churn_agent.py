"""
Customer Churn Prediction Agent - FINAL VERSION
Uses Real Telco Customer Churn Dataset
Optimized for MacBook Air with dataset present
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionAgent:
    """
    Production-ready churn prediction agent using real Telco dataset
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def load_data(self, filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        """Load Telco Customer Churn dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   Churn Rate: {(df['Churn']=='Yes').sum() / len(df) * 100:.2f}%")
            return df
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found!")
            print(f"   Please ensure the dataset is in the same directory.")
            return None
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        df = df.copy()
        
        print("\n🔧 Preprocessing data...")
        
        # Handle TotalCharges - convert to numeric (has spaces for some customers)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
        
        # Drop customerID
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Convert target to binary
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        print(f"   ✓ Handled missing values: {df.isnull().sum().sum()} nulls remaining")
        print(f"   ✓ Target converted to binary (0/1)")
        
        return df
    
    def feature_engineering(self, df):
        """Engineer additional features"""
        df = df.copy()
        
        print("   ✓ Engineering features...")
        
        # Tenure groups
        df['tenure_group'] = pd.cut(df['tenure'], 
                                     bins=[0, 12, 24, 48, 73], 
                                     labels=['0-12', '12-24', '24-48', '48+'])
        
        # Charges per tenure month
        df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        # Total services count
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies']
        
        df['total_services'] = 0
        for col in service_cols:
            if col in df.columns:
                df['total_services'] += (df[col] == 'Yes').astype(int)
        
        # Has tech support
        df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
        
        # Has security features
        df['has_security'] = ((df['OnlineSecurity'] == 'Yes') | 
                             (df['DeviceProtection'] == 'Yes')).astype(int)
        
        return df
    
    def encode_features(self, df, fit=True):
        """Encode categorical variables"""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col not in ['Churn']:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        df[col] = le.transform(df[col])
        
        return df
    
    def train_model(self, df, model_type='gradient_boosting'):
        """Train the churn prediction model"""
        
        print("\n" + "="*70)
        print("🚀 TRAINING CHURN PREDICTION MODEL")
        print("="*70)
        
        # Preprocess
        df = self.preprocess_data(df)
        df = self.feature_engineering(df)
        df = self.encode_features(df, fit=True)
        
        # Split features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        self.feature_columns = X.columns.tolist()
        print(f"\n📊 Using {len(self.feature_columns)} features for training")
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\n⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\n🎯 Training {model_type} model...")
        
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n📈 Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "="*70)
        print("📊 MODEL PERFORMANCE")
        print("="*70)
        print(f"\n   Accuracy:  {accuracy*100:.2f}%")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], digits=3)}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 10 Most Important Features:")
            for idx, row in feature_imp.head(10).iterrows():
                print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        self.is_trained = True
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETED!")
        print("="*70)
        
        return {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict_single_customer(self, customer_data):
        """
        Predict churn for a single customer
        
        Args:
            customer_data: dict with customer features
        
        Returns:
            dict with prediction results
        """
        if not self.is_trained:
            raise Exception("❌ Model not trained! Please train the model first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        df = self.feature_engineering(df)
        df = self.encode_features(df, fit=False)
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        # Scale
        X_scaled = self.scaler.transform(df)
        
        # Predict
        churn_prob = self.model.predict_proba(X_scaled)[0][1]
        churn_pred = self.model.predict(X_scaled)[0]
        
        # Risk category
        if churn_prob >= 0.7:
            risk_category = "High"
        elif churn_prob >= 0.4:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        # Get feature importance for this customer
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            top_feature_names = [f[0] for f in top_features]
        else:
            top_feature_names = []
        
        result = {
            'customer_id': customer_data.get('customerID', 'N/A'),
            'churn_probability': round(float(churn_prob), 4),
            'will_churn': bool(churn_pred),
            'risk_category': risk_category,
            'confidence': 'High' if abs(churn_prob - 0.5) > 0.3 else 'Medium' if abs(churn_prob - 0.5) > 0.15 else 'Low',
            'top_risk_factors': top_feature_names,
            'recommendation': self._generate_recommendation(risk_category, customer_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _generate_recommendation(self, risk_category, customer_data):
        """Generate retention recommendation based on risk"""
        if risk_category == "High":
            return {
                "action": "Immediate Intervention Required",
                "priority": "Critical",
                "suggestions": [
                    "Offer 20% discount for next 3 months",
                    "Assign dedicated account manager",
                    "Provide free tech support for 6 months",
                    "Upgrade to annual contract with incentives"
                ]
            }
        elif risk_category == "Medium":
            return {
                "action": "Proactive Engagement",
                "priority": "High",
                "suggestions": [
                    "Send customer satisfaction survey",
                    "Offer 10-15% discount on upgrade",
                    "Provide personalized service recommendations",
                    "Schedule check-in call"
                ]
            }
        else:
            return {
                "action": "Regular Monitoring",
                "priority": "Low",
                "suggestions": [
                    "Continue excellent service",
                    "Send loyalty rewards",
                    "Quarterly engagement emails"
                ]
            }
    
    def save_model(self, filepath='churn_model_trained.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")
    
    def load_model(self, filepath='churn_model_trained.pkl'):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            print(f"✅ Model loaded from: {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file '{filepath}' not found!")
            return False


# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 CUSTOMER CHURN PREDICTION AGENT")
    print("="*70)
    
    # Initialize agent
    agent = ChurnPredictionAgent()
    
    # Load dataset
    print("\n📂 Loading Telco Customer Churn dataset...")
    df = agent.load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    if df is None:
        print("\n❌ Cannot proceed without dataset. Exiting...")
        exit(1)
    
    # Train model
    metrics = agent.train_model(df, model_type='gradient_boosting')
    
    # Save model
    agent.save_model('churn_model_trained.pkl')
    
    # Example predictions
    print("\n" + "="*70)
    print("🧪 TESTING WITH SAMPLE CUSTOMERS")
    print("="*70)
    
    # High risk customer
    print("\n📍 Test Case 1: HIGH RISK Customer")
    high_risk = {
        'gender': 'Female',
        'SeniorCitizen': 1,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.5,
        'TotalCharges': 95.5
    }
    
    result1 = agent.predict_single_customer(high_risk)
    print(f"   Churn Probability: {result1['churn_probability']*100:.2f}%")
    print(f"   Risk Category: {result1['risk_category']}")
    print(f"   Will Churn: {'YES ⚠️' if result1['will_churn'] else 'NO ✅'}")
    
    # Low risk customer
    print("\n📍 Test Case 2: LOW RISK Customer")
    low_risk = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 70,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 105.5,
        'TotalCharges': 7385.0
    }
    
    result2 = agent.predict_single_customer(low_risk)
    print(f"   Churn Probability: {result2['churn_probability']*100:.2f}%")
    print(f"   Risk Category: {result2['risk_category']}")
    print(f"   Will Churn: {'YES ⚠️' if result2['will_churn'] else 'NO ✅'}")
    
    print("\n" + "="*70)
    print("✅ AGENT READY FOR DEPLOYMENT!")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Run test_agent.py for comprehensive testing")
    print("   2. Use the model for predictions")
    print("   3. Integrate with your application")
    print("\n" + "="*70)