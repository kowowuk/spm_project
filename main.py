"""
Enhanced FastAPI Churn Prediction API
Features: Multiple prediction types, customizable recommendations, batch processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import traceback

# Import your agent
from churn_agent import ChurnPredictionAgent

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API with multiple prediction types and recommendations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent = None
MODEL_PATH = 'churn_model_trained.pkl'
DATASET_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Enums for better validation
class PredictionType(str, Enum):
    standard = "standard"  # Standard prediction with recommendations
    probability_only = "probability_only"  # Just churn probability
    detailed = "detailed"  # Full analysis with top features and recommendations

class RecommendationType(str, Enum):
    aggressive = "aggressive"  # Strong retention offers
    balanced = "balanced"  # Standard recommendations
    conservative = "conservative"  # Minimal intervention

class RiskLevel(str, Enum):
    high = "High"
    medium = "Medium"
    low = "Low"

# Pydantic Models
class CustomerData(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=850.0)
    customerID: Optional[str] = Field(None, example="CUST-001")

class PredictionRequest(BaseModel):
    customer: CustomerData
    prediction_type: PredictionType = Field(
        default=PredictionType.standard,
        description="Type of prediction analysis"
    )
    recommendation_type: RecommendationType = Field(
        default=RecommendationType.balanced,
        description="Level of retention recommendations"
    )

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]
    prediction_type: PredictionType = Field(default=PredictionType.standard)
    recommendation_type: RecommendationType = Field(default=RecommendationType.balanced)

class RetrainRequest(BaseModel):
    file_path: Optional[str] = Field(
        default=DATASET_PATH,
        description="Path to training dataset"
    )
    model_type: Literal["gradient_boosting", "random_forest"] = Field(
        default="gradient_boosting",
        description="Type of model to train"
    )

# Enhanced recommendation generator
def generate_custom_recommendation(
    risk_category: str,
    customer_data: dict,
    recommendation_type: str
) -> dict:
    """Generate customized recommendations based on risk and type"""
    
    recommendations = {
        "High": {
            "aggressive": {
                "action": "URGENT: Immediate Executive Intervention",
                "priority": "Critical",
                "discount": "30%",
                "duration": "6 months",
                "suggestions": [
                    "Offer 30% discount for next 6 months",
                    "Assign VP-level account manager",
                    "Free premium tech support for 12 months",
                    "Waive all installation/upgrade fees",
                    "Lock-in 2-year contract with 25% lifetime discount",
                    "Provide free device upgrades",
                    "Personal quarterly business reviews"
                ]
            },
            "balanced": {
                "action": "Immediate Intervention Required",
                "priority": "Critical",
                "discount": "20%",
                "duration": "3 months",
                "suggestions": [
                    "Offer 20% discount for next 3 months",
                    "Assign dedicated account manager",
                    "Provide free tech support for 6 months",
                    "Upgrade to annual contract with incentives",
                    "Schedule retention call within 48 hours"
                ]
            },
            "conservative": {
                "action": "Proactive Customer Engagement",
                "priority": "High",
                "discount": "10%",
                "duration": "2 months",
                "suggestions": [
                    "Send personalized retention offer (10% discount)",
                    "Schedule satisfaction survey call",
                    "Offer service plan optimization",
                    "Provide loyalty rewards"
                ]
            }
        },
        "Medium": {
            "aggressive": {
                "action": "Proactive Premium Engagement",
                "priority": "High",
                "discount": "20%",
                "duration": "3 months",
                "suggestions": [
                    "Offer 20% discount on service upgrade",
                    "Free premium features trial for 3 months",
                    "Dedicated support line access",
                    "Quarterly check-in calls",
                    "Early access to new services"
                ]
            },
            "balanced": {
                "action": "Proactive Engagement",
                "priority": "High",
                "discount": "10-15%",
                "duration": "2 months",
                "suggestions": [
                    "Send customer satisfaction survey",
                    "Offer 10-15% discount on upgrade",
                    "Provide personalized service recommendations",
                    "Schedule check-in call",
                    "Send quarterly engagement emails"
                ]
            },
            "conservative": {
                "action": "Standard Monitoring",
                "priority": "Medium",
                "discount": "5%",
                "duration": "1 month",
                "suggestions": [
                    "Send service optimization tips",
                    "Offer minor upgrade incentive (5% discount)",
                    "Include in regular feedback surveys"
                ]
            }
        },
        "Low": {
            "aggressive": {
                "action": "Loyalty Enhancement Program",
                "priority": "Medium",
                "discount": "5%",
                "duration": "ongoing",
                "suggestions": [
                    "Enroll in premium loyalty program",
                    "Offer exclusive early access to features",
                    "Provide referral incentives (5% discount)",
                    "VIP customer service tier upgrade"
                ]
            },
            "balanced": {
                "action": "Regular Monitoring",
                "priority": "Low",
                "discount": "N/A",
                "duration": "ongoing",
                "suggestions": [
                    "Continue excellent service",
                    "Send loyalty rewards",
                    "Quarterly engagement emails",
                    "Include in satisfaction surveys"
                ]
            },
            "conservative": {
                "action": "Maintain Status Quo",
                "priority": "Low",
                "discount": "N/A",
                "duration": "ongoing",
                "suggestions": [
                    "Continue current service level",
                    "Annual satisfaction check",
                    "Standard communication cadence"
                ]
            }
        }
    }
    
    rec = recommendations.get(risk_category, {}).get(recommendation_type, {})
    
    # Add customer-specific insights
    insights = []
    if customer_data.get('Contract') == 'Month-to-month':
        insights.append("Contract type is month-to-month - high flexibility risk")
    if customer_data.get('tenure', 0) < 12:
        insights.append("New customer (tenure < 12 months) - critical retention period")
    if customer_data.get('TechSupport') == 'No':
        insights.append("No tech support - consider offering support package")
    if customer_data.get('PaymentMethod') == 'Electronic check':
        insights.append("Electronic check payment - consider auto-pay incentive")
    
    rec['customer_insights'] = insights
    return rec

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global agent
    try:
        agent = ChurnPredictionAgent()
        
        if os.path.exists(MODEL_PATH):
            agent.load_model(MODEL_PATH)
            print("✅ Model loaded successfully on startup")
        else:
            print("⚠️  Model file not found. Will train on first request.")
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        agent = ChurnPredictionAgent()

# Root endpoint
@app.get("/")
async def root():
    """API health check and info"""
    return {
        "status": "active",
        "service": "Customer Churn Prediction API v2.0",
        "model_loaded": agent.is_trained if agent else False,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "prediction_types": ["standard", "probability_only", "detailed"],
            "recommendation_types": ["aggressive", "balanced", "conservative"],
            "batch_processing": True,
            "model_retraining": True
        },
        "endpoints": {
            "docs": "/docs",
            "health": "GET /health",
            "predict": "POST /api/v1/predict",
            "batch_predict": "POST /api/v1/predict/batch",
            "retrain": "POST /api/v1/retrain",
            "model_info": "GET /api/v1/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": agent.is_trained if agent else False,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "dataset_path_exists": os.path.exists(DATASET_PATH),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/predict")
async def predict_churn(request: PredictionRequest):
    """
    Predict churn for a single customer with customizable options
    
    - **prediction_type**: standard, probability_only, or detailed
    - **recommendation_type**: aggressive, balanced, or conservative
    """
    try:
        if not agent or not agent.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please retrain the model first."
            )
        
        customer_dict = request.customer.dict()
        
        # Get base prediction
        result = agent.predict_single_customer(customer_dict)
        
        # Customize based on prediction_type
        if request.prediction_type == PredictionType.probability_only:
            return {
                "status": "success",
                "customer_id": result['customer_id'],
                "churn_probability": result['churn_probability'],
                "will_churn": result['will_churn'],
                "timestamp": result['timestamp']
            }
        
        # Generate custom recommendations
        custom_rec = generate_custom_recommendation(
            result['risk_category'],
            customer_dict,
            request.recommendation_type.value
        )
        result['recommendation'] = custom_rec
        
        if request.prediction_type == PredictionType.detailed:
            # Add extra analysis
            result['analysis'] = {
                "tenure_category": "New" if customer_dict['tenure'] < 12 else "Established" if customer_dict['tenure'] < 48 else "Loyal",
                "contract_risk": "High" if customer_dict['Contract'] == 'Month-to-month' else "Low",
                "service_engagement": "High" if customer_dict.get('StreamingTV') == 'Yes' or customer_dict.get('StreamingMovies') == 'Yes' else "Low",
                "support_usage": "Yes" if customer_dict.get('TechSupport') == 'Yes' else "No"
            }
        
        return {
            "status": "success",
            "prediction": result,
            "request_type": request.prediction_type.value,
            "recommendation_level": request.recommendation_type.value,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/api/v1/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple customers
    """
    try:
        if not agent or not agent.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        predictions = []
        stats = {
            "total": len(request.customers),
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "churn_predicted": 0
        }
        
        for customer in request.customers:
            customer_dict = customer.dict()
            result = agent.predict_single_customer(customer_dict)
            
            # Custom recommendations
            custom_rec = generate_custom_recommendation(
                result['risk_category'],
                customer_dict,
                request.recommendation_type.value
            )
            result['recommendation'] = custom_rec
            
            predictions.append(result)
            
            # Update stats
            if result['risk_category'] == 'High':
                stats['high_risk'] += 1
            elif result['risk_category'] == 'Medium':
                stats['medium_risk'] += 1
            else:
                stats['low_risk'] += 1
            
            if result['will_churn']:
                stats['churn_predicted'] += 1
        
        # Calculate probabilities
        churn_probs = [p['churn_probability'] for p in predictions]
        stats['avg_churn_probability'] = float(np.mean(churn_probs))
        stats['max_churn_probability'] = float(np.max(churn_probs))
        stats['min_churn_probability'] = float(np.min(churn_probs))
        
        return {
            "status": "success",
            "predictions": predictions,
            "statistics": stats,
            "recommendation_type": request.recommendation_type.value,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

@app.post("/api/v1/retrain")
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Retrain the model with new data
    """
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset not found: {request.file_path}"
            )
        
        # Load data
        df = pd.read_csv(request.file_path)
        
        # Retrain
        metrics = agent.train_model(df, model_type=request.model_type)
        
        # Save model
        agent.save_model(MODEL_PATH)
        
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics,
            "model_type": request.model_type,
            "training_samples": len(df),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retraining error: {str(e)}"
        )

@app.get("/api/v1/model/info")
async def model_info():
    """Get current model information"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "status": "success",
        "model_info": {
            "is_trained": agent.is_trained,
            "feature_count": len(agent.feature_columns) if agent.feature_columns else 0,
            "features": agent.feature_columns if agent.feature_columns else [],
            "model_type": type(agent.model).__name__ if agent.model else None
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)