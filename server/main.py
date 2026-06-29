import os
import time
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import mlflow.pyfunc
import numpy as np
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

app = FastAPI(title="Production Secure Inference API Service")

# --- SECURITY ENGINE CONFIGURATION ---
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "SUPER_SECRET_MESH_HEX_KEY_DO_NOT_LEAK")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize secure dependency helpers
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Static admin database mock container for infrastructure isolation
MOCK_USERS_DB = {
    "admin_mlops": {
        "username": "admin_mlops",
        # Hashed representation of the password: "production_password_2026"
        "hashed_password": pwd_context.hash("production_password_2026")
    }
}

# Define strict input verification schemas using Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str

class InferencePayload(BaseModel):
    features: list[float]


# --- UTILITY JWT SECURITY FUNCTIONS ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency that intercepts incoming headers, parsing and decoding token strings."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Missing or malformed authentication signature.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in MOCK_USERS_DB:
            raise credentials_exception
        return username
    except jwt.PyJWTError:
        raise credentials_exception


# --- AUTHENTICATION ENDPOINT ---
@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticates infrastructure requests and distributes time-sensitive tokens."""
    user = MOCK_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password combination.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


# --- CORE ML INTERFACE ENDPOINTS (UPGRADED) ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
MODEL_URI = "models:/Core_Production_Classifier/latest"
production_model = None

@app.on_event("startup")
def load_production_model_on_boot():
    global production_model
    try:
        production_model = mlflow.pyfunc.load_model(MODEL_URI)
        print("[BOOT SUCCESS] Model successfully linked to protected inference app.")
    except Exception:
        print("[BOOT WARNING] Active registry node unavailable. Starting engine in degraded fallback mode.")

@app.post("/predict")
def get_secure_prediction(payload: InferencePayload, current_user: str = Depends(get_current_user)):
    """Processes array vectors ONLY if the client passes a valid, verified JWT signature."""
    global production_model
    
    if production_model is None:
        try:
            production_model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception:
            raise HTTPException(status_code=503, detail="ML model registry layer unreachable.")
            
    try:
        features = np.array(payload.features).reshape(1, -1)
        prediction = production_model.predict(features)
        
        return {
            "class_prediction": int(prediction),
            "processed_by_user": current_user
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution crash: {str(e)}")
