from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import datastore
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import io
import pdf2image
from datetime import datetime
import numpy as np
import uuid
import hashlib
import imgaug.augmenters as iaa
from typing import Optional
# Old auth removed — handled by Auth0
from auth0_validator import require_auth as auth0_require_auth
from certification import certify_document, verify_pdf, certify_pdf, is_certified
import certificate_store
from fastapi.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from pathlib import Path

app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."}
    )

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kwiddex.com", "https://www.kwiddex.com", "http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#JWT auth scheme
security = HTTPBearer(auto_error=False)

#enforce Auth0 JWT on protected endpoints (RS256)
async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validates Auth0-issued RS256 JWTs. Delegates to auth0_validator."""
    return await auth0_require_auth(credentials)


#gcp
GCP_PROJECT = "sentiment-analysis-379200"
MODEL_LOCAL_PATH = "best_real_fake_resnet18.pt"

#clients
datastore_client = datastore.Client(project=GCP_PROJECT)



#load model
def load_model():
    model = resnet18(weights=None)
    
    #match saved model (sequential with dropout)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    
    #saved weights
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    return model



try:
    model = load_model()
except Exception as e:
    print(f"Could not load model: {e}")
    print("Endpoints will return errors.")
    model = None

#check upload file size
async def check_file_size(file: UploadFile):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
    await file.seek(0)
    return content

#check that model has been loaded
def require_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. The model file could not be loaded at startup."
        )


#image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    confidence_interval: dict
    timestamp: str
    result_id: str
    document_id: str
    gcs_path: str

class MonteCarloResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    confidence_interval: dict
    monte_carlo_stats: dict
    timestamp: str
    result_id: str
    document_id: str
    gcs_path: str


class CertificationRequest(BaseModel):
    reviewer_id: Optional[str] = None
    client_reference: Optional[str] = None
    notes: Optional[str] = None
    add_visible_page: bool = True  #whether to add certificate page to PDF

class CertificationResponse(BaseModel):
    success: bool
    certificate_id: str
    issued_at: str
    document_hash: str
    confidence_score: float
    message: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    organization: Optional[str] = None
    verification_link: Optional[str] = None

class RegisterResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    message: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    token: Optional[str] = None
    message: str

#certification models
class DisputeInfo(BaseModel):
    dispute_id: str
    reporter_email: Optional[str] = None
    reason: str
    filed_at: Optional[str] = None
    dispute_status: str
    certifier_response: Optional[str] = None
    resolved_at: Optional[str] = None

class VerificationResponse(BaseModel):
    valid: bool
    has_certificate: bool
    signature_valid: Optional[bool] = None
    document_intact: Optional[bool] = None
    certificate_active: Optional[bool] = None
    message: str
    certificate_id: Optional[str] = None
    issued_at: Optional[str] = None
    confidence_score: Optional[float] = None
    reviewer_id: Optional[str] = None
    organization: Optional[str] = None
    verification_link: Optional[str] = None
    disputes: Optional[list] = None
    has_disputes: bool = False

class CertificateLookupResponse(BaseModel):
    found: bool
    certificate_id: Optional[str] = None
    confidence_score: Optional[float] = None
    reviewer_id: Optional[str] = None
    organization: Optional[str] = None
    verification_link: Optional[str] = None
    status: Optional[str] = None
    message: str

class RevokeResponse(BaseModel):
    success: bool
    certificate_id: str
    message: str








def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)





def compute_confidence_interval(probabilities: np.ndarray, confidence_level=0.95) -> dict:
    confidence = float(np.max(probabilities))
    predicted_class = int(np.argmax(probabilities))
    
    #confidence interval based on softmax probabilities
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    std_estimate = np.sqrt(confidence * (1 - confidence))
    margin = z_score * std_estimate
    
    return {
        "mean": confidence,
        "lower_bound": max(0.0, confidence - margin),
        "upper_bound": min(1.0, confidence + margin),
        "confidence_level": confidence_level
    }


def save_to_datastore(prediction: int, label: str, confidence: float, ci: dict, 
                      filename: str, document_id: str, gcs_path: str, monte_carlo_stats: dict = None) -> str:
    key = datastore_client.key("PredictionResult")
    entity = datastore.Entity(key)
    entity_data = {
        "document_id": document_id,
        "gcs_path": gcs_path,
        "original_filename": filename,
        "prediction": prediction,
        "prediction_label": label,
        "confidence": confidence,
        "confidence_interval": ci,
        "timestamp": datetime.utcnow(),
        "processed": True,
        "method": "monte_carlo" if monte_carlo_stats else "standard"
    }
    if monte_carlo_stats:
        entity_data["monte_carlo_stats"] = monte_carlo_stats
    entity.update(entity_data)
    datastore_client.put(entity)
    return str(entity.key.id)




def apply_augmentations(image: Image.Image, num_augmentations: int = 30) -> list:
    img_np = np.array(image)
    augmenter = iaa.SomeOf((1, 3), [
        iaa.Rotate((-10, 10)),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Affine(scale=(0.95, 1.05)),
        iaa.JpegCompression(compression=(70, 99))
    ], random_order=True)
    augmented_images = []
    for _ in range(num_augmentations):
        aug_img = augmenter(image=img_np)
        augmented_images.append(Image.fromarray(aug_img))
    return augmented_images




def monte_carlo_inference(image: Image.Image, num_samples: int = 30) -> dict:
    augmented_images = apply_augmentations(image, num_samples)
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for aug_img in augmented_images:
            processed = preprocess_image(aug_img)
            outputs = model(processed)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]
            all_predictions.append(int(np.argmax(probs_np)))
            all_probabilities.append(probs_np)
    
    all_probabilities = np.array(all_probabilities)
    mean_probs = np.mean(all_probabilities, axis=0)
    std_probs = np.std(all_probabilities, axis=0)
    final_prediction = int(np.argmax(mean_probs))
    final_confidence = float(mean_probs[final_prediction])
    agreement_rate = float(np.sum(np.array(all_predictions) == final_prediction) / len(all_predictions))
    percentile_lower = np.percentile(all_probabilities[:, final_prediction], 2.5)
    percentile_upper = np.percentile(all_probabilities[:, final_prediction], 97.5)
    
    

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "confidence_interval": {
            "mean": final_confidence,
            "lower_bound": float(percentile_lower),
            "upper_bound": float(percentile_upper),
            "confidence_level": 0.95
        },
        "monte_carlo_stats": {
            "num_samples": num_samples,
            "agreement_rate": agreement_rate,
            "std_dev": float(std_probs[final_prediction]),
            "class_probabilities": {
                "fake": float(mean_probs[0]),
                "real": float(mean_probs[1])
            }
        }
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    require_model()
    try:
        content = await file.read()
        
        document_id = str(uuid.uuid4())
        gcs_path = ""
        
        #process the document (pdf or image)
        if file.content_type == "application/pdf":
            #convert first page of PDF to image
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        else:
            #open as image
            image = Image.open(io.BytesIO(content))
        
        processed_image = preprocess_image(image)
        
        #run inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy()[0]
        
        #prediction and confidence
        predicted_class = int(np.argmax(probabilities_np))
        confidence = float(np.max(probabilities_np))
        
        #map to label (0=fake, 1=real)
        label = "real" if predicted_class == 1 else "fake"
        
        #confidence interval
        ci = compute_confidence_interval(probabilities_np)
        
        #save to Datastore with GCS reference
        result_id = save_to_datastore(
            prediction=predicted_class,
            label=label,
            confidence=confidence,
            ci=ci,
            filename=file.filename,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            prediction_label=label,
            confidence=confidence,
            confidence_interval=ci,
            timestamp=datetime.utcnow().isoformat(),
            result_id=result_id,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



@app.post("/monte_carlo", response_model=MonteCarloResponse)
@limiter.limit("5/minute")
async def predict_monte_carlo(request: Request, file: UploadFile = File(...), num_samples: int = 30):
    require_model()
    try:
        content = await file.read()
        document_id = str(uuid.uuid4())
        gcs_path = ""
        
        if file.content_type == "application/pdf":
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        else:
            image = Image.open(io.BytesIO(content))
        
        mc_result = monte_carlo_inference(image, num_samples)
        predicted_class = mc_result["prediction"]
        confidence = mc_result["confidence"]
        ci = mc_result["confidence_interval"]
        mc_stats = mc_result["monte_carlo_stats"]
        label = "real" if predicted_class == 1 else "fake"
        
        result_id = save_to_datastore(
            prediction=predicted_class,
            label=label,
            confidence=confidence,
            ci=ci,
            filename=file.filename,
            document_id=document_id,
            gcs_path=gcs_path,
            monte_carlo_stats=mc_stats
        )
        
        return MonteCarloResponse(
            prediction=predicted_class,
            prediction_label=label,
            confidence=confidence,
            confidence_interval=ci,
            monte_carlo_stats=mc_stats,
            timestamp=datetime.utcnow().isoformat(),
            result_id=result_id,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



@app.get("/document/{document_id}")
async def get_document_result(document_id: str, user: dict = Depends(require_auth)):
    try:
        query = datastore_client.query(kind="PredictionResult")
        query.add_filter("document_id", "=", document_id)
        results = list(query.fetch(limit=1))
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        result = results[0]
        return {
            "document_id": result.get("document_id"),
            "gcs_path": result.get("gcs_path"),
            "original_filename": result.get("original_filename"),
            "prediction": result.get("prediction"),
            "prediction_label": result.get("prediction_label"),
            "confidence": result.get("confidence"),
            "confidence_interval": result.get("confidence_interval"),
            "timestamp": result.get("timestamp").isoformat() if result.get("timestamp") else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")






@app.post("/certify")
async def certify_document_endpoint(
    file: UploadFile = File(...),
    reviewer_id: Optional[str] = None,
    client_reference: Optional[str] = None,
    notes: Optional[str] = None,
    add_visible_page: bool = True,
    user: dict = Depends(require_auth)
):
    """
    Use this AFTER human review has approved the document.
    Requires authentication, pass Bearer token from /login.
    
    Parameters:
    - file: PDF or image (JPEG, PNG) to certify. Images are auto-converted to PDF.
    - reviewer_id: Email or ID of the human reviewer who approved
    - client_reference: Client's case number or reference
    - notes: Any additional notes to include
    - add_visible_page: If true, appends a human-readable certificate page
    
    Returns certified PDF as downloadable file.
    """
    require_model()
    try:
        #read the uploaded file
        content = await file.read()
        
        #default reviewer_id to authenticated user if not provided
        if not reviewer_id:
            reviewer_id = user.get("email") or user.get("sub")
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        #determine file type and get image for model scoring
        SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/tiff", "image/bmp"}
        
        if file.content_type == "application/pdf":
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        elif file.content_type in SUPPORTED_IMAGE_TYPES:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Must be PDF or image (JPEG, PNG, GIF, TIFF, BMP)."
            )
        
        #run Monte Carlo CNN analysis (same method as /monte_carlo endpoint)
        mc_result = monte_carlo_inference(image, num_samples=30)
        confidence = mc_result["confidence"]
        
        #CNN confidence is recorded in the certificate but does not gate approval.
        #Certification requires manual human approval. No auto-reject based on labels.
        
        #create certification (certify_document handles both PDF and image input)
        certified_pdf, certificate = certify_document(
            content=content,
            confidence_score=confidence,
            reviewer_id=reviewer_id,
            client_reference=client_reference,
            notes=notes,
            add_visible_page=add_visible_page
        )
        
        #store certification record in Datastore
        certificate_store.store_certificate(
            certificate_id=certificate["certificate_id"],
            document_hash=certificate["document_hash"],
            confidence_score=confidence,
            reviewer_id=reviewer_id,
            client_reference=client_reference,
            original_filename=file.filename,
            notes=notes
        )
        
        #store hash of certified PDF for document integrity verification
        certified_hash = hashlib.sha256(certified_pdf).hexdigest()
        certificate_store.store_certified_hash(certificate["certificate_id"], certified_hash)
        
        
        #filename for certified document
        original_name = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename
        certified_filename = f"{original_name}_certified.pdf"
        
        return Response(
            content=certified_pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{certified_filename}"',
                "X-Certificate-ID": certificate["certificate_id"],
                "X-Confidence-Score": str(confidence)
            }
        )
       
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Certification error: {str(e)}")





@app.post("/verify-certificate", response_model=VerificationResponse)
@limiter.limit("10/minute")
async def verify_certificate_endpoint(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        #check if it's a PDF
        if not file.content_type == "application/pdf":
            return VerificationResponse(
                valid=False,
                has_certificate=False,
                message="Only PDF files can contain Kwiddex certificates."
            )
        
        #run cryptographic verification (signature check)
        result = verify_pdf(content)
        
        #if certificate exists and signature is valid, also check Datastore for revocation
        if result.get("has_certificate") and result.get("signature_valid"):
            cert = result.get("certificate", {})
            cert_id = cert.get("certificate_id")
            if cert_id:
                try:
                    status = certificate_store.check_revocation_status(cert_id)
                    if status == "revoked":
                        result["valid"] = False
                        result["certificate_active"] = False
                        result["message"] = "Certificate has been revoked."
                except Exception:
                    pass  #if Datastore is unreachable, fall back to embedded status
            
            #check document integrity against stored certified hash
            if cert_id:
                try:
                    file_hash = hashlib.sha256(content).hexdigest()
                    intact = certificate_store.check_certified_hash(cert_id, file_hash)
                    if intact is not None:
                        result["document_intact"] = intact
                        if not intact:
                            result["valid"] = False
                            result["message"] = "Document has been modified after certification."
                    else:
                        result["document_intact"] = None  #legacy cert, no hash stored
                except Exception:
                    pass
        
        #build response
        response = VerificationResponse(
            valid=result["valid"],
            has_certificate=result["has_certificate"],
            signature_valid=result.get("signature_valid"),
            document_intact=result.get("document_intact"),
            certificate_active=result.get("certificate_active"),
            message=result["message"]
        )
        
        #add certificate details if present
        if result.get("certificate"):
            cert = result["certificate"]
            response.certificate_id = cert.get("certificate_id")
            response.issued_at = cert.get("issued_at")
            response.confidence_score = cert.get("authenticity_confidence")
            response.reviewer_id = cert.get("reviewer_id")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")





@app.get("/certificate/{certificate_id}")
async def get_certificate_details(certificate_id: str, user: dict = Depends(require_auth)): #look up certificate using ID
    try:
        record = certificate_store.lookup_certificate(certificate_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        #remove internal _entity field
        record.pop("_entity", None)
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lookup error: {str(e)}")



@app.post("/revoke-certificate/{certificate_id}")
async def revoke_certificate_endpoint(certificate_id: str, reason: Optional[str] = None, user: dict = Depends(require_auth)):
    """
    Self-revoke a certificate. Only the original certifier can revoke.
    """
    try:
        # Check ownership
        cert = certificate_store.lookup_certificate(certificate_id)
        if cert is None:
            raise HTTPException(status_code=404, detail="Certificate not found")
        caller_id = user.get("sub", "")
        if cert.get("reviewer_id") != caller_id and cert.get("reviewer_id") != user.get("email", ""):
            raise HTTPException(status_code=403, detail="Only the original certifier can revoke this certificate.")
        
        result = certificate_store.revoke_certificate(certificate_id, reason)
        
        if not result["success"]:
            if "not found" in result["message"].lower():
                raise HTTPException(status_code=404, detail="Certificate not found")
            return result
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Revocation error: {str(e)}")






# ── Dispute System ──

class ReportRequest(BaseModel):
    reason: str

@app.post("/report-certificate/{certificate_id}")
async def report_certificate_endpoint(
    certificate_id: str,
    report: ReportRequest,
    user: dict = Depends(require_auth)
):
    """
    Report/dispute a certificate. Requires login and a reason (min 50 chars).
    One report per user per certificate. Cannot report your own certificate.
    """
    reason = report.reason.strip()
    if len(reason) < 50:
        raise HTTPException(status_code=400, detail="Reason must be at least 50 characters to ensure a substantive report.")

    # Check certificate exists and reporter is not the certifier
    cert = certificate_store.lookup_certificate(certificate_id)
    if cert is None:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    caller_id = user.get("sub", "")
    caller_email = user.get("email", caller_id)

    if cert.get("reviewer_id") == caller_id or cert.get("reviewer_id") == caller_email:
        raise HTTPException(status_code=400, detail="You cannot dispute your own certificate. Use self-revoke instead.")

    result = certificate_store.file_dispute(
        certificate_id=certificate_id,
        reporter_id=caller_id,
        reporter_email=caller_email,
        reason=reason
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.post("/resolve-dispute/{certificate_id}/{dispute_id}")
async def resolve_dispute_endpoint(
    certificate_id: str,
    dispute_id: str,
    action: str,
    response_text: str,
    user: dict = Depends(require_auth)
):
    """
    Resolve a dispute. Only the original certifier can resolve.
    action: 'dismiss' (with required written response) or 'accept' (self-revoke).
    """
    # Only the original certifier can resolve disputes
    cert = certificate_store.lookup_certificate(certificate_id)
    if cert is None:
        raise HTTPException(status_code=404, detail="Certificate not found.")

    caller_id = user.get("sub", "")
    if cert.get("reviewer_id") != caller_id and cert.get("reviewer_id") != user.get("email", ""):
        raise HTTPException(status_code=403, detail="Only the original certifier can resolve disputes.")

    if action not in ("dismiss", "accept"):
        raise HTTPException(status_code=400, detail="Action must be 'dismiss' or 'accept'.")

    if action == "dismiss" and len(response_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Dismissal response must be at least 20 characters.")

    result = certificate_store.resolve_dispute(
        certificate_id=certificate_id,
        dispute_id=dispute_id,
        action=action,
        certifier_response=response_text.strip()
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.get("/my-certificates")
async def my_certificates_endpoint(user: dict = Depends(require_auth)):
    """Get all certificates signed by the current user, with dispute info."""
    caller_id = user.get("sub", "")
    caller_email = user.get("email", caller_id)

    # Try both sub and email since older certs may use email
    certs = certificate_store.get_certificates_by_reviewer(caller_id)
    if not certs and caller_email != caller_id:
        certs = certificate_store.get_certificates_by_reviewer(caller_email)

    return {"certificates": certs}


#DISABLED — Auth0 handles registration
@app.post("/register", response_model=RegisterResponse)
async def register_user(request: RegisterRequest):
    raise HTTPException(status_code=410, detail="Registration is handled by Auth0. This endpoint is disabled.")
    # Original code below disabled:
    success, result = create_user(
        request.username, 
        request.password,
        request.organization,
        request.verification_link
    )
    return RegisterResponse(
        success=success,
        user_id=result if success else None,
        message="User registered" if success else result
    )


#DISABLED — Auth0 handles login
@app.post("/login", response_model=LoginResponse)
async def login_user(request: LoginRequest):
    raise HTTPException(status_code=410, detail="Login is handled by Auth0. This endpoint is disabled.")
    # Original code below disabled:
    success, user_id = authenticate(request.username, request.password)
    if success:
        token = create_token(user_id, request.username)
        return LoginResponse(
            success=True,
            user_id=user_id,
            token=token,
            message="Login successful"
        )
    return LoginResponse(
        success=False,
        user_id=None,
        token=None,
        message="Invalid credentials"
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "certification_ready": Path("keys/kwiddex_private.pem").exists()
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

