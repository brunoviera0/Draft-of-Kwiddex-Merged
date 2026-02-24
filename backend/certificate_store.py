from google.cloud import datastore
from datetime import datetime
from typing import Optional

GCP_PROJECT = "sentiment-analysis-379200"
DATASTORE_KIND = "Certification"

_client = None

def _get_client() -> datastore.Client:
    global _client
    if _client is None:
        _client = datastore.Client(project=GCP_PROJECT)
    return _client


def store_certificate(
    certificate_id: str,
    document_hash: str,
    confidence_score: float,
    reviewer_id: Optional[str] = None,
    client_reference: Optional[str] = None,
    original_filename: Optional[str] = None,
    notes: Optional[str] = None
) -> str:
    

    client = _get_client()
    key = client.key(DATASTORE_KIND)
    entity = datastore.Entity(key=key)
    entity.update({
        "certificate_id": certificate_id,
        "document_hash": document_hash,
        "issued_at": datetime.utcnow(),
        "confidence_score": confidence_score,
        "reviewer_id": reviewer_id,
        "client_reference": client_reference,
        "original_filename": original_filename,
        "notes": notes,
        "status": "active"
    })
    client.put(entity)
    return str(entity.key.id)


def lookup_certificate(certificate_id: str) -> Optional[dict]:
    client = _get_client()
    query = client.query(kind=DATASTORE_KIND)
    query.add_filter("certificate_id", "=", certificate_id)
    results = list(query.fetch(limit=1))

    if not results:
        return None

    cert = results[0]
    return {
        "certificate_id": cert.get("certificate_id"),
        "document_hash": cert.get("document_hash"),
        "issued_at": cert.get("issued_at").isoformat() if cert.get("issued_at") else None,
        "confidence_score": cert.get("confidence_score"),
        "reviewer_id": cert.get("reviewer_id"),
        "client_reference": cert.get("client_reference"),
        "original_filename": cert.get("original_filename"),
        "notes": cert.get("notes"),
        "status": cert.get("status"),
        "revoked_at": cert.get("revoked_at").isoformat() if cert.get("revoked_at") else None,
        "revocation_reason": cert.get("revocation_reason"),
        "_entity": cert  # keep raw entity for updates
    }


def check_revocation_status(certificate_id: str) -> str:
    record = lookup_certificate(certificate_id)
    if record is None:
        return "not_found"
    return record.get("status", "active")


def revoke_certificate(certificate_id: str, reason: Optional[str] = None) -> dict:
    client = _get_client()
    query = client.query(kind=DATASTORE_KIND)
    query.add_filter("certificate_id", "=", certificate_id)
    results = list(query.fetch(limit=1))

    if not results:
        return {"success": False, "message": "Certificate not found"}

    entity = results[0]

    if entity.get("status") == "revoked":
        return {
            "success": False,
            "message": "Certificate was already revoked",
            "revoked_at": entity.get("revoked_at").isoformat() if entity.get("revoked_at") else None
        }

    entity["status"] = "revoked"
    entity["revoked_at"] = datetime.utcnow()
    entity["revocation_reason"] = reason
    client.put(entity)

    return {
        "success": True,
        "message": "Certificate revoked successfully",
        "certificate_id": certificate_id,
        "revoked_at": entity["revoked_at"].isoformat(),
        "reason": reason
    }


def list_certificates(status: Optional[str] = None, limit: int = 50) -> list:
    client = _get_client()
    query = client.query(kind=DATASTORE_KIND)
    if status:
        query.add_filter("status", "=", status)
    query.order = ["-issued_at"]
    results = list(query.fetch(limit=limit))

    return [
        {
            "certificate_id": r.get("certificate_id"),
            "issued_at": r.get("issued_at").isoformat() if r.get("issued_at") else None,
            "confidence_score": r.get("confidence_score"),
            "reviewer_id": r.get("reviewer_id"),
            "status": r.get("status"),
            "original_filename": r.get("original_filename"),
        }
        for r in results
    ]
