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
        "certified_file_hash": cert.get("certified_file_hash"),
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


def store_certified_hash(certificate_id: str, certified_hash: str) -> None:
    """Store the hash of the final certified PDF for integrity checking."""
    client = _get_client()
    query = client.query(kind=DATASTORE_KIND)
    query.add_filter("certificate_id", "=", certificate_id)
    results = list(query.fetch(limit=1))
    if results:
        entity = results[0]
        entity["certified_file_hash"] = certified_hash
        client.put(entity)


def check_certified_hash(certificate_id: str, file_hash: str) -> Optional[bool]:
    """Compare uploaded file hash against stored certified hash.
    Returns True if match, False if mismatch, None if no hash stored (legacy cert)."""
    record = lookup_certificate(certificate_id)
    if record is None:
        return None
    stored_hash = record.get("certified_file_hash")
    if stored_hash is None:
        return None
    return stored_hash == file_hash


# ── Dispute System ──

DISPUTE_KIND = "CertificateDispute"


def file_dispute(
    certificate_id: str,
    reporter_id: str,
    reporter_email: str,
    reason: str
) -> dict:
    """File a dispute against a certificate. One per user per certificate."""
    client = _get_client()

    # Check one-per-user-per-cert
    query = client.query(kind=DISPUTE_KIND)
    query.add_filter("certificate_id", "=", certificate_id)
    query.add_filter("reporter_id", "=", reporter_id)
    existing = list(query.fetch(limit=1))
    if existing:
        return {"success": False, "message": "You have already reported this certificate."}

    # Verify certificate exists
    cert = lookup_certificate(certificate_id)
    if cert is None:
        return {"success": False, "message": "Certificate not found."}

    key = client.key(DISPUTE_KIND)
    entity = datastore.Entity(key=key)
    entity.update({
        "certificate_id": certificate_id,
        "reporter_id": reporter_id,
        "reporter_email": reporter_email,
        "reason": reason,
        "filed_at": datetime.utcnow(),
        "dispute_status": "open",
        "certifier_response": None,
        "resolved_at": None,
    })
    client.put(entity)

    return {
        "success": True,
        "message": "Dispute filed successfully.",
        "dispute_id": str(entity.key.id),
        "certificate_id": certificate_id,
    }


def get_disputes_for_certificate(certificate_id: str) -> list:
    """Get all disputes for a certificate."""
    client = _get_client()
    query = client.query(kind=DISPUTE_KIND)
    query.add_filter("certificate_id", "=", certificate_id)
    results = list(query.fetch())

    return [
        {
            "dispute_id": str(r.key.id),
            "reporter_email": r.get("reporter_email"),
            "reason": r.get("reason"),
            "filed_at": r.get("filed_at").isoformat() if r.get("filed_at") else None,
            "dispute_status": r.get("dispute_status", "open"),
            "certifier_response": r.get("certifier_response"),
            "resolved_at": r.get("resolved_at").isoformat() if r.get("resolved_at") else None,
        }
        for r in results
    ]


def resolve_dispute(
    certificate_id: str,
    dispute_id: str,
    action: str,
    certifier_response: str
) -> dict:
    """Resolve a dispute. action is 'dismiss' or 'accept'."""
    client = _get_client()

    # Find the dispute
    key = client.key(DISPUTE_KIND, int(dispute_id))
    entity = client.get(key)

    if entity is None:
        return {"success": False, "message": "Dispute not found."}

    if entity.get("certificate_id") != certificate_id:
        return {"success": False, "message": "Dispute does not belong to this certificate."}

    if entity.get("dispute_status") != "open":
        return {"success": False, "message": "Dispute has already been resolved."}

    if action == "dismiss":
        entity["dispute_status"] = "dismissed"
        entity["certifier_response"] = certifier_response
        entity["resolved_at"] = datetime.utcnow()
        client.put(entity)
        return {"success": True, "message": "Dispute dismissed.", "dispute_status": "dismissed"}

    elif action == "accept":
        entity["dispute_status"] = "accepted"
        entity["certifier_response"] = certifier_response
        entity["resolved_at"] = datetime.utcnow()
        client.put(entity)
        # Self-revoke the certificate
        revoke_certificate(certificate_id, reason=f"Certifier accepted dispute: {certifier_response}")
        return {"success": True, "message": "Dispute accepted. Certificate revoked.", "dispute_status": "accepted"}

    return {"success": False, "message": "Invalid action. Use 'dismiss' or 'accept'."}


def get_certificates_by_reviewer(reviewer_id: str) -> list:
    """Get all certificates signed by a specific reviewer."""
    client = _get_client()
    query = client.query(kind=DATASTORE_KIND)
    query.add_filter("reviewer_id", "=", reviewer_id)
    query.order = ["-issued_at"]
    results = list(query.fetch(limit=100))

    certs = []
    for r in results:
        cert_id = r.get("certificate_id")
        disputes = get_disputes_for_certificate(cert_id)
        certs.append({
            "certificate_id": cert_id,
            "issued_at": r.get("issued_at").isoformat() if r.get("issued_at") else None,
            "confidence_score": r.get("confidence_score"),
            "reviewer_id": r.get("reviewer_id"),
            "status": r.get("status"),
            "original_filename": r.get("original_filename"),
            "disputes": disputes,
            "open_disputes": sum(1 for d in disputes if d["dispute_status"] == "open"),
        })
    return certs
