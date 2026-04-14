import unittest
import io
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 700, f"Test {datetime.now()}")
    c.save()
    return buffer.getvalue()


def create_image():
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def get_client():
    from fastapi.testclient import TestClient
    from predict import app
    return TestClient(app)


class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_200(self):
        r = get_client().get("/health")
        self.assertEqual(r.status_code, 200)

    def test_health_shows_model_loaded(self):
        data = get_client().get("/health").json()
        self.assertIn("model_loaded", data)

    def test_health_shows_certification_ready(self):
        data = get_client().get("/health").json()
        self.assertIn("certification_ready", data)

    def test_health_status_healthy(self):
        data = get_client().get("/health").json()
        self.assertEqual(data["status"], "healthy")


class TestDisabledEndpoints(unittest.TestCase):
    def test_register_returns_410(self):
        r = get_client().post("/register", json={"username": "test", "password": "password123"})
        self.assertEqual(r.status_code, 410)

    def test_login_returns_410(self):
        r = get_client().post("/login", json={"username": "test", "password": "password123"})
        self.assertEqual(r.status_code, 410)


class TestProtectedEndpoints(unittest.TestCase):
    def test_certify_requires_auth(self):
        pdf = create_pdf()
        r = get_client().post("/certify", files={"file": ("test.pdf", pdf, "application/pdf")})
        self.assertEqual(r.status_code, 401)

    def test_revoke_requires_auth(self):
        r = get_client().post("/revoke-certificate/fake-id")
        self.assertEqual(r.status_code, 401)

    def test_document_requires_auth(self):
        r = get_client().get("/document/fake-id")
        self.assertEqual(r.status_code, 401)

    def test_certificate_requires_auth(self):
        r = get_client().get("/certificate/fake-id")
        self.assertEqual(r.status_code, 401)

    def test_my_certificates_requires_auth(self):
        r = get_client().get("/my-certificates")
        self.assertEqual(r.status_code, 401)

    def test_report_requires_auth(self):
        r = get_client().post("/report-certificate/fake-id", json={"reason": "x" * 50})
        self.assertEqual(r.status_code, 401)

    def test_resolve_dispute_requires_auth(self):
        r = get_client().post("/resolve-dispute/fake-id/123?action=dismiss&response_text=test")
        self.assertEqual(r.status_code, 401)

    def test_certify_rejects_invalid_token(self):
        pdf = create_pdf()
        r = get_client().post(
            "/certify",
            files={"file": ("test.pdf", pdf, "application/pdf")},
            headers={"Authorization": "Bearer invalidtoken"}
        )
        self.assertEqual(r.status_code, 401)


class TestPredictEndpoint(unittest.TestCase):
    def test_predict_accepts_image(self):
        img = create_image()
        r = get_client().post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
        self.assertEqual(r.status_code, 200)

    def test_predict_returns_prediction(self):
        img = create_image()
        data = get_client().post("/predict", files={"file": ("test.jpg", img, "image/jpeg")}).json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("prediction_label", data)

    def test_predict_confidence_in_range(self):
        img = create_image()
        data = get_client().post("/predict", files={"file": ("test.jpg", img, "image/jpeg")}).json()
        self.assertGreaterEqual(data["confidence"], 0)
        self.assertLessEqual(data["confidence"], 1)

    def test_predict_rejects_missing_file(self):
        r = get_client().post("/predict")
        self.assertEqual(r.status_code, 422)

    def test_predict_returns_result_id(self):
        img = create_image()
        data = get_client().post("/predict", files={"file": ("test.jpg", img, "image/jpeg")}).json()
        self.assertIn("result_id", data)
        self.assertTrue(len(data["result_id"]) > 0)


class TestMonteCarloEndpoint(unittest.TestCase):
    def test_monte_carlo_accepts_image(self):
        img = create_image()
        r = get_client().post("/monte_carlo", files={"file": ("test.jpg", img, "image/jpeg")})
        self.assertEqual(r.status_code, 200)

    def test_monte_carlo_returns_stats(self):
        img = create_image()
        data = get_client().post("/monte_carlo", files={"file": ("test.jpg", img, "image/jpeg")}).json()
        self.assertIn("monte_carlo_stats", data)
        stats = data["monte_carlo_stats"]
        self.assertIn("num_samples", stats)
        self.assertIn("agreement_rate", stats)
        self.assertIn("std_dev", stats)

    def test_monte_carlo_confidence_interval(self):
        img = create_image()
        data = get_client().post("/monte_carlo", files={"file": ("test.jpg", img, "image/jpeg")}).json()
        self.assertIn("confidence_interval", data)
        ci = data["confidence_interval"]
        self.assertIn("lower_bound", ci)
        self.assertIn("upper_bound", ci)
        self.assertLessEqual(ci["lower_bound"], ci["upper_bound"])


class TestVerifyCertificateEndpoint(unittest.TestCase):
    def test_non_certified_pdf(self):
        pdf = create_pdf()
        r = get_client().post("/verify-certificate", files={"file": ("test.pdf", pdf, "application/pdf")})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["has_certificate"])

    def test_response_includes_dispute_fields(self):
        pdf = create_pdf()
        data = get_client().post("/verify-certificate", files={"file": ("test.pdf", pdf, "application/pdf")}).json()
        self.assertIn("has_disputes", data)
        self.assertFalse(data["has_disputes"])


class TestUnknownRoutes(unittest.TestCase):
    def test_unknown_get_returns_404(self):
        r = get_client().get("/nonexistent")
        self.assertIn(r.status_code, [404, 405])

    def test_unknown_post_returns_404(self):
        r = get_client().post("/nonexistent")
        self.assertIn(r.status_code, [404, 405])


class TestInputValidation(unittest.TestCase):
    def test_predict_no_file_422(self):
        r = get_client().post("/predict")
        self.assertEqual(r.status_code, 422)

    def test_monte_carlo_no_file_422(self):
        r = get_client().post("/monte_carlo")
        self.assertEqual(r.status_code, 422)

    def test_verify_certificate_no_file_422(self):
        r = get_client().post("/verify-certificate")
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
