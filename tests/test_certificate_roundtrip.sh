#!/usr/bin/env bash
set -e

BASE="https://kwiddex.com"
TEST_PDF="/tmp/kwiddex_test_doc.pdf"
MODIFIED_PDF="/tmp/kwiddex_test_modified.pdf"
CERTIFIED_PDF="/tmp/kwiddex_test_certified.pdf"

echo "=== Kwiddex Certificate Roundtrip Test ==="
echo ""

# Check for auth token
if [ -z "$AUTH_TOKEN" ]; then
  echo "ERROR: Set AUTH_TOKEN environment variable first."
  echo "  Get one from browser: localStorage key starting with @@auth0spajs@@"
  echo "  Or use: export AUTH_TOKEN=\$(curl ... your auth0 token flow)"
  exit 1
fi

# Create a simple test PDF
python3 -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('$TEST_PDF')
c.drawString(100, 700, 'Kwiddex roundtrip test document')
c.drawString(100, 680, 'This is the original unmodified version.')
c.save()
print('Created test PDF')
"

# Step 1: Certify
echo ""
echo "--- Step 1: Certify ---"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$CERTIFIED_PDF" \
  -X POST "$BASE/ml/certify" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@$TEST_PDF" \
  -F "reviewer_id=test@kwiddex.com")

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /certify returned $HTTP_CODE"
  cat "$CERTIFIED_PDF"
  exit 1
fi
echo "PASS: Certified PDF received ($(wc -c < "$CERTIFIED_PDF") bytes)"

# Step 2: Verify the certified PDF (should pass)
echo ""
echo "--- Step 2: Verify original certified PDF ---"
VERIFY_RESULT=$(curl -s -X POST "$BASE/ml/verify-certificate" -F "file=@$CERTIFIED_PDF")
echo "$VERIFY_RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'  valid: {r[\"valid\"]}')
print(f'  has_certificate: {r[\"has_certificate\"]}')
print(f'  signature_valid: {r.get(\"signature_valid\")}')
print(f'  document_intact: {r.get(\"document_intact\")}')
print(f'  certificate_active: {r.get(\"certificate_active\")}')
print(f'  message: {r[\"message\"]}')
assert r['valid'] == True, 'FAIL: Expected valid=True'
assert r['has_certificate'] == True, 'FAIL: Expected has_certificate=True'
assert r.get('signature_valid') == True, 'FAIL: Expected signature_valid=True'
assert r.get('document_intact') == True, 'FAIL: Expected document_intact=True'
print('PASS: Original certified PDF verified successfully')
"

# Step 3: Modify the certified PDF and verify (should fail document_intact)
echo ""
echo "--- Step 3: Verify modified certified PDF ---"
python3 -c "
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Add a new page to the certified PDF
reader = PdfReader('$CERTIFIED_PDF')
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)

# Preserve metadata (including certificate)
if reader.metadata:
    writer.add_metadata(dict(reader.metadata))

# Add a tamper page
buf = io.BytesIO()
c = canvas.Canvas(buf, pagesize=letter)
c.drawString(100, 700, 'This page was added after certification.')
c.save()
tamper_reader = PdfReader(buf)
writer.add_page(tamper_reader.pages[0])

with open('$MODIFIED_PDF', 'wb') as f:
    writer.write(f)
print('Created modified PDF')
"

VERIFY_MOD=$(curl -s -X POST "$BASE/ml/verify-certificate" -F "file=@$MODIFIED_PDF")
echo "$VERIFY_MOD" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'  valid: {r[\"valid\"]}')
print(f'  document_intact: {r.get(\"document_intact\")}')
print(f'  message: {r[\"message\"]}')
assert r['valid'] == False, 'FAIL: Expected valid=False for modified doc'
assert r.get('document_intact') == False, 'FAIL: Expected document_intact=False'
print('PASS: Modified PDF correctly detected')
"

# Step 4: Verify a non-certified PDF (should show no certificate)
echo ""
echo "--- Step 4: Verify non-certified PDF ---"
VERIFY_PLAIN=$(curl -s -X POST "$BASE/ml/verify-certificate" -F "file=@$TEST_PDF")
echo "$VERIFY_PLAIN" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'  has_certificate: {r[\"has_certificate\"]}')
print(f'  message: {r[\"message\"]}')
assert r['has_certificate'] == False, 'FAIL: Expected no certificate'
print('PASS: Non-certified PDF handled correctly')
"

# Cleanup
rm -f "$TEST_PDF" "$CERTIFIED_PDF" "$MODIFIED_PDF"

echo ""
echo "=== All tests passed ==="
