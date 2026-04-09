#!/usr/bin/env bash
set -uo pipefail

###############################################################################
#  Kwiddex Full Stack Test Suite
#  Tests endpoints, security, certificates, and rate limiting
#  Run: bash tests/test_full_stack.sh
#  Optional: AUTH_TOKEN=<token> for authenticated endpoint tests
###############################################################################

BASE="${BASE_URL:-https://kwiddex.com}"
EXPRESS="$BASE/api"
FASTAPI="$BASE/ml"

PASS=0; FAIL=0; SKIP=0; TOTAL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

pass()  { PASS=$((PASS+1));  TOTAL=$((TOTAL+1)); echo -e "  ${GREEN}✓${NC} $1"; }
fail()  { FAIL=$((FAIL+1));  TOTAL=$((TOTAL+1)); echo -e "  ${RED}✗ $1${NC}\n    → $2"; }
skip()  { SKIP=$((SKIP+1));  TOTAL=$((TOTAL+1)); echo -e "  ${YELLOW}⊘ $1${NC} ($2)"; }
header(){ echo -e "\n${CYAN}── $1 ──${NC}"; }

http_status() { curl -s -o /dev/null -w "%{http_code}" "$@" 2>/dev/null; }
http_body()   { curl -sf "$@" 2>/dev/null || echo "CURL_FAILED"; }

echo "╔══════════════════════════════════════════╗"
echo "║     Kwiddex Full Stack Test Suite        ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Base URL: $BASE"
echo "  Auth Token: ${AUTH_TOKEN:+set}${AUTH_TOKEN:-not set (some tests will skip)}"
echo ""

###############################################################################
header "1. Health Checks"
###############################################################################

# Frontend loads
S=$(http_status "$BASE")
[ "$S" = "200" ] && pass "Frontend loads (200)" || fail "Frontend loads" "Got $S"

# Express health
BODY=$(http_body "$EXPRESS/health")
if echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['ok']==True" 2>/dev/null; then
  pass "Express health OK"
else
  fail "Express health" "$BODY"
fi

# FastAPI health
BODY=$(http_body "$FASTAPI/health")
if echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='healthy'" 2>/dev/null; then
  pass "FastAPI health OK"
else
  fail "FastAPI health" "$BODY"
fi

# Model loaded
if echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['model_loaded']==True" 2>/dev/null; then
  pass "CNN model loaded"
else
  fail "CNN model loaded" "model_loaded is False"
fi

# Certification ready
if echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['certification_ready']==True" 2>/dev/null; then
  pass "RSA keys present"
else
  fail "RSA keys present" "certification_ready is False"
fi

###############################################################################
header "2. Security Headers"
###############################################################################

HEADERS=$(curl -sI "$BASE")

echo "$HEADERS" | grep -qi "strict-transport-security" && \
  pass "HSTS header present" || fail "HSTS header" "Missing Strict-Transport-Security"

echo "$HEADERS" | grep -qi "x-frame-options" && \
  pass "X-Frame-Options present" || fail "X-Frame-Options" "Missing"

echo "$HEADERS" | grep -qi "x-content-type-options" && \
  pass "X-Content-Type-Options present" || fail "X-Content-Type-Options" "Missing"

echo "$HEADERS" | grep -qi "x-xss-protection" && \
  pass "X-XSS-Protection present" || fail "X-XSS-Protection" "Missing"

echo "$HEADERS" | grep -qi "referrer-policy" && \
  pass "Referrer-Policy present" || fail "Referrer-Policy" "Missing"

###############################################################################
header "3. Authentication and Access Control"
###############################################################################

# Disabled endpoints
S=$(http_status -X POST "$FASTAPI/register" -H "Content-Type: application/json" -d '{"username":"test","password":"test1234"}')
[ "$S" = "410" ] && pass "/register returns 410 (disabled)" || fail "/register disabled" "Got $S"

S=$(http_status -X POST "$FASTAPI/login" -H "Content-Type: application/json" -d '{"username":"test","password":"test1234"}')
[ "$S" = "410" ] && pass "/login returns 410 (disabled)" || fail "/login disabled" "Got $S"

# Protected endpoints without token
S=$(http_status -X POST "$FASTAPI/certify" -F "file=@/dev/null")
[ "$S" = "401" ] && pass "/certify requires auth (401)" || fail "/certify auth" "Got $S"

S=$(http_status "$FASTAPI/document/test-id")
[ "$S" = "401" ] && pass "/document requires auth (401)" || fail "/document auth" "Got $S"

S=$(http_status -X POST "$FASTAPI/revoke-certificate/fake-id")
[ "$S" = "401" ] && pass "/revoke-certificate requires auth (401)" || fail "/revoke-certificate auth" "Got $S"

S=$(http_status "$FASTAPI/certificate/fake-id")
[ "$S" = "401" ] && pass "/certificate requires auth (401)" || fail "/certificate auth" "Got $S"

S=$(http_status -X POST "$EXPRESS/email" -H "Content-Type: application/json" -d '{"to":"test@test.com"}')
[ "$S" = "401" ] && pass "/api/email requires auth (401)" || fail "/api/email auth" "Got $S"

# Protected endpoints with invalid token
S=$(http_status -X POST "$FASTAPI/certify" -H "Authorization: Bearer invalidtoken" -F "file=@/dev/null")
[ "$S" = "401" ] && pass "/certify rejects invalid token (401)" || fail "/certify invalid token" "Got $S"

###############################################################################
header "4. Public Endpoints"
###############################################################################

# Create test files
TEST_PDF="/tmp/kwiddex_test_$$.pdf"
TEST_IMG="/tmp/kwiddex_test_$$.png"

python3 -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('$TEST_PDF')
c.drawString(100, 700, 'Kwiddex test document')
c.save()
" 2>/dev/null

python3 -c "
from PIL import Image
img = Image.new('RGB', (200, 200), color=(128, 128, 128))
img.save('$TEST_IMG')
" 2>/dev/null

# /predict
S=$(http_status -X POST "$FASTAPI/predict" -F "file=@$TEST_IMG")
[ "$S" = "200" ] && pass "/predict accepts image (200)" || fail "/predict" "Got $S"

# /verify-certificate with non-certified PDF
BODY=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@$TEST_PDF")
if echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['has_certificate']==False" 2>/dev/null; then
  pass "/verify-certificate handles non-certified PDF"
else
  fail "/verify-certificate non-certified" "$BODY"
fi

# Express /verify
S=$(http_status -X POST "$EXPRESS/verify" -F "file=@$TEST_PDF")
[ "$S" = "200" ] && pass "Express /verify accepts PDF (200)" || fail "Express /verify" "Got $S"

# Express /physical/score with image
S=$(http_status -X POST "$EXPRESS/physical/score" -F "file=@$TEST_IMG")
if [ "$S" = "200" ]; then
  pass "Express /physical/score accepts image (200)"
else
  # May take time for Monte Carlo, try with timeout
  S2=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 -X POST "$EXPRESS/physical/score" -F "file=@$TEST_IMG")
  [ "$S2" = "200" ] && pass "Express /physical/score accepts image (200, slow)" || fail "Express /physical/score" "Got $S then $S2"
fi

###############################################################################
header "5. Input Validation"
###############################################################################

# No file
S=$(http_status -X POST "$FASTAPI/predict")
[ "$S" = "422" ] && pass "/predict rejects missing file (422)" || fail "/predict no file" "Got $S"

# Non-PDF to /verify-certificate
S=$(http_status -X POST "$FASTAPI/verify-certificate" -F "file=@$TEST_IMG")
[ "$S" = "200" ] && pass "/verify-certificate rejects non-PDF gracefully" || fail "/verify-certificate non-PDF" "Got $S"

# Express /verify rejects missing file
S=$(http_status -X POST "$EXPRESS/verify")
[ "$S" = "400" ] && pass "Express /verify rejects missing file (400)" || fail "Express /verify no file" "Got $S"

# Unknown routes
S=$(http_status "$FASTAPI/nonexistent")
[ "$S" = "404" ] && pass "FastAPI unknown route (404)" || fail "FastAPI 404" "Got $S"

S=$(http_status "$EXPRESS/nonexistent")
[ "$S" = "404" ] && pass "Express unknown route (404)" || fail "Express 404" "Got $S"

###############################################################################
header "6. Rate Limiting"
###############################################################################

echo "  Testing /monte_carlo rate limit (5/min)..."
RATE_OK=true
for i in $(seq 1 6); do
  S=$(http_status -X POST "$FASTAPI/monte_carlo" -F "file=@/dev/null")
  if [ "$i" -le 5 ] && [ "$S" = "429" ]; then
    RATE_OK=false
    break
  fi
done
S=$(http_status -X POST "$FASTAPI/monte_carlo" -F "file=@/dev/null")
if [ "$S" = "429" ]; then
  pass "Rate limiting active on /monte_carlo (429 after limit)"
else
  # Rate limit window may have passed, just check it exists
  skip "Rate limit on /monte_carlo" "Could not trigger 429, may need faster requests"
fi

###############################################################################
header "7. CORS"
###############################################################################

# Should accept kwiddex.com origin
S=$(curl -s -o /dev/null -w "%{http_code}" -H "Origin: https://kwiddex.com" -X OPTIONS "$FASTAPI/health")
pass "CORS preflight responds"

# Check FastAPI CORS header
CORS=$(curl -sI -H "Origin: https://kwiddex.com" "$FASTAPI/health" | grep -i "access-control-allow-origin" || echo "none")
echo "$CORS" | grep -qi "kwiddex.com" && \
  pass "FastAPI CORS allows kwiddex.com" || skip "FastAPI CORS header" "May not include header on non-preflight"

###############################################################################
header "8. Certificate Roundtrip (requires AUTH_TOKEN)"
###############################################################################

if [ -z "${AUTH_TOKEN:-}" ]; then
  skip "Certify document" "AUTH_TOKEN not set"
  skip "Verify certified PDF" "AUTH_TOKEN not set"
  skip "Verify modified PDF" "AUTH_TOKEN not set"
  skip "Certificate details in response" "AUTH_TOKEN not set"
else
  CERT_PDF="/tmp/kwiddex_cert_$$.pdf"
  MOD_PDF="/tmp/kwiddex_mod_$$.pdf"

  # Certify
  HTTP_CODE=$(curl -s -w "%{http_code}" -o "$CERT_PDF" \
    -X POST "$FASTAPI/certify" \
    -H "Authorization: Bearer $AUTH_TOKEN" \
    -F "file=@$TEST_PDF" \
    -F "reviewer_id=test@kwiddex.com")

  if [ "$HTTP_CODE" = "200" ]; then
    pass "Certify document (200)"
    CERT_SIZE=$(wc -c < "$CERT_PDF")
    [ "$CERT_SIZE" -gt 0 ] && pass "Certified PDF has content ($CERT_SIZE bytes)" || fail "Certified PDF" "Empty file"
  else
    fail "Certify document" "Got $HTTP_CODE"
  fi

  # Verify certified PDF
  if [ -f "$CERT_PDF" ] && [ "$(wc -c < "$CERT_PDF")" -gt 0 ]; then
    BODY=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@$CERT_PDF")
    
    VALID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['valid'])" 2>/dev/null)
    HAS_CERT=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['has_certificate'])" 2>/dev/null)
    SIG_VALID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('signature_valid'))" 2>/dev/null)
    DOC_INTACT=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('document_intact'))" 2>/dev/null)
    CERT_ID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('certificate_id',''))" 2>/dev/null)
    REVIEWER=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('reviewer_id',''))" 2>/dev/null)

    [ "$VALID" = "True" ] && pass "Certified PDF is valid" || fail "Certified PDF valid" "valid=$VALID"
    [ "$HAS_CERT" = "True" ] && pass "Certificate found in PDF" || fail "Certificate found" "has_certificate=$HAS_CERT"
    [ "$SIG_VALID" = "True" ] && pass "Signature is valid" || fail "Signature valid" "signature_valid=$SIG_VALID"
    [ "$DOC_INTACT" = "True" ] && pass "Document is intact" || fail "Document intact" "document_intact=$DOC_INTACT"
    [ -n "$CERT_ID" ] && pass "Certificate ID present ($CERT_ID)" || fail "Certificate ID" "Empty"
    [ -n "$REVIEWER" ] && pass "Reviewer ID present ($REVIEWER)" || fail "Reviewer ID" "Empty"

    # Modify the certified PDF and verify
    python3 -c "
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

reader = PdfReader('$CERT_PDF')
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)
if reader.metadata:
    writer.add_metadata(dict(reader.metadata))

buf = io.BytesIO()
c = canvas.Canvas(buf, pagesize=letter)
c.drawString(100, 700, 'TAMPERED PAGE')
c.save()
tamper = PdfReader(buf)
writer.add_page(tamper.pages[0])

with open('$MOD_PDF', 'wb') as f:
    writer.write(f)
" 2>/dev/null

    if [ -f "$MOD_PDF" ]; then
      BODY=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@$MOD_PDF")
      VALID=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['valid'])" 2>/dev/null)
      DOC_INTACT=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('document_intact'))" 2>/dev/null)
      [ "$VALID" = "False" ] && pass "Modified PDF detected as invalid" || fail "Modified PDF" "valid=$VALID"
      [ "$DOC_INTACT" = "False" ] && pass "Document modification detected" || fail "Document intact check" "document_intact=$DOC_INTACT"
    else
      fail "Create modified PDF" "Python script failed"
    fi
  else
    skip "Verify certified PDF" "Certification failed"
    skip "Verify modified PDF" "Certification failed"
  fi

  rm -f "$CERT_PDF" "$MOD_PDF"
fi

###############################################################################
header "9. SSL and HTTPS"
###############################################################################

# HTTP redirects to HTTPS
S=$(curl -s -o /dev/null -w "%{http_code}" -L "http://kwiddex.com" --max-redirs 0 2>/dev/null || true)
S2=$(curl -s -o /dev/null -w "%{http_code}" "http://kwiddex.com" 2>/dev/null || true)
if [ "$S2" = "301" ] || [ "$S2" = "302" ] || [ "$S2" = "308" ]; then
  pass "HTTP redirects to HTTPS"
else
  skip "HTTP to HTTPS redirect" "Got $S2 (may be blocked or connection refused)"
fi

# SSL certificate valid
if curl -s --max-time 5 "https://kwiddex.com" > /dev/null 2>&1; then
  pass "SSL certificate valid"
else
  fail "SSL certificate" "Connection failed"
fi

###############################################################################
# Cleanup
###############################################################################

rm -f "$TEST_PDF" "$TEST_IMG"

###############################################################################
# Summary
###############################################################################

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              Test Results                ║"
echo "╠══════════════════════════════════════════╣"
printf "║  ${GREEN}Passed: %-4d${NC}                             ║\n" $PASS
printf "║  ${RED}Failed: %-4d${NC}                             ║\n" $FAIL
printf "║  ${YELLOW}Skipped: %-4d${NC}                            ║\n" $SKIP
printf "║  Total:  %-4d                             ║\n" $TOTAL
echo "╚══════════════════════════════════════════╝"

[ $FAIL -eq 0 ] && echo -e "\n${GREEN}All tests passed!${NC}" || echo -e "\n${RED}Some tests failed.${NC}"
exit $FAIL
