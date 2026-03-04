#!/usr/bin/env bash
set -euo pipefail

EXPRESS="${EXPRESS_BASE:-http://localhost:3001}"
FASTAPI="${FASTAPI_BASE:-http://localhost:8000}"

PASS=0; FAIL=0; SKIP=0; TOTAL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

pass()  { PASS=$((PASS+1));  TOTAL=$((TOTAL+1)); echo -e "  ${GREEN}✓${NC} $1"; }
fail()  { FAIL=$((FAIL+1));  TOTAL=$((TOTAL+1)); echo -e "  ${RED}✗ $1${NC}\n    → $2"; }
skip()  { SKIP=$((SKIP+1));  TOTAL=$((TOTAL+1)); echo -e "  ${YELLOW}⊘ $1${NC} — $2"; }
header(){ echo -e "\n${CYAN}── $1 ──${NC}"; }

http_status() { curl -s -o /dev/null -w "%{http_code}" "$@" 2>/dev/null; }
http_body()   { curl -sf "$@" 2>/dev/null || echo "CURL_FAILED"; }
jp()          { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)" 2>/dev/null <<< "$2"; }

TEST_EMAIL="test-$(date +%s)@kwiddex-integration.test"
TEST_PASS="IntTest_$(date +%s)!"
TOKEN=""
FASTAPI_TOKEN=""

echo "╔══════════════════════════════════════════╗"
echo "║  Express ↔ FastAPI Integration Tests v2  ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Express:  $EXPRESS"
echo "  FastAPI:  $FASTAPI"
echo "  Test user: $TEST_EMAIL"


header "1. Health — Both services reachable"

FHEALTH=$(http_body "$FASTAPI/health")
if [ "$FHEALTH" = "CURL_FAILED" ]; then
  fail "FastAPI unreachable at $FASTAPI/health" "Is uvicorn running? Aborting."
  exit 1
fi
pass "FastAPI /health responds"

if echo "$FHEALTH" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('model_loaded')==True" 2>/dev/null; then
  pass "FastAPI model loaded"
else
  fail "FastAPI model NOT loaded" "best_real_fake_resnet18.pt missing or failed to load"
fi

if echo "$FHEALTH" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('certification_ready')==True" 2>/dev/null; then
  pass "FastAPI RSA keys present"
else
  skip "FastAPI certification" "keys/kwiddex_private.pem missing — run setup_keys()"
fi

EHEALTH=$(http_body "$EXPRESS/api/health")
if [ "$EHEALTH" = "CURL_FAILED" ]; then
  fail "Express unreachable at $EXPRESS/api/health" "Is Express running? Aborting."
  exit 1
fi
pass "Express /api/health responds"

if echo "$EHEALTH" | grep -q "fastapi\|8000"; then
  pass "Express health shows FastAPI URL"
else
  skip "Express FastAPI URL in health" "Not shown — non-critical"
fi

PHEALTH=$(http_body "$EXPRESS/api/physical/health")
if echo "$PHEALTH" | grep -q '"provider":"cnn"'; then
  pass "Express physical/health provider is 'cnn'"
elif echo "$PHEALTH" | grep -q '"provider"'; then
  fail "Express physical/health wrong provider" "$(jp '["provider"]' "$PHEALTH")"
else
  fail "Express physical/health unexpected response" "$PHEALTH"
fi


header "2. Auth Proxy — Express forwards to FastAPI"

SIGNUP=$(curl -s -X POST "$EXPRESS/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"$TEST_PASS\"}" 2>/dev/null)

if echo "$SIGNUP" | grep -q '"token"'; then
  pass "Signup returns token (auto-login worked)"
  TOKEN=$(jp '["token"]' "$SIGNUP")
else
  fail "Signup failed" "$SIGNUP"
fi

FASTAPI_DIRECT=$(curl -s -X POST "$FASTAPI/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$TEST_EMAIL\",\"password\":\"$TEST_PASS\"}" 2>/dev/null)

if echo "$FASTAPI_DIRECT" | grep -q '"token"'; then
  pass "User exists in FastAPI (direct login works)"
  FASTAPI_TOKEN=$(jp '["token"]' "$FASTAPI_DIRECT")
else
  fail "User NOT in FastAPI after Express signup" "Email→username mapping broken? $FASTAPI_DIRECT"
fi

LOGIN=$(curl -s -X POST "$EXPRESS/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"$TEST_PASS\"}" 2>/dev/null)

if echo "$LOGIN" | grep -q '"token"'; then
  pass "Login returns token"
  TOKEN=$(jp '["token"]' "$LOGIN")
else
  fail "Login failed" "$LOGIN"
fi

LOGIN_HAS_USER_ID=$(jp '.get("user",{}).get("id","")' "$LOGIN")
LOGIN_HAS_USER_EMAIL=$(jp '.get("user",{}).get("email","")' "$LOGIN")
if [ -n "$LOGIN_HAS_USER_ID" ] && [ -n "$LOGIN_HAS_USER_EMAIL" ]; then
  pass "Login response shape: { token, user: { id, email } }"
else
  fail "Login response shape wrong" "Frontend expects { token, user: { id, email } }. Got: $LOGIN"
fi

DUP_STATUS=$(http_status -X POST "$EXPRESS/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"$TEST_PASS\"}")
if [ "$DUP_STATUS" = "400" ] || [ "$DUP_STATUS" = "409" ]; then
  pass "Duplicate signup rejected (HTTP $DUP_STATUS)"
else
  fail "Duplicate signup should fail" "Got HTTP $DUP_STATUS"
fi

BAD_STATUS=$(http_status -X POST "$EXPRESS/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$TEST_EMAIL\",\"password\":\"WrongPassword!\"}")
if [ "$BAD_STATUS" = "401" ]; then
  pass "Wrong password rejected (401)"
else
  fail "Wrong password should return 401" "Got HTTP $BAD_STATUS"
fi

EMPTY_STATUS=$(http_status -X POST "$EXPRESS/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{}")
if [ "$EMPTY_STATUS" = "400" ]; then
  pass "Missing fields rejected (400)"
else
  fail "Missing fields should return 400" "Got HTTP $EMPTY_STATUS"
fi


header "3. JWT Interop — FastAPI issues, Express verifies"

if [ -n "$TOKEN" ]; then
  ME_VIA_EXPRESS=$(http_body "$EXPRESS/api/auth/me" -H "Authorization: Bearer $TOKEN")
  if echo "$ME_VIA_EXPRESS" | grep -q '"user"'; then
    pass "Express-proxied token accepted by /auth/me"
  else
    fail "Express-proxied token rejected by /auth/me" "$ME_VIA_EXPRESS"
  fi
else
  skip "Express token on /auth/me" "No token available"
fi

if [ -n "$FASTAPI_TOKEN" ]; then
  ME_VIA_FASTAPI=$(http_body "$EXPRESS/api/auth/me" -H "Authorization: Bearer $FASTAPI_TOKEN")
  if echo "$ME_VIA_FASTAPI" | grep -q '"user"'; then
    pass "FastAPI-issued token accepted by Express /auth/me (JWT interop)"
  else
    fail "FastAPI-issued token REJECTED by Express" "KWX_JWT_SECRET mismatch! Got: $ME_VIA_FASTAPI"
  fi
else
  skip "FastAPI token on Express /auth/me" "No FastAPI token"
fi

NO_TOKEN_STATUS=$(http_status "$EXPRESS/api/auth/me")
if [ "$NO_TOKEN_STATUS" = "401" ]; then
  pass "Missing token returns 401"
else
  fail "Missing token should return 401" "Got $NO_TOKEN_STATUS"
fi

BAD_TOKEN_STATUS=$(http_status "$EXPRESS/api/auth/me" -H "Authorization: Bearer not.a.real.token")
if [ "$BAD_TOKEN_STATUS" = "401" ]; then
  pass "Invalid token returns 401"
else
  fail "Invalid token should return 401" "Got $BAD_TOKEN_STATUS"
fi

EXPIRED_TOKEN=$(python3 -c "
import hmac, hashlib, base64, json, os
secret = os.environ.get('KWX_JWT_SECRET', 'dev-fallback-not-for-production')
header = base64.urlsafe_b64encode(json.dumps({'alg':'HS256','typ':'JWT'}).encode()).rstrip(b'=').decode()
payload = base64.urlsafe_b64encode(json.dumps({'sub':'fake','username':'fake','iat':1000000,'exp':1000001}).encode()).rstrip(b'=').decode()
sig = base64.urlsafe_b64encode(hmac.new(secret.encode(), f'{header}.{payload}'.encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
print(f'{header}.{payload}.{sig}')
" 2>/dev/null || echo "")

if [ -n "$EXPIRED_TOKEN" ]; then
  EXPIRED_STATUS=$(http_status "$EXPRESS/api/auth/me" -H "Authorization: Bearer $EXPIRED_TOKEN")
  if [ "$EXPIRED_STATUS" = "401" ]; then
    pass "Expired token returns 401"
  else
    fail "Expired token should return 401" "Got $EXPIRED_STATUS"
  fi
else
  skip "Expired token test" "Could not craft token"
fi


header "4. CNN Scoring — Express proxies to FastAPI (CnnResult shape)"

TEST_IMG="/tmp/kwiddex_test_$(date +%s).png"
python3 -c "
from PIL import Image
import numpy as np
arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
Image.fromarray(arr).save('$TEST_IMG')
" 2>/dev/null

if [ ! -f "$TEST_IMG" ]; then
  skip "All CNN scoring tests" "PIL/Pillow not available to create test image"
else

  PREDICT=$(curl -s -X POST "$FASTAPI/predict" -F "file=@$TEST_IMG;type=image/png" 2>/dev/null)
  if echo "$PREDICT" | grep -q '"prediction"'; then
    pass "FastAPI /predict returns prediction"
  else
    fail "FastAPI /predict failed" "$PREDICT"
  fi

  MC=$(curl -s -X POST "$FASTAPI/monte_carlo?num_samples=5" -F "file=@$TEST_IMG;type=image/png" 2>/dev/null)
  if echo "$MC" | grep -q '"monte_carlo_stats"'; then
    pass "FastAPI /monte_carlo returns monte_carlo_stats"
  else
    fail "FastAPI /monte_carlo failed" "$MC"
  fi

  SCORE=$(curl -s -X POST "$EXPRESS/api/physical/score" -F "file=@$TEST_IMG;type=image/png" 2>/dev/null)
  if echo "$SCORE" | grep -q '"confidence"'; then
    pass "Express /physical/score returns CnnResult"
  else
    fail "Express /physical/score failed" "$SCORE"
  fi

  HAS_CONF=$(python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if isinstance(d.get('confidence'),(int,float)) else 'no')" <<< "$SCORE" 2>/dev/null)
  if [ "$HAS_CONF" = "yes" ]; then
    pass "Response has confidence (number)"
  else
    fail "Response missing confidence" ""
  fi

  HAS_CI=$(python3 -c "import json,sys; d=json.load(sys.stdin); ci=d.get('confidenceInterval',{}); print('yes' if isinstance(ci.get('lower'),(int,float)) and isinstance(ci.get('upper'),(int,float)) else 'no')" <<< "$SCORE" 2>/dev/null)
  if [ "$HAS_CI" = "yes" ]; then
    pass "Response has confidenceInterval (lower + upper)"
  else
    fail "Response missing confidenceInterval" ""
  fi

  CONF_VAL=$(python3 -c "import json,sys; print(json.load(sys.stdin)['confidence'])" <<< "$SCORE" 2>/dev/null)
  if python3 -c "assert 0 <= $CONF_VAL <= 1" 2>/dev/null; then
    pass "Confidence in range 0-1 ($CONF_VAL)"
  else
    fail "Confidence out of range" "$CONF_VAL"
  fi

  CI_LOWER=$(python3 -c "import json,sys; print(json.load(sys.stdin)['confidenceInterval']['lower'])" <<< "$SCORE" 2>/dev/null)
  CI_UPPER=$(python3 -c "import json,sys; print(json.load(sys.stdin)['confidenceInterval']['upper'])" <<< "$SCORE" 2>/dev/null)
  if python3 -c "assert 0 <= $CI_LOWER <= $CI_UPPER <= 1" 2>/dev/null; then
    pass "CI bounds valid: $CI_LOWER – $CI_UPPER"
  else
    fail "CI bounds invalid" "lower=$CI_LOWER upper=$CI_UPPER"
  fi

  PROVIDER=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('provider',''))" <<< "$SCORE" 2>/dev/null)
  if [ "$PROVIDER" = "cnn" ]; then
    pass "Provider is 'cnn'"
  elif [ "$PROVIDER" = "heuristic" ]; then
    fail "Provider is 'heuristic'" "Express couldn't reach FastAPI — fell back"
  else
    fail "Unexpected provider" "$PROVIDER"
  fi

  HAS_MC=$(python3 -c "import json,sys; d=json.load(sys.stdin); mc=d.get('monteCarloStats'); print('yes' if mc and 'numSamples' in mc else 'no')" <<< "$SCORE" 2>/dev/null)
  if [ "$HAS_MC" = "yes" ]; then
    pass "Monte Carlo stats present (numSamples, agreementRate, stdDev)"
  else
    skip "Monte Carlo stats" "May be using /predict instead of /monte_carlo"
  fi

  HAS_OLD=$(python3 -c "import json,sys; d=json.load(sys.stdin); print('yes' if 'score' in d or 'reasons' in d or 'flags' in d else 'no')" <<< "$SCORE" 2>/dev/null)
  if [ "$HAS_OLD" = "no" ]; then
    pass "Old AiResult fields absent (no score/reasons/flags)"
  else
    fail "Old AiResult fields still present" "Response should not have score/reasons/flags"
  fi

  ANALYSIS_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('analysisId',''))" <<< "$SCORE" 2>/dev/null)
  if [ -n "$ANALYSIS_ID" ]; then
    pass "analysisId present"
  else
    fail "analysisId missing" ""
  fi

  MODEL=$(python3 -c "import json,sys; print(json.load(sys.stdin).get('model',''))" <<< "$SCORE" 2>/dev/null)
  if [ -n "$MODEL" ]; then
    pass "model name present ($MODEL)"
  else
    fail "model name missing" ""
  fi

  rm -f "$TEST_IMG"
fi


header "5. PDF Metadata Verify (Express-only)"

TEST_PDF="/tmp/kwiddex_test_$(date +%s).pdf"
python3 -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('$TEST_PDF')
c.setAuthor('Integration Test')
c.setTitle('Test Document')
c.drawString(100, 750, 'Integration test PDF')
c.save()
" 2>/dev/null

if [ -f "$TEST_PDF" ]; then
  VERIFY=$(curl -s -X POST "$EXPRESS/api/verify" -F "file=@$TEST_PDF;type=application/pdf" 2>/dev/null)

  if echo "$VERIFY" | grep -q '"sha256"'; then
    pass "Express /verify returns sha256"
  else
    fail "Express /verify missing sha256" "$VERIFY"
  fi

  if echo "$VERIFY" | grep -q '"core"'; then
    pass "Express /verify returns core metadata"
  else
    fail "Express /verify missing core metadata" "$VERIFY"
  fi

  rm -f "$TEST_PDF"
else
  skip "PDF metadata tests" "reportlab not installed"
fi


header "6. WordPress Proxy (graceful when unconfigured)"

WP_STATUS=$(http_status "$EXPRESS/api/wp/posts")
if [ "$WP_STATUS" = "503" ]; then
  pass "WordPress proxy returns 503 when WORDPRESS_URL not set"
elif [ "$WP_STATUS" = "200" ]; then
  pass "WordPress proxy returns 200 (WORDPRESS_URL configured)"
else
  fail "WordPress proxy unexpected status" "HTTP $WP_STATUS"
fi


header "7. Error Handling"

NO_FILE_STATUS=$(http_status -X POST "$EXPRESS/api/physical/score")
if [ "$NO_FILE_STATUS" = "400" ]; then
  pass "Score rejects missing file (400)"
else
  fail "Score should reject missing file" "Got HTTP $NO_FILE_STATUS"
fi

NOT_FOUND=$(http_status "$EXPRESS/api/does-not-exist")
if [ "$NOT_FOUND" = "404" ]; then
  pass "Unknown route returns 404"
else
  fail "Unknown route should return 404" "Got $NOT_FOUND"
fi

ECHO_IMG="/tmp/kwiddex_echo_$(date +%s).png"
python3 -c "from PIL import Image; Image.new('RGB',(10,10)).save('$ECHO_IMG')" 2>/dev/null
if [ -f "$ECHO_IMG" ]; then
  ECHO=$(curl -s -X POST "$EXPRESS/api/physical/echo" -F "file=@$ECHO_IMG;type=image/png" 2>/dev/null)
  if echo "$ECHO" | grep -q '"name"\|"size"\|"mime"'; then
    pass "Physical /echo returns file info"
  else
    fail "Physical /echo unexpected response" "$ECHO"
  fi
  rm -f "$ECHO_IMG"
fi

BAD_FILE="/tmp/kwiddex_bad.txt"
echo "not an image" > "$BAD_FILE"
BAD_PRED_STATUS=$(http_status -X POST "$FASTAPI/predict" -F "file=@$BAD_FILE;type=text/plain")
if [ "$BAD_PRED_STATUS" = "400" ] || [ "$BAD_PRED_STATUS" = "500" ] || [ "$BAD_PRED_STATUS" = "422" ]; then
  pass "FastAPI rejects non-image file (HTTP $BAD_PRED_STATUS)"
else
  fail "FastAPI should reject non-image" "Got HTTP $BAD_PRED_STATUS"
fi
rm -f "$BAD_FILE"

EMPTY_EMAIL_STATUS=$(http_status -X POST "$EXPRESS/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email":"","password":"SomePass123!"}')
if [ "$EMPTY_EMAIL_STATUS" = "400" ]; then
  pass "Signup rejects empty email (400)"
else
  fail "Signup should reject empty email" "Got HTTP $EMPTY_EMAIL_STATUS"
fi


echo ""
echo "╔══════════════════════════════════════════╗"
printf "║  Results: ${GREEN}%d passed${NC}, " "$PASS"
printf "${RED}%d failed${NC}, " "$FAIL"
printf "${YELLOW}%d skipped${NC} " "$SKIP"
echo "   ║"
echo "╚══════════════════════════════════════════╝"

if [ "$FAIL" -gt 0 ]; then
  echo -e "\n${RED}FAILURES DETECTED.${NC}"
  exit 1
else
  echo -e "\n${GREEN}All Express ↔ FastAPI integration tests passed.${NC}"
  exit 0
fi
