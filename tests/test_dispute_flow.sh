#!/usr/bin/env bash
###############################################################################
#  Kwiddex Certificate Dispute System — End-to-End Demo
#
#  Demonstrates the full dispute lifecycle:
#    1. Certify a document
#    2. Verify it (clean, no disputes)
#    3. File a dispute with substantive reason
#    4. Verify again (dispute visible to anyone)
#    5. View certifier's profile (dispute notification)
#    6. Certifier dismisses with written response
#    7. Verify final state (dismissed but permanently visible)
#
#  Usage:
#    export AUTH_TOKEN="your_auth0_token"
#    bash tests/test_dispute_flow.sh
#
#  Note: Because this uses an M2M token (single identity), the same
#  token acts as both certifier and reporter. In production these
#  would be different users. The one-report-per-user check is
#  bypassed for demo purposes by using the same identity.
###############################################################################

set -uo pipefail

BASE="${BASE_URL:-https://kwiddex.com}"
FASTAPI="$BASE/ml"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
BOLD='\033[1m'

step() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${BOLD}  STEP $1: $2${NC}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
info()  { echo -e "  ${YELLOW}→${NC} $1"; }
ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; }
show()  { echo -e "  ${CYAN}Response:${NC}"; echo "$1" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    $1"; }

if [ -z "${AUTH_TOKEN:-}" ]; then
  echo -e "${RED}ERROR: AUTH_TOKEN not set.${NC}"
  echo "  Get a token:"
  echo "    curl -s --request POST --url 'https://dev-jamm61acuiu8yfq6.us.auth0.com/oauth/token' \\"
  echo "      --header 'content-type: application/json' \\"
  echo "      --data '{\"client_id\":\"M2M_CLIENT_ID\",\"client_secret\":\"M2M_SECRET\",\"audience\":\"https://api.kwiddex.com\",\"grant_type\":\"client_credentials\"}'"
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Kwiddex Certificate Dispute System — Live Demo    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

###############################################################################
step "1" "Create and certify a test document"
###############################################################################
info "Creating test PDF..."
python3 -c "
from reportlab.pdfgen import canvas
c = canvas.Canvas('/tmp/dispute_demo.pdf')
c.drawString(100, 700, 'Dispute Demo Document')
c.drawString(100, 680, 'Created: $(date)')
c.save()
print('    Test PDF created')
"

info "Certifying document..."
HTTP=$(curl -s -o /tmp/dispute_certified.pdf -w "%{http_code}" \
  -X POST "$FASTAPI/certify" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@/tmp/dispute_demo.pdf" \
  -F "reviewer_id=demo-examiner@kwiddex.com")

if [ "$HTTP" = "200" ]; then
  CERT_SIZE=$(wc -c < /tmp/dispute_certified.pdf)
  ok "Document certified successfully ($CERT_SIZE bytes)"
else
  fail "Certification failed (HTTP $HTTP)"
  exit 1
fi

###############################################################################
step "2" "Verify the certified document (should be clean)"
###############################################################################
info "Running verification..."
VERIFY1=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@/tmp/dispute_certified.pdf")

VALID=$(echo "$VERIFY1" | python3 -c "import json,sys; print(json.load(sys.stdin)['valid'])" 2>/dev/null)
HAS_DISPUTES=$(echo "$VERIFY1" | python3 -c "import json,sys; print(json.load(sys.stdin).get('has_disputes', False))" 2>/dev/null)
CERT_ID=$(echo "$VERIFY1" | python3 -c "import json,sys; print(json.load(sys.stdin)['certificate_id'])" 2>/dev/null)

echo ""
info "What anyone sees when verifying this document:"
show "$VERIFY1"

if [ "$VALID" = "True" ] && [ "$HAS_DISPUTES" = "False" ]; then
  ok "Certificate is valid with no disputes"
  ok "Certificate ID: $CERT_ID"
else
  fail "Unexpected state: valid=$VALID, has_disputes=$HAS_DISPUTES"
fi

###############################################################################
step "3" "A third party files a dispute"
###############################################################################
info "Reporter submits dispute with substantive reason (min 50 chars)..."
echo ""
info "Dispute reason:"
echo -e "    ${YELLOW}\"This document appears to contain forged signatures based on"
echo -e "    independent ink analysis conducted by our forensic lab. The ink"
echo -e "    dating results are inconsistent with the purported document date.\"${NC}"
echo ""

REPORT=$(curl -s -X POST "$FASTAPI/report-certificate/$CERT_ID" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"reason\": \"This document appears to contain forged signatures based on independent ink analysis conducted by our forensic lab. The ink dating results are inconsistent with the purported document date.\"}")

SUCCESS=$(echo "$REPORT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)

show "$REPORT"

if [ "$SUCCESS" = "True" ]; then
  ok "Dispute filed successfully"
else
  # M2M token may be the same identity as certifier
  MSG=$(echo "$REPORT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('detail', ''))" 2>/dev/null)
  if echo "$MSG" | grep -qi "your own"; then
    info "Note: M2M token is same identity as certifier. In production, reporter would be a different user."
    info "The endpoint correctly blocks self-reporting. Dispute system is working as designed."
    echo ""
    echo -e "${YELLOW}  To see the full dispute flow with visible disputes, use two"
    echo -e "  different Auth0 accounts (one certifier, one reporter).${NC}"
    
    ok "Self-report prevention verified"
    
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║              Demo Complete (Partial)                ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  Demonstrated:                                      ║"
    echo "║    ✓ Document certification                         ║"
    echo "║    ✓ Clean verification (no disputes)               ║"
    echo "║    ✓ Self-report prevention                         ║"
    echo "║                                                     ║"
    echo "║  Requires two accounts to demonstrate:              ║"
    echo "║    • Dispute filing and visibility                  ║"
    echo "║    • Certifier notification via /my-certificates    ║"
    echo "║    • Dispute dismissal/acceptance                   ║"
    echo "║    • Permanent dispute audit trail                  ║"
    echo "╚══════════════════════════════════════════════════════╝"
    
    rm -f /tmp/dispute_demo.pdf /tmp/dispute_certified.pdf
    exit 0
  fi
  fail "Dispute filing failed: $MSG"
fi

###############################################################################
step "4" "Verify again (dispute now visible to ANYONE)"
###############################################################################
info "Anyone verifying this document now sees the dispute..."
VERIFY2=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@/tmp/dispute_certified.pdf")

HAS_DISPUTES2=$(echo "$VERIFY2" | python3 -c "import json,sys; print(json.load(sys.stdin).get('has_disputes', False))" 2>/dev/null)
DISPUTE_REASON=$(echo "$VERIFY2" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['disputes'][0]['reason'] if d.get('disputes') else 'none')" 2>/dev/null)
DISPUTE_STATUS=$(echo "$VERIFY2" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['disputes'][0]['dispute_status'] if d.get('disputes') else 'none')" 2>/dev/null)
DISPUTE_ID=$(echo "$VERIFY2" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['disputes'][0]['dispute_id'] if d.get('disputes') else '')" 2>/dev/null)

echo ""
info "What anyone now sees when verifying:"
show "$VERIFY2"

if [ "$HAS_DISPUTES2" = "True" ]; then
  ok "Dispute is visible on verification"
  ok "Dispute status: $DISPUTE_STATUS"
  ok "Reporter's reason is publicly visible"
else
  fail "Dispute not showing on verification"
fi

###############################################################################
step "5" "Certifier checks their profile (sees the dispute)"
###############################################################################
info "Certifier views /my-certificates to see disputes on their work..."
MY_CERTS=$(curl -s "$FASTAPI/my-certificates" -H "Authorization: Bearer $AUTH_TOKEN")

OPEN_DISPUTES=$(echo "$MY_CERTS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for cert in data.get('certificates', []):
    if cert.get('certificate_id') == '$CERT_ID':
        print(cert.get('open_disputes', 0))
        break
" 2>/dev/null)

echo ""
info "Certifier's dashboard shows:"
show "$MY_CERTS"

if [ "$OPEN_DISPUTES" = "1" ]; then
  ok "Certifier sees 1 open dispute on their certificate"
else
  info "Open disputes count: $OPEN_DISPUTES"
fi

###############################################################################
step "6" "Certifier responds — dismisses with explanation"
###############################################################################
info "Certifier writes a substantive dismissal response..."
echo ""
info "Certifier's response:"
echo -e "    ${YELLOW}\"Our examination used a VSC 8000 spectral imaging system."
echo -e "    The ink analysis confirms the signatures are consistent with"
echo -e "    the document date. We stand by our certification.\"${NC}"
echo ""

RESOLVE=$(curl -s -X POST "$FASTAPI/resolve-dispute/$CERT_ID/$DISPUTE_ID?action=dismiss&response_text=Our+examination+used+a+VSC+8000+spectral+imaging+system.+The+ink+analysis+confirms+the+signatures+are+consistent+with+the+document+date.+We+stand+by+our+certification." \
  -H "Authorization: Bearer $AUTH_TOKEN")

show "$RESOLVE"

RESOLVED=$(echo "$RESOLVE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)
if [ "$RESOLVED" = "True" ]; then
  ok "Dispute dismissed with written response"
else
  fail "Resolution failed"
fi

###############################################################################
step "7" "Final verification (dispute dismissed but permanently visible)"
###############################################################################
info "The audit trail is permanent. Anyone verifying sees the full history..."
VERIFY3=$(curl -s -X POST "$FASTAPI/verify-certificate" -F "file=@/tmp/dispute_certified.pdf")

FINAL_STATUS=$(echo "$VERIFY3" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['disputes'][0]['dispute_status'] if d.get('disputes') else 'none')" 2>/dev/null)
CERTIFIER_RESP=$(echo "$VERIFY3" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['disputes'][0].get('certifier_response','none') if d.get('disputes') else 'none')" 2>/dev/null)

echo ""
info "Final verification result with full audit trail:"
show "$VERIFY3"

if [ "$FINAL_STATUS" = "dismissed" ]; then
  ok "Dispute status: dismissed"
  ok "Certifier's response is permanently visible"
  ok "Certificate remains valid"
  ok "Full audit trail preserved for all parties"
else
  fail "Unexpected final state: $FINAL_STATUS"
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              Dispute Demo Complete                  ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Demonstrated:                                      ║"
echo "║    ✓ Document certification                         ║"
echo "║    ✓ Clean verification (no disputes)               ║"
echo "║    ✓ Dispute filing with substantive reason          ║"
echo "║    ✓ Dispute visible to anyone who verifies          ║"
echo "║    ✓ Certifier notification via /my-certificates    ║"
echo "║    ✓ Certifier dismissal with written response       ║"
echo "║    ✓ Permanent audit trail after resolution          ║"
echo "║                                                     ║"
echo "║  Key design decisions:                               ║"
echo "║    • Disputes are informational, not blocks           ║"
echo "║    • One report per user per certificate              ║"
echo "║    • Cannot dispute your own certificate              ║"
echo "║    • Minimum 50 chars for dispute reason              ║"
echo "║    • Minimum 20 chars for dismissal response          ║"
echo "║    • Dispute history is permanent and public          ║"
echo "║    • Kwiddex never revokes — only the certifier can  ║"
echo "╚══════════════════════════════════════════════════════╝"

rm -f /tmp/dispute_demo.pdf /tmp/dispute_certified.pdf
