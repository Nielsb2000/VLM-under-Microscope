#!/usr/bin/env bash
# test_api.sh — smoke-test the paint-service REST API
# Usage: bash test_api.sh [BASE_URL]
set -e

BASE="${1:-http://localhost:3000}"
PASS=0; FAIL=0

ok()   { echo "  ✅ $1"; ((PASS++)); }
fail() { echo "  ❌ $1"; ((FAIL++)); }

check_field() {
  local label="$1" json="$2" key="$3" expected="$4"
  local got
  got=$(echo "$json" | grep -o "\"${key}\":[^,}]*" | head -1 | sed 's/.*: *//' | tr -d '"')
  if [[ "$got" == "$expected" ]]; then ok "$label ($key=$got)"; else fail "$label: expected $key=$expected got $got"; fi
}

echo ""
echo "=== paint-service API test ==="
echo "Target: $BASE"
echo ""

# ---- 1. New canvas ----
echo "--- 1. Create canvas 800×600 ---"
R=$(curl -sf -X POST "$BASE/api/canvas/new" -H 'Content-Type: application/json' \
  -d '{"width":800,"height":600}') || { fail "POST /api/canvas/new"; exit 1; }
check_field "canvas new" "$R" "width" "800"
check_field "canvas new" "$R" "height" "600"

# ---- 2. State ----
echo ""
echo "--- 2. GET canvas state ---"
S=$(curl -sf "$BASE/api/canvas/state") || { fail "GET /api/canvas/state"; exit 1; }
NOBJ=$(echo "$S" | grep -o '"objects":\[\]' | wc -l | tr -d ' ')
[[ "$NOBJ" -ge 1 ]] && ok "objects array present and empty" || fail "objects not empty after new canvas"

# ---- 3. Load image from URL ----
echo ""
echo "--- 3. Load a test image from URL ---"
IMG_URL="https://via.placeholder.com/800x600.png"
R=$(curl -sf -X POST "$BASE/api/canvas/load-image" \
  -H 'Content-Type: application/json' \
  -d "{\"url\":\"${IMG_URL}\"}") || { fail "POST /api/canvas/load-image (url)"; }
echo "  Response: $R"

# ---- 4. Draw rect ----
echo ""
echo "--- 4. Draw rect ---"
R=$(curl -sf -X POST "$BASE/api/draw/rect" -H 'Content-Type: application/json' \
  -d '{"x":50,"y":60,"width":180,"height":120,"stroke":"#ff0000","strokeWidth":3,"label":"defect","createdBy":"model"}')
RECT_ID=$(echo "$R" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[[ -n "$RECT_ID" ]] && ok "rect created id=$RECT_ID" || fail "rect id missing"
check_field "rect createdBy" "$R" "createdBy" "model"

# ---- 5. Draw arrow ----
echo ""
echo "--- 5. Draw arrow ---"
R=$(curl -sf -X POST "$BASE/api/draw/arrow" -H 'Content-Type: application/json' \
  -d '{"x1":10,"y1":10,"x2":200,"y2":200,"stroke":"#00ff00","strokeWidth":2,"createdBy":"model"}')
ARROW_ID=$(echo "$R" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[[ -n "$ARROW_ID" ]] && ok "arrow created id=$ARROW_ID" || fail "arrow id missing"

# ---- 6. Draw dot ----
echo ""
echo "--- 6. Draw dot ---"
R=$(curl -sf -X POST "$BASE/api/draw/dot" -H 'Content-Type: application/json' \
  -d '{"cx":400,"cy":300,"radius":8,"fill":"#ff00ff","createdBy":"model"}')
DOT_ID=$(echo "$R" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[[ -n "$DOT_ID" ]] && ok "dot created id=$DOT_ID" || fail "dot id missing"

# ---- 7. Draw text ----
echo ""
echo "--- 7. Draw text ---"
R=$(curl -sf -X POST "$BASE/api/draw/text" -H 'Content-Type: application/json' \
  -d '{"x":60,"y":40,"text":"mark this area","fontSize":20,"fill":"#ffffff","createdBy":"model"}')
TEXT_ID=$(echo "$R" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[[ -n "$TEXT_ID" ]] && ok "text created id=$TEXT_ID" || fail "text id missing"

# ---- 8. Bulk ops ----
echo ""
echo "--- 8. Bulk ops ---"
R=$(curl -sf -X POST "$BASE/api/canvas/ops" -H 'Content-Type: application/json' \
  -d '{"operations":[{"type":"ellipse","cx":600,"cy":400,"rx":60,"ry":40,"stroke":"#0000ff","createdBy":"model"},{"type":"line","x1":0,"y1":0,"x2":800,"y2":600,"stroke":"#ffff00","strokeWidth":1,"createdBy":"model"}]}')
BULK_N=$(echo "$R" | grep -o '"type"' | wc -l | tr -d ' ')
[[ "$BULK_N" -ge 2 ]] && ok "bulk ops returned $BULK_N objects" || fail "bulk ops returned <2 objects"

# ---- 9. List objects ----
echo ""
echo "--- 9. List objects ---"
R=$(curl -sf "$BASE/api/objects")
COUNT=$(echo "$R" | grep -o '"id"' | wc -l | tr -d ' ')
[[ "$COUNT" -ge 4 ]] && ok "$COUNT objects in state" || fail "Expected ≥4 objects, got $COUNT"

# ---- 10. PATCH object ----
echo ""
echo "--- 10. Update rect label ---"
R=$(curl -sf -X PATCH "$BASE/api/objects/$RECT_ID" -H 'Content-Type: application/json' \
  -d '{"label":"confirmed defect"}')
check_field "patch label" "$R" "label" "confirmed defect"

# ---- 11. createdBy metadata in state ----
echo ""
echo "--- 11. Verify createdBy metadata ---"
S=$(curl -sf "$BASE/api/canvas/state")
MODEL_COUNT=$(echo "$S" | grep -o '"createdBy":"model"' | wc -l | tr -d ' ')
[[ "$MODEL_COUNT" -ge 4 ]] && ok "found $MODEL_COUNT model-created objects" || fail "expected ≥4 model objects, got $MODEL_COUNT"

# ---- 12. Export JSON ----
echo ""
echo "--- 12. Export JSON ---"
J=$(curl -sf "$BASE/api/export/json")
HAS_OBJECTS=$(echo "$J" | grep -c '"objects"') || true
[[ "$HAS_OBJECTS" -ge 1 ]] && ok "JSON export has objects key" || fail "JSON export missing objects key"

# ---- 13. Export PNG ----
echo ""
echo "--- 13. Export PNG ---"
TMP=$(mktemp /tmp/paint_test_XXXX.png)
HTTP=$(curl -sf -w "%{http_code}" -o "$TMP" "$BASE/api/export/png") || { fail "PNG export request failed"; }
SIZE=$(wc -c < "$TMP" | tr -d ' ')
rm -f "$TMP"
[[ "$SIZE" -gt 1024 ]] && ok "PNG $SIZE bytes" || fail "PNG too small ($SIZE bytes)"

# ---- 14. DELETE object ----
echo ""
echo "--- 14. Delete dot ---"
R=$(curl -sf -X DELETE "$BASE/api/objects/$DOT_ID")
echo "$R" | grep -q '"ok":true' && ok "dot deleted" || fail "delete failed"

# ---- Summary ----
echo ""
echo "================================"
echo "Passed: $PASS   Failed: $FAIL"
[[ "$FAIL" -eq 0 ]] && echo "All tests passed!" && exit 0 || exit 1
