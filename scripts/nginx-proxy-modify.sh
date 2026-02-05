#!/usr/bin/env bash
# Modify an existing proxy host in nginx-proxy-manager
#
# This script updates an existing proxy host in nginx-proxy-manager via its REST API.
# Only specified fields are modified; others remain unchanged.
#
# Usage:
#   nginx-proxy-modify.sh --domain example.com [options]
#
# Required environment variables:
#   NPM_API_URL      - nginx-proxy-manager API base URL (e.g., http://localhost:81/api)
#   NPM_EMAIL        - nginx-proxy-manager admin email
#   NPM_PASSWORD     - nginx-proxy-manager admin password

set -euo pipefail

# Color output functions
color_green() { echo -e "\033[0;32m$1\033[0m"; }
color_red() { echo -e "\033[0;31m$1\033[0m"; }
color_yellow() { echo -e "\033[0;33m$1\033[0m"; }
color_blue() { echo -e "\033[0;34m$1\033[0m"; }

# Error handling
error_exit() {
    color_red "ERROR: $1" >&2
    exit "${2:-1}"
}

# Usage information
show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Modify an existing proxy host in nginx-proxy-manager.

Required Options:
  --domain DOMAIN              Domain name of the proxy host to modify
                               (or use --id to specify by ID)
  --id ID                      Proxy host ID to modify (alternative to --domain)

Modification Options (at least one required):
  --target-host HOST           New target upstream host
  --target-port PORT           New target upstream port
  --enable                     Enable the proxy host
  --disable                    Disable the proxy host
  --ssl                        Enable SSL/force HTTPS redirect
  --no-ssl                     Disable SSL/force HTTPS redirect
  --websockets                 Enable WebSocket upgrade support
  --no-websockets              Disable WebSocket upgrade support
  --block-exploits             Enable common exploit blocking
  --no-block-exploits          Disable common exploit blocking
  --http2                      Enable HTTP/2 support
  --no-http2                   Disable HTTP/2 support
  --hsts                       Enable HSTS
  --no-hsts                    Disable HSTS

General Options:
  -h, --help                   Show this help message

Environment Variables (Required):
  NPM_API_URL                  nginx-proxy-manager API URL (e.g., http://localhost:81/api)
  NPM_EMAIL                    Admin email for authentication
  NPM_PASSWORD                 Admin password for authentication

Examples:
  # Change target host and port
  $(basename "$0") --domain app.example.com --target-host 192.168.1.20 --target-port 9000

  # Enable SSL and WebSocket support
  $(basename "$0") --domain app.example.com --ssl --websockets

  # Disable proxy host temporarily
  $(basename "$0") --domain app.example.com --disable

  # Multiple modifications at once
  $(basename "$0") --id 42 --target-port 8080 --enable --http2

Exit Codes:
  0 - Success
  1 - General error
  2 - Invalid arguments
  3 - Authentication failed
  4 - API request failed
  5 - Proxy host not found

EOF
    exit 0
}

# Default values
DOMAIN=""
PROXY_ID=""
MODIFICATIONS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --id)
            PROXY_ID="$2"
            shift 2
            ;;
        --target-host)
            MODIFICATIONS+=("forward_host=$2")
            shift 2
            ;;
        --target-port)
            MODIFICATIONS+=("forward_port=$2")
            shift 2
            ;;
        --enable)
            MODIFICATIONS+=("enabled=true")
            shift
            ;;
        --disable)
            MODIFICATIONS+=("enabled=false")
            shift
            ;;
        --ssl)
            MODIFICATIONS+=("ssl_forced=true")
            shift
            ;;
        --no-ssl)
            MODIFICATIONS+=("ssl_forced=false")
            shift
            ;;
        --websockets)
            MODIFICATIONS+=("allow_websocket_upgrade=true")
            shift
            ;;
        --no-websockets)
            MODIFICATIONS+=("allow_websocket_upgrade=false")
            shift
            ;;
        --block-exploits)
            MODIFICATIONS+=("block_exploits=true")
            shift
            ;;
        --no-block-exploits)
            MODIFICATIONS+=("block_exploits=false")
            shift
            ;;
        --http2)
            MODIFICATIONS+=("http2_support=true")
            shift
            ;;
        --no-http2)
            MODIFICATIONS+=("http2_support=false")
            shift
            ;;
        --hsts)
            MODIFICATIONS+=("hsts_enabled=true")
            shift
            ;;
        --no-hsts)
            MODIFICATIONS+=("hsts_enabled=false")
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            error_exit "Unknown option: $1\nUse --help for usage information" 2
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DOMAIN" && -z "$PROXY_ID" ]]; then
    error_exit "Either --domain or --id must be specified" 2
fi

if [[ -n "$DOMAIN" && -n "$PROXY_ID" ]]; then
    error_exit "Cannot specify both --domain and --id" 2
fi

if [[ ${#MODIFICATIONS[@]} -eq 0 ]]; then
    error_exit "At least one modification option must be specified" 2
fi

# Validate environment variables
[[ -z "${NPM_API_URL:-}" ]] && error_exit "Missing required environment variable: NPM_API_URL" 2
[[ -z "${NPM_EMAIL:-}" ]] && error_exit "Missing required environment variable: NPM_EMAIL" 2
[[ -z "${NPM_PASSWORD:-}" ]] && error_exit "Missing required environment variable: NPM_PASSWORD" 2

# Check dependencies
command -v curl >/dev/null 2>&1 || error_exit "curl is required but not installed" 1
command -v jq >/dev/null 2>&1 || error_exit "jq is required but not installed" 1

color_blue "=== nginx-proxy-manager: Modify Proxy Host ==="
echo ""

# Step 1: Authenticate with NPM API
color_blue "[1/4] Authenticating with nginx-proxy-manager..."
AUTH_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${NPM_API_URL}/tokens" \
    -H "Content-Type: application/json" \
    -d "{\"identity\":\"${NPM_EMAIL}\",\"secret\":\"${NPM_PASSWORD}\"}" \
    2>&1) || error_exit "Authentication request failed" 3

HTTP_CODE=$(echo "$AUTH_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$AUTH_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    error_exit "Authentication failed (HTTP $HTTP_CODE): $RESPONSE_BODY" 3
fi

TOKEN=$(echo "$RESPONSE_BODY" | jq -r '.token // empty')
[[ -z "$TOKEN" ]] && error_exit "Failed to extract authentication token" 3

color_green "✓ Authentication successful"

# Step 2: Find proxy host if domain specified
if [[ -n "$DOMAIN" ]]; then
    color_blue "[2/4] Finding proxy host by domain: $DOMAIN"

    HOSTS_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "${NPM_API_URL}/nginx/proxy-hosts" \
        -H "Authorization: Bearer ${TOKEN}" \
        2>&1) || error_exit "Failed to fetch proxy hosts" 4

    HTTP_CODE=$(echo "$HOSTS_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$HOSTS_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "200" ]]; then
        error_exit "Failed to fetch proxy hosts (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
    fi

    # Find proxy host with matching domain
    PROXY_ID=$(echo "$RESPONSE_BODY" | jq -r --arg domain "$DOMAIN" \
        '.[] | select(.domain_names[] == $domain) | .id' | head -n1)

    if [[ -z "$PROXY_ID" ]]; then
        error_exit "No proxy host found with domain: $DOMAIN" 5
    fi

    color_green "✓ Found proxy host (ID: $PROXY_ID)"
else
    color_blue "[2/4] Using proxy host ID: $PROXY_ID"
fi

# Step 3: Fetch current proxy host configuration
color_blue "[3/4] Fetching current proxy host configuration..."

DETAIL_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "${NPM_API_URL}/nginx/proxy-hosts/${PROXY_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    2>&1) || error_exit "Failed to fetch proxy host details" 4

HTTP_CODE=$(echo "$DETAIL_RESPONSE" | tail -n1)
CURRENT_CONFIG=$(echo "$DETAIL_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    error_exit "Failed to fetch proxy host details (HTTP $HTTP_CODE): $CURRENT_CONFIG" 4
fi

color_green "✓ Current configuration retrieved"

# Step 4: Apply modifications and update
color_blue "[4/4] Applying modifications..."

# Start with current config
UPDATED_CONFIG="$CURRENT_CONFIG"

# Apply each modification
for mod in "${MODIFICATIONS[@]}"; do
    FIELD=$(echo "$mod" | cut -d= -f1)
    VALUE=$(echo "$mod" | cut -d= -f2-)

    # Handle boolean values
    if [[ "$VALUE" == "true" || "$VALUE" == "false" ]]; then
        UPDATED_CONFIG=$(echo "$UPDATED_CONFIG" | jq --argjson val "$VALUE" ".${FIELD} = \$val")
    # Handle numeric values
    elif [[ "$VALUE" =~ ^[0-9]+$ ]]; then
        UPDATED_CONFIG=$(echo "$UPDATED_CONFIG" | jq --argjson val "$VALUE" ".${FIELD} = \$val")
    # Handle string values
    else
        UPDATED_CONFIG=$(echo "$UPDATED_CONFIG" | jq --arg val "$VALUE" ".${FIELD} = \$val")
    fi

    echo "  - Setting $FIELD = $VALUE"
done

# Send update request
UPDATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "${NPM_API_URL}/nginx/proxy-hosts/${PROXY_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$UPDATED_CONFIG" \
    2>&1) || error_exit "Failed to update proxy host" 4

HTTP_CODE=$(echo "$UPDATE_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$UPDATE_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    error_exit "Failed to update proxy host (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
fi

color_green "✓ Proxy host updated successfully"

# Display updated configuration summary
PROXY_DOMAINS=$(echo "$RESPONSE_BODY" | jq -r '.domain_names | join(", ")')
FORWARD_HOST=$(echo "$RESPONSE_BODY" | jq -r '.forward_host')
FORWARD_PORT=$(echo "$RESPONSE_BODY" | jq -r '.forward_port')
ENABLED=$(echo "$RESPONSE_BODY" | jq -r '.enabled')
SSL_FORCED=$(echo "$RESPONSE_BODY" | jq -r '.ssl_forced')

echo ""
color_green "=== Success ==="
color_green "Proxy host modified successfully!"
echo ""
echo "Updated Configuration:"
echo "  ID:          $PROXY_ID"
echo "  Domains:     $PROXY_DOMAINS"
echo "  Target:      $FORWARD_HOST:$FORWARD_PORT"
echo "  Enabled:     $ENABLED"
echo "  SSL Forced:  $SSL_FORCED"
echo ""

exit 0
