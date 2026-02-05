#!/usr/bin/env bash
# Add a new proxy host to nginx-proxy-manager
#
# This script creates a new proxy host in nginx-proxy-manager via its REST API,
# with optional SSL certificate provisioning via Let's Encrypt.
#
# Usage:
#   nginx-proxy-add.sh --domain example.com --target-host 192.168.1.10 --target-port 8080 [options]
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

Add a new proxy host to nginx-proxy-manager.

Required Options:
  --domain DOMAIN              Domain name for the proxy host
  --target-host HOST           Target upstream host (IP or hostname)
  --target-port PORT           Target upstream port

Optional Options:
  --ssl                        Enable SSL with Let's Encrypt (default: false)
  --email EMAIL                Email for Let's Encrypt notifications (required if --ssl)
  --websockets                 Enable WebSocket upgrade support (default: true)
  --block-exploits             Enable common exploit blocking (default: true)
  --http2                      Enable HTTP/2 support (default: true)
  --hsts                       Enable HSTS (default: false)
  --force-ssl                  Force SSL redirect (default: true if --ssl)
  -h, --help                   Show this help message

Environment Variables (Required):
  NPM_API_URL                  nginx-proxy-manager API URL (e.g., http://localhost:81/api)
  NPM_EMAIL                    Admin email for authentication
  NPM_PASSWORD                 Admin password for authentication

Examples:
  # Add simple HTTP proxy
  $(basename "$0") --domain app.example.com --target-host 192.168.1.10 --target-port 8080

  # Add HTTPS proxy with SSL certificate
  $(basename "$0") --domain app.example.com --target-host 192.168.1.10 --target-port 8080 \\
    --ssl --email admin@example.com

  # Add proxy with custom settings
  $(basename "$0") --domain app.example.com --target-host 192.168.1.10 --target-port 8080 \\
    --ssl --email admin@example.com --hsts --no-websockets

Exit Codes:
  0 - Success
  1 - General error
  2 - Invalid arguments
  3 - Authentication failed
  4 - API request failed

EOF
    exit 0
}

# Default values
DOMAIN=""
TARGET_HOST=""
TARGET_PORT=""
SSL_ENABLED=false
SSL_EMAIL=""
WEBSOCKETS=true
BLOCK_EXPLOITS=true
HTTP2=true
HSTS=false
FORCE_SSL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --target-host)
            TARGET_HOST="$2"
            shift 2
            ;;
        --target-port)
            TARGET_PORT="$2"
            shift 2
            ;;
        --ssl)
            SSL_ENABLED=true
            FORCE_SSL=true
            shift
            ;;
        --email)
            SSL_EMAIL="$2"
            shift 2
            ;;
        --websockets)
            WEBSOCKETS=true
            shift
            ;;
        --no-websockets)
            WEBSOCKETS=false
            shift
            ;;
        --block-exploits)
            BLOCK_EXPLOITS=true
            shift
            ;;
        --no-block-exploits)
            BLOCK_EXPLOITS=false
            shift
            ;;
        --http2)
            HTTP2=true
            shift
            ;;
        --no-http2)
            HTTP2=false
            shift
            ;;
        --hsts)
            HSTS=true
            shift
            ;;
        --no-hsts)
            HSTS=false
            shift
            ;;
        --force-ssl)
            FORCE_SSL=true
            shift
            ;;
        --no-force-ssl)
            FORCE_SSL=false
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
[[ -z "$DOMAIN" ]] && error_exit "Missing required argument: --domain" 2
[[ -z "$TARGET_HOST" ]] && error_exit "Missing required argument: --target-host" 2
[[ -z "$TARGET_PORT" ]] && error_exit "Missing required argument: --target-port" 2

# Validate environment variables
[[ -z "${NPM_API_URL:-}" ]] && error_exit "Missing required environment variable: NPM_API_URL" 2
[[ -z "${NPM_EMAIL:-}" ]] && error_exit "Missing required environment variable: NPM_EMAIL" 2
[[ -z "${NPM_PASSWORD:-}" ]] && error_exit "Missing required environment variable: NPM_PASSWORD" 2

# Validate SSL requirements
if [[ "$SSL_ENABLED" == true && -z "$SSL_EMAIL" ]]; then
    error_exit "SSL enabled but --email not provided" 2
fi

# Check dependencies
command -v curl >/dev/null 2>&1 || error_exit "curl is required but not installed" 1
command -v jq >/dev/null 2>&1 || error_exit "jq is required but not installed" 1

color_blue "=== nginx-proxy-manager: Add Proxy Host ==="
echo "Domain:      $DOMAIN"
echo "Target:      $TARGET_HOST:$TARGET_PORT"
echo "SSL:         $SSL_ENABLED"
echo "WebSockets:  $WEBSOCKETS"
echo ""

# Step 1: Authenticate with NPM API
color_blue "[1/3] Authenticating with nginx-proxy-manager..."
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

# Step 2: Create proxy host
color_blue "[2/3] Creating proxy host..."

# Build JSON payload
JSON_PAYLOAD=$(jq -n \
    --arg domain "$DOMAIN" \
    --arg forward_host "$TARGET_HOST" \
    --argjson forward_port "$TARGET_PORT" \
    --argjson websockets "$WEBSOCKETS" \
    --argjson block_exploits "$BLOCK_EXPLOITS" \
    --argjson http2 "$HTTP2" \
    --argjson hsts "$HSTS" \
    --argjson force_ssl "$FORCE_SSL" \
    '{
        domain_names: [$domain],
        forward_host: $forward_host,
        forward_port: $forward_port,
        forward_scheme: "http",
        access_list_id: 0,
        certificate_id: 0,
        ssl_forced: $force_ssl,
        caching_enabled: false,
        block_exploits: $block_exploits,
        allow_websocket_upgrade: $websockets,
        http2_support: $http2,
        hsts_enabled: $hsts,
        hsts_subdomains: false,
        advanced_config: "",
        enabled: true
    }')

CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${NPM_API_URL}/nginx/proxy-hosts" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" \
    2>&1) || error_exit "Proxy host creation request failed" 4

HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$CREATE_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "201" ]]; then
    error_exit "Proxy host creation failed (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
fi

PROXY_HOST_ID=$(echo "$RESPONSE_BODY" | jq -r '.id // empty')
[[ -z "$PROXY_HOST_ID" ]] && error_exit "Failed to extract proxy host ID" 4

color_green "✓ Proxy host created (ID: $PROXY_HOST_ID)"

# Step 3: Request SSL certificate if enabled
if [[ "$SSL_ENABLED" == true ]]; then
    color_blue "[3/3] Requesting Let's Encrypt SSL certificate..."

    CERT_PAYLOAD=$(jq -n \
        --arg email "$SSL_EMAIL" \
        --arg domain "$DOMAIN" \
        '{
            provider: "letsencrypt",
            nice_name: $domain,
            domain_names: [$domain],
            meta: {
                letsencrypt_email: $email,
                letsencrypt_agree: true
            }
        }')

    CERT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${NPM_API_URL}/nginx/certificates" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$CERT_PAYLOAD" \
        2>&1) || error_exit "SSL certificate request failed" 4

    HTTP_CODE=$(echo "$CERT_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$CERT_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "201" ]]; then
        color_yellow "⚠ SSL certificate request failed (HTTP $HTTP_CODE): $RESPONSE_BODY"
        color_yellow "You may need to request the certificate manually from the NPM interface"
    else
        CERT_ID=$(echo "$RESPONSE_BODY" | jq -r '.id // empty')
        color_green "✓ SSL certificate issued (ID: $CERT_ID)"

        # Update proxy host with certificate ID
        UPDATE_PAYLOAD=$(echo "$JSON_PAYLOAD" | jq --argjson cert_id "$CERT_ID" '.certificate_id = $cert_id')

        curl -s -X PUT "${NPM_API_URL}/nginx/proxy-hosts/${PROXY_HOST_ID}" \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$UPDATE_PAYLOAD" \
            >/dev/null 2>&1 || color_yellow "⚠ Failed to attach certificate to proxy host"
    fi
else
    color_blue "[3/3] Skipping SSL certificate (not requested)"
fi

echo ""
color_green "=== Success ==="
color_green "Proxy host added successfully!"
echo ""
echo "Details:"
echo "  Domain:     $DOMAIN"
echo "  Target:     $TARGET_HOST:$TARGET_PORT"
echo "  Proxy ID:   $PROXY_HOST_ID"
echo "  SSL:        $SSL_ENABLED"
echo ""

if [[ "$SSL_ENABLED" == true ]]; then
    echo "Access your site at: https://$DOMAIN"
else
    echo "Access your site at: http://$DOMAIN"
fi

exit 0
