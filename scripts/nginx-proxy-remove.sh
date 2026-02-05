#!/usr/bin/env bash
# Remove a proxy host from nginx-proxy-manager
#
# This script deletes a proxy host from nginx-proxy-manager via its REST API.
# Can identify proxy host by domain name or ID.
#
# Usage:
#   nginx-proxy-remove.sh --domain example.com [options]
#   nginx-proxy-remove.sh --id 42 [options]
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

Remove a proxy host from nginx-proxy-manager.

Required Options (one of):
  --domain DOMAIN              Domain name of the proxy host to remove
  --id ID                      Proxy host ID to remove

Optional Options:
  --force                      Skip confirmation prompt
  --delete-cert                Also delete associated SSL certificate
  -h, --help                   Show this help message

Environment Variables (Required):
  NPM_API_URL                  nginx-proxy-manager API URL (e.g., http://localhost:81/api)
  NPM_EMAIL                    Admin email for authentication
  NPM_PASSWORD                 Admin password for authentication

Examples:
  # Remove proxy host by domain (with confirmation)
  $(basename "$0") --domain app.example.com

  # Remove proxy host by ID without confirmation
  $(basename "$0") --id 42 --force

  # Remove proxy host and its SSL certificate
  $(basename "$0") --domain app.example.com --delete-cert --force

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
FORCE=false
DELETE_CERT=false

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
        --force)
            FORCE=true
            shift
            ;;
        --delete-cert)
            DELETE_CERT=true
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

# Validate environment variables
[[ -z "${NPM_API_URL:-}" ]] && error_exit "Missing required environment variable: NPM_API_URL" 2
[[ -z "${NPM_EMAIL:-}" ]] && error_exit "Missing required environment variable: NPM_EMAIL" 2
[[ -z "${NPM_PASSWORD:-}" ]] && error_exit "Missing required environment variable: NPM_PASSWORD" 2

# Check dependencies
command -v curl >/dev/null 2>&1 || error_exit "curl is required but not installed" 1
command -v jq >/dev/null 2>&1 || error_exit "jq is required but not installed" 1

color_blue "=== nginx-proxy-manager: Remove Proxy Host ==="
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

# Step 3: Fetch proxy host details
color_blue "[3/4] Fetching proxy host details..."

DETAIL_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "${NPM_API_URL}/nginx/proxy-hosts/${PROXY_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    2>&1) || error_exit "Failed to fetch proxy host details" 4

HTTP_CODE=$(echo "$DETAIL_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$DETAIL_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    error_exit "Failed to fetch proxy host details (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
fi

PROXY_DOMAINS=$(echo "$RESPONSE_BODY" | jq -r '.domain_names | join(", ")')
FORWARD_HOST=$(echo "$RESPONSE_BODY" | jq -r '.forward_host')
FORWARD_PORT=$(echo "$RESPONSE_BODY" | jq -r '.forward_port')
CERT_ID=$(echo "$RESPONSE_BODY" | jq -r '.certificate_id // 0')

color_green "✓ Proxy host details retrieved"
echo ""
echo "Proxy Host Details:"
echo "  ID:         $PROXY_ID"
echo "  Domains:    $PROXY_DOMAINS"
echo "  Target:     $FORWARD_HOST:$FORWARD_PORT"
echo "  Cert ID:    $CERT_ID"
echo ""

# Confirmation prompt
if [[ "$FORCE" != true ]]; then
    color_yellow "⚠ This action cannot be undone!"
    read -p "Are you sure you want to delete this proxy host? (yes/no): " -r CONFIRM
    echo ""

    if [[ "$CONFIRM" != "yes" ]]; then
        color_blue "Deletion cancelled"
        exit 0
    fi
fi

# Step 4: Delete proxy host
color_blue "[4/4] Deleting proxy host..."

DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "${NPM_API_URL}/nginx/proxy-hosts/${PROXY_ID}" \
    -H "Authorization: Bearer ${TOKEN}" \
    2>&1) || error_exit "Failed to delete proxy host" 4

HTTP_CODE=$(echo "$DELETE_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$DELETE_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    error_exit "Failed to delete proxy host (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
fi

color_green "✓ Proxy host deleted successfully"

# Optional: Delete SSL certificate
if [[ "$DELETE_CERT" == true && "$CERT_ID" != "0" ]]; then
    color_blue "Deleting SSL certificate (ID: $CERT_ID)..."

    CERT_DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "${NPM_API_URL}/nginx/certificates/${CERT_ID}" \
        -H "Authorization: Bearer ${TOKEN}" \
        2>&1) || color_yellow "⚠ Failed to delete SSL certificate"

    HTTP_CODE=$(echo "$CERT_DELETE_RESPONSE" | tail -n1)

    if [[ "$HTTP_CODE" == "200" ]]; then
        color_green "✓ SSL certificate deleted"
    else
        color_yellow "⚠ SSL certificate may still exist (HTTP $HTTP_CODE)"
        color_yellow "  You may need to delete it manually from the NPM interface"
    fi
fi

echo ""
color_green "=== Success ==="
color_green "Proxy host removed successfully!"
echo ""
echo "Removed:"
echo "  Domains:  $PROXY_DOMAINS"
echo "  Target:   $FORWARD_HOST:$FORWARD_PORT"

exit 0
