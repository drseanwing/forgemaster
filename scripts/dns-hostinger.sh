#!/usr/bin/env bash
# Manage DNS records via Hostinger API
#
# This script manages DNS records for domains hosted on Hostinger using their REST API.
# Supports adding, removing, and listing DNS records.
#
# Usage:
#   dns-hostinger.sh --action list --domain example.com
#   dns-hostinger.sh --action add --domain example.com --type A --name app --value 192.168.1.10
#   dns-hostinger.sh --action remove --domain example.com --record-id abc123
#
# Required environment variables:
#   HOSTINGER_API_KEY  - Hostinger API authentication key

set -euo pipefail

# Color output functions
color_green() { echo -e "\033[0;32m$1\033[0m"; }
color_red() { echo -e "\033[0;31m$1\033[0m"; }
color_yellow() { echo -e "\033[0;33m$1\033[0m"; }
color_blue() { echo -e "\033[0;34m$1\033[0m"; }
color_cyan() { echo -e "\033[0;36m$1\033[0m"; }

# Error handling
error_exit() {
    color_red "ERROR: $1" >&2
    exit "${2:-1}"
}

# Usage information
show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Manage DNS records via Hostinger API.

Required Options:
  --action ACTION              Action to perform: add, remove, list, update
  --domain DOMAIN              Domain name (e.g., example.com)

Action-Specific Options:

For --action=add:
  --type TYPE                  DNS record type (A, AAAA, CNAME, MX, TXT)
  --name NAME                  Record name/subdomain (use @ for root)
  --value VALUE                Record value (IP, hostname, text)
  --ttl TTL                    Time to live in seconds (default: 3600)
  --priority PRIORITY          Priority for MX records (default: 10)

For --action=remove:
  --record-id ID               DNS record ID to remove
                               (or use --name to find and remove by name)
  --name NAME                  Record name to remove (finds first match)

For --action=update:
  --record-id ID               DNS record ID to update
  --value VALUE                New record value
  --ttl TTL                    New TTL (optional)

For --action=list:
  --type TYPE                  Filter by record type (optional)
  --name NAME                  Filter by record name (optional)

General Options:
  --force                      Skip confirmation prompts
  -h, --help                   Show this help message

Environment Variables (Required):
  HOSTINGER_API_KEY            Hostinger API authentication key

Examples:
  # List all DNS records for a domain
  $(basename "$0") --action list --domain example.com

  # List only A records
  $(basename "$0") --action list --domain example.com --type A

  # Add an A record for subdomain
  $(basename "$0") --action add --domain example.com --type A --name app --value 192.168.1.10

  # Add a CNAME record
  $(basename "$0") --action add --domain example.com --type CNAME --name www --value example.com

  # Add root domain A record
  $(basename "$0") --action add --domain example.com --type A --name @ --value 192.168.1.10

  # Remove a DNS record by ID
  $(basename "$0") --action remove --domain example.com --record-id abc123

  # Remove a DNS record by name
  $(basename "$0") --action remove --domain example.com --name app --force

  # Update a DNS record
  $(basename "$0") --action update --domain example.com --record-id abc123 --value 192.168.1.20

Exit Codes:
  0 - Success
  1 - General error
  2 - Invalid arguments
  3 - Authentication failed
  4 - API request failed
  5 - Record not found

EOF
    exit 0
}

# Default values
ACTION=""
DOMAIN=""
RECORD_TYPE=""
RECORD_NAME=""
RECORD_VALUE=""
RECORD_ID=""
TTL=3600
PRIORITY=10
FORCE=false

# Hostinger API base URL
HOSTINGER_API_BASE="https://api.hostinger.com/dns/v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            ACTION="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --type)
            RECORD_TYPE="$2"
            shift 2
            ;;
        --name)
            RECORD_NAME="$2"
            shift 2
            ;;
        --value)
            RECORD_VALUE="$2"
            shift 2
            ;;
        --record-id)
            RECORD_ID="$2"
            shift 2
            ;;
        --ttl)
            TTL="$2"
            shift 2
            ;;
        --priority)
            PRIORITY="$2"
            shift 2
            ;;
        --force)
            FORCE=true
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
[[ -z "$ACTION" ]] && error_exit "Missing required argument: --action" 2
[[ -z "$DOMAIN" ]] && error_exit "Missing required argument: --domain" 2

# Validate action
case "$ACTION" in
    add|remove|list|update)
        ;;
    *)
        error_exit "Invalid action: $ACTION (must be: add, remove, list, update)" 2
        ;;
esac

# Validate action-specific requirements
case "$ACTION" in
    add)
        [[ -z "$RECORD_TYPE" ]] && error_exit "Missing required argument for add: --type" 2
        [[ -z "$RECORD_NAME" ]] && error_exit "Missing required argument for add: --name" 2
        [[ -z "$RECORD_VALUE" ]] && error_exit "Missing required argument for add: --value" 2
        ;;
    remove)
        if [[ -z "$RECORD_ID" && -z "$RECORD_NAME" ]]; then
            error_exit "Either --record-id or --name must be specified for remove" 2
        fi
        ;;
    update)
        [[ -z "$RECORD_ID" ]] && error_exit "Missing required argument for update: --record-id" 2
        [[ -z "$RECORD_VALUE" ]] && error_exit "Missing required argument for update: --value" 2
        ;;
esac

# Validate environment variables
[[ -z "${HOSTINGER_API_KEY:-}" ]] && error_exit "Missing required environment variable: HOSTINGER_API_KEY" 2

# Check dependencies
command -v curl >/dev/null 2>&1 || error_exit "curl is required but not installed" 1
command -v jq >/dev/null 2>&1 || error_exit "jq is required but not installed" 1

# List DNS records
list_records() {
    color_blue "=== Hostinger DNS: List Records ==="
    echo "Domain: $DOMAIN"
    echo ""

    LIST_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "${HOSTINGER_API_BASE}/domains/${DOMAIN}/records" \
        -H "Authorization: Bearer ${HOSTINGER_API_KEY}" \
        -H "Accept: application/json" \
        2>&1) || error_exit "Failed to fetch DNS records" 4

    HTTP_CODE=$(echo "$LIST_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$LIST_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "200" ]]; then
        error_exit "Failed to fetch DNS records (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
    fi

    # Filter records if type or name specified
    FILTERED_RECORDS="$RESPONSE_BODY"
    if [[ -n "$RECORD_TYPE" ]]; then
        FILTERED_RECORDS=$(echo "$FILTERED_RECORDS" | jq --arg type "$RECORD_TYPE" '[.[] | select(.type == $type)]')
    fi
    if [[ -n "$RECORD_NAME" ]]; then
        FILTERED_RECORDS=$(echo "$FILTERED_RECORDS" | jq --arg name "$RECORD_NAME" '[.[] | select(.name == $name)]')
    fi

    # Display records in table format
    RECORD_COUNT=$(echo "$FILTERED_RECORDS" | jq 'length')

    if [[ "$RECORD_COUNT" -eq 0 ]]; then
        color_yellow "No DNS records found"
        exit 0
    fi

    color_green "Found $RECORD_COUNT record(s):"
    echo ""

    # Print table header
    printf "%-12s %-30s %-8s %-40s %-8s\n" "ID" "NAME" "TYPE" "VALUE" "TTL"
    printf "%-12s %-30s %-8s %-40s %-8s\n" "------------" "------------------------------" "--------" "----------------------------------------" "--------"

    # Print records
    echo "$FILTERED_RECORDS" | jq -r '.[] | "\(.id)\t\(.name)\t\(.type)\t\(.value)\t\(.ttl)"' | \
    while IFS=$'\t' read -r id name type value ttl; do
        printf "%-12s %-30s %-8s %-40s %-8s\n" "$id" "$name" "$type" "$value" "$ttl"
    done
}

# Add DNS record
add_record() {
    color_blue "=== Hostinger DNS: Add Record ==="
    echo "Domain:  $DOMAIN"
    echo "Type:    $RECORD_TYPE"
    echo "Name:    $RECORD_NAME"
    echo "Value:   $RECORD_VALUE"
    echo "TTL:     $TTL"
    echo ""

    # Build JSON payload
    JSON_PAYLOAD=$(jq -n \
        --arg type "$RECORD_TYPE" \
        --arg name "$RECORD_NAME" \
        --arg value "$RECORD_VALUE" \
        --argjson ttl "$TTL" \
        '{
            type: $type,
            name: $name,
            content: $value,
            ttl: $ttl
        }')

    # Add priority for MX records
    if [[ "$RECORD_TYPE" == "MX" ]]; then
        JSON_PAYLOAD=$(echo "$JSON_PAYLOAD" | jq --argjson priority "$PRIORITY" '.priority = $priority')
    fi

    ADD_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${HOSTINGER_API_BASE}/domains/${DOMAIN}/records" \
        -H "Authorization: Bearer ${HOSTINGER_API_KEY}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "$JSON_PAYLOAD" \
        2>&1) || error_exit "Failed to add DNS record" 4

    HTTP_CODE=$(echo "$ADD_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$ADD_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "201" && "$HTTP_CODE" != "200" ]]; then
        error_exit "Failed to add DNS record (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
    fi

    NEW_RECORD_ID=$(echo "$RESPONSE_BODY" | jq -r '.id // empty')

    color_green "✓ DNS record added successfully"
    echo ""
    echo "Record Details:"
    echo "  ID:     $NEW_RECORD_ID"
    echo "  Type:   $RECORD_TYPE"
    echo "  Name:   $RECORD_NAME"
    echo "  Value:  $RECORD_VALUE"
    echo ""
    color_yellow "Note: DNS propagation may take 5-15 minutes"
}

# Remove DNS record
remove_record() {
    # If name specified instead of ID, find the record
    if [[ -z "$RECORD_ID" && -n "$RECORD_NAME" ]]; then
        color_blue "Finding DNS record by name: $RECORD_NAME"

        LIST_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "${HOSTINGER_API_BASE}/domains/${DOMAIN}/records" \
            -H "Authorization: Bearer ${HOSTINGER_API_KEY}" \
            -H "Accept: application/json" \
            2>&1) || error_exit "Failed to fetch DNS records" 4

        HTTP_CODE=$(echo "$LIST_RESPONSE" | tail -n1)
        RESPONSE_BODY=$(echo "$LIST_RESPONSE" | sed '$d')

        if [[ "$HTTP_CODE" != "200" ]]; then
            error_exit "Failed to fetch DNS records (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
        fi

        RECORD_ID=$(echo "$RESPONSE_BODY" | jq -r --arg name "$RECORD_NAME" \
            '.[] | select(.name == $name) | .id' | head -n1)

        if [[ -z "$RECORD_ID" ]]; then
            error_exit "No DNS record found with name: $RECORD_NAME" 5
        fi

        color_green "✓ Found record (ID: $RECORD_ID)"
    fi

    color_blue "=== Hostinger DNS: Remove Record ==="
    echo "Domain:     $DOMAIN"
    echo "Record ID:  $RECORD_ID"
    echo ""

    # Confirmation prompt
    if [[ "$FORCE" != true ]]; then
        color_yellow "⚠ This action cannot be undone!"
        read -p "Are you sure you want to delete this DNS record? (yes/no): " -r CONFIRM
        echo ""

        if [[ "$CONFIRM" != "yes" ]]; then
            color_blue "Deletion cancelled"
            exit 0
        fi
    fi

    DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "${HOSTINGER_API_BASE}/domains/${DOMAIN}/records/${RECORD_ID}" \
        -H "Authorization: Bearer ${HOSTINGER_API_KEY}" \
        -H "Accept: application/json" \
        2>&1) || error_exit "Failed to delete DNS record" 4

    HTTP_CODE=$(echo "$DELETE_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$DELETE_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "204" && "$HTTP_CODE" != "200" ]]; then
        error_exit "Failed to delete DNS record (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
    fi

    color_green "✓ DNS record deleted successfully"
    echo ""
    color_yellow "Note: DNS changes may take 5-15 minutes to propagate"
}

# Update DNS record
update_record() {
    color_blue "=== Hostinger DNS: Update Record ==="
    echo "Domain:     $DOMAIN"
    echo "Record ID:  $RECORD_ID"
    echo "New Value:  $RECORD_VALUE"
    echo ""

    # Build JSON payload
    JSON_PAYLOAD=$(jq -n \
        --arg value "$RECORD_VALUE" \
        --argjson ttl "$TTL" \
        '{
            content: $value,
            ttl: $ttl
        }')

    UPDATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "${HOSTINGER_API_BASE}/domains/${DOMAIN}/records/${RECORD_ID}" \
        -H "Authorization: Bearer ${HOSTINGER_API_KEY}" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "$JSON_PAYLOAD" \
        2>&1) || error_exit "Failed to update DNS record" 4

    HTTP_CODE=$(echo "$UPDATE_RESPONSE" | tail -n1)
    RESPONSE_BODY=$(echo "$UPDATE_RESPONSE" | sed '$d')

    if [[ "$HTTP_CODE" != "200" ]]; then
        error_exit "Failed to update DNS record (HTTP $HTTP_CODE): $RESPONSE_BODY" 4
    fi

    color_green "✓ DNS record updated successfully"
    echo ""
    color_yellow "Note: DNS changes may take 5-15 minutes to propagate"
}

# Execute action
case "$ACTION" in
    list)
        list_records
        ;;
    add)
        add_record
        ;;
    remove)
        remove_record
        ;;
    update)
        update_record
        ;;
esac

exit 0
