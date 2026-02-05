# Nginx Proxy Manager Automation

This guide covers the automation scripts for managing nginx-proxy-manager (NPM) proxy hosts and Hostinger DNS records. These scripts provide a command-line interface for common infrastructure operations.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Script Reference](#script-reference)
  - [nginx-proxy-add.sh](#nginx-proxy-addsh)
  - [nginx-proxy-remove.sh](#nginx-proxy-removesh)
  - [nginx-proxy-modify.sh](#nginx-proxy-modifysh)
  - [dns-hostinger.sh](#dns-hostingersh)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

## Overview

The nginx automation suite consists of four bash scripts:

| Script | Purpose |
|--------|---------|
| `nginx-proxy-add.sh` | Create new proxy hosts in NPM with optional SSL |
| `nginx-proxy-remove.sh` | Remove existing proxy hosts from NPM |
| `nginx-proxy-modify.sh` | Modify proxy host configuration |
| `dns-hostinger.sh` | Manage DNS records via Hostinger API |

All scripts use REST APIs for automation and provide color-coded output for better readability.

## Prerequisites

### Software Requirements

- **bash** - Shell interpreter (included in Linux/macOS/WSL)
- **curl** - HTTP client for API requests
- **jq** - JSON processor for parsing API responses

Install dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install curl jq

# macOS
brew install curl jq

# Windows (WSL)
sudo apt-get install curl jq
```

### Service Requirements

1. **nginx-proxy-manager** - Must be running and accessible
   - Default API endpoint: `http://localhost:81/api`
   - Admin account configured
   - API accessible from the machine running scripts

2. **Hostinger Account** - For DNS management
   - Active domain hosted on Hostinger
   - API key generated from Hostinger control panel

## Quick Start

### 1. Set Environment Variables

Create a `.env` file or export directly:

```bash
# nginx-proxy-manager credentials
export NPM_API_URL="http://localhost:81/api"
export NPM_EMAIL="admin@example.com"
export NPM_PASSWORD="your-admin-password"

# Hostinger API key
export HOSTINGER_API_KEY="your-hostinger-api-key"
```

**Security Note**: Store these credentials securely. Consider using a secrets manager or encrypted environment file.

### 2. Make Scripts Executable

```bash
cd scripts
chmod +x nginx-proxy-add.sh nginx-proxy-remove.sh nginx-proxy-modify.sh dns-hostinger.sh
```

### 3. Test Connection

```bash
# List existing DNS records
./dns-hostinger.sh --action list --domain your-domain.com
```

## Environment Setup

### nginx-proxy-manager Configuration

1. **Access NPM Admin Interface**
   - Navigate to `http://your-npm-host:81`
   - Default credentials: `admin@example.com` / `changeme`
   - Change default password immediately

2. **Verify API Access**
   ```bash
   curl -X POST http://localhost:81/api/tokens \
     -H "Content-Type: application/json" \
     -d '{"identity":"admin@example.com","secret":"your-password"}'
   ```

3. **Configure Environment Variables**
   ```bash
   export NPM_API_URL="http://localhost:81/api"
   export NPM_EMAIL="admin@example.com"
   export NPM_PASSWORD="your-admin-password"
   ```

### Hostinger API Configuration

1. **Generate API Key**
   - Log in to Hostinger control panel
   - Navigate to API settings
   - Generate new API key
   - Copy key immediately (shown only once)

2. **Verify API Access**
   ```bash
   curl -X GET https://api.hostinger.com/dns/v1/domains/your-domain.com/records \
     -H "Authorization: Bearer your-api-key"
   ```

3. **Configure Environment Variable**
   ```bash
   export HOSTINGER_API_KEY="your-hostinger-api-key"
   ```

### Secure Storage of Credentials

#### Option 1: Environment File

```bash
# Create .env file (add to .gitignore)
cat > ~/.forgemaster.env << 'EOF'
export NPM_API_URL="http://localhost:81/api"
export NPM_EMAIL="admin@example.com"
export NPM_PASSWORD="secure-password"
export HOSTINGER_API_KEY="your-api-key"
EOF

# Secure permissions
chmod 600 ~/.forgemaster.env

# Source before running scripts
source ~/.forgemaster.env
```

#### Option 2: Shell Profile

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export NPM_API_URL="http://localhost:81/api"' >> ~/.bashrc
echo 'export NPM_EMAIL="admin@example.com"' >> ~/.bashrc
echo 'export NPM_PASSWORD="secure-password"' >> ~/.bashrc
echo 'export HOSTINGER_API_KEY="your-api-key"' >> ~/.bashrc

# Reload
source ~/.bashrc
```

## Script Reference

### nginx-proxy-add.sh

Create a new proxy host in nginx-proxy-manager.

#### Syntax

```bash
./nginx-proxy-add.sh --domain DOMAIN --target-host HOST --target-port PORT [OPTIONS]
```

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--domain` | Domain name for the proxy | `--domain app.example.com` |
| `--target-host` | Upstream target host/IP | `--target-host 192.168.1.10` |
| `--target-port` | Upstream target port | `--target-port 8080` |

#### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ssl` | Enable SSL with Let's Encrypt | `false` |
| `--email` | Email for SSL notifications | Required if `--ssl` |
| `--websockets` | Enable WebSocket support | `true` |
| `--no-websockets` | Disable WebSocket support | - |
| `--block-exploits` | Enable exploit blocking | `true` |
| `--http2` | Enable HTTP/2 support | `true` |
| `--hsts` | Enable HSTS headers | `false` |
| `--force-ssl` | Force HTTPS redirect | `true` if `--ssl` |

#### Examples

**Basic HTTP proxy:**
```bash
./nginx-proxy-add.sh \
  --domain app.example.com \
  --target-host 192.168.1.10 \
  --target-port 8080
```

**HTTPS proxy with SSL certificate:**
```bash
./nginx-proxy-add.sh \
  --domain app.example.com \
  --target-host 192.168.1.10 \
  --target-port 8080 \
  --ssl \
  --email admin@example.com
```

**WebSocket application:**
```bash
./nginx-proxy-add.sh \
  --domain ws.example.com \
  --target-host 192.168.1.10 \
  --target-port 3000 \
  --websockets \
  --ssl \
  --email admin@example.com
```

**Security-hardened proxy:**
```bash
./nginx-proxy-add.sh \
  --domain secure.example.com \
  --target-host 192.168.1.10 \
  --target-port 8080 \
  --ssl \
  --email admin@example.com \
  --hsts \
  --block-exploits \
  --http2
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication failed |
| 4 | API request failed |

---

### nginx-proxy-remove.sh

Remove an existing proxy host from nginx-proxy-manager.

#### Syntax

```bash
./nginx-proxy-remove.sh --domain DOMAIN [OPTIONS]
./nginx-proxy-remove.sh --id ID [OPTIONS]
```

#### Required Arguments (one of)

| Argument | Description | Example |
|----------|-------------|---------|
| `--domain` | Domain name to remove | `--domain app.example.com` |
| `--id` | Proxy host ID to remove | `--id 42` |

#### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--force` | Skip confirmation prompt | `false` |
| `--delete-cert` | Also delete SSL certificate | `false` |

#### Examples

**Remove by domain (with confirmation):**
```bash
./nginx-proxy-remove.sh --domain app.example.com
```

**Remove by ID without confirmation:**
```bash
./nginx-proxy-remove.sh --id 42 --force
```

**Remove proxy and certificate:**
```bash
./nginx-proxy-remove.sh \
  --domain app.example.com \
  --delete-cert \
  --force
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication failed |
| 4 | API request failed |
| 5 | Proxy host not found |

---

### nginx-proxy-modify.sh

Modify an existing proxy host configuration.

#### Syntax

```bash
./nginx-proxy-modify.sh --domain DOMAIN [MODIFICATIONS...]
./nginx-proxy-modify.sh --id ID [MODIFICATIONS...]
```

#### Required Arguments (one of)

| Argument | Description | Example |
|----------|-------------|---------|
| `--domain` | Domain name to modify | `--domain app.example.com` |
| `--id` | Proxy host ID to modify | `--id 42` |

#### Modification Options (at least one required)

| Argument | Description |
|----------|-------------|
| `--target-host HOST` | Change upstream host |
| `--target-port PORT` | Change upstream port |
| `--enable` | Enable the proxy host |
| `--disable` | Disable the proxy host |
| `--ssl` | Enable SSL/force HTTPS |
| `--no-ssl` | Disable SSL/force HTTPS |
| `--websockets` | Enable WebSocket support |
| `--no-websockets` | Disable WebSocket support |
| `--block-exploits` | Enable exploit blocking |
| `--no-block-exploits` | Disable exploit blocking |
| `--http2` | Enable HTTP/2 |
| `--no-http2` | Disable HTTP/2 |
| `--hsts` | Enable HSTS |
| `--no-hsts` | Disable HSTS |

#### Examples

**Change target host and port:**
```bash
./nginx-proxy-modify.sh \
  --domain app.example.com \
  --target-host 192.168.1.20 \
  --target-port 9000
```

**Enable SSL and WebSockets:**
```bash
./nginx-proxy-modify.sh \
  --domain app.example.com \
  --ssl \
  --websockets
```

**Temporarily disable proxy:**
```bash
./nginx-proxy-modify.sh \
  --domain app.example.com \
  --disable
```

**Multiple modifications:**
```bash
./nginx-proxy-modify.sh \
  --id 42 \
  --target-port 8080 \
  --enable \
  --http2 \
  --hsts
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication failed |
| 4 | API request failed |
| 5 | Proxy host not found |

---

### dns-hostinger.sh

Manage DNS records via Hostinger API.

#### Syntax

```bash
./dns-hostinger.sh --action ACTION --domain DOMAIN [OPTIONS]
```

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--action` | Action: `add`, `remove`, `list`, `update` | `--action add` |
| `--domain` | Domain name | `--domain example.com` |

#### Action-Specific Arguments

##### For `--action=add`

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--type` | Record type (A, AAAA, CNAME, MX, TXT) | Yes | - |
| `--name` | Record name/subdomain (use `@` for root) | Yes | - |
| `--value` | Record value (IP, hostname, text) | Yes | - |
| `--ttl` | Time to live in seconds | No | `3600` |
| `--priority` | Priority for MX records | No | `10` |

##### For `--action=remove`

| Argument | Description | Required |
|----------|-------------|----------|
| `--record-id` | DNS record ID to remove | One of |
| `--name` | Record name to remove (finds first match) | these two |

##### For `--action=update`

| Argument | Description | Required |
|----------|-------------|----------|
| `--record-id` | DNS record ID to update | Yes |
| `--value` | New record value | Yes |
| `--ttl` | New TTL | No |

##### For `--action=list`

| Argument | Description | Required |
|----------|-------------|----------|
| `--type` | Filter by record type | No |
| `--name` | Filter by record name | No |

#### Examples

**List all DNS records:**
```bash
./dns-hostinger.sh \
  --action list \
  --domain example.com
```

**List only A records:**
```bash
./dns-hostinger.sh \
  --action list \
  --domain example.com \
  --type A
```

**Add A record for subdomain:**
```bash
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type A \
  --name app \
  --value 192.168.1.10
```

**Add root domain A record:**
```bash
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type A \
  --name @ \
  --value 192.168.1.10
```

**Add CNAME record:**
```bash
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type CNAME \
  --name www \
  --value example.com
```

**Add MX record:**
```bash
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type MX \
  --name @ \
  --value mail.example.com \
  --priority 10
```

**Remove record by ID:**
```bash
./dns-hostinger.sh \
  --action remove \
  --domain example.com \
  --record-id abc123
```

**Remove record by name:**
```bash
./dns-hostinger.sh \
  --action remove \
  --domain example.com \
  --name app \
  --force
```

**Update record:**
```bash
./dns-hostinger.sh \
  --action update \
  --domain example.com \
  --record-id abc123 \
  --value 192.168.1.20
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication failed |
| 4 | API request failed |
| 5 | Record not found |

---

## Common Workflows

### Workflow 1: Deploy New Application with SSL

Complete workflow to expose a new application with HTTPS.

```bash
# Step 1: Add DNS A record pointing to your server
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type A \
  --name myapp \
  --value 203.0.113.10

# Step 2: Wait for DNS propagation (5-15 minutes)
# Check with: dig myapp.example.com +short
# Or: nslookup myapp.example.com

# Step 3: Add nginx proxy host with SSL
./nginx-proxy-add.sh \
  --domain myapp.example.com \
  --target-host 192.168.1.10 \
  --target-port 8080 \
  --ssl \
  --email admin@example.com \
  --websockets

# Step 4: Verify HTTPS access
# Visit: https://myapp.example.com
```

### Workflow 2: Update Application Target

Change where an existing proxy points to.

```bash
# Modify proxy to point to new host/port
./nginx-proxy-modify.sh \
  --domain myapp.example.com \
  --target-host 192.168.1.20 \
  --target-port 9000

# Verify change
curl -I https://myapp.example.com
```

### Workflow 3: Migrate Application to New Subdomain

Move application from old to new subdomain.

```bash
# Step 1: Add new DNS record
./dns-hostinger.sh \
  --action add \
  --domain example.com \
  --type A \
  --name newapp \
  --value 203.0.113.10

# Step 2: Wait for DNS propagation
sleep 600  # 10 minutes

# Step 3: Add new proxy host
./nginx-proxy-add.sh \
  --domain newapp.example.com \
  --target-host 192.168.1.10 \
  --target-port 8080 \
  --ssl \
  --email admin@example.com

# Step 4: Test new subdomain
curl -I https://newapp.example.com

# Step 5: Remove old proxy and DNS (after verification)
./nginx-proxy-remove.sh --domain oldapp.example.com --force
./dns-hostinger.sh --action remove --domain example.com --name oldapp --force
```

### Workflow 4: Temporary Maintenance Mode

Disable proxy temporarily without deleting configuration.

```bash
# Disable proxy
./nginx-proxy-modify.sh \
  --domain myapp.example.com \
  --disable

# Perform maintenance...

# Re-enable proxy
./nginx-proxy-modify.sh \
  --domain myapp.example.com \
  --enable
```

### Workflow 5: Bulk DNS Record Setup

Set up multiple subdomains for a microservices architecture.

```bash
#!/bin/bash
# setup-microservices.sh

DOMAIN="example.com"
SERVER_IP="203.0.113.10"

# Services and their internal ports
declare -A SERVICES=(
    ["api"]="8080"
    ["web"]="3000"
    ["admin"]="3001"
    ["docs"]="8081"
)

# Add DNS records
for subdomain in "${!SERVICES[@]}"; do
    echo "Setting up $subdomain.$DOMAIN..."

    # Add DNS A record
    ./dns-hostinger.sh \
        --action add \
        --domain "$DOMAIN" \
        --type A \
        --name "$subdomain" \
        --value "$SERVER_IP"
done

# Wait for DNS propagation
echo "Waiting 10 minutes for DNS propagation..."
sleep 600

# Add nginx proxy hosts
for subdomain in "${!SERVICES[@]}"; do
    port="${SERVICES[$subdomain]}"

    echo "Creating proxy for $subdomain.$DOMAIN -> 192.168.1.10:$port"

    ./nginx-proxy-add.sh \
        --domain "$subdomain.$DOMAIN" \
        --target-host "192.168.1.10" \
        --target-port "$port" \
        --ssl \
        --email "admin@example.com" \
        --websockets
done

echo "All services configured!"
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Failed (Exit Code 3)

**Symptoms:**
```
ERROR: Authentication failed (HTTP 401): Invalid credentials
```

**Solutions:**
- Verify `NPM_EMAIL` and `NPM_PASSWORD` are correct
- Check NPM is accessible at `$NPM_API_URL`
- Ensure admin account is not locked
- Try logging in via web interface first

**Debug:**
```bash
# Test authentication manually
curl -X POST $NPM_API_URL/tokens \
  -H "Content-Type: application/json" \
  -d "{\"identity\":\"$NPM_EMAIL\",\"secret\":\"$NPM_PASSWORD\"}"
```

#### 2. SSL Certificate Request Failed

**Symptoms:**
```
⚠ SSL certificate request failed (HTTP 400): Invalid domain
```

**Solutions:**
- Ensure DNS record exists and has propagated
- Verify domain is publicly accessible on port 80
- Check firewall allows inbound HTTP/HTTPS
- Ensure port 80 is not blocked by ISP

**Verification:**
```bash
# Check DNS resolution
dig +short myapp.example.com

# Check port 80 accessibility (from external host)
curl -I http://myapp.example.com
```

#### 3. Hostinger API Key Invalid

**Symptoms:**
```
ERROR: Authentication failed (HTTP 403): Invalid API key
```

**Solutions:**
- Regenerate API key in Hostinger control panel
- Ensure no extra spaces in `HOSTINGER_API_KEY`
- Check API key hasn't expired
- Verify API access is enabled for your account

**Debug:**
```bash
# Test API key manually
curl -X GET https://api.hostinger.com/dns/v1/domains \
  -H "Authorization: Bearer $HOSTINGER_API_KEY"
```

#### 4. Proxy Host Not Found (Exit Code 5)

**Symptoms:**
```
ERROR: No proxy host found with domain: myapp.example.com
```

**Solutions:**
- List all proxy hosts to verify domain name
- Check for typos in domain name
- Use `--id` instead of `--domain` if you know the ID
- Verify you're connected to correct NPM instance

**Debug:**
```bash
# List all proxy hosts
curl -X GET $NPM_API_URL/nginx/proxy-hosts \
  -H "Authorization: Bearer $(get-npm-token)"
```

#### 5. DNS Propagation Issues

**Symptoms:**
- SSL certificate fails because domain not found
- Proxy returns 502 Bad Gateway
- DNS lookup returns old IP

**Solutions:**
- Wait longer for DNS propagation (can take up to 48 hours)
- Check propagation status: `dig @8.8.8.8 myapp.example.com`
- Clear local DNS cache: `sudo systemd-resolve --flush-caches`
- Use DNS propagation checker: https://dnschecker.org

**Check Propagation:**
```bash
# Query different DNS servers
dig @8.8.8.8 myapp.example.com      # Google DNS
dig @1.1.1.1 myapp.example.com      # Cloudflare DNS
dig @208.67.222.222 myapp.example.com  # OpenDNS
```

#### 6. Missing Dependencies

**Symptoms:**
```
ERROR: jq is required but not installed
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install curl jq

# macOS
brew install curl jq

# Check versions
curl --version
jq --version
```

#### 7. NPM API Unreachable

**Symptoms:**
```
ERROR: Failed to fetch proxy hosts: Connection refused
```

**Solutions:**
- Verify NPM is running: `docker ps | grep nginx-proxy-manager`
- Check `NPM_API_URL` is correct (default: `http://localhost:81/api`)
- Ensure port 81 is accessible
- Check firewall rules

**Debug:**
```bash
# Test NPM accessibility
curl -I http://localhost:81

# Check NPM container
docker logs nginx-proxy-manager
```

---

### Debug Mode

Enable verbose output for troubleshooting:

```bash
# Run with bash debug mode
bash -x ./nginx-proxy-add.sh --domain test.example.com --target-host 127.0.0.1 --target-port 8080

# Inspect HTTP responses
RESPONSE=$(curl -v -X POST "$NPM_API_URL/tokens" \
  -H "Content-Type: application/json" \
  -d "{\"identity\":\"$NPM_EMAIL\",\"secret\":\"$NPM_PASSWORD\"}" \
  2>&1)
echo "$RESPONSE"
```

---

### Getting Help

If issues persist:

1. **Check Script Version**: Ensure you have latest scripts
2. **Review Logs**: Check NPM logs in Docker
3. **Test APIs**: Use curl to test API endpoints directly
4. **Verify Environment**: Double-check all environment variables
5. **Check Network**: Ensure all services are reachable

**NPM Logs:**
```bash
docker logs nginx-proxy-manager -f
```

**API Health Check:**
```bash
# NPM health
curl -I $NPM_API_URL/../health

# Hostinger health (check any endpoint)
curl -I https://api.hostinger.com/dns/v1/domains
```

---

## Security Considerations

### Credential Storage

1. **Never commit credentials** to version control
2. **Use environment files** with restricted permissions (600)
3. **Rotate credentials regularly** (every 90 days)
4. **Use separate admin accounts** for automation vs. human access
5. **Consider secrets management** (HashiCorp Vault, AWS Secrets Manager)

### API Access

1. **Restrict NPM API access** to internal network only
2. **Use HTTPS** for NPM API in production
3. **Enable rate limiting** on NPM if possible
4. **Monitor API usage** for suspicious activity
5. **Revoke unused API keys** immediately

### DNS Security

1. **Enable DNSSEC** if supported by Hostinger
2. **Use CAA records** to restrict certificate authorities
3. **Monitor DNS changes** for unauthorized modifications
4. **Use separate API keys** per environment
5. **Set minimum TTL** to allow quick updates

---

## Best Practices

### Naming Conventions

```bash
# Use consistent subdomain patterns
myapp-dev.example.com    # Development
myapp-staging.example.com # Staging
myapp.example.com         # Production

# Avoid special characters in subdomains
# Use hyphens, not underscores
# Keep names short and meaningful
```

### SSL Certificates

```bash
# Always use SSL for production
--ssl --email admin@example.com

# Set reasonable SSL options
--force-ssl    # Redirect HTTP to HTTPS
--hsts         # Enable HSTS for browsers
--http2        # Enable HTTP/2 for performance
```

### Proxy Configuration

```bash
# Enable WebSocket support for real-time apps
--websockets

# Enable exploit blocking for security
--block-exploits

# Use appropriate TTL for DNS records
--ttl 3600     # 1 hour for frequently changing IPs
--ttl 86400    # 24 hours for stable IPs
```

### Automation Scripts

```bash
# Always include error handling
set -euo pipefail

# Use --force for automation
./script.sh --force

# Log all operations
./script.sh 2>&1 | tee -a automation.log

# Verify after changes
if curl -I https://myapp.example.com | grep -q "200 OK"; then
    echo "Deployment successful"
else
    echo "Deployment failed"
    exit 1
fi
```

---

## Additional Resources

- [nginx-proxy-manager Documentation](https://nginxproxymanager.com/guide/)
- [Hostinger API Documentation](https://hostinger.com/api-docs)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [DNS Best Practices](https://www.icann.org/resources/pages/dns-best-practices-2013-03-21-en)

---

## Version History

- **v1.0** (2026-02-05) - Initial release
  - nginx-proxy-add.sh
  - nginx-proxy-remove.sh
  - nginx-proxy-modify.sh
  - dns-hostinger.sh

---

**Note**: This documentation assumes nginx-proxy-manager v2.x and Hostinger DNS API v1. API endpoints and behavior may change in future versions.
