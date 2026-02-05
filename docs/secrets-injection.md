# Secrets Injection System

## Overview

The Forgemaster secrets injection system provides a secure mechanism to automatically inject environment variables into Docker Compose commands executed through Claude Code's Bash tool. This prevents hardcoding secrets in scripts and keeps sensitive credentials out of version control.

### How It Works

1. A PreToolUse hook intercepts Bash tool calls before execution
2. The hook detects Docker Compose commands (docker compose, docker-compose)
3. Environment variables are injected from a secure local store
4. The modified command is passed to execution with secrets included
5. Secrets are never logged or stored in command history

### Security Features

- **Local-only storage**: Secrets stored in user's home directory (`~/.claude/hooks/`)
- **Restricted permissions**: Hook script enforced to `chmod 600` (read/write by owner only)
- **No logging**: Secrets are not echoed or logged during execution
- **Selective injection**: Only Docker Compose commands receive secrets, not all Bash calls
- **Environment variable sourcing**: Supports reading from existing environment when available

## Installation

### Step 1: Copy the Hook Script

Place the hook script in your Claude Code hooks directory:

```bash
mkdir -p ~/.claude/hooks
cp scripts/hooks/inject-secrets.sh ~/.claude/hooks/inject-secrets.sh
chmod 600 ~/.claude/hooks/inject-secrets.sh
```

The `chmod 600` command restricts file permissions so only the owner can read/write it.

### Step 2: Configure Claude Code Settings

Edit or create `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "~/.claude/hooks/inject-secrets.sh"
      }
    ]
  }
}
```

If the file already exists, add the `hooks` section to the existing JSON structure.

### Step 3: Set Environment Variables

The hook reads secrets from your environment. Export them in your shell profile:

```bash
# Add to ~/.bashrc, ~/.zshrc, or equivalent
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export GITHUB_TOKEN="your_github_token_here"
export DOCKERHUB_USERNAME="your_dockerhub_username"
export DOCKERHUB_TOKEN="your_dockerhub_token_here"
export OPENAI_API_KEY="your_openai_key_here"
export HOSTINGER_API_TOKEN="your_hostinger_token_here"
```

Or set them in your current shell session:

```bash
export ANTHROPIC_API_KEY="your_key_here"
# then run claude code
```

### Step 4: Verify Installation

Test the hook by running a simple Docker Compose command:

```bash
docker compose version
```

The hook intercepts this and prepends environment variables if any are set. You can verify in logs that secrets are being injected (though the values themselves will be masked).

## Adding New Secrets

To add a new environment variable to the secrets registry:

### In the Hook Script

Edit `scripts/hooks/inject-secrets.sh` and add to the `SECRETS` associative array:

```bash
declare -A SECRETS=(
    ["ANTHROPIC_API_KEY"]="${ANTHROPIC_API_KEY:-}"
    ["GITHUB_TOKEN"]="${GITHUB_TOKEN:-}"
    ["DOCKERHUB_USERNAME"]="${DOCKERHUB_USERNAME:-}"
    ["DOCKERHUB_TOKEN"]="${DOCKERHUB_TOKEN:-}"
    ["OPENAI_API_KEY"]="${OPENAI_API_KEY:-}"
    ["HOSTINGER_API_TOKEN"]="${HOSTINGER_API_TOKEN:-}"
    ["NEW_SECRET_NAME"]="${NEW_SECRET_NAME:-}"      # Add new entry here
)
```

The syntax `${VAR_NAME:-}` means: use the environment variable `VAR_NAME` if set, otherwise use an empty string.

### Export the Environment Variable

Add to your shell profile:

```bash
export NEW_SECRET_NAME="your_secret_value_here"
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc on macOS
```

## Usage Examples

### Docker Compose Up with Secrets

```bash
docker compose up -d
```

The hook automatically injects:

```bash
env ANTHROPIC_API_KEY=<value> GITHUB_TOKEN=<value> ... docker compose up -d
```

### Docker Compose Build with Secrets

```bash
docker compose build
```

Secrets are injected for build operations that need them.

### Non-Docker Commands Unaffected

```bash
npm install
git status
ls -la
```

These commands pass through unchanged - only Docker Compose commands are intercepted.

## Security Considerations

### Best Practices

1. **Never commit secrets**: Do not add actual secret values to any file in the repository
2. **Use environment variables**: Store secrets in your shell profile, not in scripts
3. **Restrict permissions**: Always run `chmod 600` on the hook script
4. **Review changes**: When updating the hook, verify no secrets are included
5. **Rotate regularly**: Periodically regenerate and update your API keys
6. **Use strong tokens**: Generate API keys with minimal required scopes

### What Happens to Secrets?

- **Before hook**: Secrets exist only in your shell environment
- **During execution**: Secrets are passed via command-line environment variables
- **After execution**: Command history may contain the injected command (without values, if properly configured)
- **Logging**: Claude Code and Docker logs should NOT contain unmasked secret values

### If a Secret is Compromised

1. Immediately revoke the token/key in the service's dashboard
2. Generate a new token/key
3. Update your shell profile with the new secret
4. Clear shell history if necessary: `history -c && history -w`
5. Restart any running services

## Troubleshooting

### Secrets Not Being Injected

**Problem**: Docker Compose commands don't have access to secrets

**Solutions**:
1. Verify hook script exists and has correct permissions:
   ```bash
   ls -l ~/.claude/hooks/inject-secrets.sh
   # Should show: -rw------- (600 permissions)
   ```

2. Check environment variables are exported:
   ```bash
   echo $ANTHROPIC_API_KEY
   # Should print your key, not empty
   ```

3. Verify `~/.claude/settings.json` has correct hook configuration:
   ```bash
   cat ~/.claude/settings.json | grep -A 5 "PreToolUse"
   ```

4. Test hook manually:
   ```bash
   echo '{"tool_name":"Bash","tool_input":{"command":"docker compose up"}}' | bash ~/.claude/hooks/inject-secrets.sh
   ```

### "Command not found" Errors

**Problem**: jq is not installed

**Solution**: Install jq (required for JSON parsing):

```bash
# macOS with Homebrew
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# CentOS/RHEL
sudo yum install jq

# Windows (with Git Bash)
choco install jq
```

### Hook Not Running

**Problem**: Hook command isn't being executed

**Solutions**:
1. Verify Claude Code recognizes the settings file:
   ```bash
   # Check Claude Code's documentation for settings file location
   # Usually: ~/.claude/settings.json or platform-specific location
   ```

2. Check file syntax:
   ```bash
   jq . ~/.claude/settings.json
   # Should output valid JSON, not a syntax error
   ```

3. Restart Claude Code completely after updating settings

### Secrets in Shell History

**Problem**: Secrets appear in bash history

**Solution**: Add to your shell profile to exclude commands with secrets:

```bash
# For bash/zsh - don't log commands starting with 'env '
export HISTCONTROL=ignorespace
# Then use: " docker compose up" (leading space)
```

Or disable history for sensitive operations:

```bash
set +o history
docker compose up -d
set -o history
```

## Environment Variables Reference

| Variable | Purpose | Example Source |
|----------|---------|-----------------|
| `ANTHROPIC_API_KEY` | Claude API authentication | Anthropic console |
| `GITHUB_TOKEN` | GitHub API access and authentication | GitHub Settings > Developer settings > Personal access tokens |
| `DOCKERHUB_USERNAME` | Docker Hub username for image pulls | Docker Hub account |
| `DOCKERHUB_TOKEN` | Docker Hub API token | Docker Hub account settings |
| `OPENAI_API_KEY` | OpenAI API authentication (if used) | OpenAI dashboard |
| `HOSTINGER_API_TOKEN` | Hostinger API token for domain/hosting management | Hostinger control panel |

## Advanced Configuration

### Custom Hook Script

You can modify the hook script to:
- Add additional secrets
- Change Docker Compose command detection patterns
- Add logging (to file only, never stdout)
- Support additional tools or patterns

### Multiple Hook Scripts

You can chain multiple hooks in `settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "~/.claude/hooks/inject-secrets.sh"
      },
      {
        "matcher": "Bash",
        "command": "~/.claude/hooks/audit-commands.sh"
      }
    ]
  }
}
```

Each hook receives the output of the previous one.

### Conditional Injection

To only inject secrets for specific commands, modify the hook script's grep pattern:

```bash
# Current pattern - all docker compose variations
if echo "$COMMAND" | grep -qE '(docker compose|docker-compose)\s+(up|run|exec|build)'; then

# More restrictive - only docker compose up
if echo "$COMMAND" | grep -qE '(docker compose|docker-compose)\s+up'; then
```

## Support and Reporting Issues

If you encounter issues with the secrets injection system:

1. Verify all installation steps completed
2. Check the troubleshooting section above
3. Test the hook script manually with sample input
4. Review file permissions and JSON syntax
5. Check shell environment variables are properly exported

For bugs or feature requests, open an issue in the Forgemaster repository.
