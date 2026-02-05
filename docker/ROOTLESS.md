# Rootless Docker Deployment

FORGEMASTER is designed to work with rootless Docker for enhanced security.

## Rootless Docker Compatibility

All services in this deployment are configured for rootless Docker:

### Non-Privileged Containers
- No containers run with `privileged: true`
- All services use standard user namespaces
- No elevated capabilities required

### Volume Permissions
- All application directories use proper ownership via `--chown` flags
- Non-root user `forgemaster:forgemaster` (UID/GID 1000)
- PostgreSQL and Ollama use their default non-root users

### Docker Socket Access
The orchestrator container mounts the Docker socket as read-only:
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro
```

For rootless Docker, this path needs adjustment:

```bash
# Find your rootless Docker socket location
echo $DOCKER_HOST

# Common locations:
# - unix:///run/user/1000/docker.sock
# - unix://$HOME/.docker/run/docker.sock
```

Update the volume mount in `docker-compose.yml`:
```yaml
volumes:
  - ${XDG_RUNTIME_DIR}/docker.sock:/var/run/docker.sock:ro
```

Or use environment variable:
```bash
export DOCKER_HOST=unix:///run/user/1000/docker.sock
docker compose up
```

## Setup Rootless Docker

### Ubuntu/Debian
```bash
# Install rootless Docker
curl -fsSL https://get.docker.com/rootless | sh

# Add to PATH
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
echo 'export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock' >> ~/.bashrc
source ~/.bashrc

# Enable at boot
systemctl --user enable docker
sudo loginctl enable-linger $(whoami)
```

### Verify Installation
```bash
docker context ls
docker info | grep rootless
```

## Resource Limits

Rootless Docker may have lower default resource limits. Adjust via:

```bash
# Increase user namespaces
echo "user.max_user_namespaces=28633" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Increase inotify watches (for file monitoring)
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Known Limitations

1. **Port Binding**: Ports below 1024 require privileged ports permission:
   ```bash
   sudo setcap cap_net_bind_service=ep $(which rootlesskit)
   ```

2. **Volume Performance**: Rootless volumes may have slightly slower I/O
   - For PostgreSQL, consider using `pgdata` volume with `z` flag for SELinux

3. **Network Access**: Bridge networks work, but macvlan/ipvlan are not supported

## Security Benefits

- Containers run without root privileges
- No setuid/setgid binaries in containers
- Host root cannot be compromised from containers
- Isolated user namespaces
- Read-only Docker socket prevents privilege escalation

## Testing Rootless Deployment

```bash
# Build and start
docker compose build
docker compose up -d

# Verify all services are running
docker compose ps

# Check that processes are non-root
docker compose exec orchestrator whoami  # Should be 'forgemaster'
docker compose exec postgres whoami      # Should be 'postgres'

# Verify health checks
docker compose exec orchestrator curl http://localhost:8000/health
```

## Troubleshooting

### Permission Denied on Docker Socket
```bash
# Ensure Docker socket path is correct
ls -la $DOCKER_HOST

# Check socket permissions
stat $(echo $DOCKER_HOST | sed 's/unix:\/\///')
```

### Volume Mount Issues
```bash
# Ensure directories exist with correct permissions
mkdir -p logs context
chmod 755 logs context
```

### Port Already in Use
```bash
# Check if ports are already bound
ss -tlnp | grep -E ':(8000|5432|11434)'

# Use different ports in .env file
echo "FORGEMASTER_PORT=8001" >> .env
```
