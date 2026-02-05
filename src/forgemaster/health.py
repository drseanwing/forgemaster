"""Health check and systemd watchdog notification support.

This module provides:
- A standalone HTTP health check endpoint (no FastAPI dependency)
- systemd watchdog notification support via sd_notify protocol
- Service readiness signaling

The health endpoint returns JSON status information and is used by:
- systemd watchdog monitoring (Type=notify)
- Docker HEALTHCHECK commands
- External monitoring systems
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


class HealthCheckServer:
    """Standalone HTTP health check server using asyncio.

    This server provides a simple /health endpoint that returns JSON status
    without requiring FastAPI or other frameworks. It's designed to be
    lightweight and suitable for use in systemd Type=notify services.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """Initialize health check server.

        Args:
            host: Host to bind to (default: 127.0.0.1)
            port: Port to bind to (default: 8000)
        """
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.is_healthy = True
        self._shutdown_event = asyncio.Event()

    async def handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming HTTP request.

        Args:
            reader: StreamReader for incoming data
            writer: StreamWriter for outgoing data
        """
        try:
            # Read the request line
            line = await reader.readline()
            request_line = line.decode().strip()

            # Parse the request
            parts = request_line.split()
            if len(parts) < 2:
                self._send_error(writer, 400, "Bad Request")
                return

            method, path = parts[0], parts[1]

            # Skip headers
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break

            # Route to handler
            if method == "GET" and path == "/health":
                await self._handle_health(writer)
            elif method == "GET" and path == "/ready":
                await self._handle_ready(writer)
            else:
                self._send_error(writer, 404, "Not Found")

        except Exception as e:
            log.error(f"Error handling request: {e}")
            self._send_error(writer, 500, "Internal Server Error")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """Handle /health endpoint.

        Returns JSON with status and timestamp.

        Args:
            writer: StreamWriter for response
        """
        status = "healthy" if self.is_healthy else "unhealthy"
        response_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response_json = json.dumps(response_data)
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(response_json)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{response_json}"
        )

        writer.write(response.encode())
        await writer.drain()

    async def _handle_ready(self, writer: asyncio.StreamWriter) -> None:
        """Handle /ready endpoint (alias for /health).

        Args:
            writer: StreamWriter for response
        """
        await self._handle_health(writer)

    def _send_error(self, writer: asyncio.StreamWriter, code: int, message: str) -> None:
        """Send HTTP error response.

        Args:
            writer: StreamWriter for response
            code: HTTP status code
            message: Status message
        """
        response = (
            f"HTTP/1.1 {code} {message}\r\n"
            f"Content-Type: text/plain\r\n"
            f"Content-Length: {len(message)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{message}"
        )
        writer.write(response.encode())
        asyncio.create_task(self._drain_and_close(writer))

    async def _drain_and_close(self, writer: asyncio.StreamWriter) -> None:
        """Drain writer and close connection.

        Args:
            writer: StreamWriter to close
        """
        try:
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def start(self) -> None:
        """Start the health check server.

        Runs indefinitely until stop() is called or the server encounters
        a fatal error.
        """
        self.server = await asyncio.start_server(
            self.handle_request, self.host, self.port
        )

        addr = self.server.sockets[0].getsockname()
        log.info(f"Health check server started on {addr[0]}:{addr[1]}")

        async with self.server:
            # Wait for shutdown signal
            await self._shutdown_event.wait()

    def stop(self) -> None:
        """Stop the health check server."""
        self._shutdown_event.set()

    def set_health_status(self, healthy: bool) -> None:
        """Update health status.

        Args:
            healthy: True if service is healthy, False otherwise
        """
        self.is_healthy = healthy


class SystemdNotifier:
    """Systemd watchdog and service notification support.

    Provides communication with systemd via the sd_notify protocol.
    Used with Type=notify services for health monitoring and lifecycle signaling.

    Only works if NOTIFY_SOCKET environment variable is set (set by systemd).
    """

    def __init__(self):
        """Initialize systemd notifier."""
        self.notify_socket = os.environ.get("NOTIFY_SOCKET")
        self.is_enabled = self.notify_socket is not None
        self._watchdog_usec = os.environ.get("WATCHDOG_USEC")

    async def notify_ready(self) -> bool:
        """Signal to systemd that service is ready.

        Used with Type=notify services. Tells systemd that the service
        has completed its startup sequence and is ready for work.

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            return await self._send_notify("READY=1")
        except Exception as e:
            log.warning(f"Failed to send READY notification: {e}")
            return False

    async def notify_watchdog(self) -> bool:
        """Send watchdog keep-alive notification to systemd.

        Should be called periodically (at least twice per WatchdogSec interval)
        to prevent systemd from restarting the service.

        Used with WatchdogSec in systemd service file. If notifications
        don't arrive frequently enough, systemd will restart the service.

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            return await self._send_notify("WATCHDOG=1")
        except Exception as e:
            log.warning(f"Failed to send WATCHDOG notification: {e}")
            return False

    async def notify_stopping(self) -> bool:
        """Signal to systemd that service is stopping.

        Args:
            Returns:
                True if notification was sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            return await self._send_notify("STOPPING=1")
        except Exception as e:
            log.warning(f"Failed to send STOPPING notification: {e}")
            return False

    async def notify_status(self, message: str) -> bool:
        """Send status message to systemd.

        Args:
            message: Status message to send

        Returns:
            True if notification was sent successfully, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            return await self._send_notify(f"STATUS={message}")
        except Exception as e:
            log.warning(f"Failed to send STATUS notification: {e}")
            return False

    async def _send_notify(self, message: str) -> bool:
        """Send notification message via sd_notify protocol.

        Args:
            message: Notification message to send

        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.notify_socket or not self.is_enabled:
            return False

        try:
            # Handle abstract socket (starts with @)
            if self.notify_socket.startswith("@"):
                # Replace @ with null byte for abstract socket
                socket_path = "\0" + self.notify_socket[1:]
            else:
                socket_path = self.notify_socket

            # Create and send via Unix domain socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.sendto(message.encode(), socket_path)
            sock.close()

            log.debug(f"Sent systemd notification: {message}")
            return True

        except Exception as e:
            log.debug(f"Error sending sd_notify message: {e}")
            return False

    def get_watchdog_interval(self) -> Optional[float]:
        """Get recommended watchdog interval in seconds.

        Returns:
            Interval in seconds, or None if not available
        """
        if not self._watchdog_usec:
            return None

        try:
            # WATCHDOG_USEC is in microseconds
            return int(self._watchdog_usec) / 1_000_000
        except (ValueError, TypeError):
            return None


async def start_health_server(host: str = "127.0.0.1", port: int = 8000) -> HealthCheckServer:
    """Start the health check server in background.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8000)

    Returns:
        HealthCheckServer instance
    """
    server = HealthCheckServer(host, port)
    asyncio.create_task(server.start())
    await asyncio.sleep(0.1)  # Allow server to start
    return server


async def notify_systemd_ready() -> bool:
    """Convenience function to notify systemd that service is ready.

    Returns:
        True if notification was sent, False otherwise
    """
    notifier = SystemdNotifier()
    return await notifier.notify_ready()


async def start_watchdog_loop(notifier: SystemdNotifier, interval_override: Optional[float] = None) -> None:
    """Start background watchdog notification loop.

    Sends periodic WATCHDOG=1 notifications to systemd. Should be called
    once at service startup.

    Args:
        notifier: SystemdNotifier instance
        interval_override: Override calculated interval (in seconds)
    """
    if not notifier.is_enabled:
        log.debug("Watchdog not enabled (NOTIFY_SOCKET not set)")
        return

    # Get watchdog interval from systemd
    interval = interval_override or notifier.get_watchdog_interval()

    if interval is None:
        log.warning("WATCHDOG_USEC not set, watchdog loop disabled")
        return

    # Send notifications at half the watchdog interval
    notify_interval = interval / 2

    log.info(f"Starting watchdog loop with {notify_interval:.1f}s interval")

    try:
        while True:
            await asyncio.sleep(notify_interval)
            success = await notifier.notify_watchdog()
            if not success:
                log.warning("Failed to send watchdog notification")
    except asyncio.CancelledError:
        log.debug("Watchdog loop cancelled")
        raise
