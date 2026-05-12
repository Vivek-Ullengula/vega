# coaction_agent_platform/adapters/aws/boto3_factory.py
"""Centralized Boto3 client factory.

All AWS adapters use this factory. Prevents each adapter from independently
creating clients and makes retry, timeout, and region behavior consistent.
"""

import boto3
from botocore.config import Config


class Boto3SessionFactory:
    """Creates boto3 clients with consistent retry, timeout, and region config."""

    def __init__(self, region_name: str) -> None:
        self.region_name = region_name
        self.config = Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=3,
            read_timeout=30,
        )
        self._clients: dict[str, object] = {}

    def client(self, service_name: str):
        """Get or create a cached boto3 client for the given service."""
        if service_name not in self._clients:
            self._clients[service_name] = boto3.client(
                service_name,
                region_name=self.region_name,
                config=self.config,
            )
        return self._clients[service_name]

    def resource(self, service_name: str):
        """Get or create a cached boto3 resource for the given service."""
        key = f"{service_name}_resource"
        if key not in self._clients:
            self._clients[key] = boto3.resource(
                service_name,
                region_name=self.region_name,
            )
        return self._clients[key]
