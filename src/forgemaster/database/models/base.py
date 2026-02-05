"""SQLAlchemy declarative base and common column mixins for Forgemaster.

This module defines the DeclarativeBase class and a TimestampMixin that
provides id, created_at, and updated_at columns shared across all models.

All models in the Forgemaster database should inherit from Base and
include TimestampMixin for consistent primary key and timestamp handling.

Example:
    >>> class MyModel(TimestampMixin, Base):
    ...     __tablename__ = "my_table"
    ...     name: Mapped[str] = mapped_column(Text, nullable=False)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base for all Forgemaster models."""

    pass


class TimestampMixin:
    """Mixin providing id (UUID), created_at, and updated_at columns.

    This mixin should be listed before Base in the class hierarchy
    to ensure the columns are included in the model's table definition.

    Attributes:
        id: UUID primary key with server-side default via gen_random_uuid().
        created_at: Timestamp set by the database on row creation.
        updated_at: Timestamp set by the database on row creation and
                    updated on each modification.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
