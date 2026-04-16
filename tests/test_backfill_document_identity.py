from __future__ import annotations

import asyncio

from scripts.backfill_document_identity import needs_identity_update, resolve_entity_metadata


class _EntityRepo:
    async def get_document(self, *, entity_id: str):
        if entity_id == "entity-123":
            return {"id": "entity-123", "Name": "Employee Handbook", "DocExtension": ".pdf"}
        return None

    async def find_document_by_filename(self, *, filename: str):
        if filename in {"Employee Handbook.pdf", "Employee Handbook"}:
            return {"id": "entity-123", "Name": "Employee Handbook", "DocExtension": ".pdf"}
        return None


def test_resolve_entity_metadata_from_blob_basename():
    async def _run():
        repo = _EntityRepo()
        resolved = await resolve_entity_metadata(
            repo,
            raw_document_id=None,
            raw_document_name="entity-123.pdf",
            blob_name="0000-0000-0000-0000/entity-123.pdf",
        )

        assert resolved is not None
        assert resolved.id == "entity-123"
        assert resolved.name == "Employee Handbook.pdf"

    asyncio.run(_run())


def test_resolve_entity_metadata_from_filename():
    async def _run():
        repo = _EntityRepo()
        resolved = await resolve_entity_metadata(
            repo,
            raw_document_id=None,
            raw_document_name="Employee Handbook.pdf",
            blob_name=None,
        )

        assert resolved is not None
        assert resolved.id == "entity-123"
        assert resolved.name == "Employee Handbook.pdf"

    asyncio.run(_run())


def test_needs_identity_update_detects_canonical_change():
    changed = needs_identity_update(
        raw_document_id="hashed-id",
        raw_document_name="hashed-id.pdf",
        canonical=type("Canonical", (), {"id": "entity-123", "name": "Employee Handbook.pdf"})(),
    )

    unchanged = needs_identity_update(
        raw_document_id="entity-123",
        raw_document_name="Employee Handbook.pdf",
        canonical=type("Canonical", (), {"id": "entity-123", "name": "Employee Handbook.pdf"})(),
    )

    assert changed is True
    assert unchanged is False