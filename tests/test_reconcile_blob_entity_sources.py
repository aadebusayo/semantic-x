from __future__ import annotations

from scripts.reconcile_blob_entity_sources import build_reconciliation_plan


def test_plan_deletes_blob_without_matching_entity():
    plan = build_reconciliation_plan(
        blob_names=["folder/orphan-file.pdf"],
        file_entities=[],
        all_entity_ids=[],
    )

    assert plan["summary"]["deleteBlobCount"] == 1
    assert plan["deleteBlobs"][0]["reason"] == "no_matching_entity"


def test_plan_deletes_entity_without_matching_blob():
    plan = build_reconciliation_plan(
        blob_names=[],
        file_entities=[
            {"id": "doc-1", "Name": "Manual", "DocExtension": ".pdf", "ObjectType": 1},
        ],
        all_entity_ids=["doc-1"],
    )

    assert plan["summary"]["deleteEntityCount"] == 1
    assert plan["deleteEntities"][0]["reason"] == "no_matching_blob"


def test_plan_keeps_blob_when_matching_file_entity_exists():
    plan = build_reconciliation_plan(
        blob_names=["folder/doc-1.pdf"],
        file_entities=[
            {"id": "doc-1", "Name": "Manual", "DocExtension": ".pdf", "ObjectType": 1},
        ],
        all_entity_ids=["doc-1"],
    )

    assert plan["summary"]["deleteBlobCount"] == 0
    assert plan["summary"]["deleteEntityCount"] == 0
    assert plan["summary"]["keptBlobCount"] == 1
    assert plan["summary"]["keptEntityCount"] == 1


def test_plan_deletes_blob_and_entity_when_entity_has_no_name():
    plan = build_reconciliation_plan(
        blob_names=["folder/doc-1.pdf"],
        file_entities=[
            {"id": "doc-1", "Name": "", "DocExtension": ".pdf", "ObjectType": 1},
        ],
        all_entity_ids=["doc-1"],
    )

    assert plan["summary"]["deleteBlobCount"] == 1
    assert plan["summary"]["deleteEntityCount"] == 1
    assert plan["deleteBlobs"][0]["reason"] == "entity_missing_name"
    assert plan["deleteEntities"][0]["reason"] == "missing_name"


def test_plan_keeps_folder_marker_blob_when_entity_id_exists():
    plan = build_reconciliation_plan(
        blob_names=["folder-entity-id"],
        file_entities=[],
        all_entity_ids=["folder-entity-id"],
        directory_like_blob_names=["folder-entity-id"],
    )

    assert plan["summary"]["deleteBlobCount"] == 0
    assert plan["summary"]["keptBlobCount"] == 1


def test_plan_deletes_directory_like_blob_without_entity():
    plan = build_reconciliation_plan(
        blob_names=["orphan-directory-id"],
        file_entities=[],
        all_entity_ids=[],
        directory_like_blob_names=["orphan-directory-id"],
    )

    assert plan["summary"]["deleteBlobCount"] == 1
    assert plan["summary"]["deleteDirectoryLikeBlobCount"] == 1
    assert plan["deleteBlobs"][0]["reason"] == "directory_like_without_entity"
    assert plan["deleteBlobs"][0]["isDirectory"] is True