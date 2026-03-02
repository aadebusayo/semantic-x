from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
)


def _required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value or ("<" in value and ">" in value):
        raise RuntimeError(f"Missing required env: {name}")
    return value


def _optional(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def _optional_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    return int(value)


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, ResourceNotFoundError):
        return True
    if isinstance(exc, HttpResponseError):
        return getattr(exc, "status_code", None) == 404
    return False


def _reset_resources(
    *,
    index_client: SearchIndexClient,
    indexer_client: SearchIndexerClient,
    index_name: str,
    data_source_name: str,
    skillset_name: str,
    indexer_name: str,
) -> None:
    print("Reset mode enabled: deleting existing search resources...")

    for label, deleter in (
        ("indexer", lambda: indexer_client.delete_indexer(indexer_name)),
        ("skillset", lambda: indexer_client.delete_skillset(skillset_name)),
        ("data source", lambda: indexer_client.delete_data_source_connection(data_source_name)),
        ("index", lambda: index_client.delete_index(index_name)),
    ):
        try:
            deleter()
            print(f"Deleted {label}: {indexer_name if label == 'indexer' else skillset_name if label == 'skillset' else data_source_name if label == 'data source' else index_name}")
        except Exception as exc:
            if _is_not_found(exc):
                print(f"{label.capitalize()} not found, skipping")
            else:
                raise


def _ensure_index(index_client: SearchIndexClient, *, index_name: str, vector_dimensions: int, aoai_endpoint: str, aoai_api_key: str, embedding_deployment: str, embedding_model: str) -> None:
    try:
        index_client.get_index(index_name)
        print(f"Index exists: {index_name}")
        return
    except ResourceNotFoundError:
        pass

    fields = [
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            searchable=True,
            filterable=True,
            analyzer_name="keyword",
        ),
        SimpleField(name="documentId", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="documentName", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="chunkIndex", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="chunkStart", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="chunkEnd", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw",
                    vectorizer_name="default-aoai-vectorizer",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="default-aoai-vectorizer",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=aoai_endpoint,
                        deployment_name=embedding_deployment,
                        api_key=aoai_api_key,
                        model_name=embedding_model,
                    ),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="documentName"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ]
        ),
    )

    index_client.create_or_update_index(index)
    print(f"Index created: {index_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update Azure AI Search index + indexer pipeline.")
    parser.add_argument("--reset", action="store_true", help="Delete indexer/skillset/datasource/index before recreating")
    args = parser.parse_args()

    load_dotenv()

    endpoint = _required("AZURE_SEARCH_ENDPOINT")
    api_key = _required("AZURE_SEARCH_API_KEY")
    index_name = _required("AZURE_SEARCH_INDEX")

    blob_connection_string = _required("AZURE_STORAGE_CONNECTION_STRING")
    blob_container = _required("AZURE_STORAGE_CONTAINER")
    blob_prefix = (os.getenv("AZURE_STORAGE_PREFIX") or "").strip()

    aoai_endpoint = _required("AZURE_OPENAI_ENDPOINT")
    aoai_api_key = _required("AZURE_OPENAI_API_KEY")
    embedding_deployment = _required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embedding_model = _optional("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    vector_dimensions = _optional_int("SEARCH_VECTOR_DIMENSIONS", 1536)
    chunk_size = _optional_int("CHUNK_SIZE_CHARS", 1200)
    chunk_overlap = _optional_int("CHUNK_OVERLAP_CHARS", 150)

    data_source_name = f"{index_name}-blob-ds"
    skillset_name = f"{index_name}-skillset"
    indexer_name = f"{index_name}-indexer"

    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    indexer_client = SearchIndexerClient(endpoint=endpoint, credential=credential)

    if args.reset:
        _reset_resources(
            index_client=index_client,
            indexer_client=indexer_client,
            index_name=index_name,
            data_source_name=data_source_name,
            skillset_name=skillset_name,
            indexer_name=indexer_name,
        )

    _ensure_index(
        index_client,
        index_name=index_name,
        vector_dimensions=vector_dimensions,
        aoai_endpoint=aoai_endpoint,
        aoai_api_key=aoai_api_key,
        embedding_deployment=embedding_deployment,
        embedding_model=embedding_model,
    )

    data_source = SearchIndexerDataSourceConnection(
        name=data_source_name,
        type="azureblob",
        connection_string=blob_connection_string,
        container=SearchIndexerDataContainer(name=blob_container, query=blob_prefix or None),
    )
    indexer_client.create_or_update_data_source_connection(data_source)
    print(f"Data source ready: {data_source_name}")

    split_skill = SplitSkill(
        name="split-text",
        context="/document",
        text_split_mode="pages",
        maximum_page_length=chunk_size,
        page_overlap_length=chunk_overlap,
        inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
        outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
    )

    embedding_skill = AzureOpenAIEmbeddingSkill(
        name="embed-pages",
        context="/document/pages/*",
        resource_url=aoai_endpoint,
        deployment_name=embedding_deployment,
        api_key=aoai_api_key,
        model_name=embedding_model,
        dimensions=vector_dimensions,
        inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs=[OutputFieldMappingEntry(name="embedding", target_name="vector")],
    )

    projection = SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="documentId",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="content", source="/document/pages/*"),
                    InputFieldMappingEntry(name="contentVector", source="/document/pages/*/vector"),
                    InputFieldMappingEntry(name="documentName", source="/document/metadata_storage_name"),
                    InputFieldMappingEntry(name="chunkIndex", source="/document/pages/*/ordinalPosition"),
                ],
            )
        ],
        parameters=SearchIndexerIndexProjectionsParameters(projection_mode="skipIndexingParentDocuments"),
    )

    skillset = SearchIndexerSkillset(
        name=skillset_name,
        description="Chunk + embed from blob to search index.",
        skills=[split_skill, embedding_skill],
        index_projection=projection,
    )
    indexer_client.create_or_update_skillset(skillset)
    print(f"Skillset ready: {skillset_name}")

    indexer = SearchIndexer(
        name=indexer_name,
        data_source_name=data_source_name,
        skillset_name=skillset_name,
        target_index_name=index_name,
    )
    indexer_client.create_or_update_indexer(indexer)
    print(f"Indexer ready: {indexer_name}")

    run_now = (os.getenv("AZURE_SEARCH_RUN_INDEXER_NOW") or "true").strip().lower() in {"1", "true", "yes"}
    if run_now:
        indexer_client.run_indexer(indexer_name)
        print(f"Indexer run started: {indexer_name}")


if __name__ == "__main__":
    main()
