from __future__ import annotations

import os

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
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


def main() -> None:
    load_dotenv()

    endpoint = _required("AZURE_SEARCH_ENDPOINT")
    api_key = _required("AZURE_SEARCH_API_KEY")
    index_name = _required("AZURE_SEARCH_INDEX")

    content_field = _optional("SEARCH_CONTENT_FIELD", "content")
    vector_field = _optional("SEARCH_VECTOR_FIELD", "contentVector")
    doc_id_field = _optional("SEARCH_DOC_ID_FIELD", "documentId")
    doc_name_field = _optional("SEARCH_DOC_NAME_FIELD", "documentName")
    chunk_key_field = _optional("SEARCH_CHUNK_KEY_FIELD", "id")
    chunk_index_field = _optional("SEARCH_CHUNK_INDEX_FIELD", "chunkIndex")
    chunk_start_field = _optional("SEARCH_CHUNK_START_FIELD", "chunkStart")
    chunk_end_field = _optional("SEARCH_CHUNK_END_FIELD", "chunkEnd")
    semantic_name = _optional("AZURE_SEARCH_SEMANTIC_CONFIG", "default")

    vector_dimensions = int((os.getenv("SEARCH_VECTOR_DIMENSIONS") or "1536").strip())
    vectorizer_name = _optional("SEARCH_VECTOR_PROFILE_VECTORIZER", "default-aoai-vectorizer")
    aoai_resource_url = _required("AZURE_OPENAI_ENDPOINT")
    aoai_api_key = _required("AZURE_OPENAI_API_KEY")
    aoai_embedding_deployment = _required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    aoai_embedding_model = _optional("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    fields = [
        SimpleField(name=chunk_key_field, type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name=doc_id_field, type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name=doc_name_field, type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name=content_field, type=SearchFieldDataType.String),
        SimpleField(name=chunk_index_field, type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name=chunk_start_field, type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name=chunk_end_field, type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name=vector_field,
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
                    vectorizer_name=vectorizer_name,
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name=vectorizer_name,
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=aoai_resource_url,
                        deployment_name=aoai_embedding_deployment,
                        api_key=aoai_api_key,
                        model_name=aoai_embedding_model,
                    ),
                )
            ],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=semantic_name,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name=doc_name_field),
                        content_fields=[SemanticField(field_name=content_field)],
                    ),
                )
            ]
        ),
    )

    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
    created = client.create_or_update_index(index)
    print(f"Index ready: {created.name}")


if __name__ == "__main__":
    main()
