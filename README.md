# Infosearch API

Lean REST backend for your Infosearch project:

- Queries Azure AI Search (semantic + vector) for relevant document chunks
- Returns chat answers from Azure OpenAI grounded on retrieved citations metadata (doc name/id, rank, excerpt)
- Stores chat history in Azure Cosmos DB (title + turns)
- Generates and stores “quick action questions” per uploaded document (Cosmos DB)
- Optional pull-based ingestion worker: polls Azure Blob Storage and upserts chunk docs into Azure AI Search

## Quick start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Configure env

- Copy `env.example` to `.env`
- Fill in Azure AI Search + Azure OpenAI + Cosmos settings
- Optional: enable Blob ingestion (see below)

3) Run

```bash
python main.py
```

## Create Azure AI Search index

Use the Azure AI Search **index client** to create/update the index schema this app expects:

```bash
python scripts/create_search_index.py
```

This uses `create_or_update_index` and your `.env` values (`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX`, and field-name settings).

For query-time `VectorizableTextQuery` support, set these as well before running the script:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- Optional: `AZURE_OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)

## Create Azure AI Search indexer pipeline (chunk + embed)

To use Azure AI Search indexers (instead of only push-ingestion), run:

```bash
python scripts/create_search_indexer.py
```

This provisions/updates:

- Blob data source
- Skillset (`SplitSkill` + `AzureOpenAIEmbeddingSkill`)
- Indexer

The script also ensures your index has vector profile + Azure OpenAI vectorizer wiring.

Required env for this script:

- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_CONTAINER`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

Optional env:

- `AZURE_STORAGE_PREFIX`
- `AZURE_SEARCH_DATASOURCE_NAME`
- `AZURE_SEARCH_SKILLSET_NAME`
- `AZURE_SEARCH_INDEXER_NAME`
- `AZURE_SEARCH_RUN_INDEXER_NOW=true` (to trigger an immediate run)

## Blob ingestion worker (optional)

This repo includes a simple pull-based ingestion worker (no queues/messaging):

- Polls a Blob container periodically
- Detects new/changed blobs via ETag
- Extracts text (supports `.txt`, `.md`, `.csv`, `.pdf`)
- Chunks text and upserts chunk documents into your Azure AI Search index
- Generates + stores quick questions in Cosmos DB

To enable it, set these in `.env`:

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_CONTAINER`
- Optional: `AZURE_STORAGE_PREFIX`
- Optional: `ENABLE_INGESTION_WORKER=true` (default)

`AZURE_STORAGE_PREFIX` limits blob listing to names that start with that prefix (for example `folderA/`), so only that logical path is scanned.

### Cosmos ingestion container

Create a Cosmos container named by `COSMOS_INGESTION_CONTAINER` with partition key `/document_id`.

### Cosmos partition keys used by this app

- Chat container (`COSMOS_CHAT_CONTAINER`): `/id`
- Suggestions container (`COSMOS_SUGGESTIONS_CONTAINER`): `/document_id`
- Ingestion container (`COSMOS_INGESTION_CONTAINER`): `/document_id`

### Azure AI Search index expectations

Indexing upserts chunk documents containing:

- Key field (default `id`)
- `documentId`, `documentName`
- `content`
- `chunkIndex`, `chunkStart`, `chunkEnd`

If your index uses different field names, configure them via the `SEARCH_*_FIELD` and `SEARCH_CHUNK_*_FIELD` settings in `.env`.

## One-time backfill indexing script

If you want a one-time index population (not continuously wired into the app), run:

```bash
python scripts/backfill_search_from_entity.py
```

The script:

- Reads upload metadata from Cosmos container `COSMOS_ENTITY_CONTAINER` (default `Entity`)
- Uses `BasePathId` rules to resolve blob path from IDs:
    - `BasePathId == ENTITY_ROOT_BASEPATH_ID` (default `0000-0000-0000-0000`) means blob is at root: `<id>`
    - Otherwise it builds nested path by following parent `BasePathId` chain: `<parentId>/<childId>/.../<id>`
- Downloads each resolved blob from `AZURE_STORAGE_CONTAINER`
- Extracts text, chunks it, and upserts chunk documents into Azure AI Search

Set these env vars before running:

- `COSMOS_ENTITY_CONTAINER` (default `Entity`)
- `ENTITY_ROOT_BASEPATH_ID` (default `0000-0000-0000-0000`)
- `ENTITY_FILE_OBJECT_TYPE` (default `0`, to index only object type 0)
- `BACKFILL_LOG_LEVEL` (default `INFO`, set `DEBUG` for per-document detail)
- `BACKFILL_LOG_EVERY` (default `50`, periodic progress interval)

## API

- `POST /api/v1/chat/messages`
    - Body: `{ message, chat_id?, user_id?, document?: {id?, name?}, top_k? }`
    - Returns: `{ chat_id, title, answer, citations[] }`

- `GET /api/v1/chats?user_id=&limit=`
    - Returns: list of chat history items

- `GET /api/v1/chats/{chat_id}?user_id=`
    - Returns: chat transcript (turns)

- `POST /api/v1/documents/suggestions`
    - Body: `{ document: {id?, name?}, text? }`
    - Returns: `{ questions[] }`

- `GET /api/v1/suggestions/recent?limit=`
    - Returns: recent quick questions by recency

## Local frontend playground

After starting the API, open:

- `http://127.0.0.1:8000/playground`

The playground lets you test locally:

- Send chat messages (`POST /api/v1/chat/messages`)
- List chats (`GET /api/v1/chats`)
- Load a chat transcript (`GET /api/v1/chats/{chat_id}`)
- Create quick questions (`POST /api/v1/documents/suggestions`)
- View recent quick questions (`GET /api/v1/suggestions/recent`)
