# Feature Specification: Embeddings & Qdrant Integration

**Feature Branch**: `001-embeddings-qdrant`
**Created**: 2025-12-10
**Status**: Draft
**Input**: User description: "# SPEC-1: Generate Embeddings & Store in Qdrant

## Goal
Extract content from the deployed Docusaurus book, generate embeddings using Cohere, and store them in Qdrant Cloud for RAG chatbot use.

## Steps

1. **Extract Book Content**
   - Scrape deployed pages or use original markdown files.
   - Clean text (remove HTML, navigation, code blocks).
   - Save each page as JSON: { \"page_url\": \"/docs/intro\", \"title\": \"Introduction\", \"content\": \"Cleaned text...\" }

2. **Chunk Text**
   - Split text into 300–1000 token chunks.
   - Add metadata: page_url, heading, chunk_index, text.
   - Example: { \"page_url\": \"/docs/intro\", \"heading\": \"Intro\", \"chunk_index\": 0, \"text\": \"chunk content...\" }

3. **Generate Embeddings**
   - Use Cohere model: \`embed-multilingual-v3.0\` or \`embed-english-light-v3.0\`.
   - Generate embeddings for all chunks.
   - Verify vector dimensions match model.

4. **Store in Qdrant**
   - Create collection with cosine distance and correct vector size.
   - Upload embeddings with metadata.

5. **Validate**
   - Run test similarity queries (e.g., \"What is a humanoid robot?\").
   - Ensure returned text matches book content.
   - Confirm vector count = total chunks.

## Deliverables
- Cleaned text JSON
- Chunked JSON
- Embedding + ingestion script
- Populated Qdrant collection"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - RAG Chatbot Content Retrieval (Priority: P1)

As a user of the humanoid robotics chatbot, I want to ask questions about the book content and receive accurate, contextually relevant answers based on the book's information.

**Why this priority**: This is the core value proposition of the feature - enabling intelligent search and retrieval of information from the humanoid robotics book content.

**Independent Test**: The system can be tested by querying the chatbot with specific questions about the book content and verifying that the returned information is accurate and relevant to the original book.

**Acceptance Scenarios**:

1. **Given** the book content has been processed and stored in Qdrant, **When** a user asks "What is a humanoid robot?", **Then** the system returns relevant text chunks from the book that explain what a humanoid robot is.

2. **Given** multiple text chunks exist in the Qdrant collection, **When** a user asks a technical question about ROS 2, **Then** the system retrieves the most relevant chunks related to ROS 2 from the book content.

---

### User Story 2 - Content Extraction and Processing (Priority: P2)

As a system administrator, I want to extract content from the deployed Docusaurus book and convert it into searchable embeddings so that users can query the content effectively.

**Why this priority**: This is the foundational step that enables the RAG functionality. Without proper content extraction and embedding, the chatbot cannot function.

**Independent Test**: The system can be tested by verifying that all book pages have been extracted, cleaned, and converted to embeddings that are stored in Qdrant with appropriate metadata.

**Acceptance Scenarios**:

1. **Given** a deployed Docusaurus book site, **When** the extraction process runs, **Then** all content pages are retrieved and cleaned of HTML/navigation elements.

2. **Given** raw book content, **When** the chunking process runs, **Then** the content is split into 300-1000 token chunks with proper metadata (page_url, heading, chunk_index).

---

### User Story 3 - Embedding Validation and Quality Assurance (Priority: P3)

As a quality assurance engineer, I want to validate that the embeddings accurately represent the original content and that similarity searches return relevant results.

**Why this priority**: This ensures the reliability and accuracy of the RAG system, preventing users from receiving incorrect or irrelevant information.

**Independent Test**: The system can be tested by running similarity queries with known questions and verifying that the returned text matches the original book content.

**Acceptance Scenarios**:

1. **Given** embeddings stored in Qdrant, **When** a validation query is run, **Then** the system confirms that vector count matches total chunks and dimensions match the Cohere model.

2. **Given** a test question about humanoid robotics, **When** similarity search is performed, **Then** the returned text chunks contain relevant information from the book.

---

### Edge Cases

- What happens when the Docusaurus site is temporarily unavailable during content extraction?
- How does the system handle pages with heavy mathematical formulas or code blocks during text cleaning?
- What occurs when the Qdrant Cloud service is temporarily unavailable during embedding storage?
- How does the system handle very long pages that result in many chunks?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract content from the deployed Docusaurus book site or original markdown files
- **FR-002**: System MUST clean extracted text by removing HTML, navigation, and code blocks
- **FR-003**: System MUST save each page as JSON with page_url, title, and cleaned content
- **FR-004**: System MUST chunk text into 300-1000 token segments with metadata
- **FR-005**: System MUST generate embeddings using Cohere's embed-multilingual-v3.0 or embed-english-light-v3.0 model
- **FR-006**: System MUST store embeddings in Qdrant Cloud with cosine distance and proper vector size
- **FR-007**: System MUST associate metadata (page_url, heading, chunk_index) with each embedding
- **FR-008**: System MUST validate that vector dimensions match the Cohere model specifications
- **FR-009**: System MUST allow similarity queries to retrieve relevant text chunks from the book
- **FR-010**: System MUST verify that the total vector count matches the total number of chunks

### Key Entities *(include if feature involves data)*

- **Text Chunk**: A segment of book content (300-1000 tokens) with associated metadata including page_url, heading, and chunk_index
- **Embedding Vector**: A numerical representation of text content generated by the Cohere model, stored in Qdrant with associated metadata
- **Qdrant Collection**: A container for embeddings with cosine distance metric, designed for similarity search operations
- **Book Content**: The original text from the humanoid robotics book, extracted from Docusaurus pages and processed for RAG use

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All book content pages MUST be successfully extracted from the Docusaurus site or markdown files.
- **SC-002**: Text cleaning process MUST remove HTML tags, navigation elements, and code blocks while preserving meaningful content.
- **SC-003**: Content chunking MUST produce segments within the 300-1000 token range with appropriate metadata.
- **SC-004**: Embedding generation MUST produce vectors with dimensions matching the Cohere model specifications.
- **SC-005**: All embeddings MUST be successfully stored in Qdrant Cloud with associated metadata.
- **SC-006**: Similarity queries MUST return relevant text chunks from the book content based on user questions.
- **SC-007**: Vector count in Qdrant MUST match the total number of text chunks generated from the book.
- **SC-008**: The system MUST successfully process and store at least 95% of the book content without errors.
