# Data Model: Retrieval & Test Embeddings

## Entities

### RetrievedChunk
- **id**: string - Unique identifier for the retrieved chunk
- **text**: string - The actual text content retrieved from the book
- **source_location**: string - Reference to the original location in the book (e.g., chapter, section)
- **similarity_score**: float - Cosine similarity score between query and chunk
- **metadata**: object - Additional metadata about the retrieval context

### Query
- **id**: string - Unique identifier for the query
- **text**: string - The input query text
- **top_k**: integer - Number of results requested (default: 3)
- **similarity_threshold**: float - Minimum similarity threshold for results
- **timestamp**: datetime - When the query was executed

### ValidationResult
- **query_id**: string - Reference to the associated query
- **chunk_id**: string - Reference to the retrieved chunk
- **accuracy_score**: float - Measure of how well the chunk matches expected content
- **original_text**: string - The original book text for comparison
- **validation_passed**: boolean - Whether the validation criteria were met
- **validation_details**: string - Explanation of validation results

### ValidationLog
- **id**: string - Unique identifier for the validation log entry
- **query**: Query - The query that was executed
- **results**: array[RetrievedChunk] - The chunks retrieved by the query
- **validations**: array[ValidationResult] - Validation results for each chunk
- **overall_accuracy**: float - Overall accuracy percentage for the query
- **timestamp**: datetime - When the validation was performed
- **book_section**: string - Which book section was tested

## Relationships

- Query → 1..* RetrievedChunk (one query returns multiple chunks)
- Query → 1 ValidationResult (one query has validation results)
- RetrievedChunk → 1 ValidationResult (each chunk is validated)
- ValidationLog → 1 Query (one log entry per query execution)
- ValidationLog → * RetrievedChunk (log contains all retrieved chunks)
- ValidationLog → * ValidationResult (log contains all validation results)

## Validation Rules

### RetrievedChunk
- text must not be empty
- similarity_score must be between 0 and 1
- source_location must follow format "chapter.section" or similar hierarchical reference

### Query
- text must not be empty
- top_k must be positive integer
- similarity_threshold must be between 0 and 1

### ValidationResult
- accuracy_score must be between 0 and 1
- validation_passed must be true if accuracy_score >= 0.95 (as per success criteria)

### ValidationLog
- overall_accuracy must be between 0 and 1
- must contain at least one result and validation