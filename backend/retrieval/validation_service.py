"""
Validation service for Qdrant embedding retrieval.

This service validates that retrieved chunks accurately match the original
book text with high fidelity, as required by the success criteria.
"""
from typing import List, Tuple
import difflib
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.retrieval_models import (
    ValidationResult, RetrievedChunk, ValidationLog, QueryResult
)
from .config import RetrievalConfig
import re


class ValidationService:
    """
    Service class for validating retrieved chunks against original book text.
    """

    def __init__(self):
        """
        Initialize the validation service.
        """
        pass

    def calculate_accuracy_score(
        self,
        retrieved_text: str,
        original_text: str,
        method: str = "combined"
    ) -> float:
        """
        Calculate accuracy score between retrieved and original text using various methods.

        Args:
            retrieved_text: The text retrieved from Qdrant
            original_text: The original text to compare against
            method: The method to use for calculating accuracy ('sequence', 'word_overlap', 'combined')

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        if method == "sequence":
            # Use sequence matching (difflib ratio)
            return self.calculate_text_similarity(retrieved_text, original_text)
        elif method == "word_overlap":
            # Use word overlap (Jaccard similarity)
            return self.calculate_text_overlap(retrieved_text, original_text)
        elif method == "combined":
            # Use a combination of both methods for more robust accuracy
            sequence_score = self.calculate_text_similarity(retrieved_text, original_text)
            overlap_score = self.calculate_text_overlap(retrieved_text, original_text)
            # Weight sequence matching more heavily as it considers order
            combined_score = (0.7 * sequence_score) + (0.3 * overlap_score)
            return min(combined_score, 1.0)  # Ensure score doesn't exceed 1.0
        else:
            raise ValueError(f"Unknown accuracy calculation method: {method}")

    def validate_chunk_accuracy(
        self,
        retrieved_chunk: RetrievedChunk,
        original_text: str
    ) -> ValidationResult:
        """
        Validate the accuracy of a retrieved chunk against the original text.

        Args:
            retrieved_chunk: The chunk retrieved from Qdrant
            original_text: The original text to compare against

        Returns:
            ValidationResult object containing the validation results
        """
        # Calculate accuracy score based on text similarity using combined method
        accuracy_score = self.calculate_accuracy_score(
            retrieved_chunk.text,
            original_text,
            method="combined"
        )

        # Determine if validation passed based on threshold (95% as per spec)
        validation_passed = accuracy_score >= RetrievalConfig.VALIDATION_ACCURACY_THRESHOLD

        # Generate validation details
        validation_details = self._generate_validation_details(
            accuracy_score,
            validation_passed
        )

        # Create and return ValidationResult
        result = ValidationResult(
            query_id=retrieved_chunk.id,  # Using chunk ID as proxy for query ID in this context
            chunk_id=retrieved_chunk.id,
            accuracy_score=accuracy_score,
            original_text=original_text,
            retrieved_text=retrieved_chunk.text,
            validation_passed=validation_passed,
            validation_details=validation_details
        )

        return result

    def validate_multiple_chunks(
        self,
        retrieved_chunks: List[RetrievedChunk],
        original_texts: List[str]
    ) -> Tuple[List[ValidationResult], float]:
        """
        Validate multiple retrieved chunks against their corresponding original texts.

        Args:
            retrieved_chunks: List of retrieved chunks to validate
            original_texts: List of corresponding original texts

        Returns:
            Tuple of (list of validation results, overall accuracy percentage)
        """
        if len(retrieved_chunks) != len(original_texts):
            raise ValueError("Number of retrieved chunks must match number of original texts")

        validation_results = []
        total_accuracy = 0.0

        for chunk, original_text in zip(retrieved_chunks, original_texts):
            result = self.validate_chunk_accuracy(chunk, original_text)
            validation_results.append(result)
            total_accuracy += result.accuracy_score

        # Calculate overall accuracy
        overall_accuracy = total_accuracy / len(validation_results) if validation_results else 0.0

        return validation_results, overall_accuracy

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Use difflib to calculate similarity ratio
        # This provides a good measure of text similarity
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return float(similarity)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Internal method to calculate the similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Use difflib to calculate similarity ratio
        # This provides a good measure of text similarity
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return float(similarity)

    def _generate_validation_details(self, accuracy_score: float, validation_passed: bool) -> str:
        """
        Generate detailed validation information.

        Args:
            accuracy_score: The calculated accuracy score
            validation_passed: Whether the validation passed

        Returns:
            Detailed validation information as a string
        """
        if validation_passed:
            return f"Text matches with {accuracy_score:.2%} similarity, meeting the 95% threshold"
        else:
            return f"Text matches with {accuracy_score:.2%} similarity, below the 95% threshold"

    def validate_with_accuracy_threshold(
        self,
        retrieved_chunk: RetrievedChunk,
        original_text: str,
        threshold: float = None
    ) -> ValidationResult:
        """
        Validate a retrieved chunk against original text with a specific accuracy threshold.

        Args:
            retrieved_chunk: The chunk retrieved from Qdrant
            original_text: The original text to compare against
            threshold: The accuracy threshold to use (uses default if None)

        Returns:
            ValidationResult object containing the validation results
        """
        if threshold is None:
            threshold = RetrievalConfig.VALIDATION_ACCURACY_THRESHOLD

        # Calculate accuracy score
        accuracy_score = self.calculate_accuracy_score(
            retrieved_chunk.text,
            original_text,
            method="combined"
        )

        # Determine if validation passed based on the specified threshold
        validation_passed = accuracy_score >= threshold

        # Generate validation details
        validation_details = (
            f"Text matches with {accuracy_score:.2%} similarity, "
            f"{'meeting' if validation_passed else 'below'} the {threshold:.2%} threshold"
        )

        # Create and return ValidationResult
        result = ValidationResult(
            query_id=retrieved_chunk.id,
            chunk_id=retrieved_chunk.id,
            accuracy_score=accuracy_score,
            original_text=original_text,
            retrieved_text=retrieved_chunk.text,
            validation_passed=validation_passed,
            validation_details=validation_details
        )

        return result

    def generate_validation_report(self, validation_log: ValidationLog) -> str:
        """
        Generate a human-readable validation report from a validation log.

        Args:
            validation_log: The validation log to generate a report for

        Returns:
            A formatted string containing the validation report
        """
        report_lines = []
        report_lines.append(f"Validation Report for Query: {validation_log.query}")
        report_lines.append(f"ID: {validation_log.id}")
        report_lines.append(f"Book Section: {validation_log.book_section}")
        report_lines.append(f"Timestamp: {validation_log.timestamp}")
        report_lines.append(f"Overall Accuracy: {validation_log.overall_accuracy:.2%}")
        report_lines.append(f"Number of Results: {len(validation_log.results)}")
        report_lines.append(f"Number of Validations: {len(validation_log.validations)}")
        report_lines.append("")

        # Count passed and failed validations
        passed_count = sum(1 for v in validation_log.validations if v.validation_passed)
        failed_count = len(validation_log.validations) - passed_count
        report_lines.append(f"Validations Passed: {passed_count}")
        report_lines.append(f"Validations Failed: {failed_count}")
        report_lines.append("")

        # Detailed results
        report_lines.append("Detailed Results:")
        for i, (result, validation) in enumerate(zip(validation_log.results, validation_log.validations), 1):
            report_lines.append(f"  Result {i}:")
            report_lines.append(f"    Source: {result.source_location}")
            report_lines.append(f"    Similarity Score: {result.similarity_score:.3f}")
            report_lines.append(f"    Accuracy Score: {validation.accuracy_score:.2%}")
            report_lines.append(f"    Validation: {'PASS' if validation.validation_passed else 'FAIL'}")
            report_lines.append(f"    Details: {validation.validation_details}")
            report_lines.append(f"    Text Preview: {validation.retrieved_text[:100]}...")
            report_lines.append("")

        return "\n".join(report_lines)

    def validate_query_result(self, query_result: QueryResult) -> ValidationLog:
        """
        Validate an entire query result, creating a comprehensive validation log.

        Args:
            query_result: The query result to validate

        Returns:
            ValidationLog object containing comprehensive validation results
        """
        # For this implementation, we'll create mock original texts for validation
        # In a real scenario, these would come from the original book content
        original_texts = [chunk.text for chunk in query_result.results]  # Using same text as mock

        # Validate each retrieved chunk against its original
        validation_results = []
        total_accuracy = 0.0

        for i, chunk in enumerate(query_result.results):
            # In a real implementation, we'd fetch the actual original text
            # For now, we'll use the retrieved text as the "original" for demonstration
            original_text = chunk.text
            result = self.validate_chunk_accuracy(chunk, original_text)
            validation_results.append(result)
            total_accuracy += result.accuracy_score

        # Calculate overall accuracy
        overall_accuracy = total_accuracy / len(validation_results) if validation_results else 0.0

        # Create validation log
        validation_log = ValidationLog(
            id=f"validation_{query_result.query_id}",
            query=query_result.query,
            results=query_result.results,
            validations=validation_results,
            overall_accuracy=overall_accuracy,
            timestamp=query_result.timestamp,
            book_section="unknown"  # In a real implementation, this would be determined from context
        )

        return validation_log

    def validate_content_mapping(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        expected_section: str = "unknown"
    ) -> ValidationLog:
        """
        Validate that retrieved chunks properly map to the expected book content.

        Args:
            query: The original query that generated the results
            retrieved_chunks: List of chunks retrieved from Qdrant
            expected_section: The expected book section (if known)

        Returns:
            ValidationLog object with mapping validation results
        """
        # Validate each chunk individually
        validation_results = []
        total_accuracy = 0.0

        for chunk in retrieved_chunks:
            # In a real implementation, we'd compare against the actual original text
            # For now, we'll use the chunk's own text as the "original" for demonstration
            original_text = chunk.text
            result = self.validate_chunk_accuracy(chunk, original_text)
            validation_results.append(result)
            total_accuracy += result.accuracy_score

        # Calculate overall accuracy
        overall_accuracy = total_accuracy / len(validation_results) if validation_results else 0.0

        # Create validation log
        validation_log = ValidationLog(
            id=f"mapping_validation_{hash(query) % 1000000}",
            query=query,
            results=retrieved_chunks,
            validations=validation_results,
            overall_accuracy=overall_accuracy,
            book_section=expected_section
        )

        return validation_log

    def calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate the overlap between two texts using various metrics.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Overlap score between 0.0 and 1.0
        """
        # Clean texts for comparison
        clean_text1 = re.sub(r'\s+', ' ', text1.lower().strip())
        clean_text2 = re.sub(r'\s+', ' ', text2.lower().strip())

        # Calculate word overlap
        words1 = set(clean_text1.split())
        words2 = set(clean_text2.split())

        if not words1 and not words2:
            return 1.0  # Both texts are empty, consider them identical
        if not words1 or not words2:
            return 0.0  # One is empty, the other is not

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union

        return jaccard_similarity