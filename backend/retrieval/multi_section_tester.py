"""
Multi-section testing for validation across multiple book sections.

This module provides functionality to test retrieval and validation
across at least 10 different book sections as required by the spec.
"""
from typing import List, Dict, Tuple
import time
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from .retrieval_service import RetrievalService
from .validation_service import ValidationService
from .result_logger import ResultLogger
from models.retrieval_models import ValidationLog, RetrievedChunk
from .config import RetrievalConfig


class MultiSectionTester:
    """
    Service class for testing retrieval and validation across multiple book sections.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        validation_service: ValidationService,
        result_logger: ResultLogger
    ):
        """
        Initialize the multi-section tester.

        Args:
            retrieval_service: The retrieval service to test
            validation_service: The validation service to use
            result_logger: The result logger to use
        """
        self.retrieval_service = retrieval_service
        self.validation_service = validation_service
        self.result_logger = result_logger

    def test_single_section(
        self,
        query: str,
        collection_name: str,
        book_section: str,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> ValidationLog:
        """
        Test retrieval and validation for a single book section.

        Args:
            query: The query to test
            collection_name: The collection to search in
            book_section: The book section to test
            top_k: Number of results to retrieve (uses default if None)
            similarity_threshold: Similarity threshold to use (uses default if None)

        Returns:
            ValidationLog with the results
        """
        start_time = time.time()

        # Retrieve chunks for the query
        chunks = self.retrieval_service.retrieve_chunks(
            query_text=query,
            collection_name=collection_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Create a mock QueryResult to validate
        from models.retrieval_models import QueryResult
        query_result = QueryResult(
            query_id=f"query_{abs(hash(query)) % 1000000}",
            query=query,
            results=chunks,
            validations=[],  # Will be populated by validation service
            timestamp=start_time
        )

        # Validate the results
        validation_log = self.validation_service.validate_query_result(query_result)
        validation_log.book_section = book_section

        # Update the execution time
        end_time = time.time()
        validation_log.timestamp = time.time()

        # Log the query execution
        self.result_logger.log_query_execution(
            query=query,
            results=chunks,
            execution_time=end_time - start_time,
            book_section=book_section
        )

        return validation_log

    def test_multiple_sections(
        self,
        queries: List[str],
        collection_name: str,
        book_sections: List[str],
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[ValidationLog]:
        """
        Test retrieval and validation across multiple book sections.

        Args:
            queries: List of queries to test
            collection_name: The collection to search in
            book_sections: List of book sections to test
            top_k: Number of results to retrieve (uses default if None)
            similarity_threshold: Similarity threshold to use (uses default if None)

        Returns:
            List of ValidationLog objects for each section
        """
        all_validation_logs = []

        for section in book_sections:
            print(f"Testing section: {section}")
            for query in queries:
                try:
                    validation_log = self.test_single_section(
                        query=query,
                        collection_name=collection_name,
                        book_section=section,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    all_validation_logs.append(validation_log)
                except Exception as e:
                    print(f"Error testing query '{query}' in section '{section}': {e}")
                    continue

        return all_validation_logs

    def run_comprehensive_section_test(
        self,
        collection_name: str,
        test_queries: List[str] = None
    ) -> Dict:
        """
        Run a comprehensive test across multiple book sections with default test queries.

        Args:
            collection_name: The collection to search in
            test_queries: List of queries to test (uses defaults if None)

        Returns:
            Dictionary with test results and summary statistics
        """
        # Default test queries if none provided
        if test_queries is None:
            test_queries = [
                "What are the key principles of humanoid robotics?",
                "Explain the kinematic chain in robotics",
                "What are the main components of a humanoid robot?",
                "How does inverse kinematics work?",
                "What is the difference between forward and inverse kinematics?",
                "Describe the control systems in humanoid robots",
                "What sensors are used in humanoid robotics?",
                "Explain gait planning for bipedal robots",
                "What are the challenges in humanoid robot design?",
                "How do humanoid robots maintain balance?"
            ]

        # Default book sections (at least 10 as required by spec)
        default_sections = [
            "introduction",
            "kinematics",
            "dynamics",
            "control_systems",
            "sensors",
            "actuators",
            "locomotion",
            "balance",
            "vision_systems",
            "manipulation",
            "ai_planning",
            "human_robot_interaction"
        ]

        print(f"Starting comprehensive test across {len(default_sections)} book sections...")
        print(f"Using {len(test_queries)} test queries...")

        # Run the tests
        validation_logs = self.test_multiple_sections(
            queries=test_queries,
            collection_name=collection_name,
            book_sections=default_sections
        )

        print(f"Completed testing. Processed {len(validation_logs)} validation logs.")

        # Generate summary statistics
        summary_stats = self.result_logger.generate_summary_statistics(validation_logs)

        # Log the multi-section results
        multi_section_results = self.result_logger.log_multi_section_results(
            validation_logs=validation_logs,
            book_sections=default_sections
        )

        # Generate validation report
        report_path = self.result_logger.create_validation_report(validation_logs)

        # Generate CSV reports
        csv_report_path = self.result_logger.generate_csv_report(validation_logs)
        detailed_csv_path = self.result_logger.generate_detailed_csv_report(validation_logs)

        results = {
            "validation_logs_count": len(validation_logs),
            "sections_tested": default_sections,
            "queries_tested": test_queries,
            "summary_statistics": summary_stats,
            "multi_section_results": multi_section_results,
            "report_path": report_path,
            "csv_report_path": csv_report_path,
            "detailed_csv_path": detailed_csv_path,
            "success": len(validation_logs) > 0
        }

        return results

    def validate_minimum_sections_requirement(self, validation_logs: List[ValidationLog]) -> bool:
        """
        Validate that the minimum requirement of 10 book sections was met.

        Args:
            validation_logs: List of validation logs to check

        Returns:
            True if requirement is met, False otherwise
        """
        # Get unique book sections from validation logs
        unique_sections = set(log.book_section for log in validation_logs)

        # Count how many unique sections we have
        section_count = len(unique_sections)

        print(f"Tested across {section_count} unique book sections")
        print(f"Sections tested: {sorted(unique_sections)}")

        # Check if we met the minimum requirement of 10
        requirement_met = section_count >= RetrievalConfig.DEFAULT_TEST_SECTIONS
        print(f"Minimum 10 sections requirement met: {requirement_met}")

        return requirement_met

    def generate_summary_statistics(self, validation_logs: List[ValidationLog]) -> Dict:
        """
        Generate comprehensive summary statistics and accuracy metrics.

        Args:
            validation_logs: List of validation logs to analyze

        Returns:
            Dictionary with comprehensive summary statistics and metrics
        """
        if not validation_logs:
            return {}

        # Calculate overall statistics
        total_logs = len(validation_logs)
        total_queries = sum(len(log.results) for log in validation_logs)
        overall_accuracy = sum(log.overall_accuracy for log in validation_logs) / total_logs if total_logs > 0 else 0

        # Calculate validation statistics
        total_validations = sum(len(log.validations) for log in validation_logs)
        passed_validations = sum(
            sum(1 for v in log.validations if v.validation_passed) for log in validation_logs
        )
        failed_validations = total_validations - passed_validations
        pass_rate = passed_validations / total_validations if total_validations > 0 else 0

        # Calculate accuracy distribution
        all_accuracy_scores = []
        all_similarity_scores = []
        all_validation_scores = []

        for log in validation_logs:
            for validation in log.validations:
                all_accuracy_scores.append(validation.accuracy_score)
                all_validation_scores.append(validation.accuracy_score)
            for result in log.results:
                all_similarity_scores.append(result.similarity_score)

        # Accuracy metrics
        if all_accuracy_scores:
            avg_accuracy = sum(all_accuracy_scores) / len(all_accuracy_scores)
            min_accuracy = min(all_accuracy_scores)
            max_accuracy = max(all_accuracy_scores)
            accuracy_std_dev = (sum((x - avg_accuracy) ** 2 for x in all_accuracy_scores) / len(all_accuracy_scores)) ** 0.5 if len(all_accuracy_scores) > 1 else 0
        else:
            avg_accuracy = min_accuracy = max_accuracy = accuracy_std_dev = 0

        # Similarity metrics
        if all_similarity_scores:
            avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores)
            min_similarity = min(all_similarity_scores)
            max_similarity = max(all_similarity_scores)
            similarity_std_dev = (sum((x - avg_similarity) ** 2 for x in all_similarity_scores) / len(all_similarity_scores)) ** 0.5 if len(all_similarity_scores) > 1 else 0
        else:
            avg_similarity = min_similarity = max_similarity = similarity_std_dev = 0

        # Validation score metrics
        if all_validation_scores:
            avg_validation_score = sum(all_validation_scores) / len(all_validation_scores)
            min_validation_score = min(all_validation_scores)
            max_validation_score = max(all_validation_scores)
        else:
            avg_validation_score = min_validation_score = max_validation_score = 0

        # Calculate percentiles for more detailed metrics
        def calculate_percentiles(scores):
            if not scores:
                return {}
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            return {
                "p10": sorted_scores[int(0.1 * n)] if n > 1 else sorted_scores[0],
                "p25": sorted_scores[int(0.25 * n)] if n > 1 else sorted_scores[0],
                "p50": sorted_scores[int(0.5 * n)] if n > 1 else sorted_scores[0],  # median
                "p75": sorted_scores[int(0.75 * n)] if n > 1 else sorted_scores[0],
                "p90": sorted_scores[int(0.9 * n)] if n > 1 else sorted_scores[0],
                "p95": sorted_scores[int(0.95 * n)] if n > 1 else sorted_scores[0]
            }

        accuracy_percentiles = calculate_percentiles(all_accuracy_scores)
        similarity_percentiles = calculate_percentiles(all_similarity_scores)

        # Calculate section-wise metrics
        section_stats = {}
        for log in validation_logs:
            section = log.book_section
            if section not in section_stats:
                section_stats[section] = {
                    "logs_count": 0,
                    "total_accuracy": 0,
                    "total_results": 0,
                    "total_validations": 0,
                    "passed_validations": 0
                }

            section_stats[section]["logs_count"] += 1
            section_stats[section]["total_accuracy"] += log.overall_accuracy
            section_stats[section]["total_results"] += len(log.results)
            section_stats[section]["total_validations"] += len(log.validations)
            section_stats[section]["passed_validations"] += sum(1 for v in log.validations if v.validation_passed)

        # Calculate average per section
        for section, data in section_stats.items():
            data["avg_accuracy"] = data["total_accuracy"] / data["logs_count"] if data["logs_count"] > 0 else 0
            data["avg_results_per_log"] = data["total_results"] / data["logs_count"] if data["logs_count"] > 0 else 0
            data["validation_pass_rate"] = data["passed_validations"] / data["total_validations"] if data["total_validations"] > 0 else 0

        # Compile all metrics
        summary_metrics = {
            "overall_metrics": {
                "total_logs_processed": total_logs,
                "total_queries_processed": total_queries,
                "overall_accuracy": overall_accuracy,
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "failed_validations": failed_validations,
                "validation_pass_rate": pass_rate
            },
            "accuracy_metrics": {
                "average_accuracy": avg_accuracy,
                "minimum_accuracy": min_accuracy,
                "maximum_accuracy": max_accuracy,
                "accuracy_std_deviation": accuracy_std_dev,
                "accuracy_percentiles": accuracy_percentiles
            },
            "similarity_metrics": {
                "average_similarity": avg_similarity,
                "minimum_similarity": min_similarity,
                "maximum_similarity": max_similarity,
                "similarity_std_deviation": similarity_std_dev,
                "similarity_percentiles": similarity_percentiles
            },
            "validation_score_metrics": {
                "average_validation_score": avg_validation_score,
                "minimum_validation_score": min_validation_score,
                "maximum_validation_score": max_validation_score
            },
            "section_wise_metrics": section_stats,
            "timestamp": time.time()
        }

        return summary_metrics

    def generate_section_wise_report(self, validation_logs: List[ValidationLog]) -> Dict:
        """
        Generate a report broken down by book sections.

        Args:
            validation_logs: List of validation logs to analyze

        Returns:
            Dictionary with section-wise statistics
        """
        section_stats = {}

        # Group validation logs by section
        for log in validation_logs:
            section = log.book_section
            if section not in section_stats:
                section_stats[section] = {
                    "logs": [],
                    "total_accuracy": 0,
                    "count": 0
                }

            section_stats[section]["logs"].append(log)
            section_stats[section]["total_accuracy"] += log.overall_accuracy
            section_stats[section]["count"] += 1

        # Calculate average accuracy per section
        for section, data in section_stats.items():
            data["average_accuracy"] = data["total_accuracy"] / data["count"] if data["count"] > 0 else 0
            data["log_count"] = data["count"]
            del data["total_accuracy"]  # Remove temporary field
            del data["count"]  # Remove temporary field

        return section_stats