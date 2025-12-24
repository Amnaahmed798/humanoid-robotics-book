"""
Result logging service for retrieval validation.

This service handles comprehensive logging of query execution and validation results,
as well as generating validation reports for multiple book sections.
"""
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.retrieval_models import ValidationLog, ValidationResult, RetrievedChunk
from .config import RetrievalConfig


class ResultLogger:
    """
    Service class for logging retrieval and validation results.
    """

    def __init__(self, log_directory: str = "logs"):
        """
        Initialize the result logger.

        Args:
            log_directory: Directory to store log files
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)

    def log_query_execution(
        self,
        query: str,
        results: List[RetrievedChunk],
        execution_time: float,
        book_section: str = "unknown"
    ) -> str:
        """
        Log the execution of a query with its results and metadata.

        Args:
            query: The query that was executed
            results: The results returned by the query
            execution_time: Time taken to execute the query in seconds
            book_section: The book section being queried

        Returns:
            Path to the saved log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = abs(hash(query)) % 1000000  # Simple hash for unique ID
        filename = f"query_execution_{timestamp}_{query_hash}.json"
        filepath = self.log_directory / filename

        log_data = {
            "log_type": "query_execution",
            "id": f"query_{timestamp}_{query_hash}",
            "query": query,
            "book_section": book_section,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "results_count": len(results),
            "results": [
                {
                    "id": chunk.id,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "source_location": chunk.source_location,
                    "similarity_score": chunk.similarity_score
                }
                for chunk in results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def log_validation_result(self, validation_log: ValidationLog) -> str:
        """
        Log a single validation result to a JSON file.

        Args:
            validation_log: The validation log to save

        Returns:
            Path to the saved log file
        """
        timestamp = validation_log.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{timestamp}_{validation_log.id[:8]}.json"
        filepath = self.log_directory / filename

        log_data = {
            "log_type": "validation_result",
            "id": validation_log.id,
            "query": validation_log.query,
            "book_section": validation_log.book_section,
            "timestamp": validation_log.timestamp.isoformat(),
            "overall_accuracy": validation_log.overall_accuracy,
            "results_count": len(validation_log.results),
            "validations_count": len(validation_log.validations),
            "results": [
                {
                    "id": chunk.id,
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "source_location": chunk.source_location,
                    "similarity_score": chunk.similarity_score
                }
                for chunk in validation_log.results
            ],
            "validations": [
                {
                    "chunk_id": val.chunk_id,
                    "accuracy_score": val.accuracy_score,
                    "validation_passed": val.validation_passed,
                    "validation_details": val.validation_details
                }
                for val in validation_log.validations
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def log_batch_validation_results(self, validation_logs: List[ValidationLog]) -> str:
        """
        Log multiple validation results to a single JSON file.

        Args:
            validation_logs: List of validation logs to save

        Returns:
            Path to the saved log file
        """
        if not validation_logs:
            raise ValueError("No validation logs to save")

        # Use the timestamp of the first log for the filename
        timestamp = validation_logs[0].timestamp.strftime("%Y%m%d_%H%M%S") if validation_logs[0].timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_validation_{timestamp}.json"
        filepath = self.log_directory / filename

        batch_data = {
            "batch_id": f"batch_{timestamp}",
            "total_logs": len(validation_logs),
            "timestamp": datetime.now().isoformat(),
            "logs": []
        }

        for log in validation_logs:
            log_timestamp = log.timestamp.isoformat() if log.timestamp else datetime.now().isoformat()
            log_data = {
                "id": log.id,
                "query": log.query,
                "book_section": log.book_section,
                "timestamp": log_timestamp,
                "overall_accuracy": log.overall_accuracy,
                "results_count": len(log.results),
                "validations_count": len(log.validations)
            }
            batch_data["logs"].append(log_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def log_multi_section_results(
        self,
        validation_logs: List[ValidationLog],
        book_sections: List[str]
    ) -> str:
        """
        Log results across multiple book sections to a single JSON file.

        Args:
            validation_logs: List of validation logs for different sections
            book_sections: List of book sections that were tested

        Returns:
            Path to the saved log file
        """
        if not validation_logs:
            raise ValueError("No validation logs to save")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_section_validation_{timestamp}.json"
        filepath = self.log_directory / filename

        # Create a mapping of section to its validation log
        section_logs = dict(zip(book_sections, validation_logs))

        # Calculate summary statistics
        total_logs = len(validation_logs)
        total_queries = sum(len(log.results) for log in validation_logs)
        overall_accuracy = sum(log.overall_accuracy for log in validation_logs) / total_logs if total_logs > 0 else 0
        total_validations = sum(len(log.validations) for log in validation_logs)
        passed_validations = sum(
            sum(1 for v in log.validations if v.validation_passed) for log in validation_logs
        )
        failed_validations = total_validations - passed_validations
        pass_rate = passed_validations / total_validations if total_validations > 0 else 0

        # Generate section-specific statistics
        section_stats = {}
        for section, log in section_logs.items():
            section_passed = sum(1 for v in log.validations if v.validation_passed)
            section_failed = len(log.validations) - section_passed
            section_stats[section] = {
                "overall_accuracy": log.overall_accuracy,
                "results_count": len(log.results),
                "validations_count": len(log.validations),
                "passed_count": section_passed,
                "failed_count": section_failed,
                "pass_rate": section_passed / len(log.validations) if log.validations else 0
            }

        summary_data = {
            "summary": {
                "total_sections_tested": total_logs,
                "total_queries": total_queries,
                "overall_accuracy": overall_accuracy,
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "failed_validations": failed_validations,
                "pass_rate": pass_rate,
                "timestamp": datetime.now().isoformat()
            },
            "section_stats": section_stats,
            "sections_tested": book_sections,
            "detailed_logs": [
                {
                    "id": log.id,
                    "query": log.query,
                    "book_section": log.book_section,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else datetime.now().isoformat(),
                    "overall_accuracy": log.overall_accuracy,
                    "results_count": len(log.results),
                    "validations_count": len(log.validations)
                }
                for log in validation_logs
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def generate_csv_report(self, validation_logs: List[ValidationLog]) -> str:
        """
        Generate a CSV report from validation logs.

        Args:
            validation_logs: List of validation logs to include in the report

        Returns:
            Path to the generated CSV file
        """
        if not validation_logs:
            raise ValueError("No validation logs to generate report from")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.csv"
        filepath = self.log_directory / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'log_id', 'query', 'book_section', 'timestamp', 'overall_accuracy',
                'results_count', 'validations_passed', 'validations_failed'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for log in validation_logs:
                passed_count = sum(1 for v in log.validations if v.validation_passed)
                failed_count = len(log.validations) - passed_count

                writer.writerow({
                    'log_id': log.id,
                    'query': log.query.replace('\n', ' ').replace('\r', ' '),  # Clean newlines
                    'book_section': log.book_section,
                    'timestamp': log.timestamp.isoformat(),
                    'overall_accuracy': f"{log.overall_accuracy:.4f}",
                    'results_count': len(log.results),
                    'validations_passed': passed_count,
                    'validations_failed': failed_count
                })

        return str(filepath)

    def generate_detailed_csv_report(self, validation_logs: List[ValidationLog]) -> str:
        """
        Generate a detailed CSV report with individual validation results.

        Args:
            validation_logs: List of validation logs to include in the report

        Returns:
            Path to the generated CSV file
        """
        if not validation_logs:
            raise ValueError("No validation logs to generate report from")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_validation_report_{timestamp}.csv"
        filepath = self.log_directory / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'log_id', 'query', 'book_section', 'chunk_id', 'source_location',
                'similarity_score', 'accuracy_score', 'validation_passed', 'validation_details'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for log in validation_logs:
                for result, validation in zip(log.results, log.validations):
                    writer.writerow({
                        'log_id': log.id,
                        'query': log.query.replace('\n', ' ').replace('\r', ' '),  # Clean newlines
                        'book_section': log.book_section,
                        'chunk_id': result.id,
                        'source_location': result.source_location,
                        'similarity_score': f"{result.similarity_score:.4f}",
                        'accuracy_score': f"{validation.accuracy_score:.4f}",
                        'validation_passed': validation.validation_passed,
                        'validation_details': validation.validation_details.replace('\n', ' ').replace('\r', ' ')  # Clean newlines
                    })

        return str(filepath)

    def log_multi_section_results(
        self,
        validation_logs: List[ValidationLog],
        book_sections: List[str]
    ) -> Dict[str, Any]:
        """
        Log results across multiple book sections and generate summary statistics.

        Args:
            validation_logs: List of validation logs for different sections
            book_sections: List of book sections that were tested

        Returns:
            Dictionary with summary statistics
        """
        if len(validation_logs) != len(book_sections):
            raise ValueError("Number of validation logs must match number of book sections")

        # Create a mapping of section to its validation log
        section_logs = dict(zip(book_sections, validation_logs))

        # Calculate summary statistics
        total_logs = len(validation_logs)
        total_queries = sum(len(log.results) for log in validation_logs)
        overall_accuracy = sum(log.overall_accuracy for log in validation_logs) / total_logs if total_logs > 0 else 0
        total_validations = sum(len(log.validations) for log in validation_logs)
        passed_validations = sum(
            sum(1 for v in log.validations if v.validation_passed) for log in validation_logs
        )
        failed_validations = total_validations - passed_validations
        pass_rate = passed_validations / total_validations if total_validations > 0 else 0

        # Generate section-specific statistics
        section_stats = {}
        for section, log in section_logs.items():
            section_passed = sum(1 for v in log.validations if v.validation_passed)
            section_failed = len(log.validations) - section_passed
            section_stats[section] = {
                "overall_accuracy": log.overall_accuracy,
                "results_count": len(log.results),
                "validations_count": len(log.validations),
                "passed_count": section_passed,
                "failed_count": section_failed,
                "pass_rate": section_passed / len(log.validations) if log.validations else 0
            }

        # Save the multi-section results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_section_validation_{timestamp}.json"
        filepath = self.log_directory / filename

        summary_data = {
            "summary": {
                "total_sections_tested": total_logs,
                "total_queries": total_queries,
                "overall_accuracy": overall_accuracy,
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "failed_validations": failed_validations,
                "pass_rate": pass_rate,
                "timestamp": datetime.now().isoformat()
            },
            "section_stats": section_stats,
            "sections_tested": book_sections
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        return summary_data

    def generate_summary_statistics(self, validation_logs: List[ValidationLog]) -> Dict[str, Any]:
        """
        Generate summary statistics from validation logs.

        Args:
            validation_logs: List of validation logs to analyze

        Returns:
            Dictionary with summary statistics
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
        for log in validation_logs:
            for validation in log.validations:
                all_accuracy_scores.append(validation.accuracy_score)

        if all_accuracy_scores:
            avg_accuracy = sum(all_accuracy_scores) / len(all_accuracy_scores)
            min_accuracy = min(all_accuracy_scores)
            max_accuracy = max(all_accuracy_scores)
        else:
            avg_accuracy = min_accuracy = max_accuracy = 0

        # Calculate similarity score statistics
        all_similarity_scores = []
        for log in validation_logs:
            for result in log.results:
                all_similarity_scores.append(result.similarity_score)

        if all_similarity_scores:
            avg_similarity = sum(all_similarity_scores) / len(all_similarity_scores)
            min_similarity = min(all_similarity_scores)
            max_similarity = max(all_similarity_scores)
        else:
            avg_similarity = min_similarity = max_similarity = 0

        return {
            "total_logs": total_logs,
            "total_queries": total_queries,
            "overall_accuracy": overall_accuracy,
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "pass_rate": pass_rate,
            "accuracy_statistics": {
                "average": avg_accuracy,
                "minimum": min_accuracy,
                "maximum": max_accuracy
            },
            "similarity_statistics": {
                "average": avg_similarity,
                "minimum": min_similarity,
                "maximum": max_similarity
            },
            "timestamp": datetime.now().isoformat()
        }

    def generate_validation_report(self, validation_logs: List[ValidationLog]) -> str:
        """
        Generate a validation report in text format (alias for create_validation_report).

        Args:
            validation_logs: List of validation logs to include in the report

        Returns:
            Path to the generated report file
        """
        return self.create_validation_report(validation_logs)

    def create_validation_report(self, validation_logs: List[ValidationLog]) -> str:
        """
        Create a comprehensive validation report in text format.

        Args:
            validation_logs: List of validation logs to include in the report

        Returns:
            Path to the generated report file
        """
        if not validation_logs:
            raise ValueError("No validation logs to generate report from")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_validation_report_{timestamp}.txt"
        filepath = self.log_directory / filename

        # Generate summary statistics
        stats = self.generate_summary_statistics(validation_logs)

        report_lines = []
        report_lines.append("Comprehensive Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total validation logs: {stats['total_logs']}")
        report_lines.append(f"Total queries processed: {stats['total_queries']}")
        report_lines.append(f"Overall accuracy: {stats['overall_accuracy']:.2%}")
        report_lines.append(f"Total validations: {stats['total_validations']}")
        report_lines.append(f"Passed validations: {stats['passed_validations']}")
        report_lines.append(f"Failed validations: {stats['failed_validations']}")
        report_lines.append(f"Pass rate: {stats['pass_rate']:.2%}")
        report_lines.append("")

        report_lines.append("Accuracy Statistics:")
        report_lines.append(f"  Average accuracy: {stats['accuracy_statistics']['average']:.2%}")
        report_lines.append(f"  Minimum accuracy: {stats['accuracy_statistics']['minimum']:.2%}")
        report_lines.append(f"  Maximum accuracy: {stats['accuracy_statistics']['maximum']:.2%}")
        report_lines.append("")

        report_lines.append("Similarity Statistics:")
        report_lines.append(f"  Average similarity: {stats['similarity_statistics']['average']:.2%}")
        report_lines.append(f"  Minimum similarity: {stats['similarity_statistics']['minimum']:.2%}")
        report_lines.append(f"  Maximum similarity: {stats['similarity_statistics']['maximum']:.2%}")
        report_lines.append("")

        # Add details for each validation log (limit to first 5 to keep report manageable)
        report_lines.append("Validation Log Details (First 5):")
        report_lines.append("-" * 30)
        for i, log in enumerate(validation_logs[:5]):
            report_lines.append(f"Log {i+1}:")
            report_lines.append(f"  ID: {log.id}")
            report_lines.append(f"  Query: {log.query[:100]}{'...' if len(log.query) > 100 else ''}")
            report_lines.append(f"  Book Section: {log.book_section}")
            report_lines.append(f"  Accuracy: {log.overall_accuracy:.2%}")
            report_lines.append(f"  Results: {len(log.results)}")
            report_lines.append(f"  Validations: {len(log.validations)}")
            report_lines.append("")

        if len(validation_logs) > 5:
            report_lines.append(f"... and {len(validation_logs) - 5} more logs")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        return str(filepath)