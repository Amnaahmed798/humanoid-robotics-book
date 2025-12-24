"""
Performance metrics collection for different retrieval configurations.

This module provides functionality to collect and analyze performance metrics
for different top-k and similarity threshold configurations.
"""
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class PerformanceMetrics:
    """
    Data class to hold performance metrics for a specific configuration.
    """
    top_k: int
    similarity_threshold: float
    execution_time: float  # in seconds
    results_count: int
    average_similarity_score: float
    timestamp: datetime
    query_complexity: Optional[str] = None  # simple, medium, complex


class PerformanceMetricsCollector:
    """
    Service class for collecting and analyzing performance metrics.
    """

    def __init__(self):
        """
        Initialize the metrics collector.
        """
        self.metrics_history: List[PerformanceMetrics] = []

    def collect_metrics(
        self,
        top_k: int,
        similarity_threshold: float,
        execution_time: float,
        results_count: int,
        average_similarity_score: float,
        query_complexity: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Collect performance metrics for a specific configuration.

        Args:
            top_k: The top-k value used
            similarity_threshold: The similarity threshold used
            execution_time: Time taken for the operation in seconds
            results_count: Number of results returned
            average_similarity_score: Average similarity score of results
            query_complexity: Complexity of the query (optional)

        Returns:
            PerformanceMetrics object containing the collected metrics
        """
        metrics = PerformanceMetrics(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            execution_time=execution_time,
            results_count=results_count,
            average_similarity_score=average_similarity_score,
            timestamp=datetime.now(),
            query_complexity=query_complexity
        )

        self.metrics_history.append(metrics)
        return metrics

    def get_metrics_by_config(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[PerformanceMetrics]:
        """
        Get metrics filtered by configuration parameters.

        Args:
            top_k: Filter by specific top_k value (None to ignore)
            similarity_threshold: Filter by specific similarity threshold (None to ignore)

        Returns:
            List of PerformanceMetrics matching the criteria
        """
        filtered_metrics = self.metrics_history

        if top_k is not None:
            filtered_metrics = [m for m in filtered_metrics if m.top_k == top_k]

        if similarity_threshold is not None:
            filtered_metrics = [
                m for m in filtered_metrics
                if m.similarity_threshold == similarity_threshold
            ]

        return filtered_metrics

    def get_average_metrics_by_config(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get average metrics for a specific configuration.

        Args:
            top_k: Filter by specific top_k value (None to ignore)
            similarity_threshold: Filter by specific similarity threshold (None to ignore)

        Returns:
            Dictionary with average metrics, or None if no metrics match
        """
        metrics = self.get_metrics_by_config(top_k, similarity_threshold)

        if not metrics:
            return None

        avg_execution_time = statistics.mean([m.execution_time for m in metrics])
        avg_results_count = statistics.mean([m.results_count for m in metrics])
        avg_similarity_score = statistics.mean([m.average_similarity_score for m in metrics])

        return {
            "avg_execution_time": avg_execution_time,
            "avg_results_count": avg_results_count,
            "avg_similarity_score": avg_similarity_score,
            "sample_count": len(metrics)
        }

    def get_optimal_configuration(self) -> Optional[Tuple[int, float]]:
        """
        Determine the optimal configuration based on performance metrics.

        Returns:
            Tuple of (optimal_top_k, optimal_similarity_threshold) or None if no metrics
        """
        if not self.metrics_history:
            return None

        # For now, we'll define optimal as having a good balance of:
        # - Reasonable execution time (not too slow)
        # - Good number of relevant results
        # - High similarity scores
        best_score = float('-inf')
        best_config = None

        for metrics in self.metrics_history:
            # Calculate a composite score
            # Lower execution time is better (penalize slow queries)
            time_score = max(0, 1 - (metrics.execution_time / 5.0))  # Assume 5s is too slow

            # More results might be better, but capped
            results_score = min(1.0, metrics.results_count / 10.0)  # Cap at 10 results

            # Higher similarity is better
            similarity_score = metrics.average_similarity_score

            # Weighted composite score
            composite_score = (0.3 * time_score) + (0.3 * results_score) + (0.4 * similarity_score)

            if composite_score > best_score:
                best_score = composite_score
                best_config = (metrics.top_k, metrics.similarity_threshold)

        return best_config

    def get_configuration_rankings(self) -> List[Dict]:
        """
        Get rankings of all configurations based on performance.

        Returns:
            List of dictionaries with configuration and score rankings
        """
        if not self.metrics_history:
            return []

        config_scores = {}
        config_counts = {}

        for metrics in self.metrics_history:
            config = (metrics.top_k, metrics.similarity_threshold)

            # Calculate score for this metric entry
            time_score = max(0, 1 - (metrics.execution_time / 5.0))
            results_score = min(1.0, metrics.results_count / 10.0)
            similarity_score = metrics.average_similarity_score
            composite_score = (0.3 * time_score) + (0.3 * results_score) + (0.4 * similarity_score)

            if config not in config_scores:
                config_scores[config] = []
                config_counts[config] = 0

            config_scores[config].append(composite_score)
            config_counts[config] += 1

        # Calculate average score for each configuration
        rankings = []
        for config, scores in config_scores.items():
            avg_score = sum(scores) / len(scores)
            rankings.append({
                "top_k": config[0],
                "similarity_threshold": config[1],
                "avg_composite_score": avg_score,
                "sample_count": config_counts[config],
                "avg_execution_time": statistics.mean([
                    m.execution_time for m in self.metrics_history
                    if (m.top_k, m.similarity_threshold) == config
                ]),
                "avg_results_count": statistics.mean([
                    m.results_count for m in self.metrics_history
                    if (m.top_k, m.similarity_threshold) == config
                ]),
                "avg_similarity_score": statistics.mean([
                    m.average_similarity_score for m in self.metrics_history
                    if (m.top_k, m.similarity_threshold) == config
                ])
            })

        # Sort by average composite score (descending)
        rankings.sort(key=lambda x: x["avg_composite_score"], reverse=True)
        return rankings

    def reset_metrics(self):
        """
        Reset all collected metrics.
        """
        self.metrics_history = []