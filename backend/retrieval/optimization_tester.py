"""
Optimization testing for different parameter combinations.

This module provides functionality to test and evaluate different top-k
and similarity threshold parameter combinations to optimize retrieval performance.
"""
from typing import List, Tuple, Dict, Optional
import time
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from .retrieval_service import RetrievalService
from .performance_metrics import PerformanceMetricsCollector
from models.retrieval_models import RetrievedChunk


class OptimizationTester:
    """
    Service class for testing and optimizing retrieval parameters.
    """

    def __init__(self, retrieval_service: RetrievalService):
        """
        Initialize the optimization tester.

        Args:
            retrieval_service: The retrieval service to test
        """
        self.retrieval_service = retrieval_service
        self.metrics_collector = PerformanceMetricsCollector()

    def test_parameter_combinations(
        self,
        query: str,
        collection_name: str,
        top_k_values: List[int],
        similarity_thresholds: List[float],
        iterations: int = 1
    ) -> List[Dict]:
        """
        Test different parameter combinations for retrieval.

        Args:
            query: The query to test with
            collection_name: The collection to search in
            top_k_values: List of top_k values to test
            similarity_thresholds: List of similarity thresholds to test
            iterations: Number of iterations to run for each combination (for averaging)

        Returns:
            List of dictionaries containing test results for each combination
        """
        results = []

        for top_k in top_k_values:
            for threshold in similarity_thresholds:
                print(f"Testing combination: top_k={top_k}, threshold={threshold}")

                # Run multiple iterations to get average performance
                total_time = 0
                total_results = 0
                total_similarity = 0
                successful_runs = 0

                for _ in range(iterations):
                    try:
                        start_time = time.time()

                        # Perform retrieval
                        chunks = self.retrieval_service.retrieve_chunks(
                            query_text=query,
                            collection_name=collection_name,
                            top_k=top_k,
                            similarity_threshold=threshold
                        )

                        end_time = time.time()
                        execution_time = end_time - start_time

                        # Calculate average similarity score
                        if chunks:
                            avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
                        else:
                            avg_similarity = 0.0

                        # Collect metrics
                        self.metrics_collector.collect_metrics(
                            top_k=top_k,
                            similarity_threshold=threshold,
                            execution_time=execution_time,
                            results_count=len(chunks),
                            average_similarity_score=avg_similarity
                        )

                        total_time += execution_time
                        total_results += len(chunks)
                        total_similarity += avg_similarity
                        successful_runs += 1

                    except Exception as e:
                        print(f"Error testing combination top_k={top_k}, threshold={threshold}: {e}")
                        continue

                if successful_runs > 0:
                    avg_time = total_time / successful_runs
                    avg_results = total_results / successful_runs
                    avg_similarity = total_similarity / successful_runs

                    result = {
                        "top_k": top_k,
                        "similarity_threshold": threshold,
                        "avg_execution_time": avg_time,
                        "avg_results_count": avg_results,
                        "avg_similarity_score": avg_similarity,
                        "successful_runs": successful_runs,
                        "total_runs": iterations
                    }
                    results.append(result)

        return results

    def find_optimal_parameters(
        self,
        query: str,
        collection_name: str,
        top_k_range: Tuple[int, int] = (1, 10),
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        step_size: int = 1
    ) -> Tuple[int, float]:
        """
        Find optimal parameters through systematic testing.

        Args:
            query: The query to optimize for
            collection_name: The collection to search in
            top_k_range: Range of top_k values to test (min, max)
            threshold_range: Range of similarity thresholds to test (min, max)
            step_size: Step size for parameter increments

        Returns:
            Tuple of (optimal_top_k, optimal_similarity_threshold)
        """
        # Generate parameter ranges
        top_k_values = list(range(top_k_range[0], top_k_range[1] + 1, step_size))

        # Generate threshold values
        threshold_values = []
        current = threshold_range[0]
        while current <= threshold_range[1]:
            threshold_values.append(round(current, 2))
            current += 0.1  # Use 0.1 step for thresholds

        print(f"Testing {len(top_k_values)} top_k values and {len(threshold_values)} threshold values")
        print(f"Total combinations to test: {len(top_k_values) * len(threshold_values)}")

        # Test all combinations
        results = self.test_parameter_combinations(
            query=query,
            collection_name=collection_name,
            top_k_values=top_k_values,
            similarity_thresholds=threshold_values,
            iterations=1
        )

        # Find the best combination based on a composite score
        best_score = float('-inf')
        best_params = (3, 0.3)  # Default values

        for result in results:
            # Calculate a composite score that balances:
            # - Fast execution time (higher score for faster)
            # - Good number of relevant results (not too few, not too many)
            # - High similarity scores
            time_score = max(0, 1 - (result["avg_execution_time"] / 5.0))  # 5s is considered slow
            results_score = min(1.0, result["avg_results_count"] / 10.0)  # Cap at 10 results
            similarity_score = result["avg_similarity_score"]

            # Weighted composite score
            composite_score = (0.3 * time_score) + (0.3 * results_score) + (0.4 * similarity_score)

            if composite_score > best_score:
                best_score = composite_score
                best_params = (result["top_k"], result["similarity_threshold"])

        return best_params

    def get_parameter_recommendations(
        self,
        query: str,
        collection_name: str,
        use_case: str = "general"
    ) -> Dict:
        """
        Get parameter recommendations based on use case.

        Args:
            query: The query to optimize for
            collection_name: The collection to search in
            use_case: The use case ('precision', 'recall', 'balanced', 'general')

        Returns:
            Dictionary with recommended parameters and reasoning
        """
        # For different use cases, we might want different optimization strategies
        if use_case == "precision":
            # Higher threshold, lower top_k for more precise results
            top_k_values = [1, 2, 3]
            threshold_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        elif use_case == "recall":
            # Lower threshold, higher top_k for more comprehensive results
            top_k_values = [5, 7, 10]
            threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        else:  # balanced/general
            # Balanced approach
            top_k_values = [2, 3, 4, 5]
            threshold_values = [0.2, 0.3, 0.4, 0.5, 0.6]

        results = self.test_parameter_combinations(
            query=query,
            collection_name=collection_name,
            top_k_values=top_k_values,
            similarity_thresholds=threshold_values,
            iterations=1
        )

        if not results:
            return {
                "recommended_top_k": 3,
                "recommended_threshold": 0.3,
                "reasoning": "No test results available, using default values",
                "use_case": use_case
            }

        # Find the best based on the use case
        if use_case == "precision":
            # Prioritize higher similarity scores
            best_result = max(results, key=lambda x: x["avg_similarity_score"])
        elif use_case == "recall":
            # Prioritize higher number of results
            best_result = max(results, key=lambda x: x["avg_results_count"])
        else:  # balanced/general
            # Use the composite scoring approach
            best_score = float('-inf')
            best_result = results[0]

            for result in results:
                time_score = max(0, 1 - (result["avg_execution_time"] / 5.0))
                results_score = min(1.0, result["avg_results_count"] / 10.0)
                similarity_score = result["avg_similarity_score"]
                composite_score = (0.3 * time_score) + (0.3 * results_score) + (0.4 * similarity_score)

                if composite_score > best_score:
                    best_score = composite_score
                    best_result = result

        return {
            "recommended_top_k": best_result["top_k"],
            "recommended_threshold": best_result["similarity_threshold"],
            "reasoning": f"Optimized for {use_case} use case based on test results",
            "avg_execution_time": best_result["avg_execution_time"],
            "avg_results_count": best_result["avg_results_count"],
            "avg_similarity_score": best_result["avg_similarity_score"],
            "use_case": use_case
        }

    def run_comprehensive_optimization(
        self,
        queries: List[str],
        collection_name: str
    ) -> Dict:
        """
        Run comprehensive optimization across multiple queries.

        Args:
            queries: List of queries to test with
            collection_name: The collection to search in

        Returns:
            Dictionary with overall optimization recommendations
        """
        all_results = []

        for i, query in enumerate(queries):
            print(f"Optimizing for query {i+1}/{len(queries)}: {query[:50]}...")
            result = self.get_parameter_recommendations(query, collection_name)
            all_results.append(result)

        # Aggregate results to find overall best parameters
        top_k_values = [r["recommended_top_k"] for r in all_results]
        threshold_values = [r["recommended_threshold"] for r in all_results]

        # Find the most common recommendations
        from collections import Counter
        top_k_counter = Counter(top_k_values)
        threshold_counter = Counter(threshold_values)

        overall_top_k = top_k_counter.most_common(1)[0][0]
        overall_threshold = threshold_counter.most_common(1)[0][0]

        return {
            "overall_recommended_top_k": overall_top_k,
            "overall_recommended_threshold": overall_threshold,
            "individual_results": all_results,
            "query_count": len(queries)
        }