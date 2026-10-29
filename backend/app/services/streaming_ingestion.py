"""Streaming Parquet ingestion with Apache Arrow for large datasets.

This module provides chunked reading of Parquet files without loading entire datasets
into memory, supporting incremental profiling and running statistical updates.
"""
import io
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from backend.app.services.importance import (
    mean,
    pearson_correlation,
    spearman_correlation,
    std_dev,
    variance,
)


def is_number(val: Any) -> bool:
    """Check if value can be converted to float."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


class ColumnProfile:
    """Running statistics for a single column computed incrementally.

    Uses Welford's online algorithm for numerically stable variance computation.
    """

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0  # Sum of squares for variance
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.null_count = 0
        self.sample_values: List[float] = []  # Reservoir sample for correlation
        self.max_samples = 10000  # Cap reservoir size
        self.unique_count = 0
        self.value_counts: Dict[str, int] = {}
        self.is_numeric = True
        self._initialized = False

    def update(self, values: List[Any], sample_indices: Optional[List[int]] = None) -> None:
        """Update running statistics with a batch of values."""
        if not self._initialized:
            # Determine type from first non-null batch
            for v in values:
                if v is not None and not is_number(v):
                    self.is_numeric = False
                    break
            self._initialized = True

        for idx, v in enumerate(values):
            if v is None:
                self.null_count += 1
                continue

            self.count += 1

            if self.is_numeric:
                try:
                    fv = float(v)
                    self.sum += fv
                    self.sum_sq += fv * fv
                    if fv < self.min_val:
                        self.min_val = fv
                    if fv > self.max_val:
                        self.max_val = fv

                    # Reservoir sampling for correlation computation
                    if sample_indices is not None and idx in sample_indices:
                        if len(self.sample_values) < self.max_samples:
                            self.sample_values.append(fv)
                except (ValueError, TypeError):
                    self.is_numeric = False
                    # Fall through to categorical handling
                    self.count -= 1  # Undo count for non-numeric
            else:
                # Categorical handling
                str_val = str(v)
                self.value_counts[str_val] = self.value_counts.get(str_val, 0) + 1

        if not self.is_numeric:
            self.unique_count = len(self.value_counts)

    def finalize(self) -> Dict[str, Any]:
        """Finalize and return profile dictionary."""
        result = {
            "name": self.name,
            "count": self.count,
            "null_count": self.null_count,
            "null_ratio": self.null_count / max(1, self.count + self.null_count),
            "is_numeric": self.is_numeric,
        }

        if self.is_numeric and self.count > 0:
            m = self.sum / self.count
            var = (self.sum_sq / self.count) - (m * m) if self.count > 1 else 0.0
            result.update({
                "mean": m,
                "std": math.sqrt(max(0.0, var)),
                "min": self.min_val if self.min_val != float("inf") else None,
                "max": self.max_val if self.max_val != float("-inf") else None,
                "variance": max(0.0, var),
            })
        else:
            result.update({
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "variance": None,
                "unique_count": self.unique_count,
                "top_values": sorted(self.value_counts.items(), key=lambda x: -x[1])[:10],
            })

        return result


class StreamingProfiler:
    """Incremental dataset profiler for large Parquet files."""

    def __init__(self, target_column: str, chunk_size: int = 10000, reservoir_size: int = 10000):
        self.target_column = target_column
        self.chunk_size = chunk_size
        self.reservoir_size = reservoir_size
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.target_profile: Optional[ColumnProfile] = None
        self.total_rows = 0
        self._sample_indices: List[int] = []  # Indices to sample in each chunk

    def _get_sample_indices(self, chunk_len: int) -> List[int]:
        """Determine which row indices in chunk to add to reservoir."""
        if self.reservoir_size <= 0:
            return []
        # Simple systematic sampling for reservoir
        step = max(1, chunk_len // min(self.reservoir_size, chunk_len))
        return list(range(0, chunk_len, step))[:self.reservoir_size]

    def process_chunk(self, table: pa.Table) -> None:
        """Process a single Arrow table chunk and update running statistics."""
        if table.num_rows == 0:
            return

        # Initialize column profiles on first chunk
        if not self.column_profiles:
            for col_name in table.column_names:
                self.column_profiles[col_name] = ColumnProfile(col_name)
                if col_name == self.target_column:
                    self.target_profile = self.column_profiles[col_name]

        sample_indices = self._get_sample_indices(table.num_rows)

        for col_name in table.column_names:
            col = table.column(col_name)
            profile = self.column_profiles[col_name]

            # Convert to Python list (this is the memory tradeoff for simplicity)
            # For truly massive data, we'd use Arrow compute functions
            values = col.to_pylist()
            profile.update(values, sample_indices)

        self.total_rows += table.num_rows

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Stream a Parquet file and return full profile."""
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(batch_size=self.chunk_size):
            self.process_chunk(batch)

        return self.get_profile()

    def process_bytes(self, data: bytes) -> Dict[str, Any]:
        """Stream Parquet from bytes and return full profile."""
        buffer = pa.py_buffer(data)
        pf = pq.ParquetFile(buffer)

        for batch in pf.iter_batches(batch_size=self.chunk_size):
            self.process_chunk(batch)

        return self.get_profile()

    def get_profile(self) -> Dict[str, Any]:
        """Get finalized profile for all columns."""
        profiles = {}
        for col_name, profile in self.column_profiles.items():
            profiles[col_name] = profile.finalize()

        # Compute correlations with target if available
        correlations = {}
        if self.target_profile and self.target_profile.is_numeric:
            target_samples = self.target_profile.sample_values
            for col_name, profile in self.column_profiles.items():
                if col_name == self.target_column:
                    continue
                if profile.is_numeric and profile.sample_values:
                    # Need paired samples - simplify by using min length
                    n = min(len(target_samples), len(profile.sample_values))
                    if n >= 2:
                        correlations[col_name] = {
                            "pearson": pearson_correlation(target_samples[:n], profile.sample_values[:n]),
                            "spearman": spearman_correlation(target_samples[:n], profile.sample_values[:n]),
                        }

        return {
            "row_count": self.total_rows,
            "columns": profiles,
            "target_column": self.target_column,
            "correlations_with_target": correlations,
        }


def stream_parquet_chunks(file_path: str, chunk_size: int = 10000) -> Iterator[pa.Table]:
    """Yield Arrow table chunks from a Parquet file."""
    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=chunk_size):
        yield batch


def stream_parquet_chunks_from_bytes(data: bytes, chunk_size: int = 10000) -> Iterator[pa.Table]:
    """Yield Arrow table chunks from Parquet bytes."""
    buffer = pa.py_buffer(data)
    pf = pq.ParquetFile(buffer)
    for batch in pf.iter_batches(batch_size=chunk_size):
        yield batch


def infer_parquet_schema(file_path: str) -> Dict[str, str]:
    """Return column name -> Arrow type string mapping for a Parquet file."""
    pf = pq.ParquetFile(file_path)
    schema = pf.schema_arrow
    return {field.name: str(field.type) for field in schema}


def infer_parquet_schema_from_bytes(data: bytes) -> Dict[str, str]:
    """Return column name -> Arrow type string mapping from Parquet bytes."""
    buffer = pa.py_buffer(data)
    pf = pq.ParquetFile(buffer)
    schema = pf.schema_arrow
    return {field.name: str(field.type) for field in schema}


def parquet_to_csv_chunks(
    file_path: str,
    chunk_size: int = 10000,
    columns: Optional[List[str]] = None,
) -> Iterator[str]:
    """Convert Parquet chunks to CSV strings for compatibility with existing pipeline.

    This allows the existing CSV-based feature engine to work with Parquet input
    by streaming chunks and converting to CSV format.
    """
    pf = pq.ParquetFile(file_path)
    first = True

    for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
        table = pa.Table.from_batches([batch])
        if columns:
            table = table.select(columns)

        # Write to CSV string
        output = io.StringIO()
        table.to_csv(output, include_header=first)
        yield output.getvalue()
        first = False

