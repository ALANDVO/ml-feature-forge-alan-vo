"""Tests for streaming Parquet ingestion with Apache Arrow."""

import io
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.app.services.streaming_ingestion import (
    StreamingProfiler,
    stream_parquet_chunks,
    stream_parquet_chunks_from_bytes,
    infer_parquet_schema,
    infer_parquet_schema_from_bytes,
    parquet_to_csv_chunks,
    ColumnProfile,
)


def create_test_parquet(num_rows: int = 1000) -> bytes:
    """Create a test Parquet file in memory with numeric and categorical columns."""
    import random

    random.seed(42)
    data = {
        "id": list(range(num_rows)),
        "age": [random.randint(18, 80) for _ in range(num_rows)],
        "income": [random.uniform(20000, 200000) for _ in range(num_rows)],
        "credit_score": [random.randint(300, 850) for _ in range(num_rows)],
        "loan_amount": [random.uniform(5000, 50000) for _ in range(num_rows)],
        "debt_to_income": [random.uniform(0.1, 0.5) for _ in range(num_rows)],
        "months_employed": [random.randint(0, 360) for _ in range(num_rows)],
        "city_tier": [random.choice(["tier_1", "tier_2", "tier_3"]) for _ in range(num_rows)],
        "approved": [random.randint(0, 1) for _ in range(num_rows)],
    }
    table = pa.table(data)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def create_parquet_file(data: dict, path: str) -> None:
    """Write a dictionary as a Parquet file."""
    table = pa.table(data)
    pq.write_table(table, path)


class TestColumnProfile:
    """Tests for the ColumnProfile running statistics class."""

    def test_numeric_column_profile(self):
        profile = ColumnProfile("test_numeric")
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        profile.update(values)
        result = profile.finalize()

        assert result["name"] == "test_numeric"
        assert result["count"] == 5
        assert result["null_count"] == 0
        assert result["is_numeric"] is True
        assert abs(result["mean"] - 3.0) < 0.001
        # Population std dev (dividing by n): sqrt(2.0) ≈ 1.414
        assert abs(result["std"] - 1.414) < 0.01
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["variance"] > 0

    def test_categorical_column_profile(self):
        profile = ColumnProfile("test_categorical")
        values = ["a", "b", "a", "c", "b", "a"]
        profile.update(values)
        result = profile.finalize()

        assert result["is_numeric"] is False
        assert result["count"] == 6
        assert result["unique_count"] == 3
        assert result["top_values"][0][0] == "a"
        assert result["top_values"][0][1] == 3

    def test_null_handling(self):
        profile = ColumnProfile("test_nulls")
        values = [1.0, None, 3.0, None, 5.0]
        profile.update(values)
        result = profile.finalize()

        assert result["count"] == 3
        assert result["null_count"] == 2
        assert result["null_ratio"] == 2 / 5

    def test_mixed_types_become_categorical(self):
        profile = ColumnProfile("test_mixed")
        profile.update([1.0, 2.0, "not_a_number", 4.0])
        result = profile.finalize()

        assert result["is_numeric"] is False


class TestStreamingProfiler:
    """Tests for the StreamingProfiler incremental profiler."""

    def test_profile_from_bytes(self):
        parquet_bytes = create_test_parquet(5000)
        profiler = StreamingProfiler(target_column="approved", chunk_size=1000)
        profile = profiler.process_bytes(parquet_bytes)

        assert profile["row_count"] == 5000
        assert "columns" in profile
        assert "approved" in profile["columns"]
        assert profile["target_column"] == "approved"
        assert "correlations_with_target" in profile

    def test_profile_from_file(self):
        parquet_bytes = create_test_parquet(5000)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(parquet_bytes)
            temp_path = f.name

        try:
            profiler = StreamingProfiler(target_column="approved", chunk_size=1000)
            profile = profiler.process_file(temp_path)

            assert profile["row_count"] == 5000
            assert "columns" in profile
        finally:
            import os
            os.unlink(temp_path)

    def test_chunked_processing_consistency(self):
        """Verify that chunked processing produces same results as full load."""
        parquet_bytes = create_test_parquet(10000)

        # Process in chunks of 1000
        profiler_chunked = StreamingProfiler(target_column="approved", chunk_size=1000)
        profile_chunked = profiler_chunked.process_bytes(parquet_bytes)

        # Process in one large chunk
        profiler_full = StreamingProfiler(target_column="approved", chunk_size=10000)
        profile_full = profiler_full.process_bytes(parquet_bytes)

        # Row counts should match
        assert profile_chunked["row_count"] == profile_full["row_count"] == 10000

        # Column stats should be very close (allow small numerical differences)
        for col_name in profile_chunked["columns"]:
            chunked_col = profile_chunked["columns"][col_name]
            full_col = profile_full["columns"][col_name]

            if chunked_col["is_numeric"]:
                assert abs(chunked_col["mean"] - full_col["mean"]) < 0.01
                assert abs(chunked_col["std"] - full_col["std"]) < 0.01
                assert chunked_col["count"] == full_col["count"]

    def test_target_column_correlation(self):
        parquet_bytes = create_test_parquet(2000)
        profiler = StreamingProfiler(target_column="approved", chunk_size=500)
        profile = profiler.process_bytes(parquet_bytes)

        # Target column should be numeric (0/1)
        assert profile["columns"]["approved"]["is_numeric"] is True

        # Should have correlations with target
        correlations = profile["correlations_with_target"]
        assert isinstance(correlations, dict)

        # Numeric columns should have correlations computed
        for col_name in ["age", "income", "credit_score", "loan_amount", "debt_to_income", "months_employed"]:
            if col_name in correlations:
                assert "pearson" in correlations[col_name]
                assert "spearman" in correlations[col_name]
                assert -1 <= correlations[col_name]["pearson"] <= 1
                assert -1 <= correlations[col_name]["spearman"] <= 1


class TestStreamParquetChunks:
    """Tests for the parquet chunk streaming functions."""

    def test_stream_parquet_chunks_from_bytes(self):
        parquet_bytes = create_test_parquet(1000)
        chunks = list(stream_parquet_chunks_from_bytes(parquet_bytes, chunk_size=250))

        assert len(chunks) == 4
        total_rows = sum(chunk.num_rows for chunk in chunks)
        assert total_rows == 1000

    def test_stream_parquet_chunks_from_file(self):
        parquet_bytes = create_test_parquet(1000)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(parquet_bytes)
            temp_path = f.name

        try:
            chunks = list(stream_parquet_chunks(temp_path, chunk_size=250))

            assert len(chunks) == 4
            total_rows = sum(chunk.num_rows for chunk in chunks)
            assert total_rows == 1000
        finally:
            import os
            os.unlink(temp_path)


class TestSchemaInference:
    """Tests for Parquet schema inference."""

    def test_infer_parquet_schema_from_bytes(self):
        parquet_bytes = create_test_parquet(100)
        schema = infer_parquet_schema_from_bytes(parquet_bytes)

        assert isinstance(schema, dict)
        assert "id" in schema
        assert "age" in schema
        assert "income" in schema
        assert "city_tier" in schema
        assert "approved" in schema

    def test_infer_parquet_schema_from_file(self):
        parquet_bytes = create_test_parquet(100)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(parquet_bytes)
            temp_path = f.name

        try:
            schema = infer_parquet_schema(temp_path)

            assert isinstance(schema, dict)
            assert "id" in schema
            assert "approved" in schema
        finally:
            import os
            os.unlink(temp_path)


class TestParquetToCsvChunks:
    """Tests for Parquet to CSV chunk conversion."""

    def test_parquet_to_csv_chunks(self):
        parquet_bytes = create_test_parquet(1000)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(parquet_bytes)
            temp_path = f.name

        try:
            chunks = list(parquet_to_csv_chunks(temp_path, chunk_size=250))

            assert len(chunks) == 4
            # First chunk should have header (quoted by default)
            assert '"id","age","income"' in chunks[0]
            # Subsequent chunks should not have header
            assert '"id","age","income"' not in chunks[1]

            # Verify total rows (header + data rows) - allow small variation due to newline handling
            total_data_rows = sum(chunk.count("\n") - 1 for chunk in chunks)
            assert abs(total_data_rows - 1000) <= 3
        finally:
            import os
            os.unlink(temp_path)

    def test_parquet_to_csv_chunks_with_columns(self):
        parquet_bytes = create_test_parquet(1000)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            f.write(parquet_bytes)
            temp_path = f.name

        try:
            chunks = list(parquet_to_csv_chunks(temp_path, chunk_size=500, columns=["age", "income", "approved"]))

            assert len(chunks) == 2
            # Should only have selected columns
            assert '"age","income","approved"' in chunks[0]
            assert "credit_score" not in chunks[0]
        finally:
            import os
            os.unlink(temp_path)


class TestLargeDatasetStreaming:
    """Tests for streaming large datasets that don't fit in memory."""

    def test_large_dataset_memory_efficiency(self):
        """Test that we can profile a large dataset in chunks without OOM."""
        # Create a moderately large dataset (100k rows)
        parquet_bytes = create_test_parquet(100000)

        profiler = StreamingProfiler(target_column="approved", chunk_size=10000, reservoir_size=5000)
        profile = profiler.process_bytes(parquet_bytes)

        assert profile["row_count"] == 100000
        assert profile["columns"]["approved"]["count"] > 0
        assert len(profile["correlations_with_target"]) > 0

    def test_zero_row_chunk_handling(self):
        """Test that empty chunks are handled gracefully."""
        # Create parquet with 0 rows
        table = pa.table({"id": [], "value": []})
        buffer = io.BytesIO()
        pq.write_table(table, buffer)

        profiler = StreamingProfiler(target_column="value", chunk_size=1000)
        # This should not raise
        profile = profiler.process_bytes(buffer.getvalue())

        assert profile["row_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

