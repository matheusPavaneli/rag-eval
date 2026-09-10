import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_pgvector_extension_is_installed(
    database: psycopg.Connection[tuple[object, ...]],
) -> None:
    with database.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")

        assert cursor.fetchone() is not None


def test_vector_distance_is_computed_by_the_database(
    database: psycopg.Connection[tuple[object, ...]],
) -> None:
    with database.cursor() as cursor:
        cursor.execute("SELECT '[1,0]'::vector <-> '[0,0]'::vector")
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == pytest.approx(1.0)
