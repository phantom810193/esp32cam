import sqlite3
from pathlib import Path


def main() -> None:
    db_path = Path("mvp.sql")
    sql_files = sorted(Path.cwd().glob("*.sql"))

    if not sql_files:
        print("No SQL files found in the current directory.")
        return

    with sqlite3.connect(db_path) as conn:
        for sql_file in sql_files:
            statements = sql_file.read_text(encoding="utf-8")
            conn.executescript(statements)
            conn.commit()
            print(f"Successfully executed {sql_file.name}.")


if __name__ == "__main__":
    main()
