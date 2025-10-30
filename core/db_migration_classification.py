"""
Database Migration: Add Classification Fields
purpose, risk_category, technique, modality 필드 추가
"""

import sqlite3
from pathlib import Path


def migrate():
    """Add classification fields to prompts and media_arsenal tables"""
    project_root = Path(__file__).parent.parent
    db_path = str(project_root / "arsenal.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("🔄 Starting classification migration...")

    # Check and add columns to prompts table
    cursor.execute("PRAGMA table_info(prompts)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    migrations_applied = 0

    if 'purpose' not in existing_columns:
        cursor.execute("ALTER TABLE prompts ADD COLUMN purpose TEXT DEFAULT 'defensive'")
        migrations_applied += 1
        print("  ✓ Added 'purpose' column to prompts")

    if 'risk_category' not in existing_columns:
        cursor.execute("ALTER TABLE prompts ADD COLUMN risk_category TEXT")
        migrations_applied += 1
        print("  ✓ Added 'risk_category' column to prompts")

    if 'technique' not in existing_columns:
        cursor.execute("ALTER TABLE prompts ADD COLUMN technique TEXT")
        migrations_applied += 1
        print("  ✓ Added 'technique' column to prompts")

    if 'modality' not in existing_columns:
        cursor.execute("ALTER TABLE prompts ADD COLUMN modality TEXT DEFAULT 'text_only'")
        migrations_applied += 1
        print("  ✓ Added 'modality' column to prompts")

    # Check and add columns to media_arsenal table
    cursor.execute("PRAGMA table_info(media_arsenal)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    if 'purpose' not in existing_columns:
        cursor.execute("ALTER TABLE media_arsenal ADD COLUMN purpose TEXT DEFAULT 'offensive'")
        migrations_applied += 1
        print("  ✓ Added 'purpose' column to media_arsenal")

    if 'risk_category' not in existing_columns:
        cursor.execute("ALTER TABLE media_arsenal ADD COLUMN risk_category TEXT")
        migrations_applied += 1
        print("  ✓ Added 'risk_category' column to media_arsenal")

    if 'modality' not in existing_columns:
        cursor.execute("ALTER TABLE media_arsenal ADD COLUMN modality TEXT DEFAULT 'multimodal'")
        migrations_applied += 1
        print("  ✓ Added 'modality' column to media_arsenal")

    conn.commit()
    conn.close()

    if migrations_applied > 0:
        print(f"\n✅ Migration complete! Applied {migrations_applied} changes.")
    else:
        print("\n✅ All columns already exist. No migration needed.")


if __name__ == "__main__":
    migrate()
