import sqlite3
import json

DB_FILE = "triage.db"

# --- Common Utilities ---

def find_matching_slots(resource_type, resource_id, preferred_start):
    """Find the preferred slot and up to 5 alternative available slots."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Preferred slot
        c.execute(
            """
            SELECT * FROM availability
            WHERE resource_type=? AND resource_id=? AND slot_start=?
            """,
            (resource_type, resource_id, preferred_start),
        )
        preferred = c.fetchone()

        # Alternatives (excluding preferred)
        c.execute(
            """
            SELECT * FROM availability
            WHERE resource_type=? AND resource_id=? AND is_available=1
              AND slot_start <> ?
            ORDER BY slot_start ASC
            LIMIT 5
            """,
            (resource_type, resource_id, preferred_start),
        )
        alternatives = c.fetchall()

    return preferred, alternatives


def suggest_alternatives_json(alts):
    """Convert alternative slots into JSON format."""
    if not alts:
        return "[]"
    return json.dumps([dict(r) for r in alts], indent=2)


def book_slot(resource_type, resource_id, slot_start):
    """Book a slot by marking it unavailable. Returns True if successful."""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            """
            UPDATE availability
            SET is_available=0
            WHERE resource_type=? AND resource_id=? AND slot_start=? AND is_available=1
            """,
            (resource_type, resource_id, slot_start),
        )
        ok = (c.rowcount == 1)
        conn.commit()
    return ok