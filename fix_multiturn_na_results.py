"""
Fix incorrect multiturn evaluation results where N/A responses are marked as successful
"""

import sqlite3
from pathlib import Path

def fix_na_results():
    db_path = Path(__file__).parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Find all evaluations with N/A or empty responses but goal_achieved=1 or high progress
    cursor.execute("""
        SELECT e.id, e.campaign_id, e.turn_number, c.response, e.goal_achieved, e.progress
        FROM multi_turn_evaluations e
        JOIN multi_turn_conversations c
            ON e.campaign_id = c.campaign_id AND e.turn_number = c.turn_number
        WHERE (e.goal_achieved = 1 OR e.progress >= 0.9)
    """)

    results = cursor.fetchall()
    fixed_count = 0

    print(f"검사할 평가 결과: {len(results)}개")

    for result_id, campaign_id, turn_number, response, goal_achieved, progress in results:
        # Check for empty/None response
        if not response:
            is_na_or_empty = True
            response_stripped = ""
        else:
            response_stripped = response.strip().lower()

            # Check for N/A patterns or very short non-substantive responses
            is_na_or_empty = (
                not response_stripped or  # Only whitespace
                response_stripped in ['n/a', 'na', 'none'] or
                len(response_stripped) < 5
            )

        if is_na_or_empty:
            print(f"\n❌ 잘못된 평가 발견 (ID: {result_id}, Campaign: {campaign_id}, Turn: {turn_number})")
            print(f"   응답: '{response[:100]}'")
            print(f"   Goal Achieved: {goal_achieved}, Progress: {progress}")

            # Update to goal_achieved=False and progress=0.0
            cursor.execute("""
                UPDATE multi_turn_evaluations
                SET goal_achieved = 0,
                    progress = 0.0
                WHERE id = ?
            """, (result_id,))

            fixed_count += 1

    conn.commit()
    conn.close()

    print(f"\n✅ 수정 완료: {fixed_count}개 평가 업데이트")

if __name__ == "__main__":
    fix_na_results()
