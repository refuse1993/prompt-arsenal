"""
Update Existing Prompts with Classification
기존 프롬프트에 분류 정보 추가
"""

import sqlite3
from pathlib import Path


# 기존 카테고리 → 새로운 분류 매핑
CLASSIFICATION_MAP = {
    # Offensive: Jailbreak
    "jailbreak": {
        "purpose": "offensive",
        "risk_category": "security",
        "technique": "jailbreak",
        "modality": "text_only"
    },

    # Offensive: Prompt Injection
    "prompt_injection": {
        "purpose": "offensive",
        "risk_category": "security",
        "technique": "prompt_injection",
        "modality": "text_only"
    },

    # Offensive: Adversarial
    "adversarial": {
        "purpose": "offensive",
        "risk_category": "safety",
        "technique": "adversarial",
        "modality": "text_only"
    },
    "harmful_content": {
        "purpose": "offensive",
        "risk_category": "safety",
        "technique": "adversarial",
        "modality": "text_only"
    },

    # Offensive: Fuzzing
    "fuzzing": {
        "purpose": "offensive",
        "risk_category": "security",
        "technique": "fuzzing",
        "modality": "text_only"
    },

    # Defensive: Security
    "malicious_use": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "robustness_test",
        "modality": "text_only"
    },
    "information_hazard": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "robustness_test",
        "modality": "text_only"
    },
    "illegal_activity": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "malware_generation": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "fraud": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "privacy_violation": {
        "purpose": "defensive",
        "risk_category": "security",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },

    # Defensive: Safety
    "physical_harm": {
        "purpose": "defensive",
        "risk_category": "safety",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "economic_harm": {
        "purpose": "defensive",
        "risk_category": "safety",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "health_consultation": {
        "purpose": "defensive",
        "risk_category": "safety",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },

    # Defensive: Ethics
    "toxic_content": {
        "purpose": "defensive",
        "risk_category": "ethics",
        "technique": "robustness_test",
        "modality": "text_only"
    },
    "profanity": {
        "purpose": "defensive",
        "risk_category": "ethics",
        "technique": "content_filter_test",
        "modality": "text_only"
    },
    "offensive": {
        "purpose": "defensive",
        "risk_category": "ethics",
        "technique": "content_filter_test",
        "modality": "text_only"
    },
    "hate_speech": {
        "purpose": "defensive",
        "risk_category": "ethics",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },

    # Defensive: Misinformation
    "misinformation": {
        "purpose": "defensive",
        "risk_category": "misinformation",
        "technique": "robustness_test",
        "modality": "text_only"
    },
    "human_impersonation": {
        "purpose": "defensive",
        "risk_category": "misinformation",
        "technique": "robustness_test",
        "modality": "text_only"
    },

    # Defensive: Compliance
    "sexual_content": {
        "purpose": "defensive",
        "risk_category": "compliance",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "political_lobbying": {
        "purpose": "defensive",
        "risk_category": "compliance",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "legal_opinion": {
        "purpose": "defensive",
        "risk_category": "compliance",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "financial_advice": {
        "purpose": "defensive",
        "risk_category": "compliance",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    },
    "gov_decision": {
        "purpose": "defensive",
        "risk_category": "compliance",
        "technique": "safety_benchmark",
        "modality": "multimodal"
    }
}


def update_existing_prompts():
    """기존 프롬프트에 분류 정보 업데이트"""
    project_root = Path(__file__).parent.parent
    db_path = str(project_root / "arsenal.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("🔄 Updating existing prompts with classification...")

    updated = 0
    skipped = 0

    # 모든 프롬프트 가져오기
    cursor.execute("SELECT id, category FROM prompts")
    prompts = cursor.fetchall()

    for prompt_id, category in prompts:
        if category in CLASSIFICATION_MAP:
            classification = CLASSIFICATION_MAP[category]

            cursor.execute('''
                UPDATE prompts
                SET purpose = ?, risk_category = ?, technique = ?, modality = ?
                WHERE id = ?
            ''', (
                classification['purpose'],
                classification['risk_category'],
                classification['technique'],
                classification['modality'],
                prompt_id
            ))
            updated += 1
        else:
            # Unknown category - 기본값 유지
            skipped += 1

    conn.commit()
    conn.close()

    print(f"\n✅ Update complete!")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")

    # 통계 출력
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n📊 Updated Statistics:")

    print(f"\n🎯 Purpose:")
    cursor.execute('SELECT purpose, COUNT(*) FROM prompts GROUP BY purpose')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print(f"\n⚠️  Risk Category:")
    cursor.execute('SELECT risk_category, COUNT(*) FROM prompts GROUP BY risk_category')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print(f"\n🔧 Technique:")
    cursor.execute('SELECT technique, COUNT(*) FROM prompts GROUP BY technique')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print(f"\n📱 Modality:")
    cursor.execute('SELECT modality, COUNT(*) FROM prompts GROUP BY modality')
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()


if __name__ == "__main__":
    update_existing_prompts()
