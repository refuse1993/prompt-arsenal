"""
윤리적 사용 검증 시스템 (Ethics Validator)
악의적 사용 방지 및 연구 목적 확인
"""

from typing import Dict, Optional
from datetime import datetime
import json
import os


class EthicsValidator:
    """
    윤리적 사용 검증

    목적:
    - 연구 목적 확인
    - 동의 확인 (Deepfake, Voice Cloning)
    - 사용 로그 기록
    - 악의적 사용 감지 및 차단
    """

    def __init__(self, log_file: str = 'logs/ethics_log.json'):
        """
        Args:
            log_file: 윤리 검증 로그 파일 경로
        """
        self.log_file = log_file
        self.ensure_log_dir()

        # 고위험 작업 목록
        self.high_risk_operations = [
            'deepfake',
            'voice_clone',
            'face_swap',
            'full_multimedia',
            'universal_perturbation'
        ]

        # 동의 필요 작업
        self.consent_required_operations = [
            'deepfake',
            'voice_clone',
            'face_swap',
            'lip_sync',
            'full_multimedia'
        ]

    def ensure_log_dir(self):
        """로그 디렉토리 생성"""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def validate_usage(self, operation: str, target: str = None,
                      user_consent: bool = False,
                      research_purpose: str = None) -> Dict:
        """
        사용 목적 검증

        Args:
            operation: 작업 유형 ('deepfake', 'voice_clone' 등)
            target: 타겟 (선택적)
            user_consent: 사용자 동의 여부
            research_purpose: 연구 목적 (선택적)

        Returns:
            {
                'allowed': bool,
                'reason': str,
                'warning': str (선택적),
                'consent_required': bool
            }
        """
        result = {
            'allowed': True,
            'reason': 'Operation permitted for security research',
            'warning': None,
            'consent_required': False
        }

        # 1. 고위험 작업 체크
        if operation in self.high_risk_operations:
            result['warning'] = f"⚠️ High-risk operation: {operation}"

        # 2. 동의 필요 작업 체크
        if operation in self.consent_required_operations:
            result['consent_required'] = True

            if not user_consent:
                result['allowed'] = False
                result['reason'] = f"User consent required for {operation}"
                return result

        # 3. 연구 목적 체크 (권장)
        if operation in self.high_risk_operations and not research_purpose:
            result['warning'] = "⚠️ Research purpose not specified. Please provide justification."

        # 4. 로그 기록
        self.log_usage(operation, target, user_consent, research_purpose, result['allowed'])

        return result

    def require_consent(self, operation: str, details: str = None) -> bool:
        """
        사용자 동의 요청 (콘솔 프롬프트)

        Args:
            operation: 작업 유형
            details: 세부 정보

        Returns:
            동의 여부 (True/False)
        """
        print("\n" + "=" * 60)
        print(f"⚠️  ETHICS CONSENT REQUIRED")
        print("=" * 60)
        print(f"Operation: {operation}")

        if details:
            print(f"Details: {details}")

        print("\nThis operation involves:")
        if operation in ['deepfake', 'face_swap']:
            print("  - Facial manipulation (Deepfake)")
            print("  - Potential identity impersonation")
        elif operation in ['voice_clone']:
            print("  - Voice cloning and synthesis")
            print("  - Potential speaker impersonation")
        elif operation in ['full_multimedia']:
            print("  - Multi-modal manipulation (Image + Audio + Video)")
            print("  - Advanced adversarial attack generation")

        print("\n📜 TERMS OF USE:")
        print("  1. This tool is for DEFENSIVE SECURITY RESEARCH ONLY")
        print("  2. You must have proper authorization for all targets")
        print("  3. Generated content must NOT be used maliciously")
        print("  4. You agree to responsible disclosure practices")
        print("  5. All usage is logged for security purposes")

        print("\n" + "=" * 60)

        # 실제 프로덕션에서는 더 강력한 검증 필요
        # 여기서는 간단한 yes/no 프롬프트
        try:
            response = input("Do you agree to these terms? (yes/no): ").strip().lower()
            consent = response in ['yes', 'y']

            if consent:
                print("✓ Consent granted. Proceeding with operation...")
            else:
                print("✗ Consent denied. Operation cancelled.")

            return consent

        except (KeyboardInterrupt, EOFError):
            print("\n✗ Operation cancelled by user.")
            return False

    def log_usage(self, operation: str, target: Optional[str],
                 consent: bool, purpose: Optional[str], allowed: bool):
        """
        사용 로그 기록

        Args:
            operation: 작업 유형
            target: 타겟
            consent: 동의 여부
            purpose: 연구 목적
            allowed: 허용 여부
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'target': target,
            'consent': consent,
            'purpose': purpose,
            'allowed': allowed
        }

        # JSON 로그 파일에 추가
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            # 로그 실패는 작업을 차단하지 않음
            print(f"Warning: Failed to log usage: {e}")

    def get_usage_stats(self) -> Dict:
        """
        사용 통계 조회

        Returns:
            {
                'total_operations': int,
                'high_risk_operations': int,
                'consent_granted': int,
                'consent_denied': int,
                'operations_by_type': dict
            }
        """
        if not os.path.exists(self.log_file):
            return {
                'total_operations': 0,
                'high_risk_operations': 0,
                'consent_granted': 0,
                'consent_denied': 0,
                'operations_by_type': {}
            }

        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)

            stats = {
                'total_operations': len(logs),
                'high_risk_operations': sum(1 for log in logs if log['operation'] in self.high_risk_operations),
                'consent_granted': sum(1 for log in logs if log.get('consent')),
                'consent_denied': sum(1 for log in logs if not log.get('consent') and log['operation'] in self.consent_required_operations),
                'operations_by_type': {}
            }

            # 작업 유형별 통계
            for log in logs:
                op = log['operation']
                if op not in stats['operations_by_type']:
                    stats['operations_by_type'][op] = 0
                stats['operations_by_type'][op] += 1

            return stats

        except Exception as e:
            print(f"Error loading usage stats: {e}")
            return {}


class ResearchPurposeValidator:
    """
    연구 목적 검증 (추가 레이어)

    실제 프로덕션에서는:
    - 기관 IRB (Institutional Review Board) 승인 확인
    - 연구 프로토콜 검증
    - 데이터 사용 동의서 확인
    """

    @staticmethod
    def validate_research_purpose(purpose: str) -> bool:
        """
        연구 목적 검증

        Args:
            purpose: 연구 목적 설명

        Returns:
            유효 여부
        """
        if not purpose or len(purpose.strip()) < 10:
            return False

        # 악의적 키워드 체크 (간단한 예시)
        malicious_keywords = [
            'hack', 'steal', 'fraud', 'scam', 'illegal',
            'blackmail', 'extortion', 'impersonation'
        ]

        purpose_lower = purpose.lower()
        for keyword in malicious_keywords:
            if keyword in purpose_lower:
                return False

        return True


# ========================================
# 고수준 API
# ========================================

def check_ethics(operation: str, require_consent: bool = True) -> bool:
    """
    간편한 윤리 검증 API

    Args:
        operation: 작업 유형
        require_consent: 동의 요청 여부

    Returns:
        허용 여부 (True/False)
    """
    validator = EthicsValidator()

    # 동의 필요한 작업인 경우
    if require_consent and operation in validator.consent_required_operations:
        consent = validator.require_consent(operation)
        if not consent:
            return False

    # 검증
    result = validator.validate_usage(operation, user_consent=True if require_consent else False)

    return result['allowed']


if __name__ == "__main__":
    # 테스트
    print("Ethics Validation System Test")
    print("=" * 60)

    validator = EthicsValidator()

    # 일반 작업 검증
    result1 = validator.validate_usage('fgsm', user_consent=False)
    print(f"\n1. FGSM Attack: {result1}")

    # 고위험 작업 검증
    result2 = validator.validate_usage('deepfake', user_consent=False)
    print(f"\n2. Deepfake (no consent): {result2}")

    # 동의 포함 검증
    result3 = validator.validate_usage('deepfake', user_consent=True, research_purpose="AI security research")
    print(f"\n3. Deepfake (with consent): {result3}")

    # 사용 통계
    stats = validator.get_usage_stats()
    print(f"\n4. Usage Stats: {json.dumps(stats, indent=2)}")

    print("\n" + "=" * 60)
    print("Ethics Validation System ready!")
