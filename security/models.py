"""
Security Scanner Data Models
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Finding:
    """보안 취약점 발견 사항"""

    # CWE 정보
    cwe_id: str  # 'CWE-89', 'CWE-79', etc.
    cwe_name: str  # 'SQL Injection', 'XSS', etc.
    severity: str  # 'Critical', 'High', 'Medium', 'Low'
    confidence: float  # 0.0 - 1.0

    # 위치 정보
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None

    # 상세 정보
    title: str = ""
    description: str = ""
    attack_scenario: str = ""
    remediation: str = ""
    remediation_code: str = ""  # 개선된 코드 예시

    # 증거
    code_snippet: str = ""
    context_code: str = ""  # 주변 코드 (LLM 검증용)
    payload: Optional[str] = None  # 동적 분석 시 사용된 페이로드
    response_evidence: Optional[str] = None  # 동적 분석 응답

    # 검증 정보
    verified_by: str = "tool"  # 'semgrep', 'bandit', 'ruff', 'llm', 'llm+tool'
    is_false_positive: bool = False
    llm_reasoning: Optional[str] = None

    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (DB 저장용)"""
        return {
            'cwe_id': self.cwe_id,
            'cwe_name': self.cwe_name,
            'severity': self.severity,
            'confidence': self.confidence,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column_number': self.column_number,
            'function_name': self.function_name,
            'title': self.title,
            'description': self.description,
            'attack_scenario': self.attack_scenario,
            'remediation': self.remediation,
            'remediation_code': self.remediation_code,
            'code_snippet': self.code_snippet,
            'payload': self.payload,
            'response_evidence': self.response_evidence,
            'verified_by': self.verified_by,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class SecurityReport:
    """보안 스캔 리포트"""

    findings: List[Finding] = field(default_factory=list)
    scan_id: Optional[int] = None
    scan_type: str = "static"  # 'static', 'dynamic'
    target: str = ""
    mode: str = "rule_only"  # 'rule_only', 'verify_with_llm', 'llm_detect', 'hybrid'

    # 통계
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # 성능
    scan_duration: float = 0.0  # seconds
    llm_calls: int = 0
    llm_cost: float = 0.0  # USD

    # 검증 통계
    tool_only_confirmed: int = 0  # 도구만으로 확정
    llm_verified: int = 0  # LLM 검증 필요
    false_positives_removed: int = 0  # 오탐 제거

    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def calculate_stats(self):
        """통계 재계산"""
        # False positive 제외한 실제 발견사항만 카운트
        valid_findings = [f for f in self.findings if not f.is_false_positive]

        self.total_findings = len(valid_findings)

        self.critical_count = sum(1 for f in valid_findings if f.severity == 'Critical')
        self.high_count = sum(1 for f in valid_findings if f.severity == 'High')
        self.medium_count = sum(1 for f in valid_findings if f.severity == 'Medium')
        self.low_count = sum(1 for f in valid_findings if f.severity == 'Low')

        # 검증 통계
        self.tool_only_confirmed = sum(
            1 for f in valid_findings
            if f.verified_by in ['semgrep', 'bandit', 'ruff'] and f.confidence >= 0.8
        )

        self.llm_verified = sum(
            1 for f in valid_findings
            if 'llm' in f.verified_by
        )

        # False positive 제거 통계
        self.false_positives_removed = sum(1 for f in self.findings if f.is_false_positive)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        self.calculate_stats()

        return {
            'scan_id': self.scan_id,
            'scan_type': self.scan_type,
            'target': self.target,
            'mode': self.mode,
            'total_findings': self.total_findings,
            'critical_count': self.critical_count,
            'high_count': self.high_count,
            'medium_count': self.medium_count,
            'low_count': self.low_count,
            'scan_duration': self.scan_duration,
            'llm_calls': self.llm_calls,
            'llm_cost': self.llm_cost,
            'tool_only_confirmed': self.tool_only_confirmed,
            'llm_verified': self.llm_verified,
            'false_positives_removed': self.false_positives_removed,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class ScanConfig:
    """스캔 설정"""

    target: str  # 파일 경로 or 디렉토리
    mode: str = "hybrid"  # 'rule_only', 'verify_with_llm', 'llm_detect', 'hybrid'
    profile_name: Optional[str] = None  # API 프로필 (LLM 사용 시)

    # 도구 선택
    use_semgrep: bool = True
    use_bandit: bool = True
    use_ruff: bool = True

    # LLM 설정
    llm_confidence_threshold: float = 0.8  # 이 값 미만이면 LLM 검증
    enable_llm_verification: bool = True  # True면 LLM 사용 (mode에 따라)

    # CWE 필터
    cwe_categories: Optional[List[str]] = None  # None이면 전체

    # 출력 설정
    output_format: str = "rich"  # 'rich', 'json', 'html', 'sarif'
    output_file: Optional[str] = None
