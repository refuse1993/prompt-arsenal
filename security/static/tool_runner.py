"""
Static Analysis Tool Runner
Semgrep, Bandit, Ruff 실행 및 결과 파싱
"""

import subprocess
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import Finding


class ToolRunner:
    """정적 분석 도구 실행기"""

    def __init__(self):
        self.semgrep_available = self._check_tool("semgrep")
        self.bandit_available = self._check_tool("bandit")
        self.ruff_available = self._check_tool("ruff")

    def _count_files(self, target: str) -> int:
        """스캔 대상 파일 개수 세기"""
        target_path = Path(target)

        if target_path.is_file():
            return 1

        # Python 파일만 카운트 (대부분의 도구가 Python 중심)
        py_files = list(target_path.rglob("*.py"))
        return len(py_files)

    def _check_tool(self, tool_name: str) -> bool:
        """도구가 설치되어 있는지 확인"""
        try:
            result = subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def run_semgrep(self, target: str) -> List[Finding]:
        """
        Semgrep 실행

        Args:
            target: 스캔 대상 (파일 or 디렉토리)

        Returns:
            Finding 객체 리스트
        """
        if not self.semgrep_available:
            return []

        findings = []

        try:
            # 파일 개수 세기
            file_count = self._count_files(target)
            print(f"🔍 Semgrep 스캔 시작... (약 {file_count}개 파일)")

            # Semgrep 실행
            start_time = time.time()
            result = subprocess.run(
                ["semgrep", "--config=auto", "--json", target],
                capture_output=True,
                text=True,
                timeout=600  # 10분으로 증가 (큰 프로젝트 대응)
            )
            elapsed = time.time() - start_time
            print(f"✅ Semgrep 스캔 완료 ({elapsed:.1f}초 소요)")

            # JSON 파싱
            data = json.loads(result.stdout)

            # 결과 변환
            for r in data.get('results', []):
                # CWE 추출
                metadata = r.get('extra', {}).get('metadata', {})
                cwe_list = metadata.get('cwe', [])
                cwe_id = cwe_list[0] if isinstance(cwe_list, list) and cwe_list else "CWE-Unknown"

                # 심각도 매핑
                severity = r.get('extra', {}).get('severity', 'medium').capitalize()
                if severity == 'Error':
                    severity = 'High'
                elif severity == 'Warning':
                    severity = 'Medium'
                elif severity == 'Info':
                    severity = 'Low'

                findings.append(Finding(
                    cwe_id=cwe_id,
                    cwe_name=metadata.get('owasp', ''),
                    severity=severity,
                    confidence=0.8,  # Semgrep은 높은 신뢰도
                    file_path=r['path'],
                    line_number=r['start']['line'],
                    column_number=r['start']['col'],
                    title=r['check_id'],
                    description=r['extra'].get('message', ''),
                    code_snippet=r['extra'].get('lines', ''),
                    verified_by='semgrep'
                ))

            print(f"  📊 Semgrep: {len(findings)}개 발견")

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Semgrep error: {e}")

        return findings

    async def run_bandit(self, target: str) -> List[Finding]:
        """
        Bandit 실행 (Python 전용)

        Args:
            target: 스캔 대상

        Returns:
            Finding 객체 리스트
        """
        if not self.bandit_available:
            return []

        findings = []

        try:
            # 파일 개수 세기
            file_count = self._count_files(target)
            print(f"🔍 Bandit 스캔 시작... (약 {file_count}개 파일)")

            # Bandit 실행
            start_time = time.time()
            result = subprocess.run(
                ["bandit", "-r", target, "-f", "json"],
                capture_output=True,
                text=True,
                timeout=600  # 10분으로 증가 (큰 프로젝트 대응)
            )
            elapsed = time.time() - start_time
            print(f"✅ Bandit 스캔 완료 ({elapsed:.1f}초 소요)")

            # JSON 파싱
            data = json.loads(result.stdout)

            # CWE 매핑 (Bandit test_id → CWE)
            cwe_mapping = self._get_bandit_cwe_mapping()

            # 결과 변환
            for r in data.get('results', []):
                test_id = r['test_id']
                cwe_id = cwe_mapping.get(test_id, f"CWE-{test_id}")

                # 심각도 매핑
                severity_map = {
                    'HIGH': 'Critical',
                    'MEDIUM': 'High',
                    'LOW': 'Medium'
                }
                severity = severity_map.get(r['issue_severity'], 'Medium')

                # 신뢰도 매핑
                confidence_map = {
                    'HIGH': 0.9,
                    'MEDIUM': 0.7,
                    'LOW': 0.5
                }
                confidence = confidence_map.get(r['issue_confidence'], 0.7)

                findings.append(Finding(
                    cwe_id=cwe_id,
                    cwe_name=r['test_name'],
                    severity=severity,
                    confidence=confidence,
                    file_path=r['filename'],
                    line_number=r['line_number'],
                    title=r['test_name'],
                    description=r['issue_text'],
                    code_snippet=r['code'],
                    verified_by='bandit'
                ))

            print(f"  📊 Bandit: {len(findings)}개 발견")

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Bandit error: {e}")

        return findings

    async def run_ruff(self, target: str) -> List[Finding]:
        """
        Ruff 실행 (Python 보안 규칙)

        Args:
            target: 스캔 대상

        Returns:
            Finding 객체 리스트
        """
        if not self.ruff_available:
            return []

        findings = []

        try:
            # 파일 개수 세기
            file_count = self._count_files(target)
            print(f"🔍 Ruff 스캔 시작... (약 {file_count}개 파일)")

            # Ruff 실행 (보안 규칙만: --select S)
            start_time = time.time()
            result = subprocess.run(
                ["ruff", "check", "--select", "S", "--output-format", "json", target],
                capture_output=True,
                text=True,
                timeout=120  # 2분으로 증가 (큰 프로젝트 대응)
            )
            elapsed = time.time() - start_time
            print(f"✅ Ruff 스캔 완료 ({elapsed:.1f}초 소요)")

            # JSON 파싱
            data = json.loads(result.stdout)

            # 결과 변환
            for r in data:
                # Ruff 규칙 → CWE 매핑
                rule_code = r['code']
                cwe_id = self._ruff_rule_to_cwe(rule_code)

                findings.append(Finding(
                    cwe_id=cwe_id,
                    cwe_name=r['code'],
                    severity='Medium',  # Ruff는 대부분 Medium
                    confidence=0.7,
                    file_path=str(r['filename']),
                    line_number=r['location']['row'],
                    column_number=r['location']['column'],
                    title=r['code'],
                    description=r['message'],
                    code_snippet='',  # Ruff는 코드 스니펫 제공 안함
                    verified_by='ruff'
                ))

            print(f"  📊 Ruff: {len(findings)}개 발견")

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            print(f"❌ Ruff error: {e}")

        return findings

    async def run_all_tools(self, target: str) -> Dict[str, List[Finding]]:
        """
        모든 도구 병렬 실행

        Args:
            target: 스캔 대상

        Returns:
            도구별 Finding 리스트
        """
        tasks = []
        enabled_tools = []

        if self.semgrep_available:
            tasks.append(('semgrep', self.run_semgrep(target)))
            enabled_tools.append('Semgrep')

        if self.bandit_available:
            tasks.append(('bandit', self.run_bandit(target)))
            enabled_tools.append('Bandit')

        if self.ruff_available:
            tasks.append(('ruff', self.run_ruff(target)))
            enabled_tools.append('Ruff')

        # 전체 진행 상황 표시
        if tasks:
            print(f"\n📊 정적 분석 도구 실행 중... ({len(tasks)}개 도구: {', '.join(enabled_tools)})")

        # 병렬 실행
        results = {}
        if tasks:
            tool_results = await asyncio.gather(*[task[1] for task in tasks])

            for i, (tool_name, _) in enumerate(tasks):
                results[tool_name] = tool_results[i]

            # 전체 결과 요약
            total_findings = sum(len(findings) for findings in results.values())
            print(f"\n✅ 정적 분석 완료: 총 {total_findings}개 발견")

        return results

    def _get_bandit_cwe_mapping(self) -> Dict[str, str]:
        """Bandit test_id → CWE 매핑"""
        return {
            'B101': 'CWE-20',   # assert_used
            'B102': 'CWE-78',   # exec_used
            'B103': 'CWE-78',   # set_bad_file_permissions
            'B104': 'CWE-200',  # hardcoded_bind_all_interfaces
            'B105': 'CWE-798',  # hardcoded_password_string
            'B106': 'CWE-798',  # hardcoded_password_funcarg
            'B107': 'CWE-798',  # hardcoded_password_default
            'B108': 'CWE-22',   # hardcoded_tmp_directory
            'B110': 'CWE-703',  # try_except_pass
            'B112': 'CWE-703',  # try_except_continue
            'B201': 'CWE-502',  # flask_debug_true
            'B301': 'CWE-502',  # pickle
            'B302': 'CWE-327',  # marshal
            'B303': 'CWE-327',  # md5
            'B304': 'CWE-327',  # des
            'B305': 'CWE-327',  # blowfish
            'B306': 'CWE-327',  # mktemp_q
            'B307': 'CWE-502',  # eval
            'B308': 'CWE-377',  # mark_safe
            'B310': 'CWE-22',   # urllib_urlopen
            'B311': 'CWE-330',  # random
            'B312': 'CWE-78',   # telnetlib
            'B313': 'CWE-327',  # xml_bad_cElementTree
            'B314': 'CWE-327',  # xml_bad_ElementTree
            'B315': 'CWE-327',  # xml_bad_expatreader
            'B316': 'CWE-327',  # xml_bad_expatbuilder
            'B317': 'CWE-327',  # xml_bad_sax
            'B318': 'CWE-327',  # xml_bad_minidom
            'B319': 'CWE-327',  # xml_bad_pulldom
            'B320': 'CWE-327',  # xml_bad_etree
            'B321': 'CWE-78',   # ftplib
            'B323': 'CWE-502',  # unverified_context
            'B324': 'CWE-327',  # hashlib_new_insecure_functions
            'B325': 'CWE-327',  # tempnam
            'B401': 'CWE-20',   # import_telnetlib
            'B402': 'CWE-20',   # import_ftplib
            'B403': 'CWE-502',  # import_pickle
            'B404': 'CWE-78',   # import_subprocess
            'B405': 'CWE-327',  # import_xml_etree
            'B406': 'CWE-327',  # import_xml_sax
            'B407': 'CWE-327',  # import_xml_expat
            'B408': 'CWE-327',  # import_xml_minidom
            'B409': 'CWE-327',  # import_xml_pulldom
            'B410': 'CWE-502',  # import_lxml
            'B411': 'CWE-327',  # import_xmlrpclib
            'B412': 'CWE-327',  # import_httpoxy
            'B413': 'CWE-327',  # import_pycrypto
            'B501': 'CWE-295',  # request_with_no_cert_validation
            'B502': 'CWE-295',  # ssl_with_bad_version
            'B503': 'CWE-295',  # ssl_with_bad_defaults
            'B504': 'CWE-295',  # ssl_with_no_version
            'B505': 'CWE-327',  # weak_cryptographic_key
            'B506': 'CWE-377',  # yaml_load
            'B507': 'CWE-78',   # ssh_no_host_key_verification
            'B601': 'CWE-78',   # paramiko_calls
            'B602': 'CWE-78',   # subprocess_popen_with_shell_equals_true
            'B603': 'CWE-78',   # subprocess_without_shell_equals_true
            'B604': 'CWE-78',   # any_other_function_with_shell_equals_true
            'B605': 'CWE-78',   # start_process_with_a_shell
            'B606': 'CWE-78',   # start_process_with_no_shell
            'B607': 'CWE-78',   # start_process_with_partial_path
            'B608': 'CWE-89',   # hardcoded_sql_expressions
            'B609': 'CWE-78',   # linux_commands_wildcard_injection
            'B610': 'CWE-89',   # django_extra_used
            'B611': 'CWE-89',   # django_rawsql_used
            'B701': 'CWE-117',  # jinja2_autoescape_false
            'B702': 'CWE-117',  # use_of_mako_templates
            'B703': 'CWE-352',  # django_mark_safe
        }

    def _ruff_rule_to_cwe(self, rule_code: str) -> str:
        """Ruff 규칙 → CWE 매핑"""
        # S 카테고리 (보안) 규칙들
        mapping = {
            'S101': 'CWE-703',  # assert
            'S102': 'CWE-78',   # exec
            'S103': 'CWE-22',   # bad_file_permissions
            'S104': 'CWE-200',  # hardcoded_bind_all_interfaces
            'S105': 'CWE-798',  # hardcoded_password_string
            'S106': 'CWE-798',  # hardcoded_password_funcarg
            'S107': 'CWE-798',  # hardcoded_password_default
            'S108': 'CWE-22',   # hardcoded_tmp_directory
            'S301': 'CWE-502',  # pickle
            'S302': 'CWE-327',  # marshal
            'S303': 'CWE-327',  # md5
            'S304': 'CWE-327',  # des
            'S305': 'CWE-327',  # blowfish
            'S306': 'CWE-327',  # mktemp
            'S307': 'CWE-502',  # eval
            'S308': 'CWE-377',  # mark_safe
            'S310': 'CWE-22',   # urllib_urlopen
            'S311': 'CWE-330',  # random
            'S312': 'CWE-78',   # telnetlib
            'S324': 'CWE-327',  # hashlib_insecure
            'S501': 'CWE-295',  # request_no_cert_validation
            'S506': 'CWE-377',  # yaml_load
            'S508': 'CWE-78',   # snmp_insecure_version
            'S601': 'CWE-78',   # paramiko
            'S602': 'CWE-78',   # shell_true
            'S603': 'CWE-78',   # subprocess_without_shell
            'S604': 'CWE-78',   # call_with_shell_true
            'S605': 'CWE-78',   # start_process_with_shell
            'S606': 'CWE-78',   # start_process_without_shell
            'S607': 'CWE-78',   # start_process_with_partial_path
            'S608': 'CWE-89',   # sql_injection_vector
        }

        return mapping.get(rule_code, 'CWE-Unknown')
