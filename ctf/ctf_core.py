"""
CTF Core - CTF 자동 풀이 통합 엔진
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime

from .llm_reasoner import LLMReasoner
from .tool_executor import ToolExecutor
from .web_solver import WebSolver
from .forensics_solver import ForensicsSolver
from .pwn_solver import PwnSolver
from .crypto_solver import CryptoSolver
from .reversing_solver import ReversingSolver


class CTFSolver:
    """CTF 자동 풀이 통합 엔진"""

    def __init__(
        self,
        db,
        provider: str,
        model: str,
        api_key: str
    ):
        """
        Args:
            db: ArsenalDB 인스턴스
            provider: 'openai' or 'anthropic'
            model: 모델 이름
            api_key: API 키
        """
        self.db = db
        self.llm = LLMReasoner(provider, model, api_key)
        self.executor = ToolExecutor()

        # 카테고리별 Solver 초기화
        self.web_solver = WebSolver(self.llm, self.executor)
        self.forensics_solver = ForensicsSolver(self.llm, self.executor)
        self.pwn_solver = PwnSolver(self.llm, self.executor)
        self.crypto_solver = CryptoSolver(self.llm, self.executor)
        self.reversing_solver = ReversingSolver(self.llm, self.executor)

    async def solve_challenge(self, challenge_id: int, max_retries: int = 3) -> Dict:
        """
        CTF 문제 자동 풀이

        Args:
            challenge_id: 문제 ID (DB에서 조회)
            max_retries: 최대 재시도 횟수

        Returns:
            풀이 결과
        """
        # 1. DB에서 문제 조회
        challenge = self.db.get_ctf_challenge(challenge_id)

        if not challenge:
            return {
                'success': False,
                'error': f'Challenge not found: {challenge_id}'
            }

        print(f"\n{'='*60}")
        print(f"🎯 CTF Challenge: {challenge['title']}")
        print(f"{'='*60}")
        print(f"Category: {challenge['category']}")
        print(f"Difficulty: {challenge['difficulty']}")
        print(f"Description: {challenge['description'][:200]}")
        print(f"{'='*60}\n")

        start_time = datetime.now()

        # 2. 카테고리별 Solver 실행
        result = None
        attempt = 0

        while attempt < max_retries and (not result or not result.get('success')):
            attempt += 1
            print(f"  🔄 Attempt {attempt}/{max_retries}")

            try:
                result = await self._execute_solver(challenge)

                if result.get('success'):
                    print(f"\n  ✅ FLAG FOUND: {result.get('flag')}")
                    break
                else:
                    print(f"\n  ❌ Attempt {attempt} failed")

                    if attempt < max_retries:
                        print(f"  🔄 Retrying...")
                        await asyncio.sleep(2)

            except Exception as e:
                print(f"\n  ❌ Error: {e}")

                if attempt < max_retries:
                    await asyncio.sleep(2)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 3. 결과 저장
        execution_log = {
            'challenge_id': challenge_id,
            'category': challenge['category'],
            'success': result.get('success', False) if result else False,
            'flag': result.get('flag') if result else None,
            'attempts': attempt,
            'duration': duration,
            'tools_used': self._extract_tools_used(result) if result else [],
            'llm_provider': self.llm.provider,
            'llm_model': self.llm.model,
            'error': result.get('error') if result and not result.get('success') else None,
            'timestamp': datetime.now().isoformat()
        }

        log_id = self.db.insert_ctf_execution_log(execution_log)

        # 4. 문제 상태 업데이트
        if result and result.get('success'):
            self.db.update_ctf_challenge_status(
                challenge_id,
                status='solved',
                flag=result.get('flag')
            )

        print(f"\n{'='*60}")
        print(f"📊 Execution Summary")
        print(f"{'='*60}")
        print(f"Success: {execution_log['success']}")
        print(f"Attempts: {execution_log['attempts']}")
        print(f"Duration: {duration:.2f}s")
        print(f"Log ID: {log_id}")
        print(f"{'='*60}\n")

        return {
            **execution_log,
            'log_id': log_id,
            'result': result
        }

    async def _execute_solver(self, challenge: Dict) -> Dict:
        """카테고리별 Solver 실행"""
        category = challenge['category'].lower()

        if category == 'web':
            return await self._solve_web(challenge)
        elif category == 'forensics':
            return await self._solve_forensics(challenge)
        elif category == 'pwn':
            return await self._solve_pwn(challenge)
        elif category == 'crypto':
            return await self._solve_crypto(challenge)
        elif category == 'reversing':
            return await self._solve_reversing(challenge)
        else:
            return {
                'success': False,
                'error': f'Unknown category: {category}'
            }

    async def _solve_web(self, challenge: Dict) -> Dict:
        """Web 문제 풀이"""
        url = challenge.get('url')

        if not url:
            return {
                'success': False,
                'error': 'URL not provided'
            }

        challenge_info = {
            'title': challenge.get('title'),
            'description': challenge.get('description'),
            'hints': challenge.get('hints', [])
        }

        return await self.web_solver.solve(url, challenge_info)

    async def _solve_forensics(self, challenge: Dict) -> Dict:
        """Forensics 문제 풀이"""
        file_path = challenge.get('file_path')

        if not file_path:
            return {
                'success': False,
                'error': 'File path not provided'
            }

        challenge_info = {
            'title': challenge.get('title'),
            'description': challenge.get('description'),
            'hints': challenge.get('hints', [])
        }

        return await self.forensics_solver.solve(file_path, challenge_info)

    async def _solve_pwn(self, challenge: Dict) -> Dict:
        """Pwn 문제 풀이"""
        binary_path = challenge.get('file_path')

        target_info = {
            'title': challenge.get('title'),
            'host': challenge.get('host'),
            'port': challenge.get('port'),
            'description': challenge.get('description'),
            'hints': challenge.get('hints', [])
        }

        return await self.pwn_solver.solve(binary_path, target_info)

    async def _solve_crypto(self, challenge: Dict) -> Dict:
        """Crypto 문제 풀이"""
        ciphertext = challenge.get('ciphertext') or challenge.get('description')

        challenge_info = {
            'title': challenge.get('title'),
            'description': challenge.get('description'),
            'hints': challenge.get('hints', []),
            'key': challenge.get('key'),
            'n': challenge.get('n'),
            'e': challenge.get('e'),
            'c': challenge.get('c')
        }

        return await self.crypto_solver.solve(ciphertext, challenge_info)

    async def _solve_reversing(self, challenge: Dict) -> Dict:
        """Reversing 문제 풀이"""
        binary_path = challenge.get('file_path')

        if not binary_path:
            return {
                'success': False,
                'error': 'Binary path not provided'
            }

        challenge_info = {
            'title': challenge.get('title'),
            'description': challenge.get('description'),
            'hints': challenge.get('hints', [])
        }

        return await self.reversing_solver.solve(binary_path, challenge_info)

    def _extract_tools_used(self, result: Dict) -> list:
        """결과에서 사용된 도구 추출"""
        tools = []

        # 결과 분석해서 사용된 도구 추출
        if 'method' in result:
            method = result['method'].lower()

            tool_map = {
                'sqlmap': 'sqlmap',
                'nikto': 'nikto',
                'dirb': 'dirb',
                'binwalk': 'binwalk',
                'foremost': 'foremost',
                'strings': 'strings',
                'exiftool': 'exiftool',
                'checksec': 'checksec',
                'readelf': 'readelf',
                'objdump': 'objdump',
                'ltrace': 'ltrace',
                'strace': 'strace',
                'hashcat': 'hashcat',
                'john': 'john'
            }

            for tool_name, tool_key in tool_map.items():
                if tool_name in method:
                    tools.append(tool_key)

        return tools

    def get_statistics(self) -> Dict:
        """CTF 풀이 통계"""
        return self.db.get_ctf_statistics()

    def get_tool_recommendations(self, category: str) -> Dict:
        """카테고리별 권장 도구 목록"""
        return self.executor.get_tool_recommendations(category)
