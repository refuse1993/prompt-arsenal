"""
Crypto Solver - CTF 암호학 자동 풀이
"""

import asyncio
import base64
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass

from .llm_reasoner import LLMReasoner, CTFAnalysis
from .tool_executor import ToolExecutor, ToolResult


@dataclass
class CryptoResult:
    """암호 해독 결과"""
    cipher_type: str
    success: bool
    flag: Optional[str]
    plaintext: Optional[str]
    method: str
    confidence: float


class CryptoSolver:
    """CTF 암호학 자동 풀이 엔진"""

    def __init__(self, llm: LLMReasoner, executor: ToolExecutor):
        self.llm = llm
        self.executor = executor

    async def solve(self, ciphertext: str, challenge_info: Dict) -> Dict:
        """
        암호 문제 자동 풀이

        Args:
            ciphertext: 암호문
            challenge_info: 문제 정보

        Returns:
            풀이 결과
        """
        # 1. LLM으로 문제 분석
        analysis = await self.llm.analyze_challenge({
            'title': challenge_info.get('title', ''),
            'description': challenge_info.get('description', ''),
            'hints': challenge_info.get('hints', [])
        })

        if analysis.category != 'crypto':
            return {
                'success': False,
                'error': f'Not a crypto challenge: {analysis.category}'
            }

        # 2. 암호 유형별 처리
        cipher_type = analysis.vulnerability_type.lower()

        # 자동 암호 식별
        detected_type = self._detect_cipher_type(ciphertext)

        if detected_type:
            print(f"  🔍 자동 탐지된 암호: {detected_type}")
            cipher_type = detected_type

        # 3. 암호 유형별 해독
        if 'base64' in cipher_type or 'base 64' in cipher_type:
            result = self._solve_base64(ciphertext)
        elif 'caesar' in cipher_type or 'rot' in cipher_type:
            result = self._solve_caesar(ciphertext)
        elif 'vigenere' in cipher_type or 'vigenère' in cipher_type:
            result = await self._solve_vigenere(ciphertext, challenge_info)
        elif 'xor' in cipher_type:
            result = self._solve_xor(ciphertext, challenge_info)
        elif 'rsa' in cipher_type:
            result = await self._solve_rsa(ciphertext, challenge_info)
        elif 'aes' in cipher_type:
            result = await self._solve_aes(ciphertext, challenge_info)
        elif 'hash' in cipher_type or 'md5' in cipher_type or 'sha' in cipher_type:
            result = await self._solve_hash(ciphertext, challenge_info)
        else:
            # 모든 방법 시도
            result = await self._brute_force_all(ciphertext, challenge_info)

        return {
            'success': result.success,
            'flag': result.flag,
            'plaintext': result.plaintext,
            'cipher_type': result.cipher_type,
            'method': result.method,
            'confidence': result.confidence,
            'analysis': analysis
        }

    # === 암호 타입 탐지 ===

    def _detect_cipher_type(self, ciphertext: str) -> Optional[str]:
        """암호 타입 자동 탐지"""
        ciphertext = ciphertext.strip()

        # Base64 패턴
        if self._is_base64(ciphertext):
            return 'base64'

        # Hex 패턴
        if all(c in '0123456789abcdefABCDEF' for c in ciphertext.replace(' ', '')):
            return 'hex'

        # ROT13/Caesar (알파벳만)
        if ciphertext.isalpha():
            return 'caesar'

        # Hash 길이
        if len(ciphertext) == 32 and all(c in '0123456789abcdefABCDEF' for c in ciphertext):
            return 'md5'
        if len(ciphertext) == 40 and all(c in '0123456789abcdefABCDEF' for c in ciphertext):
            return 'sha1'

        return None

    def _is_base64(self, text: str) -> bool:
        """Base64 여부 확인"""
        try:
            # Base64는 a-zA-Z0-9+/= 문자만 사용
            if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in text):
                return False

            # 패딩 확인
            if len(text) % 4 != 0:
                return False

            # 디코딩 시도
            base64.b64decode(text, validate=True)
            return True
        except:
            return False

    # === Base64 ===

    def _solve_base64(self, ciphertext: str) -> CryptoResult:
        """Base64 디코딩"""
        print(f"  🔓 Base64 디코딩 시도")

        try:
            decoded = base64.b64decode(ciphertext).decode('utf-8', errors='ignore')

            flag = self._extract_flag(decoded)

            if flag:
                return CryptoResult(
                    cipher_type='Base64',
                    success=True,
                    flag=flag,
                    plaintext=decoded,
                    method='Base64 decoding',
                    confidence=0.95
                )

            # 재귀적 Base64 (여러 번 인코딩)
            current = decoded
            layers = 1

            while layers < 10:
                try:
                    if self._is_base64(current):
                        current = base64.b64decode(current).decode('utf-8', errors='ignore')
                        layers += 1

                        flag = self._extract_flag(current)
                        if flag:
                            return CryptoResult(
                                cipher_type='Base64',
                                success=True,
                                flag=flag,
                                plaintext=current,
                                method=f'Base64 decoding ({layers} layers)',
                                confidence=0.9
                            )
                    else:
                        break
                except:
                    break

            return CryptoResult(
                cipher_type='Base64',
                success=False,
                flag=None,
                plaintext=decoded,
                method='Base64 decoding',
                confidence=0.5
            )

        except Exception as e:
            return CryptoResult(
                cipher_type='Base64',
                success=False,
                flag=None,
                plaintext=None,
                method='Base64 decoding failed',
                confidence=0.0
            )

    # === Caesar/ROT ===

    def _solve_caesar(self, ciphertext: str) -> CryptoResult:
        """Caesar/ROT 암호 해독"""
        print(f"  🔓 Caesar/ROT 암호 해독 시도")

        best_result = None
        best_score = 0

        # 모든 shift 값 시도 (0-25)
        for shift in range(26):
            plaintext = self._caesar_shift(ciphertext, shift)

            flag = self._extract_flag(plaintext)

            if flag:
                return CryptoResult(
                    cipher_type='Caesar',
                    success=True,
                    flag=flag,
                    plaintext=plaintext,
                    method=f'Caesar cipher (shift={shift})',
                    confidence=0.9
                )

            # 영어 단어 점수 계산
            score = self._english_score(plaintext)

            if score > best_score:
                best_score = score
                best_result = plaintext

        return CryptoResult(
            cipher_type='Caesar',
            success=False,
            flag=None,
            plaintext=best_result,
            method='Caesar cipher (best guess)',
            confidence=0.4
        )

    def _caesar_shift(self, text: str, shift: int) -> str:
        """Caesar 암호 shift"""
        result = []

        for char in text:
            if 'A' <= char <= 'Z':
                result.append(chr((ord(char) - ord('A') + shift) % 26 + ord('A')))
            elif 'a' <= char <= 'z':
                result.append(chr((ord(char) - ord('a') + shift) % 26 + ord('a')))
            else:
                result.append(char)

        return ''.join(result)

    # === Vigenere ===

    async def _solve_vigenere(self, ciphertext: str, challenge_info: Dict) -> CryptoResult:
        """Vigenere 암호 해독"""
        print(f"  🔓 Vigenere 암호 해독 시도")

        # 힌트에서 키 추출
        key = challenge_info.get('key')

        if not key:
            # LLM에게 키 추측 요청
            print("  🤖 LLM으로 Vigenere 키 추측")
            # 실제 구현 필요

        if key:
            plaintext = self._vigenere_decrypt(ciphertext, key)

            flag = self._extract_flag(plaintext)

            if flag:
                return CryptoResult(
                    cipher_type='Vigenere',
                    success=True,
                    flag=flag,
                    plaintext=plaintext,
                    method=f'Vigenere cipher (key={key})',
                    confidence=0.85
                )

        return CryptoResult(
            cipher_type='Vigenere',
            success=False,
            flag=None,
            plaintext=None,
            method='Vigenere cipher (no key)',
            confidence=0.0
        )

    def _vigenere_decrypt(self, ciphertext: str, key: str) -> str:
        """Vigenere 복호화"""
        plaintext = []
        key = key.upper()
        key_length = len(key)
        key_index = 0

        for char in ciphertext:
            if 'A' <= char <= 'Z':
                shift = ord(key[key_index % key_length]) - ord('A')
                plaintext.append(chr((ord(char) - ord('A') - shift) % 26 + ord('A')))
                key_index += 1
            elif 'a' <= char <= 'z':
                shift = ord(key[key_index % key_length]) - ord('A')
                plaintext.append(chr((ord(char) - ord('a') - shift) % 26 + ord('a')))
                key_index += 1
            else:
                plaintext.append(char)

        return ''.join(plaintext)

    # === XOR ===

    def _solve_xor(self, ciphertext: str, challenge_info: Dict) -> CryptoResult:
        """XOR 암호 해독"""
        print(f"  🔓 XOR 암호 해독 시도")

        # Hex 디코딩
        try:
            cipher_bytes = bytes.fromhex(ciphertext.replace(' ', ''))
        except:
            cipher_bytes = ciphertext.encode()

        # Single-byte XOR 브루트포스
        for key in range(256):
            plaintext_bytes = bytes([b ^ key for b in cipher_bytes])

            try:
                plaintext = plaintext_bytes.decode('utf-8', errors='ignore')

                flag = self._extract_flag(plaintext)

                if flag:
                    return CryptoResult(
                        cipher_type='XOR',
                        success=True,
                        flag=flag,
                        plaintext=plaintext,
                        method=f'Single-byte XOR (key={hex(key)})',
                        confidence=0.9
                    )

                # 영어 문장인지 확인
                if self._english_score(plaintext) > 0.7:
                    return CryptoResult(
                        cipher_type='XOR',
                        success=False,
                        flag=None,
                        plaintext=plaintext,
                        method=f'Single-byte XOR (key={hex(key)})',
                        confidence=0.6
                    )
            except:
                continue

        return CryptoResult(
            cipher_type='XOR',
            success=False,
            flag=None,
            plaintext=None,
            method='XOR cipher failed',
            confidence=0.0
        )

    # === RSA ===

    async def _solve_rsa(self, ciphertext: str, challenge_info: Dict) -> CryptoResult:
        """RSA 암호 해독"""
        print(f"  🔓 RSA 암호 해독 시도")

        # RSA 파라미터 추출
        n = challenge_info.get('n')
        e = challenge_info.get('e')
        c = challenge_info.get('c') or ciphertext

        if not all([n, e, c]):
            return CryptoResult(
                cipher_type='RSA',
                success=False,
                flag=None,
                plaintext=None,
                method='RSA parameters missing',
                confidence=0.0
            )

        # 작은 n 분해 시도 (실제 구현 필요)
        # from Crypto.Util.number import long_to_bytes
        # import gmpy2

        # LLM에게 RSA 공격 전략 요청
        print("  🤖 LLM으로 RSA 공격 전략 생성")

        return CryptoResult(
            cipher_type='RSA',
            success=False,
            flag=None,
            plaintext=None,
            method='RSA attack not implemented',
            confidence=0.0
        )

    # === AES ===

    async def _solve_aes(self, ciphertext: str, challenge_info: Dict) -> CryptoResult:
        """AES 암호 해독"""
        print(f"  🔓 AES 암호 해독 시도")

        # AES 키 필요
        key = challenge_info.get('key')

        if not key:
            return CryptoResult(
                cipher_type='AES',
                success=False,
                flag=None,
                plaintext=None,
                method='AES key missing',
                confidence=0.0
            )

        # 실제 구현 필요 (pycryptodome)
        # from Crypto.Cipher import AES

        return CryptoResult(
            cipher_type='AES',
            success=False,
            flag=None,
            plaintext=None,
            method='AES decryption not implemented',
            confidence=0.0
        )

    # === Hash ===

    async def _solve_hash(self, hash_value: str, challenge_info: Dict) -> CryptoResult:
        """해시 크래킹"""
        print(f"  🔓 해시 크래킹 시도")

        # Hashcat/John 사용
        # 실제 구현에서는 wordlist 필요

        wordlist = challenge_info.get('wordlist', '/usr/share/wordlists/rockyou.txt')

        # Hashcat 실행 (실제 구현 필요)
        # hash_type 자동 감지
        hash_type = 0  # MD5

        if len(hash_value) == 40:
            hash_type = 100  # SHA1
        elif len(hash_value) == 64:
            hash_type = 1400  # SHA256

        return CryptoResult(
            cipher_type='Hash',
            success=False,
            flag=None,
            plaintext=None,
            method='Hash cracking not implemented',
            confidence=0.0
        )

    # === Brute Force All ===

    async def _brute_force_all(self, ciphertext: str, challenge_info: Dict) -> CryptoResult:
        """모든 암호 해독 방법 시도"""
        print(f"  🔍 모든 암호 해독 방법 시도")

        # 1. Base64
        result = self._solve_base64(ciphertext)
        if result.success:
            return result

        # 2. Hex
        try:
            hex_decoded = bytes.fromhex(ciphertext.replace(' ', '')).decode('utf-8', errors='ignore')
            flag = self._extract_flag(hex_decoded)
            if flag:
                return CryptoResult(
                    cipher_type='Hex',
                    success=True,
                    flag=flag,
                    plaintext=hex_decoded,
                    method='Hex decoding',
                    confidence=0.9
                )
        except:
            pass

        # 3. Caesar
        result = self._solve_caesar(ciphertext)
        if result.success:
            return result

        # 4. XOR
        result = self._solve_xor(ciphertext, challenge_info)
        if result.success:
            return result

        return CryptoResult(
            cipher_type='Unknown',
            success=False,
            flag=None,
            plaintext=None,
            method='All methods failed',
            confidence=0.0
        )

    # === Helper Methods ===

    def _extract_flag(self, text: str) -> Optional[str]:
        """텍스트에서 플래그 추출"""
        import re

        patterns = [
            r'flag\{[^}]+\}',
            r'FLAG\{[^}]+\}',
            r'CTF\{[^}]+\}',
            r'[A-Za-z0-9_]+\{[^}]+\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _english_score(self, text: str) -> float:
        """영어 문장 점수 (0.0 ~ 1.0)"""
        # 간단한 구현: 알파벳 비율
        if not text:
            return 0.0

        alpha_count = sum(1 for c in text if c.isalpha())
        total = len(text)

        if total == 0:
            return 0.0

        return alpha_count / total
