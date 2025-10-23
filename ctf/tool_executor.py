"""
Tool Executor - CTF 도구 자동 실행
"""

import subprocess
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    output: str
    error: str
    exit_code: int
    duration: float


class ToolExecutor:
    """CTF 도구 자동 실행 엔진"""

    def __init__(self):
        self.installed_tools = self._check_installed_tools()

    def _check_installed_tools(self) -> Dict[str, bool]:
        """설치된 도구 확인"""
        tools = {
            # Web
            'sqlmap': 'sqlmap',
            'nikto': 'nikto',
            'dirb': 'dirb',
            'gobuster': 'gobuster',

            # Pwn
            'checksec': 'checksec',
            'gdb': 'gdb',
            'radare2': 'r2',
            'pwntools': 'python',  # Python 패키지

            # Forensics
            'binwalk': 'binwalk',
            'foremost': 'foremost',
            'volatility': 'volatility',
            'strings': 'strings',
            'exiftool': 'exiftool',
            'file': 'file',

            # Crypto
            'hashcat': 'hashcat',
            'john': 'john',
            'openssl': 'openssl',

            # Reversing
            'ghidra': 'ghidra',
            'objdump': 'objdump',
            'readelf': 'readelf',
            'ltrace': 'ltrace',
            'strace': 'strace'
        }

        installed = {}
        for name, command in tools.items():
            installed[name] = self._is_tool_installed(command)

        return installed

    def _is_tool_installed(self, tool: str) -> bool:
        """도구 설치 여부 확인"""
        try:
            if tool == 'python':
                # pwntools 확인
                subprocess.run(
                    ['python', '-c', 'import pwn'],
                    capture_output=True,
                    timeout=2
                )
                return True
            else:
                subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    timeout=2
                )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        """
        명령어 실행

        Args:
            command: 실행할 명령어
            timeout: 타임아웃 (초)

        Returns:
            실행 결과
        """
        import time
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            duration = time.time() - start_time

            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8', errors='ignore'),
                error=stderr.decode('utf-8', errors='ignore'),
                exit_code=process.returncode,
                duration=duration
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                exit_code=-1,
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                duration=duration
            )

    # === Web Tools ===

    async def run_sqlmap(self, url: str, options: Optional[List[str]] = None) -> ToolResult:
        """SQLMap 실행"""
        if not self.installed_tools.get('sqlmap'):
            return ToolResult(False, "", "sqlmap not installed", -1, 0.0)

        cmd = ['sqlmap', '-u', url]
        if options:
            cmd.extend(options)
        else:
            cmd.extend(['--batch', '--level=1', '--risk=1'])

        return await self.execute(' '.join(cmd), timeout=120)

    async def run_nikto(self, url: str) -> ToolResult:
        """Nikto 웹 스캐너 실행"""
        if not self.installed_tools.get('nikto'):
            return ToolResult(False, "", "nikto not installed", -1, 0.0)

        cmd = f"nikto -h {url}"
        return await self.execute(cmd, timeout=300)

    async def run_dirb(self, url: str, wordlist: Optional[str] = None) -> ToolResult:
        """Dirb 디렉토리 브루트포스"""
        if not self.installed_tools.get('dirb'):
            return ToolResult(False, "", "dirb not installed", -1, 0.0)

        cmd = f"dirb {url}"
        if wordlist:
            cmd += f" {wordlist}"

        return await self.execute(cmd, timeout=180)

    # === Forensics Tools ===

    async def run_binwalk(self, file_path: str, extract: bool = False) -> ToolResult:
        """Binwalk 파일 분석"""
        if not self.installed_tools.get('binwalk'):
            return ToolResult(False, "", "binwalk not installed", -1, 0.0)

        cmd = f"binwalk {file_path}"
        if extract:
            cmd += " -e"

        return await self.execute(cmd)

    async def run_foremost(self, file_path: str, output_dir: str = "foremost_output") -> ToolResult:
        """Foremost 파일 복구"""
        if not self.installed_tools.get('foremost'):
            return ToolResult(False, "", "foremost not installed", -1, 0.0)

        cmd = f"foremost -o {output_dir} -i {file_path}"
        return await self.execute(cmd)

    async def run_strings(self, file_path: str, min_length: int = 4) -> ToolResult:
        """Strings 추출"""
        if not self.installed_tools.get('strings'):
            return ToolResult(False, "", "strings not installed", -1, 0.0)

        cmd = f"strings -n {min_length} {file_path}"
        return await self.execute(cmd)

    async def run_exiftool(self, file_path: str) -> ToolResult:
        """Exiftool 메타데이터 추출"""
        if not self.installed_tools.get('exiftool'):
            return ToolResult(False, "", "exiftool not installed", -1, 0.0)

        cmd = f"exiftool {file_path}"
        return await self.execute(cmd)

    async def run_file(self, file_path: str) -> ToolResult:
        """File 타입 확인"""
        if not self.installed_tools.get('file'):
            return ToolResult(False, "", "file not installed", -1, 0.0)

        cmd = f"file {file_path}"
        return await self.execute(cmd)

    # === Crypto Tools ===

    async def run_hashcat(self, hash_file: str, wordlist: str, hash_type: int = 0) -> ToolResult:
        """Hashcat 해시 크래킹"""
        if not self.installed_tools.get('hashcat'):
            return ToolResult(False, "", "hashcat not installed", -1, 0.0)

        cmd = f"hashcat -m {hash_type} {hash_file} {wordlist}"
        return await self.execute(cmd, timeout=600)

    async def run_john(self, hash_file: str, wordlist: Optional[str] = None) -> ToolResult:
        """John the Ripper 해시 크래킹"""
        if not self.installed_tools.get('john'):
            return ToolResult(False, "", "john not installed", -1, 0.0)

        cmd = f"john {hash_file}"
        if wordlist:
            cmd += f" --wordlist={wordlist}"

        return await self.execute(cmd, timeout=600)

    # === Reversing Tools ===

    async def run_checksec(self, binary_path: str) -> ToolResult:
        """Checksec 보안 기능 확인"""
        if not self.installed_tools.get('checksec'):
            return ToolResult(False, "", "checksec not installed", -1, 0.0)

        cmd = f"checksec --file={binary_path}"
        return await self.execute(cmd)

    async def run_readelf(self, binary_path: str, option: str = "-a") -> ToolResult:
        """Readelf ELF 분석"""
        if not self.installed_tools.get('readelf'):
            return ToolResult(False, "", "readelf not installed", -1, 0.0)

        cmd = f"readelf {option} {binary_path}"
        return await self.execute(cmd)

    async def run_objdump(self, binary_path: str, option: str = "-d") -> ToolResult:
        """Objdump 디스어셈블"""
        if not self.installed_tools.get('objdump'):
            return ToolResult(False, "", "objdump not installed", -1, 0.0)

        cmd = f"objdump {option} {binary_path}"
        return await self.execute(cmd)

    async def run_ltrace(self, binary_path: str, args: Optional[str] = None) -> ToolResult:
        """Ltrace 라이브러리 호출 추적"""
        if not self.installed_tools.get('ltrace'):
            return ToolResult(False, "", "ltrace not installed", -1, 0.0)

        cmd = f"ltrace {binary_path}"
        if args:
            cmd += f" {args}"

        return await self.execute(cmd, timeout=30)

    async def run_strace(self, binary_path: str, args: Optional[str] = None) -> ToolResult:
        """Strace 시스템 호출 추적"""
        if not self.installed_tools.get('strace'):
            return ToolResult(False, "", "strace not installed", -1, 0.0)

        cmd = f"strace {binary_path}"
        if args:
            cmd += f" {args}"

        return await self.execute(cmd, timeout=30)

    def get_tool_recommendations(self, category: str) -> List[str]:
        """카테고리별 권장 도구 목록"""
        recommendations = {
            'web': ['sqlmap', 'nikto', 'dirb', 'gobuster'],
            'pwn': ['checksec', 'gdb', 'radare2', 'pwntools'],
            'forensics': ['binwalk', 'foremost', 'strings', 'exiftool', 'file'],
            'crypto': ['hashcat', 'john', 'openssl'],
            'reversing': ['ghidra', 'objdump', 'readelf', 'ltrace', 'strace']
        }

        tools = recommendations.get(category, [])
        installed = [tool for tool in tools if self.installed_tools.get(tool)]
        missing = [tool for tool in tools if not self.installed_tools.get(tool)]

        return {
            'installed': installed,
            'missing': missing
        }
