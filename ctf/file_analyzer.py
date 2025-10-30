"""
CTF Challenge File Analyzer
첨부파일 분석 및 자동 처리 시스템
"""

import asyncio
import hashlib
import json
import subprocess
import zipfile
import tarfile
import magic
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from rich.console import Console
from dataclasses import dataclass

console = Console()


@dataclass
class FileInfo:
    """파일 정보"""
    file_path: Path
    filename: str
    file_size: int
    file_type: str
    mime_type: str
    md5_hash: str
    sha256_hash: str
    is_compressed: bool
    is_executable: bool
    created_at: datetime
    analysis: Dict = None


class FileAnalyzer:
    """CTF 챌린지 파일 분석기"""

    FILE_CATEGORIES = {
        'archive': ['.zip', '.tar', '.tar.gz', '.tgz', '.rar', '.7z', '.gz'],
        'binary': ['.exe', '.dll', '.so', '.elf', '.bin', '.out'],
        'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.ico'],
        'audio': ['.mp3', '.wav', '.ogg', '.flac', '.m4a'],
        'document': ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'],
        'network': ['.pcap', '.pcapng', '.cap'],
        'code': ['.py', '.js', '.c', '.cpp', '.java', '.go', '.sh', '.php'],
        'data': ['.json', '.xml', '.csv', '.db', '.sqlite', '.sql'],
        'web': ['.html', '.htm', '.css', '.js']
    }

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (선택)
        """
        self.llm = llm_client

    async def analyze_basic_info(self, file_path: Path) -> FileInfo:
        """기본 파일 정보 추출"""
        try:
            # MIME 타입 감지
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))

            # 해시 계산
            with open(file_path, 'rb') as f:
                file_data = f.read()
                md5_hash = hashlib.md5(file_data).hexdigest()
                sha256_hash = hashlib.sha256(file_data).hexdigest()

            # 파일 분류
            suffix = file_path.suffix.lower()
            is_compressed = suffix in self.FILE_CATEGORIES['archive']
            is_executable = suffix in self.FILE_CATEGORIES['binary'] or mime_type.startswith('application/x-')

            file_info = FileInfo(
                file_path=file_path,
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                file_type=self._get_file_category(suffix),
                mime_type=mime_type,
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                is_compressed=is_compressed,
                is_executable=is_executable,
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                analysis={}
            )

            return file_info

        except Exception as e:
            console.print(f"[red]기본 정보 추출 실패: {str(e)}[/red]")
            return None

    def _get_file_category(self, suffix: str) -> str:
        """파일 확장자로 카테고리 판단"""
        for category, extensions in self.FILE_CATEGORIES.items():
            if suffix in extensions:
                return category
        return 'unknown'

    async def extract_archive(self, file_path: Path) -> List[Path]:
        """압축 파일 해제"""
        try:
            extract_dir = file_path.parent / f"{file_path.stem}_extracted"
            extract_dir.mkdir(exist_ok=True)

            console.print(f"[cyan]📦 압축 해제 중: {file_path.name}[/cyan]")

            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path) as zf:
                    zf.extractall(extract_dir)
            elif file_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                mode = 'r:gz' if file_path.suffix in ['.tar.gz', '.tgz'] else 'r'
                with tarfile.open(file_path, mode) as tf:
                    tf.extractall(extract_dir)
            elif file_path.suffix == '.gz':
                import gzip
                import shutil
                output_file = extract_dir / file_path.stem
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                console.print(f"[yellow]지원하지 않는 압축 형식: {file_path.suffix}[/yellow]")
                return []

            # 재귀적으로 모든 파일 찾기
            extracted_files = list(extract_dir.rglob('*'))
            extracted_files = [f for f in extracted_files if f.is_file()]

            console.print(f"[green]✓ {len(extracted_files)}개 파일 추출 완료[/green]")
            return extracted_files

        except Exception as e:
            console.print(f"[red]압축 해제 실패: {str(e)}[/red]")
            return []

    async def analyze_binary(self, file_path: Path) -> Dict:
        """바이너리 파일 분석 (ELF, PE)"""
        analysis = {
            'type': 'binary',
            'file_info': '',
            'strings': '',
            'sections': [],
            'imports': [],
            'security': {}
        }

        try:
            # file 명령어
            result = subprocess.run(
                ['file', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            analysis['file_info'] = result.stdout.strip()

            # strings (처음 100줄만)
            result = subprocess.run(
                ['strings', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            lines = result.stdout.split('\n')[:100]
            analysis['strings'] = '\n'.join(lines)

            # ELF 파일이면 추가 분석
            if 'ELF' in analysis['file_info']:
                analysis['security'] = await self._analyze_elf_security(file_path)

                # readelf로 섹션 정보
                result = subprocess.run(
                    ['readelf', '-S', str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                analysis['sections'] = result.stdout[:1000]

            console.print(f"[green]✓ 바이너리 분석 완료[/green]")

        except subprocess.TimeoutExpired:
            console.print(f"[yellow]분석 타임아웃[/yellow]")
        except FileNotFoundError:
            console.print(f"[yellow]분석 도구 없음 (file, strings, readelf)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]바이너리 분석 오류: {str(e)}[/yellow]")

        return analysis

    async def _analyze_elf_security(self, file_path: Path) -> Dict:
        """ELF 보안 기능 체크"""
        security = {
            'nx': False,
            'pie': False,
            'canary': False,
            'relro': 'No'
        }

        try:
            # checksec 대체: readelf로 직접 체크
            result = subprocess.run(
                ['readelf', '-l', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout

            # NX (Non-Executable Stack)
            if 'GNU_STACK' in output:
                security['nx'] = 'RWE' not in output

            # PIE (Position Independent Executable)
            result = subprocess.run(
                ['readelf', '-h', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            security['pie'] = 'DYN' in result.stdout or 'EXEC' not in result.stdout

            # Canary (Stack Protector)
            result = subprocess.run(
                ['readelf', '-s', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            security['canary'] = '__stack_chk_fail' in result.stdout

            # RELRO
            if 'GNU_RELRO' in output:
                result = subprocess.run(
                    ['readelf', '-d', str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if 'BIND_NOW' in result.stdout:
                    security['relro'] = 'Full'
                else:
                    security['relro'] = 'Partial'

        except Exception as e:
            console.print(f"[dim yellow]ELF 보안 체크 실패: {str(e)}[/dim yellow]")

        return security

    async def analyze_image(self, file_path: Path) -> Dict:
        """이미지 파일 분석"""
        analysis = {
            'type': 'image',
            'size': None,
            'format': None,
            'mode': None,
            'exif': {},
            'strings': ''
        }

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(file_path)
            analysis['size'] = img.size
            analysis['format'] = img.format
            analysis['mode'] = img.mode

            # EXIF 데이터
            try:
                exif_data = img._getexif()
                if exif_data:
                    analysis['exif'] = {
                        TAGS.get(k, k): str(v)[:100]
                        for k, v in exif_data.items()
                    }
            except:
                pass

            # strings (숨겨진 텍스트 찾기)
            result = subprocess.run(
                ['strings', str(file_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            lines = result.stdout.split('\n')[:50]
            analysis['strings'] = '\n'.join(lines)

            console.print(f"[green]✓ 이미지 분석 완료[/green]")

        except ImportError:
            console.print(f"[yellow]Pillow 없음: pip install Pillow[/yellow]")
        except Exception as e:
            console.print(f"[yellow]이미지 분석 오류: {str(e)}[/yellow]")

        return analysis

    async def analyze_pcap(self, file_path: Path) -> Dict:
        """네트워크 캡처 파일 분석"""
        analysis = {
            'type': 'network',
            'packet_count': 0,
            'protocols': {},
            'endpoints': [],
            'suspicious': []
        }

        try:
            from scapy.all import rdpcap, IP, TCP, UDP

            packets = rdpcap(str(file_path))
            analysis['packet_count'] = len(packets)

            # 프로토콜 통계
            protocols = {}
            ip_pairs = set()

            for pkt in packets[:1000]:  # 최대 1000개만
                # 프로토콜
                if IP in pkt:
                    if TCP in pkt:
                        protocols['TCP'] = protocols.get('TCP', 0) + 1
                    elif UDP in pkt:
                        protocols['UDP'] = protocols.get('UDP', 0) + 1

                    # IP 주소
                    ip_pairs.add((pkt[IP].src, pkt[IP].dst))

            analysis['protocols'] = protocols
            analysis['endpoints'] = [f"{src} -> {dst}" for src, dst in list(ip_pairs)[:20]]

            console.print(f"[green]✓ PCAP 분석 완료[/green]")

        except ImportError:
            console.print(f"[yellow]scapy 없음: pip install scapy[/yellow]")
        except Exception as e:
            console.print(f"[yellow]PCAP 분석 오류: {str(e)}[/yellow]")

        return analysis

    async def analyze_document(self, file_path: Path) -> Dict:
        """문서 파일 분석"""
        analysis = {
            'type': 'document',
            'text_content': '',
            'metadata': {},
            'page_count': 0
        }

        try:
            if file_path.suffix.lower() == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        analysis['page_count'] = len(pdf.pages)

                        # 메타데이터
                        if pdf.metadata:
                            analysis['metadata'] = {
                                k: str(v)[:100]
                                for k, v in pdf.metadata.items()
                            }

                        # 텍스트 추출 (처음 5페이지)
                        text_parts = []
                        for page in pdf.pages[:5]:
                            text_parts.append(page.extract_text())
                        analysis['text_content'] = '\n'.join(text_parts)[:5000]

                except ImportError:
                    console.print(f"[yellow]PyPDF2 없음: pip install PyPDF2[/yellow]")

            elif file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    analysis['text_content'] = f.read()[:5000]

            console.print(f"[green]✓ 문서 분석 완료[/green]")

        except Exception as e:
            console.print(f"[yellow]문서 분석 오류: {str(e)}[/yellow]")

        return analysis

    async def analyze_by_type(self, file_path: Path, file_type: str, mime_type: str) -> Dict:
        """파일 타입별 전문 분석"""

        if file_type == 'binary' or mime_type.startswith('application/x-executable'):
            return await self.analyze_binary(file_path)

        elif file_type == 'image':
            return await self.analyze_image(file_path)

        elif file_type == 'network':
            return await self.analyze_pcap(file_path)

        elif file_type == 'document':
            return await self.analyze_document(file_path)

        elif file_type == 'code' or file_type == 'data':
            # 텍스트 파일 읽기
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # 최대 10KB
                return {
                    'type': file_type,
                    'content': content,
                    'lines': len(content.split('\n'))
                }
            except:
                return {'type': file_type, 'error': 'Cannot read file'}

        else:
            return {'type': 'unknown', 'note': 'No specific analysis available'}

    async def llm_comprehensive_analysis(
        self,
        file_info: FileInfo,
        expert_analysis: Dict,
        challenge_title: str = "",
        challenge_category: str = "",
        challenge_description: str = ""
    ) -> str:
        """LLM 종합 분석"""

        if not self.llm:
            return "LLM 비활성화 - 자동 분석 없음"

        try:
            # 분석 데이터 요약
            analysis_summary = {
                'filename': file_info.filename,
                'size': file_info.file_size,
                'type': file_info.file_type,
                'mime': file_info.mime_type,
                'is_executable': file_info.is_executable,
                'md5': file_info.md5_hash
            }

            # 전문 분석 결과 요약 (너무 길면 잘라내기)
            expert_summary = {}
            for key, value in expert_analysis.items():
                if isinstance(value, str):
                    expert_summary[key] = value[:1000] if len(value) > 1000 else value
                elif isinstance(value, dict):
                    expert_summary[key] = {k: str(v)[:200] for k, v in list(value.items())[:10]}
                elif isinstance(value, list):
                    expert_summary[key] = [str(item)[:200] for item in value[:10]]
                else:
                    expert_summary[key] = value

            prompt = f"""당신은 CTF 챌린지 파일 분석 전문가입니다.

**챌린지 정보**:
- 제목: {challenge_title or 'Unknown'}
- 카테고리: {challenge_category or 'Unknown'}
- 설명: {challenge_description[:500] if challenge_description else 'None'}

**파일 정보**:
{json.dumps(analysis_summary, indent=2, ensure_ascii=False)}

**분석 결과**:
{json.dumps(expert_summary, indent=2, ensure_ascii=False)[:3000]}

**요청사항**:
다음 형식으로 실용적인 분석을 제공하세요:

1. **파일 개요** (2-3줄)
   - 이 파일이 무엇인지
   - 챌린지와의 관계

2. **주요 발견사항** (3-5개 불릿)
   - 눈에 띄는 특징
   - 의심스러운 부분
   - 힌트가 될 만한 요소

3. **분석 접근법**
   - 도구: (3-5개 추천)
   - 확인 포인트: (3-5개)
   - 분석 순서: (3단계)

4. **보안 주의사항** (있으면)
   - 실행 위험성
   - 필요한 격리 수준

5. **다음 단계** (우선순위 3개)
   - 1. ...
   - 2. ...
   - 3. ...

**중요**: 구체적이고 실행 가능한 조언을 제공하세요. 학생이 이해할 수 있도록 쉽게 설명하세요."""

            response = await self.llm.generate(prompt)
            return response.strip()

        except Exception as e:
            console.print(f"[yellow]LLM 분석 오류: {str(e)}[/yellow]")
            return f"LLM 분석 실패: {str(e)}"

    def calculate_entropy(self, file_path: Path) -> float:
        """파일 엔트로피 계산 (암호화/패킹 감지)"""
        try:
            import math
            from collections import Counter

            with open(file_path, 'rb') as f:
                data = f.read(100000)  # 최대 100KB

            if not data:
                return 0.0

            # 바이트 빈도 계산
            counter = Counter(data)
            length = len(data)

            # Shannon entropy
            entropy = 0.0
            for count in counter.values():
                probability = count / length
                entropy -= probability * math.log2(probability)

            return entropy

        except Exception as e:
            console.print(f"[dim yellow]엔트로피 계산 실패: {str(e)}[/dim yellow]")
            return 0.0

    async def check_security(self, file_path: Path) -> Dict:
        """보안 체크"""
        checks = {
            'is_executable': False,
            'entropy': 0.0,
            'suspicious_strings': [],
            'safe_to_analyze': True,
            'warnings': []
        }

        try:
            # 실행 파일 체크
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
            checks['is_executable'] = (
                mime_type.startswith('application/x-') or
                file_path.suffix in ['.exe', '.dll', '.so', '.elf']
            )

            # 엔트로피 계산
            checks['entropy'] = self.calculate_entropy(file_path)

            # 의심스러운 문자열 체크
            suspicious_keywords = [
                'eval', 'exec', 'system', 'shell', 'cmd',
                'password', 'token', 'api_key', 'secret'
            ]

            try:
                result = subprocess.run(
                    ['strings', str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                strings_output = result.stdout.lower()

                for keyword in suspicious_keywords:
                    if keyword in strings_output:
                        checks['suspicious_strings'].append(keyword)

            except:
                pass

            # 위험도 평가
            if checks['is_executable']:
                checks['warnings'].append('실행 파일 - 샌드박스 환경에서만 실행하세요')

            if checks['entropy'] > 7.5:
                checks['warnings'].append('높은 엔트로피 - 암호화 또는 패킹된 파일 가능성')
                checks['safe_to_analyze'] = False

            if len(checks['suspicious_strings']) > 3:
                checks['warnings'].append('의심스러운 문자열 다수 포함')

        except Exception as e:
            console.print(f"[yellow]보안 체크 오류: {str(e)}[/yellow]")

        return checks


# 테스트용
async def test_analyzer():
    """분석기 테스트"""
    analyzer = FileAnalyzer()

    # 테스트 파일 경로
    test_file = Path("/path/to/test/file")

    if not test_file.exists():
        console.print("[yellow]테스트 파일이 없습니다[/yellow]")
        return

    # 기본 분석
    file_info = await analyzer.analyze_basic_info(test_file)
    console.print(f"\n파일: {file_info.filename}")
    console.print(f"크기: {file_info.file_size} bytes")
    console.print(f"타입: {file_info.file_type}")
    console.print(f"MD5: {file_info.md5_hash}")

    # 타입별 분석
    expert_analysis = await analyzer.analyze_by_type(
        test_file,
        file_info.file_type,
        file_info.mime_type
    )
    console.print(f"\n전문 분석:")
    console.print(json.dumps(expert_analysis, indent=2, ensure_ascii=False))

    # 보안 체크
    security = await analyzer.check_security(test_file)
    console.print(f"\n보안 체크:")
    console.print(json.dumps(security, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(test_analyzer())
