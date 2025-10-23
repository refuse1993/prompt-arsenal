"""
Port Scanner using python3-nmap
포트 스캔 및 서비스 버전 탐지
"""

import asyncio
import subprocess
from typing import List, Dict, Optional
from dataclasses import dataclass
import socket


@dataclass
class PortInfo:
    """포트 정보"""
    port: int
    protocol: str
    state: str
    service: str
    version: str = ""
    banner: str = ""


class PortScanner:
    """포트 스캐너 (nmap 기반)"""

    def __init__(self):
        self.nmap_available = False
        self._check_nmap()

    def _check_nmap(self):
        """nmap 설치 여부 확인"""
        try:
            result = subprocess.run(
                ['nmap', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.nmap_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.nmap_available = False

    async def scan(self, target: str, scan_type: str = "standard") -> List[PortInfo]:
        """
        포트 스캔 실행

        Args:
            target: IP 주소 또는 도메인
            scan_type: quick, standard, full

        Returns:
            포트 정보 리스트
        """
        if self.nmap_available:
            return await self._nmap_scan(target, scan_type)
        else:
            print("  ⚠️  nmap 미설치 - 기본 스캔 모드 사용")
            return await self._basic_scan(target)

    async def _nmap_scan(self, target: str, scan_type: str) -> List[PortInfo]:
        """nmap을 사용한 포트 스캔"""
        # 스캔 타입별 옵션
        scan_options = {
            "quick": "-F",  # Fast scan (100 common ports)
            "standard": "-p 1-1000",  # Top 1000 ports
            "full": "-p-"  # All 65535 ports
        }

        # nmap 명령어 구성
        cmd = [
            'nmap',
            '-sV',  # Service version detection
            '--script', 'vulners',  # Vulners CVE 스크립트
            scan_options.get(scan_type, "-F"),
            '-oX', '-',  # XML output to stdout
            target
        ]

        try:
            # nmap 실행
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"  ⚠️  nmap 오류: {stderr.decode()}")
                return await self._basic_scan(target)

            # XML 파싱
            return self._parse_nmap_xml(stdout.decode())

        except Exception as e:
            print(f"  ⚠️  nmap 실행 실패: {e}")
            return await self._basic_scan(target)

    def _parse_nmap_xml(self, xml_output: str) -> List[PortInfo]:
        """nmap XML 출력 파싱"""
        ports = []

        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_output)

            # 각 포트 정보 추출
            for port_elem in root.findall('.//port'):
                state_elem = port_elem.find('state')

                # 열린 포트만 처리
                if state_elem is not None and state_elem.get('state') == 'open':
                    service_elem = port_elem.find('service')

                    port_info = PortInfo(
                        port=int(port_elem.get('portid')),
                        protocol=port_elem.get('protocol'),
                        state='open',
                        service=service_elem.get('name', 'unknown') if service_elem is not None else 'unknown',
                        version=service_elem.get('version', '') if service_elem is not None else ''
                    )

                    # 배너 정보 추가
                    if service_elem is not None:
                        product = service_elem.get('product', '')
                        version = service_elem.get('version', '')
                        if product and version:
                            port_info.banner = f"{product} {version}"
                        elif product:
                            port_info.banner = product

                    ports.append(port_info)

        except Exception as e:
            print(f"  ⚠️  XML 파싱 오류: {e}")

        return ports

    async def _basic_scan(self, target: str) -> List[PortInfo]:
        """기본 소켓 기반 포트 스캔 (nmap 없을 때)"""
        # 일반적인 포트 목록
        common_ports = [
            (21, 'ftp'), (22, 'ssh'), (23, 'telnet'),
            (25, 'smtp'), (53, 'dns'), (80, 'http'),
            (110, 'pop3'), (143, 'imap'), (443, 'https'),
            (445, 'smb'), (3306, 'mysql'), (3389, 'rdp'),
            (5432, 'postgresql'), (6379, 'redis'), (8080, 'http-proxy'),
            (8443, 'https-alt'), (27017, 'mongodb')
        ]

        open_ports = []
        tasks = []

        # 병렬 포트 체크
        for port, service in common_ports:
            task = self._check_port(target, port, service)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # 열린 포트만 필터링
        open_ports = [port_info for port_info in results if port_info is not None]

        return open_ports

    async def _check_port(self, target: str, port: int, service: str,
                         timeout: float = 1.0) -> Optional[PortInfo]:
        """개별 포트 체크"""
        try:
            # TCP 연결 시도
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port),
                timeout=timeout
            )

            writer.close()
            await writer.wait_closed()

            # 배너 그래빙 시도
            banner = await self._grab_banner(target, port)

            return PortInfo(
                port=port,
                protocol='tcp',
                state='open',
                service=service,
                banner=banner
            )

        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return None

    async def _grab_banner(self, target: str, port: int, timeout: float = 2.0) -> str:
        """배너 그래빙"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target, port),
                timeout=timeout
            )

            # HTTP/HTTPS인 경우 GET 요청 전송
            if port in [80, 8080, 443, 8443]:
                request = f'GET / HTTP/1.1\r\nHost: {target}\r\n\r\n'
                writer.write(request.encode())
                await writer.drain()

            # 응답 읽기
            data = await asyncio.wait_for(reader.read(1024), timeout=timeout)
            banner = data.decode('utf-8', errors='ignore').strip()

            writer.close()
            await writer.wait_closed()

            # 첫 줄만 추출 (배너)
            if banner:
                first_line = banner.split('\n')[0].strip()
                return first_line[:200]  # 최대 200자

            return ""

        except Exception:
            return ""

    def get_service_info(self, port: int) -> Dict[str, str]:
        """포트 번호로 서비스 정보 조회"""
        well_known_ports = {
            21: {'service': 'ftp', 'description': 'File Transfer Protocol'},
            22: {'service': 'ssh', 'description': 'Secure Shell'},
            23: {'service': 'telnet', 'description': 'Telnet (Insecure)'},
            25: {'service': 'smtp', 'description': 'Simple Mail Transfer Protocol'},
            53: {'service': 'dns', 'description': 'Domain Name System'},
            80: {'service': 'http', 'description': 'HTTP Web Server'},
            110: {'service': 'pop3', 'description': 'Post Office Protocol v3'},
            143: {'service': 'imap', 'description': 'Internet Message Access Protocol'},
            443: {'service': 'https', 'description': 'HTTPS Secure Web'},
            445: {'service': 'smb', 'description': 'SMB/CIFS File Sharing'},
            3306: {'service': 'mysql', 'description': 'MySQL Database'},
            3389: {'service': 'rdp', 'description': 'Remote Desktop Protocol'},
            5432: {'service': 'postgresql', 'description': 'PostgreSQL Database'},
            6379: {'service': 'redis', 'description': 'Redis Database'},
            8080: {'service': 'http-proxy', 'description': 'HTTP Proxy'},
            8443: {'service': 'https-alt', 'description': 'HTTPS Alternative'},
            27017: {'service': 'mongodb', 'description': 'MongoDB Database'}
        }

        return well_known_ports.get(port, {
            'service': 'unknown',
            'description': 'Unknown service'
        })

    def check_dangerous_services(self, ports: List[PortInfo]) -> List[Dict]:
        """위험한 서비스 체크"""
        dangerous = []

        dangerous_services = {
            21: {'severity': 'high', 'reason': 'FTP는 평문 인증을 사용합니다 (SFTP 사용 권장)'},
            23: {'severity': 'critical', 'reason': 'Telnet는 암호화되지 않은 프로토콜입니다 (SSH 사용 권장)'},
            445: {'severity': 'medium', 'reason': 'SMB는 랜섬웨어 공격에 자주 사용됩니다'},
            3389: {'severity': 'medium', 'reason': 'RDP는 브루트포스 공격에 취약합니다 (강력한 인증 필요)'},
        }

        for port_info in ports:
            if port_info.port in dangerous_services:
                info = dangerous_services[port_info.port]
                dangerous.append({
                    'port': port_info.port,
                    'service': port_info.service,
                    'severity': info['severity'],
                    'reason': info['reason']
                })

        return dangerous
