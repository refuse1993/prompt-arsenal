"""
System Security Scanner
Docker, Kubernetes, Port Scanning, CVE Detection
"""

import subprocess
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class SystemScanner:
    """
    System Security Scanner

    Features:
    - Docker image vulnerability scan (Trivy)
    - Kubernetes cluster scan (kube-bench, kube-hunter)
    - Port scanning (nmap)
    - CVE search and analysis
    """

    def __init__(self, db=None):
        self.db = db
        self.scan_results = {}

    def check_tool_installed(self, tool_name: str) -> bool:
        """Check if security tool is installed"""
        try:
            result = subprocess.run(
                ['which', tool_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def scan_docker_image(self, image_name: str, severity: str = "HIGH,CRITICAL") -> Dict:
        """
        Scan Docker image for vulnerabilities using Trivy

        Parameters:
        - image_name: Docker image name (e.g., 'nginx:latest')
        - severity: Vulnerability severity filter (LOW,MEDIUM,HIGH,CRITICAL)

        Returns:
        - scan_result: {
            'image': str,
            'vulnerabilities': List[Dict],
            'summary': Dict,
            'tool': 'trivy'
        }
        """
        if not self.check_tool_installed('trivy'):
            return {
                'success': False,
                'error': 'Trivy not installed. Install: brew install trivy'
            }

        try:
            # Run Trivy scan with JSON output
            cmd = [
                'trivy', 'image',
                '--format', 'json',
                '--severity', severity,
                '--no-progress',
                image_name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'Trivy scan failed: {result.stderr}'
                }

            # Parse JSON output
            scan_data = json.loads(result.stdout)

            vulnerabilities = []
            total_high = 0
            total_critical = 0

            # Extract vulnerabilities
            for target_result in scan_data.get('Results', []):
                for vuln in target_result.get('Vulnerabilities', []):
                    vuln_info = {
                        'cve_id': vuln.get('VulnerabilityID'),
                        'package': vuln.get('PkgName'),
                        'version': vuln.get('InstalledVersion'),
                        'fixed_version': vuln.get('FixedVersion'),
                        'severity': vuln.get('Severity'),
                        'title': vuln.get('Title'),
                        'description': vuln.get('Description', '')[:200],
                        'references': vuln.get('References', [])
                    }
                    vulnerabilities.append(vuln_info)

                    if vuln.get('Severity') == 'HIGH':
                        total_high += 1
                    elif vuln.get('Severity') == 'CRITICAL':
                        total_critical += 1

            return {
                'success': True,
                'image': image_name,
                'tool': 'trivy',
                'vulnerabilities': vulnerabilities,
                'summary': {
                    'total': len(vulnerabilities),
                    'critical': total_critical,
                    'high': total_high
                },
                'scan_time': datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Scan timeout (5 minutes)'}
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'Failed to parse Trivy output: {e}'}
        except Exception as e:
            return {'success': False, 'error': f'Scan error: {e}'}

    def scan_kubernetes_cluster(self, context: str = None) -> Dict:
        """
        Scan Kubernetes cluster for security issues

        Uses:
        - kube-bench: CIS Kubernetes Benchmark checks
        - kube-hunter: Penetration testing

        Parameters:
        - context: kubectl context name (optional)

        Returns:
        - scan_result: {
            'bench_results': Dict,  # kube-bench output
            'hunter_results': Dict,  # kube-hunter output
            'summary': Dict
        }
        """
        results = {
            'success': True,
            'tool': 'kubernetes',
            'bench_results': None,
            'hunter_results': None,
            'summary': {}
        }

        # Check kube-bench
        if self.check_tool_installed('kube-bench'):
            try:
                cmd = ['kube-bench', 'run', '--json']
                if context:
                    cmd.extend(['--context', context])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    bench_data = json.loads(result.stdout)
                    results['bench_results'] = bench_data

                    # Extract summary
                    total_fail = bench_data.get('Totals', {}).get('total_fail', 0)
                    total_warn = bench_data.get('Totals', {}).get('total_warn', 0)
                    results['summary']['kube_bench_failures'] = total_fail
                    results['summary']['kube_bench_warnings'] = total_warn

            except Exception as e:
                results['bench_error'] = str(e)
        else:
            results['bench_error'] = 'kube-bench not installed'

        # Check kube-hunter
        if self.check_tool_installed('kube-hunter'):
            try:
                cmd = ['kube-hunter', '--remote', '--json']

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180
                )

                if result.returncode == 0:
                    hunter_data = json.loads(result.stdout)
                    results['hunter_results'] = hunter_data

                    # Extract vulnerabilities
                    vulnerabilities = hunter_data.get('vulnerabilities', [])
                    results['summary']['kube_hunter_vulns'] = len(vulnerabilities)

            except Exception as e:
                results['hunter_error'] = str(e)
        else:
            results['hunter_error'] = 'kube-hunter not installed'

        results['scan_time'] = datetime.now().isoformat()
        return results

    def scan_ports(self, target: str, ports: str = "1-1000") -> Dict:
        """
        Scan network ports using nmap

        Parameters:
        - target: IP address or hostname
        - ports: Port range (e.g., "1-1000", "22,80,443")

        Returns:
        - scan_result: {
            'target': str,
            'open_ports': List[Dict],
            'services': Dict,
            'summary': Dict
        }
        """
        if not self.check_tool_installed('nmap'):
            return {
                'success': False,
                'error': 'nmap not installed. Install: brew install nmap'
            }

        try:
            # Run nmap with service version detection
            cmd = [
                'nmap',
                '-sV',  # Service version detection
                '-p', ports,
                '--open',  # Only show open ports
                '-oX', '-',  # XML output to stdout
                target
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'nmap scan failed: {result.stderr}'
                }

            # Parse XML output (simple regex parsing)
            open_ports = []
            services = {}

            # Extract port information
            port_pattern = r'<port protocol="(\w+)" portid="(\d+)">.*?<state state="open".*?<service name="([^"]*)".*?product="([^"]*)"'
            for match in re.finditer(port_pattern, result.stdout, re.DOTALL):
                protocol, port, service, product = match.groups()
                port_info = {
                    'port': int(port),
                    'protocol': protocol,
                    'service': service,
                    'product': product
                }
                open_ports.append(port_info)
                services[f"{protocol}/{port}"] = f"{service} ({product})"

            return {
                'success': True,
                'target': target,
                'tool': 'nmap',
                'open_ports': open_ports,
                'services': services,
                'summary': {
                    'total_open_ports': len(open_ports),
                    'scan_range': ports
                },
                'scan_time': datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'nmap scan timeout (5 minutes)'}
        except Exception as e:
            return {'success': False, 'error': f'Scan error: {e}'}

    def search_cve(self, cve_id: str) -> Dict:
        """
        Search CVE information from NVD

        Parameters:
        - cve_id: CVE identifier (e.g., 'CVE-2024-1234')

        Returns:
        - cve_info: {
            'cve_id': str,
            'description': str,
            'severity': str,
            'score': float,
            'published': str,
            'references': List[str]
        }
        """
        # This would require NVD API integration
        # For now, return placeholder
        return {
            'success': False,
            'error': 'CVE search not implemented. Use: https://nvd.nist.gov/vuln/detail/' + cve_id
        }

    def save_scan_to_db(self, scan_type: str, target: str, result: Dict, llm_analysis: str = None) -> Optional[int]:
        """Save scan result to database"""
        if not self.db:
            return None

        try:
            # Calculate risk score
            risk_score = 0
            if scan_type == 'docker':
                risk_score = (
                    result['summary'].get('critical', 0) * 10 +
                    result['summary'].get('high', 0) * 5
                )
            elif scan_type == 'kubernetes':
                risk_score = (
                    result['summary'].get('kube_bench_failures', 0) * 5 +
                    result['summary'].get('kube_hunter_vulns', 0) * 8
                )
            elif scan_type == 'ports':
                risk_score = result['summary'].get('total_open_ports', 0) * 2

            # Extract scan times
            scan_time = result.get('scan_time', datetime.now().isoformat())

            # Extract open ports and services
            open_ports = []
            services = {}

            if scan_type == 'ports' and result.get('open_ports'):
                open_ports = result['open_ports']
                services = result.get('services', {})
            elif scan_type == 'docker' and result.get('vulnerabilities'):
                # For docker, store vulnerability packages as "services"
                for vuln in result['vulnerabilities'][:20]:  # Limit to top 20
                    services[vuln['package']] = f"{vuln['cve_id']} ({vuln['severity']})"
            elif scan_type == 'kubernetes':
                # For k8s, store summary as services
                if result.get('bench_results'):
                    services['kube-bench'] = f"Failures: {result['summary'].get('kube_bench_failures', 0)}"
                if result.get('hunter_results'):
                    services['kube-hunter'] = f"Vulnerabilities: {result['summary'].get('kube_hunter_vulns', 0)}"

            # Save to database
            scan_id = self.db.insert_system_scan(
                target=target,
                scan_type=scan_type,
                start_time=scan_time,
                end_time=scan_time,
                open_ports=json.dumps(open_ports, ensure_ascii=False),
                services=json.dumps(services, ensure_ascii=False),
                findings=json.dumps(result, ensure_ascii=False),
                risk_score=min(risk_score, 100),  # Cap at 100
                llm_analysis=llm_analysis
            )

            return scan_id

        except Exception as e:
            print(f"Failed to save scan: {e}")
            return None

    def get_scan_summary(self) -> str:
        """Generate human-readable scan summary"""
        summary_lines = []

        for scan_type, result in self.scan_results.items():
            if not result.get('success'):
                continue

            summary_lines.append(f"\n## {scan_type.upper()} Scan")

            if scan_type == 'docker':
                summary_lines.append(f"Image: {result['image']}")
                summary_lines.append(f"Total Vulnerabilities: {result['summary']['total']}")
                summary_lines.append(f"  - Critical: {result['summary']['critical']}")
                summary_lines.append(f"  - High: {result['summary']['high']}")

            elif scan_type == 'kubernetes':
                if result.get('bench_results'):
                    summary_lines.append(f"CIS Benchmark Failures: {result['summary'].get('kube_bench_failures', 0)}")
                if result.get('hunter_results'):
                    summary_lines.append(f"Security Issues: {result['summary'].get('kube_hunter_vulns', 0)}")

            elif scan_type == 'ports':
                summary_lines.append(f"Target: {result['target']}")
                summary_lines.append(f"Open Ports: {result['summary']['total_open_ports']}")
                for port_info in result['open_ports'][:10]:
                    summary_lines.append(f"  - {port_info['port']}/{port_info['protocol']}: {port_info['service']}")

        return '\n'.join(summary_lines)


def get_installed_scanners() -> Dict[str, bool]:
    """Check which security scanners are installed"""
    scanner = SystemScanner()
    return {
        'trivy': scanner.check_tool_installed('trivy'),
        'nmap': scanner.check_tool_installed('nmap'),
        'kube-bench': scanner.check_tool_installed('kube-bench'),
        'kube-hunter': scanner.check_tool_installed('kube-hunter')
    }
