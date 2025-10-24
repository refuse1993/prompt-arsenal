"""
Security API Blueprint
Security scanner results and management
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB

security_bp = Blueprint('security', __name__)
db = ArsenalDB()


@security_bp.route('/scan-results', methods=['GET'])
def get_scan_results():
    """Get security scan results"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)

    offset = (page - 1) * limit

    results = db.get_security_scan_results(limit=limit, offset=offset)
    total = db.get_security_scan_count()

    # Map database fields to frontend expected fields
    mapped_results = []
    for scan in results:
        mapped_results.append({
            'id': scan['id'],
            'tool': scan.get('mode', 'static').upper(),  # mode -> tool
            'target': scan['target'],
            'vulnerabilities_found': scan.get('total_findings', 0),  # total_findings -> vulnerabilities_found
            'status': 'completed' if scan.get('completed_at') else 'running',  # derive status
            'scanned_at': scan.get('started_at', ''),  # started_at -> scanned_at
            'critical_count': scan.get('critical_count', 0),
            'high_count': scan.get('high_count', 0),
            'medium_count': scan.get('medium_count', 0),
            'low_count': scan.get('low_count', 0),
            'scan_duration': scan.get('scan_duration', 0),
            'llm_calls': scan.get('llm_calls', 0),
            'llm_cost': scan.get('llm_cost', 0.0)
        })

    return jsonify({
        'success': True,
        'data': mapped_results,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })


@security_bp.route('/vulnerabilities', methods=['GET'])
def get_vulnerabilities():
    """Get detected vulnerabilities"""
    severity = request.args.get('severity', None)

    vulnerabilities = db.get_vulnerabilities(severity=severity)

    # Map database fields to frontend expected fields
    mapped_vulns = []
    for vuln in vulnerabilities:
        mapped_vulns.append({
            'id': vuln['id'],
            'severity': vuln['severity'],
            'type': vuln.get('cwe_name', vuln.get('cwe_id', 'Unknown')),  # cwe_name -> type
            'description': vuln.get('description', vuln.get('title', '')),
            'file': vuln.get('file_path', ''),  # file_path -> file
            'line': vuln.get('line_number', 0),  # line_number -> line
            'recommendation': vuln.get('remediation', ''),  # remediation -> recommendation
            'cwe_id': vuln.get('cwe_id', ''),
            'confidence': vuln.get('confidence', 0),
            'verified_by': vuln.get('verified_by', 'tool'),
            'is_false_positive': vuln.get('is_false_positive', 0)
        })

    return jsonify({
        'success': True,
        'data': mapped_vulns
    })


@security_bp.route('/tools', methods=['GET'])
def get_tools():
    """Get available security tools"""
    tools = [
        {
            'name': 'Garak',
            'description': 'NVIDIA LLM 취약점 스캐너',
            'version': '0.9.0',
            'enabled': True
        },
        {
            'name': 'Semgrep',
            'description': '정적 코드 분석',
            'version': '1.45.0',
            'enabled': True
        },
        {
            'name': 'Bandit',
            'description': 'Python 보안 검사',
            'version': '1.7.5',
            'enabled': True
        }
    ]

    return jsonify({
        'success': True,
        'data': tools
    })


@security_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get security statistics"""
    stats = {
        'total_scans': db.get_security_scan_count(),
        'critical_vulns': db.get_vulnerability_count('critical'),
        'high_vulns': db.get_vulnerability_count('high'),
        'medium_vulns': db.get_vulnerability_count('medium'),
        'low_vulns': db.get_vulnerability_count('low')
    }

    return jsonify({
        'success': True,
        'data': stats
    })
