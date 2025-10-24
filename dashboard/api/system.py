"""
System API Blueprint
System scanner and network security
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB

system_bp = Blueprint('system', __name__)
db = ArsenalDB()


@system_bp.route('/scan-results', methods=['GET'])
def get_scan_results():
    """Get system scan results"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)

    offset = (page - 1) * limit

    results = db.get_system_scan_results(limit=limit, offset=offset)
    total = db.get_system_scan_count()

    return jsonify({
        'success': True,
        'data': results,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })


@system_bp.route('/hosts', methods=['GET'])
def get_hosts():
    """Get scanned hosts"""
    hosts = db.get_scanned_hosts()

    return jsonify({
        'success': True,
        'data': hosts
    })


@system_bp.route('/ports', methods=['GET'])
def get_ports():
    """Get open ports"""
    host = request.args.get('host', None)

    ports = db.get_open_ports(host=host)

    return jsonify({
        'success': True,
        'data': ports
    })


@system_bp.route('/cves', methods=['GET'])
def get_cves():
    """Get detected CVEs"""
    severity = request.args.get('severity', None)

    cves = db.get_detected_cves(severity=severity)

    return jsonify({
        'success': True,
        'data': cves
    })


@system_bp.route('/tools', methods=['GET'])
def get_tools():
    """Get available system tools"""
    tools = [
        {
            'name': 'Nmap',
            'description': '네트워크 포트 스캐너',
            'version': '7.94',
            'enabled': True
        },
        {
            'name': 'Vulners',
            'description': 'CVE 취약점 데이터베이스',
            'version': '2.1.0',
            'enabled': True
        }
    ]

    return jsonify({
        'success': True,
        'data': tools
    })


@system_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system scanner statistics"""
    stats = {
        'total_scans': db.get_system_scan_count(),
        'total_hosts': db.get_host_count(),
        'total_ports': db.get_port_count(),
        'critical_cves': db.get_cve_count('critical'),
        'high_cves': db.get_cve_count('high')
    }

    return jsonify({
        'success': True,
        'data': stats
    })
