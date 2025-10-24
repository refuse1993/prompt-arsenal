"""
Stats API Blueprint
Global statistics and analytics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify
from core.database import ArsenalDB

stats_bp = Blueprint('stats', __name__)
db = ArsenalDB()


@stats_bp.route('/overview', methods=['GET'])
def get_overview():
    """Get dashboard overview statistics"""
    stats = db.get_stats()

    overview = {
        'prompts': {
            'total': stats.get('total_prompts', 0),
            'categories': stats.get('categories', 0),
            'tests': stats.get('text_tests', 0)
        },
        'multimodal': {
            'total': stats.get('total_media', 0),
            'images': stats.get('images', 0),
            'audio': stats.get('audio', 0),
            'video': stats.get('video', 0),
            'tests': stats.get('multimodal_tests', 0)
        },
        'multiturn': {
            'campaigns': stats.get('campaigns', 0),
            'avg_asr': stats.get('avg_multiturn_asr', 0)
        },
        'ctf': {
            'challenges': stats.get('ctf_challenges', 0),
            'tests': stats.get('ctf_tests', 0),
            'success_rate': stats.get('ctf_success_rate', 0)
        },
        'security': {
            'scans': stats.get('security_scans', 0),
            'vulnerabilities': stats.get('vulnerabilities', 0)
        },
        'system': {
            'scans': stats.get('system_scans', 0),
            'hosts': stats.get('hosts', 0),
            'cves': stats.get('cves', 0)
        }
    }

    return jsonify({
        'success': True,
        'data': overview
    })


@stats_bp.route('/activity', methods=['GET'])
def get_activity():
    """Get recent activity"""
    limit = 20

    activity = []

    # Recent text tests
    recent_text = db.get_recent_test_results(limit=limit)
    for test in recent_text:
        activity.append({
            'type': 'text_test',
            'timestamp': test.get('tested_at'),
            'description': f"Text test on {test.get('model_name')}",
            'success': test.get('success')
        })

    # Recent multimodal tests
    recent_multimodal = db.get_recent_multimodal_tests(limit=limit)
    for test in recent_multimodal:
        activity.append({
            'type': 'multimodal_test',
            'timestamp': test.get('tested_at'),
            'description': f"Multimodal test on {test.get('model_name')}",
            'success': test.get('success')
        })

    # Sort by timestamp
    activity.sort(key=lambda x: x['timestamp'], reverse=True)

    return jsonify({
        'success': True,
        'data': activity[:limit]
    })


@stats_bp.route('/success-rates', methods=['GET'])
def get_success_rates():
    """Get success rates by category"""
    text_rates = db.get_success_rates_by_category()
    multimodal_rates = db.get_multimodal_success_rates()
    ctf_rates = db.get_ctf_success_rates()

    return jsonify({
        'success': True,
        'data': {
            'text': text_rates,
            'multimodal': multimodal_rates,
            'ctf': ctf_rates
        }
    })


@stats_bp.route('/timeline', methods=['GET'])
def get_timeline():
    """Get activity timeline (last 30 days)"""
    days = 30

    timeline = db.get_activity_timeline(days=days)

    return jsonify({
        'success': True,
        'data': timeline
    })
