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


@stats_bp.route('/classification', methods=['GET'])
def get_classification_stats():
    """Get classification-based statistics (purpose, risk_category, technique)"""
    import sqlite3

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Purpose statistics
    cursor.execute('''
        SELECT
            p.purpose,
            COUNT(DISTINCT p.id) as prompt_count,
            COUNT(tr.id) as test_count,
            SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE p.purpose IS NOT NULL
        GROUP BY p.purpose
        ORDER BY prompt_count DESC
    ''')
    purpose_stats = [dict(row) for row in cursor.fetchall()]

    # Risk category statistics
    cursor.execute('''
        SELECT
            p.risk_category,
            COUNT(DISTINCT p.id) as prompt_count,
            COUNT(tr.id) as test_count,
            SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE p.risk_category IS NOT NULL
        GROUP BY p.risk_category
        ORDER BY prompt_count DESC
    ''')
    risk_stats = [dict(row) for row in cursor.fetchall()]

    # Technique statistics
    cursor.execute('''
        SELECT
            p.technique,
            COUNT(DISTINCT p.id) as prompt_count,
            COUNT(tr.id) as test_count,
            SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE p.technique IS NOT NULL
        GROUP BY p.technique
        ORDER BY prompt_count DESC
        LIMIT 10
    ''')
    technique_stats = [dict(row) for row in cursor.fetchall()]

    # Modality statistics
    cursor.execute('''
        SELECT
            p.modality,
            COUNT(DISTINCT p.id) as prompt_count,
            COUNT(tr.id) as test_count,
            SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
            AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE p.modality IS NOT NULL
        GROUP BY p.modality
        ORDER BY prompt_count DESC
    ''')
    modality_stats = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'purpose': purpose_stats,
            'risk_category': risk_stats,
            'technique': technique_stats,
            'modality': modality_stats
        }
    })


@stats_bp.route('/categories', methods=['GET'])
def get_categories_with_classification():
    """Get categories with classification info"""
    categories = db.get_categories()

    # Group by purpose
    offensive = [c for c in categories if c.get('purpose') == 'offensive']
    defensive = [c for c in categories if c.get('purpose') == 'defensive']

    return jsonify({
        'success': True,
        'data': {
            'all': categories,
            'offensive': offensive,
            'defensive': defensive,
            'total': len(categories)
        }
    })
