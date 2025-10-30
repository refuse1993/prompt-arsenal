"""
CTF API Blueprint
CTF challenge and test result management
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB

ctf_bp = Blueprint('ctf', __name__)
db = ArsenalDB()


@ctf_bp.route('/challenges', methods=['GET'])
def list_challenges():
    """Get all CTF challenges"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    category = request.args.get('category', None)
    competition = request.args.get('competition', None)  # Filter by source (competition name)

    offset = (page - 1) * limit

    challenges = db.get_ctf_challenges(
        category=category,
        source=competition,  # Use source field for competition filtering
        limit=limit,
        offset=offset
    )
    total = db.get_ctf_challenge_count(category=category)

    return jsonify({
        'success': True,
        'data': challenges,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })


@ctf_bp.route('/challenges/<int:challenge_id>', methods=['GET'])
def get_challenge(challenge_id):
    """Get single challenge details with files"""
    challenge = db.get_ctf_challenge_by_id(challenge_id)

    if not challenge:
        return jsonify({'success': False, 'error': 'Challenge not found'}), 404

    # Get attached files
    files = db.get_challenge_files(challenge_id)
    challenge['files'] = files

    return jsonify({
        'success': True,
        'data': challenge
    })


@ctf_bp.route('/challenges/<int:challenge_id>/files', methods=['GET'])
def get_challenge_files(challenge_id):
    """Get files attached to a challenge"""
    files = db.get_challenge_files(challenge_id)

    return jsonify({
        'success': True,
        'data': files
    })


@ctf_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get CTF categories with real data"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 실제 DB에서 카테고리별 통계 가져오기
    stats = cursor.execute('''
        SELECT category,
               COUNT(*) as count,
               SUM(CASE WHEN status = 'solved' THEN 1 ELSE 0 END) as solved
        FROM ctf_challenges
        GROUP BY category
    ''').fetchall()

    categories = []
    for row in stats:
        total = row['count']
        solved = row['solved']
        success_rate = (solved / total) if total > 0 else 0.0

        categories.append({
            'name': row['category'],
            'display': row['category'].replace('_', ' ').title(),
            'count': total,
            'solved': solved,
            'success_rate': round(success_rate, 2)
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': categories
    })


@ctf_bp.route('/test-results', methods=['GET'])
def get_test_results():
    """Get CTF test results"""
    challenge_id = request.args.get('challenge_id', type=int)

    if challenge_id:
        results = db.get_ctf_test_results(challenge_id)
    else:
        results = db.get_all_ctf_test_results()

    return jsonify({
        'success': True,
        'data': results
    })


@ctf_bp.route('/competitions', methods=['GET'])
def get_competitions():
    """Get available CTF competitions grouped by source field"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Source 필드 기반으로 그룹화 (competition name)
    stats = cursor.execute('''
        SELECT source,
               COUNT(*) as challenge_count,
               SUM(CASE WHEN status = 'solved' THEN 1 ELSE 0 END) as solved_count,
               MIN(created_at) as first_added,
               COUNT(DISTINCT category) as category_count
        FROM ctf_challenges
        WHERE source IS NOT NULL AND source != ''
        GROUP BY source
        ORDER BY first_added DESC
    ''').fetchall()

    competitions = []
    for row in stats:
        competitions.append({
            'id': row['source'],
            'name': row['source'],
            'challenges_found': row['challenge_count'],
            'solved': row['solved_count'],
            'category_count': row['category_count'],
            'first_added': row['first_added'],
            'status': 'active' if row['challenge_count'] > 0 else 'inactive',
            'url': ''  # URL is not stored in current schema
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': competitions
    })


@ctf_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get CTF statistics"""
    stats = {
        'total_challenges': db.get_ctf_challenge_count(),
        'total_tests': db.get_ctf_test_count(),
        'avg_success_rate': db.get_ctf_avg_success_rate(),
        'playwright_enabled': True
    }

    return jsonify({
        'success': True,
        'data': stats
    })
