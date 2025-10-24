"""
Multimodal API Blueprint
Multimodal attack management and testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB

multimodal_bp = Blueprint('multimodal', __name__)
db = ArsenalDB()


@multimodal_bp.route('/list', methods=['GET'])
def list_media():
    """Get all media with pagination"""
    import sqlite3
    from pathlib import Path

    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    media_type = request.args.get('media_type', None)
    attack_type = request.args.get('attack_type', None)
    model = request.args.get('model', None)

    offset = (page - 1) * limit

    # If model filter is provided, use custom query with JOIN
    if model:
        db_path = Path(__file__).parent.parent.parent / "arsenal.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT DISTINCT m.*
            FROM media_arsenal m
            LEFT JOIN multimodal_test_results t ON m.id = t.media_id
            WHERE 1=1
        '''
        params = []

        if media_type:
            query += ' AND m.media_type = ?'
            params.append(media_type)

        if attack_type:
            query += ' AND m.attack_type = ?'
            params.append(attack_type)

        if model:
            provider, model_name = model.split('/', 1) if '/' in model else (model, None)
            query += ' AND t.provider = ?'
            params.append(provider)
            if model_name:
                query += ' AND t.model = ?'
                params.append(model_name)

        query += ' ORDER BY m.created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        media = [dict(row) for row in cursor.fetchall()]

        # Get total count with same filters
        count_query = '''
            SELECT COUNT(DISTINCT m.id)
            FROM media_arsenal m
            LEFT JOIN multimodal_test_results t ON m.id = t.media_id
            WHERE 1=1
        '''
        count_params = []

        if media_type:
            count_query += ' AND m.media_type = ?'
            count_params.append(media_type)

        if attack_type:
            count_query += ' AND m.attack_type = ?'
            count_params.append(attack_type)

        if model:
            provider, model_name = model.split('/', 1) if '/' in model else (model, None)
            count_query += ' AND t.provider = ?'
            count_params.append(provider)
            if model_name:
                count_query += ' AND t.model = ?'
                count_params.append(model_name)

        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]
        conn.close()
    else:
        # Use existing database methods when no model filter
        media = db.get_all_media(
            media_type=media_type,
            attack_type=attack_type,
            limit=limit,
            offset=offset
        )
        total = db.get_media_count(media_type=media_type, attack_type=attack_type)

    return jsonify({
        'success': True,
        'data': media,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })


@multimodal_bp.route('/types', methods=['GET'])
def get_media_types():
    """Get media types with counts"""
    types = db.get_media_types()

    return jsonify({
        'success': True,
        'data': types
    })


@multimodal_bp.route('/attack-types', methods=['GET'])
def get_attack_types():
    """Get attack types with counts"""
    attack_types = db.get_attack_types()

    return jsonify({
        'success': True,
        'data': attack_types
    })


@multimodal_bp.route('/<int:media_id>', methods=['GET'])
def get_media(media_id):
    """Get single media details"""
    media = db.get_media_by_id(media_id)

    if not media:
        return jsonify({'success': False, 'error': 'Media not found'}), 404

    return jsonify({
        'success': True,
        'data': media
    })


@multimodal_bp.route('/<int:media_id>', methods=['DELETE'])
def delete_media(media_id):
    """Delete media"""
    db.delete_media(media_id)

    return jsonify({
        'success': True,
        'message': 'Media deleted'
    })


@multimodal_bp.route('/test-results/<int:media_id>', methods=['GET'])
def get_test_results(media_id):
    """Get test results for media"""
    results = db.get_multimodal_test_results(media_id)

    return jsonify({
        'success': True,
        'data': results
    })


@multimodal_bp.route('/combinations', methods=['GET'])
def get_combinations():
    """Get cross-modal combinations"""
    combinations = db.get_cross_modal_combinations()

    return jsonify({
        'success': True,
        'data': combinations
    })


@multimodal_bp.route('/stats/attack-types', methods=['GET'])
def get_attack_type_stats():
    """Get attack type distribution statistics"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 공격 타입별 개수
    stats = cursor.execute('''
        SELECT attack_type, COUNT(*) as count
        FROM media_arsenal
        GROUP BY attack_type
    ''').fetchall()

    data = [{'type': row['attack_type'], 'count': row['count']} for row in stats]
    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })


@multimodal_bp.route('/stats/success-rates', methods=['GET'])
def get_success_rates():
    """Get success rates by attack type"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 공격 타입별 테스트 성공률
    stats = cursor.execute('''
        SELECT m.attack_type,
               COUNT(t.id) as total_tests,
               SUM(CASE WHEN t.success = 1 THEN 1 ELSE 0 END) as successes
        FROM media_arsenal m
        LEFT JOIN multimodal_test_results t ON m.id = t.media_id
        GROUP BY m.attack_type
    ''').fetchall()

    data = []
    for row in stats:
        total = row['total_tests'] or 0
        success = row['successes'] or 0
        rate = (success / total * 100) if total > 0 else 0
        data.append({
            'type': row['attack_type'],
            'rate': round(rate, 1),
            'total': total,
            'successes': success
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })


@multimodal_bp.route('/models', methods=['GET'])
def get_models():
    """Get all tested models with counts"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    models = cursor.execute('''
        SELECT provider, model, COUNT(DISTINCT media_id) as count
        FROM multimodal_test_results
        GROUP BY provider, model
        ORDER BY count DESC
    ''').fetchall()

    data = [dict(row) for row in models]
    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })
