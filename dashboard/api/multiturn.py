"""
Multi-turn API Blueprint
Multi-turn campaign management and testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB

multiturn_bp = Blueprint('multiturn', __name__)
db = ArsenalDB()


@multiturn_bp.route('/campaigns', methods=['GET'])
def list_campaigns():
    """Get all multi-turn campaigns with success rates"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get campaigns with evaluation statistics
    campaigns = cursor.execute('''
        SELECT
            c.*,
            COUNT(e.id) as total_evaluations,
            SUM(CASE WHEN e.goal_achieved = 1 THEN 1 ELSE 0 END) as goals_achieved,
            AVG(e.progress) as avg_progress
        FROM multi_turn_campaigns c
        LEFT JOIN multi_turn_evaluations e ON c.id = e.campaign_id
        GROUP BY c.id
        ORDER BY c.created_at DESC
    ''').fetchall()

    data = []
    for row in campaigns:
        campaign = dict(row)
        total = campaign.get('total_evaluations', 0)
        achieved = campaign.get('goals_achieved', 0)
        campaign['success_rate'] = round((achieved / total * 100), 1) if total > 0 else 0
        campaign['avg_progress'] = round((campaign.get('avg_progress', 0) or 0) * 100, 1)
        data.append(campaign)

    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })


@multiturn_bp.route('/campaigns/<int:campaign_id>', methods=['GET'])
def get_campaign(campaign_id):
    """Get single campaign with conversations and evaluations"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 캠페인 정보
    campaign = cursor.execute('''
        SELECT * FROM multi_turn_campaigns WHERE id = ?
    ''', (campaign_id,)).fetchone()

    if not campaign:
        conn.close()
        return jsonify({'success': False, 'error': 'Campaign not found'}), 404

    # 대화 기록
    conversations = cursor.execute('''
        SELECT * FROM multi_turn_conversations
        WHERE campaign_id = ?
        ORDER BY turn_number
    ''', (campaign_id,)).fetchall()

    # 평가 결과
    evaluations = cursor.execute('''
        SELECT * FROM multi_turn_evaluations
        WHERE campaign_id = ?
        ORDER BY turn_number
    ''', (campaign_id,)).fetchall()

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'campaign': dict(campaign),
            'conversations': [dict(row) for row in conversations],
            'evaluations': [dict(row) for row in evaluations]
        }
    })


@multiturn_bp.route('/strategies', methods=['GET'])
def get_strategies():
    """Get available multi-turn strategies with real data"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 실제 DB에서 전략별 통계 가져오기
    real_stats = cursor.execute('''
        SELECT c.strategy,
               COUNT(DISTINCT c.id) as campaign_count,
               COUNT(DISTINCT e.id) as evaluation_count,
               AVG(CASE WHEN e.goal_achieved = 1 THEN 1.0 ELSE 0.0 END) as success_rate
        FROM multi_turn_campaigns c
        LEFT JOIN multi_turn_evaluations e ON c.id = e.campaign_id
        GROUP BY c.strategy
    ''').fetchall()

    # 전략별 데이터를 딕셔너리로 변환
    real_data = {}
    for row in real_stats:
        strategy = row['strategy']
        real_data[strategy] = {
            'campaigns': row['campaign_count'],
            'evaluations': row['evaluation_count'],
            'success_rate': row['success_rate'] or 0.0
        }

    # 전략 정의 (논문 데이터)
    strategy_definitions = {
        'figstep': {'description': '단계적 행동 유도', 'paper_asr': 0.60},
        'crescendo': {'description': '점진적 강도 증가', 'paper_asr': 0.722},
        'roleplay': {'description': '역할극 기반', 'paper_asr': 0.649},
        'crescendobis': {'description': 'Crescendo 변형', 'paper_asr': 0.673},
        'latent': {'description': '잠재 의도 활용', 'paper_asr': 0.825},
        'seed': {'description': '시드 심기 전략', 'paper_asr': 0.756},
        'deception': {'description': '기만 기반', 'paper_asr': 0.684},
        'mml_attack': {'description': 'MML 공격', 'paper_asr': 0.70},
        'visual_storytelling': {'description': '시각적 스토리텔링', 'paper_asr': 0.65}
    }

    strategies = []
    for strategy_key, stats in real_data.items():
        definition = strategy_definitions.get(strategy_key.lower(), {
            'description': strategy_key,
            'paper_asr': 0.0
        })

        strategies.append({
            'name': strategy_key,
            'description': definition['description'],
            'paper_asr': definition['paper_asr'],
            'actual_success_rate': round(stats['success_rate'], 3),
            'campaigns': stats['campaigns'],
            'evaluations': stats['evaluations']
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': strategies
    })


@multiturn_bp.route('/test-results', methods=['GET'])
def get_test_results():
    """Get multi-turn evaluation results from real data"""
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent.parent / "arsenal.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 캠페인별 평가 결과 요약
    results = cursor.execute('''
        SELECT
            c.id as campaign_id,
            c.name as campaign_name,
            c.strategy,
            c.target_model,
            c.status,
            c.turns_used,
            COUNT(e.id) as total_evaluations,
            SUM(CASE WHEN e.goal_achieved = 1 THEN 1 ELSE 0 END) as goals_achieved,
            AVG(e.progress) as avg_progress,
            c.created_at
        FROM multi_turn_campaigns c
        LEFT JOIN multi_turn_evaluations e ON c.id = e.campaign_id
        GROUP BY c.id
        ORDER BY c.created_at DESC
    ''').fetchall()

    data = []
    for row in results:
        total = row['total_evaluations'] or 0
        achieved = row['goals_achieved'] or 0
        success_rate = (achieved / total * 100) if total > 0 else 0

        data.append({
            'campaign_id': row['campaign_id'],
            'campaign_name': row['campaign_name'],
            'strategy': row['strategy'],
            'model': row['target_model'],
            'status': row['status'],
            'turns_used': row['turns_used'],
            'total_evaluations': total,
            'goals_achieved': achieved,
            'success_rate': round(success_rate, 1),
            'avg_progress': round(row['avg_progress'] * 100, 1) if row['avg_progress'] else 0,
            'created_at': row['created_at']
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })
