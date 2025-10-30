"""
Prompts API Blueprint
Text prompt management and testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Blueprint, jsonify, request
from core.database import ArsenalDB
import sqlite3

prompts_bp = Blueprint('prompts', __name__)
db = ArsenalDB()


@prompts_bp.route('/list', methods=['GET'])
def list_prompts():
    """Get all prompts with pagination and test statistics"""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    category = request.args.get('category', None)
    model = request.args.get('model', None)
    search = request.args.get('search', '')

    # New classification filters
    purpose = request.args.get('purpose', None)
    risk_category = request.args.get('risk_category', None)
    technique = request.args.get('technique', None)
    modality = request.args.get('modality', None)

    offset = (page - 1) * limit

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Build query with LEFT JOIN to get test statistics
    query = """
        SELECT p.id, p.category, p.payload, p.description, p.source, p.tags, p.created_at,
               p.purpose, p.risk_category, p.technique, p.modality,
               COUNT(tr.id) as test_count,
               SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
    """

    conditions = []
    params = []

    if category:
        conditions.append("p.category = ?")
        params.append(category)

    if purpose:
        conditions.append("p.purpose = ?")
        params.append(purpose)

    if risk_category:
        conditions.append("p.risk_category = ?")
        params.append(risk_category)

    if technique:
        conditions.append("p.technique = ?")
        params.append(technique)

    if modality:
        conditions.append("p.modality = ?")
        params.append(modality)

    if model:
        # Model format: "provider/model"
        provider, model_name = model.split('/', 1) if '/' in model else (model, None)
        conditions.append("tr.provider = ?")
        params.append(provider)
        if model_name:
            conditions.append("tr.model = ?")
            params.append(model_name)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " GROUP BY p.id ORDER BY test_count DESC, p.created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    prompts = []
    for row in rows:
        prompt = {
            'id': row[0],
            'category': row[1],
            'payload': row[2],
            'description': row[3] if row[3] else '',
            'source': row[4],
            'tags': row[5] if row[5] else '',
            'created_at': row[6],
            'purpose': row[7],
            'risk_category': row[8],
            'technique': row[9],
            'modality': row[10],
            'test_count': row[11],
            'success_count': row[12] if row[12] else 0,
            'success_rate': (row[12] / row[11] * 100) if row[11] > 0 and row[12] else 0
        }
        prompts.append(prompt)

    # Get total count
    count_query = "SELECT COUNT(DISTINCT p.id) FROM prompts p"
    count_conditions = []
    count_params = []

    if category or model or purpose or risk_category or technique or modality:
        if model:
            count_query += " LEFT JOIN test_results tr ON p.id = tr.prompt_id"

        if category:
            count_conditions.append("p.category = ?")
            count_params.append(category)
        if purpose:
            count_conditions.append("p.purpose = ?")
            count_params.append(purpose)
        if risk_category:
            count_conditions.append("p.risk_category = ?")
            count_params.append(risk_category)
        if technique:
            count_conditions.append("p.technique = ?")
            count_params.append(technique)
        if modality:
            count_conditions.append("p.modality = ?")
            count_params.append(modality)

        if model:
            provider, model_name = model.split('/', 1) if '/' in model else (model, None)
            count_conditions.append("tr.provider = ?")
            count_params.append(provider)
            if model_name:
                count_conditions.append("tr.model = ?")
                count_params.append(model_name)

    if count_conditions:
        count_query += " WHERE " + " AND ".join(count_conditions)

    cursor.execute(count_query, count_params)
    total = cursor.fetchone()[0]
    conn.close()

    return jsonify({
        'success': True,
        'data': prompts,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })


@prompts_bp.route('/search', methods=['GET'])
def search_prompts():
    """Search prompts by keyword"""
    keyword = request.args.get('q', '')
    category = request.args.get('category', None)
    model = request.args.get('model', None)

    # New classification filters
    purpose = request.args.get('purpose', None)
    risk_category = request.args.get('risk_category', None)
    technique = request.args.get('technique', None)
    modality = request.args.get('modality', None)

    # Always use custom query to support all filters
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = '''
        SELECT DISTINCT p.*,
               COUNT(tr.id) as test_count,
               SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE (p.payload LIKE ? OR p.description LIKE ? OR p.tags LIKE ?)
    '''
    params = [f'%{keyword}%', f'%{keyword}%', f'%{keyword}%']

    if category:
        query += ' AND p.category = ?'
        params.append(category)

    if purpose:
        query += ' AND p.purpose = ?'
        params.append(purpose)

    if risk_category:
        query += ' AND p.risk_category = ?'
        params.append(risk_category)

    if technique:
        query += ' AND p.technique = ?'
        params.append(technique)

    if modality:
        query += ' AND p.modality = ?'
        params.append(modality)

    if model:
        provider, model_name = model.split('/', 1) if '/' in model else (model, None)
        query += ' AND tr.provider = ?'
        params.append(provider)
        if model_name:
            query += ' AND tr.model = ?'
            params.append(model_name)

    query += ' GROUP BY p.id ORDER BY p.created_at DESC'

    cursor.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        result = dict(row)
        test_count = result.get('test_count', 0)
        success_count = result.get('success_count', 0)
        result['success_rate'] = (success_count / test_count * 100) if test_count > 0 else 0
        results.append(result)

    conn.close()

    return jsonify({
        'success': True,
        'data': results,
        'count': len(results)
    })


@prompts_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get all prompt categories with counts"""
    categories = db.get_prompt_categories()

    return jsonify({
        'success': True,
        'data': categories
    })


@prompts_bp.route('/add', methods=['POST'])
def add_prompt():
    """Add new prompt"""
    data = request.json

    prompt_id = db.insert_prompt(
        category=data['category'],
        payload=data['payload'],
        description=data.get('description', ''),
        source=data.get('source', 'manual'),
        tags=data.get('tags', '')
    )

    return jsonify({
        'success': True,
        'prompt_id': prompt_id
    })


@prompts_bp.route('/<int:prompt_id>', methods=['GET'])
def get_prompt(prompt_id):
    """Get single prompt details"""
    prompt = db.get_prompt_by_id(prompt_id)

    if not prompt:
        return jsonify({'success': False, 'error': 'Prompt not found'}), 404

    return jsonify({
        'success': True,
        'data': prompt
    })


@prompts_bp.route('/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    """Delete prompt"""
    db.delete_prompt(prompt_id)

    return jsonify({
        'success': True,
        'message': 'Prompt deleted'
    })


@prompts_bp.route('/test-results/<int:prompt_id>', methods=['GET'])
def get_test_results(prompt_id):
    """Get test results for a prompt"""
    results = db.get_test_results_by_prompt(prompt_id)

    return jsonify({
        'success': True,
        'data': results
    })


@prompts_bp.route('/models', methods=['GET'])
def get_models():
    """Get all tested models with counts"""
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    models = cursor.execute('''
        SELECT provider, model, COUNT(DISTINCT prompt_id) as count
        FROM test_results
        GROUP BY provider, model
        ORDER BY count DESC
    ''').fetchall()

    data = [dict(row) for row in models]
    conn.close()

    return jsonify({
        'success': True,
        'data': data
    })


@prompts_bp.route('/classification-options', methods=['GET'])
def get_classification_options():
    """Get all available classification filter options"""
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Purpose options
    purposes = cursor.execute('''
        SELECT DISTINCT purpose, COUNT(*) as count
        FROM prompts
        WHERE purpose IS NOT NULL
        GROUP BY purpose
        ORDER BY count DESC
    ''').fetchall()

    # Risk category options
    risk_categories = cursor.execute('''
        SELECT DISTINCT risk_category, COUNT(*) as count
        FROM prompts
        WHERE risk_category IS NOT NULL
        GROUP BY risk_category
        ORDER BY count DESC
    ''').fetchall()

    # Technique options
    techniques = cursor.execute('''
        SELECT DISTINCT technique, COUNT(*) as count
        FROM prompts
        WHERE technique IS NOT NULL
        GROUP BY technique
        ORDER BY count DESC
    ''').fetchall()

    # Modality options
    modalities = cursor.execute('''
        SELECT DISTINCT modality, COUNT(*) as count
        FROM prompts
        WHERE modality IS NOT NULL
        GROUP BY modality
        ORDER BY count DESC
    ''').fetchall()

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'purpose': [dict(row) for row in purposes],
            'risk_category': [dict(row) for row in risk_categories],
            'technique': [dict(row) for row in techniques],
            'modality': [dict(row) for row in modalities]
        }
    })
