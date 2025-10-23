"""
Dashboard API Server
Provides REST API for Campaign Statistics Dashboard
"""

import sys
from pathlib import Path

# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from core.database import ArsenalDB
import os
import sqlite3

# Get dashboard directory
DASHBOARD_DIR = Path(__file__).parent

app = Flask(__name__, static_folder=str(DASHBOARD_DIR))
CORS(app)

# Initialize database
db = ArsenalDB("arsenal.db")


@app.route('/')
def index():
    """Serve dashboard homepage"""
    return send_from_directory(DASHBOARD_DIR, 'index.html')


@app.route('/dashboard')
def dashboard():
    """Serve dashboard homepage"""
    return send_from_directory(DASHBOARD_DIR, 'index.html')


@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Get prompts with optional filtering"""
    category = request.args.get('category')
    limit = int(request.args.get('limit', 100))
    search = request.args.get('search', '').lower()

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT p.id, p.category, p.payload, p.description, p.source, p.tags, p.created_at,
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

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " GROUP BY p.id ORDER BY test_count DESC, p.created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    prompts = []
    for row in rows:
        prompt = {
            'id': row[0],
            'category': row[1],
            'payload': row[2],
            'description': row[3],
            'source': row[4],
            'tags': row[5],
            'created_at': row[6],
            'test_count': row[7],
            'success_count': row[8],
            'success_rate': (row[8] / row[7] * 100) if row[7] > 0 else 0
        }

        # Apply search filter
        if not search or (
            search in prompt['payload'].lower() or
            search in (prompt['description'] or '').lower() or
            search in prompt['category'].lower()
        ):
            prompts.append(prompt)

    conn.close()

    return jsonify({
        'success': True,
        'data': prompts,
        'total': len(prompts)
    })


@app.route('/api/prompts/<int:prompt_id>', methods=['GET'])
def get_prompt_detail(prompt_id):
    """Get prompt details with test results"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Get prompt
    cursor.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({
            'success': False,
            'error': 'Prompt not found'
        }), 404

    prompt = {
        'id': row[0],
        'category': row[1],
        'payload': row[2],
        'description': row[3],
        'source': row[4],
        'is_template': row[5],
        'tags': row[6],
        'created_at': row[7]
    }

    # Get test results
    cursor.execute("""
        SELECT id, provider, model, response, success, severity, confidence,
               reasoning, response_time, tested_at
        FROM test_results
        WHERE prompt_id = ?
        ORDER BY tested_at DESC
    """, (prompt_id,))

    test_results = []
    for row in cursor.fetchall():
        test_results.append({
            'id': row[0],
            'provider': row[1],
            'model': row[2],
            'response': row[3],
            'success': row[4],
            'severity': row[5],
            'confidence': row[6],
            'reasoning': row[7],
            'response_time': row[8],
            'tested_at': row[9]
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'prompt': prompt,
            'test_results': test_results
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall Prompt Arsenal statistics"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Total prompts
    cursor.execute("SELECT COUNT(*) FROM prompts")
    total_prompts = cursor.fetchone()[0]

    # Total tests
    cursor.execute("SELECT COUNT(*) FROM test_results")
    total_tests = cursor.fetchone()[0]

    # Successful tests
    cursor.execute("SELECT COUNT(*) FROM test_results WHERE success = 1")
    successful_tests = cursor.fetchone()[0]

    # Success rate
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

    # Category breakdown
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM prompts
        GROUP BY category
        ORDER BY count DESC
    """)
    category_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

    # Recent activity (last 7 days)
    cursor.execute("""
        SELECT COUNT(*)
        FROM test_results
        WHERE datetime(tested_at) >= datetime('now', '-7 days')
    """)
    recent_tests = cursor.fetchone()[0]

    # Top categories by success rate
    cursor.execute("""
        SELECT p.category,
               COUNT(DISTINCT p.id) as prompt_count,
               COUNT(tr.id) as test_count,
               SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        GROUP BY p.category
        HAVING test_count > 0
        ORDER BY (success_count * 1.0 / test_count) DESC
        LIMIT 5
    """)

    top_categories = []
    for row in cursor.fetchall():
        top_categories.append({
            'category': row[0],
            'prompt_count': row[1],
            'test_count': row[2],
            'success_rate': (row[3] / row[2] * 100) if row[2] > 0 else 0
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'total_prompts': total_prompts,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': round(success_rate, 1),
            'category_breakdown': category_breakdown,
            'recent_tests': recent_tests,
            'top_categories': top_categories
        }
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories with counts"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM prompts
        GROUP BY category
        ORDER BY count DESC
    """)

    categories = []
    for row in cursor.fetchall():
        categories.append({
            'name': row[0],
            'count': row[1]
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': categories
    })


@app.route('/api/search', methods=['GET'])
def search():
    """Search prompts by keyword"""
    query = request.args.get('q', '').lower()

    if not query:
        return jsonify({
            'success': False,
            'error': 'Search query required'
        }), 400

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Search in payload, description, category
    cursor.execute("""
        SELECT p.id, p.category, p.payload, p.description, p.source, p.tags, p.created_at,
               COUNT(tr.id) as test_count,
               SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count
        FROM prompts p
        LEFT JOIN test_results tr ON p.id = tr.prompt_id
        WHERE LOWER(p.payload) LIKE ? OR LOWER(p.description) LIKE ? OR LOWER(p.category) LIKE ?
        GROUP BY p.id
        ORDER BY p.created_at DESC
        LIMIT 100
    """, (f'%{query}%', f'%{query}%', f'%{query}%'))

    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'category': row[1],
            'payload': row[2],
            'description': row[3],
            'source': row[4],
            'tags': row[5],
            'created_at': row[6],
            'test_count': row[7],
            'success_count': row[8],
            'success_rate': (row[8] / row[7] * 100) if row[7] > 0 else 0
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': results,
        'total': len(results)
    })


@app.route('/api/multimodal', methods=['GET'])
def get_multimodal():
    """Get multimodal media with test results"""
    limit = int(request.args.get('limit', 100))

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT m.id, m.media_type, m.attack_type, m.generated_file, m.description,
               m.tags, m.created_at,
               COUNT(tr.id) as test_count,
               SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count
        FROM media_arsenal m
        LEFT JOIN multimodal_test_results tr ON m.id = tr.media_id
        GROUP BY m.id
        ORDER BY m.created_at DESC
        LIMIT ?
    """, (limit,))

    media_list = []
    for row in cursor.fetchall():
        media_list.append({
            'id': row[0],
            'media_type': row[1],
            'attack_type': row[2],
            'file_path': row[3],
            'description': row[4],
            'tags': row[5],
            'created_at': row[6],
            'test_count': row[7],
            'success_count': row[8],
            'success_rate': (row[8] / row[7] * 100) if row[7] > 0 else 0
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': media_list,
        'total': len(media_list)
    })


@app.route('/api/multimodal/<int:media_id>', methods=['GET'])
def get_multimodal_detail(media_id):
    """Get multimodal media details with test results"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Get media
    cursor.execute("SELECT * FROM media_arsenal WHERE id = ?", (media_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({
            'success': False,
            'error': 'Media not found'
        }), 404

    media = {
        'id': row[0],
        'media_type': row[1],
        'attack_type': row[2],
        'base_file': row[3],
        'file_path': row[4],
        'parameters': row[5],
        'description': row[6],
        'tags': row[7],
        'created_at': row[8]
    }

    # Get test results
    cursor.execute("""
        SELECT id, provider, model, response, vision_response, success,
               severity, confidence, reasoning, response_time, tested_at
        FROM multimodal_test_results
        WHERE media_id = ?
        ORDER BY tested_at DESC
    """, (media_id,))

    test_results = []
    for row in cursor.fetchall():
        test_results.append({
            'id': row[0],
            'provider': row[1],
            'model': row[2],
            'response': row[3],
            'vision_response': row[4],
            'success': row[5],
            'severity': row[6],
            'confidence': row[7],
            'reasoning': row[8],
            'response_time': row[9],
            'tested_at': row[10]
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'media': media,
            'test_results': test_results
        }
    })


@app.route('/api/campaigns', methods=['GET'])
def get_campaigns():
    """Get multi-turn campaigns"""
    limit = int(request.args.get('limit', 100))
    status = request.args.get('status')

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    query = """
        SELECT id, name, goal, strategy, target_provider, target_model,
               status, max_turns, turns_used, started_at, completed_at
        FROM multi_turn_campaigns
    """

    params = []
    if status:
        query += " WHERE status = ?"
        params.append(status)

    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)

    campaigns = []
    for row in cursor.fetchall():
        campaigns.append({
            'id': row[0],
            'name': row[1],
            'goal': row[2],
            'strategy': row[3],
            'target_provider': row[4],
            'target_model': row[5],
            'status': row[6],
            'max_turns': row[7],
            'turns_used': row[8],
            'started_at': row[9],
            'completed_at': row[10]
        })

    conn.close()

    return jsonify({
        'success': True,
        'data': campaigns,
        'total': len(campaigns)
    })


@app.route('/api/campaigns/<int:campaign_id>', methods=['GET'])
def get_campaign_detail(campaign_id):
    """Get campaign details with turn-by-turn results"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Get campaign
    cursor.execute("SELECT * FROM multi_turn_campaigns WHERE id = ?", (campaign_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({
            'success': False,
            'error': 'Campaign not found'
        }), 404

    campaign = {
        'id': row[0],
        'name': row[1],
        'goal': row[2],
        'strategy': row[3],
        'target_provider': row[4],
        'target_model': row[5],
        'status': row[6],
        'max_turns': row[7],
        'turns_used': row[8],
        'started_at': row[9],
        'completed_at': row[10]
    }

    # Get conversation turns
    cursor.execute("""
        SELECT turn_number, prompt_text, response, prompt_images, prompt_audio,
               prompt_video, prompt_strategy, response_time
        FROM multi_turn_conversations
        WHERE campaign_id = ?
        ORDER BY turn_number ASC
    """, (campaign_id,))

    turns = []
    for row in cursor.fetchall():
        turns.append({
            'turn_number': row[0],
            'prompt_text': row[1],
            'response': row[2],
            'prompt_images': row[3],
            'prompt_audio': row[4],
            'prompt_video': row[5],
            'prompt_strategy': row[6],
            'response_time': row[7]
        })

    # Get turn evaluations
    cursor.execute("""
        SELECT turn_number, goal_achieved, progress, toxicity,
               defense_triggered, reasoning
        FROM multi_turn_evaluations
        WHERE campaign_id = ?
        ORDER BY turn_number ASC
    """, (campaign_id,))

    evaluations = {}
    for row in cursor.fetchall():
        evaluations[row[0]] = {
            'goal_achieved': row[1],
            'progress': row[2],
            'toxicity': row[3],
            'defense_triggered': row[4],
            'reasoning': row[5]
        }

    # Merge evaluations into turns
    for turn in turns:
        turn['evaluation'] = evaluations.get(turn['turn_number'], {})

    conn.close()

    return jsonify({
        'success': True,
        'data': {
            'campaign': campaign,
            'turns': turns
        }
    })


# ============================================================================
# Prompt Management APIs
# ============================================================================

@app.route('/api/prompts', methods=['POST'])
def create_prompt():
    """Create new prompt"""
    try:
        data = request.get_json()

        category = data.get('category')
        payload = data.get('payload')
        description = data.get('description', '')
        source = data.get('source', 'web-ui')
        tags = data.get('tags', '')

        if not category or not payload:
            return jsonify({
                'success': False,
                'error': 'Category and payload are required'
            }), 400

        prompt_id = db.insert_prompt(category, payload, description, source, tags=tags)

        return jsonify({
            'success': True,
            'data': {
                'id': prompt_id,
                'message': 'Prompt created successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['PUT'])
def update_prompt(prompt_id):
    """Update existing prompt"""
    try:
        data = request.get_json()

        category = data.get('category')
        payload = data.get('payload')
        description = data.get('description')
        tags = data.get('tags')

        db.update_prompt(prompt_id, category, payload, description, tags)

        return jsonify({
            'success': True,
            'data': {
                'message': 'Prompt updated successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    """Delete prompt"""
    try:
        db.delete_prompt(prompt_id)

        return jsonify({
            'success': True,
            'data': {
                'message': 'Prompt deleted successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Testing APIs
# ============================================================================

@app.route('/api/test', methods=['POST'])
def test_prompt():
    """Test prompt against LLM"""
    try:
        data = request.get_json()

        prompt_id = data.get('prompt_id')
        prompt_text = data.get('prompt_text')
        provider = data.get('provider')
        model = data.get('model')
        api_key = data.get('api_key')

        if not prompt_text or not provider or not model:
            return jsonify({
                'success': False,
                'error': 'prompt_text, provider, and model are required'
            }), 400

        # Import LLM tester
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from llm_tester import LLMTester
        import asyncio

        # Create tester
        tester = LLMTester(db, provider, model, api_key)

        # Run test asynchronously
        async def run_test():
            result = await tester.test_prompt(prompt_text)
            return result

        result = asyncio.run(run_test())

        # Save result if prompt_id provided
        if prompt_id and result:
            db.insert_test_result(
                prompt_id=prompt_id,
                provider=provider,
                model=model,
                response=result.get('response', ''),
                success=result.get('success', False),
                severity=result.get('severity', 'low'),
                confidence=result.get('confidence', 0.0),
                reasoning=result.get('reasoning', ''),
                response_time=result.get('response_time', 0.0),
                used_input=prompt_text
            )

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Config/Profile Management APIs
# ============================================================================

@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    """Get API profiles from config"""
    try:
        import json
        config_path = Path(__file__).parent.parent / 'config.json'

        if not config_path.exists():
            return jsonify({
                'success': True,
                'data': {
                    'profiles': {},
                    'default_profile': None
                }
            })

        with open(config_path, 'r') as f:
            config = json.load(f)

        return jsonify({
            'success': True,
            'data': {
                'profiles': config.get('profiles', {}),
                'default_profile': config.get('default_profile')
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/profiles', methods=['POST'])
def create_profile():
    """Create new API profile"""
    try:
        import json
        data = request.get_json()

        name = data.get('name')
        provider = data.get('provider')
        model = data.get('model')
        api_key = data.get('api_key')
        base_url = data.get('base_url')

        if not name or not provider or not model:
            return jsonify({
                'success': False,
                'error': 'name, provider, and model are required'
            }), 400

        config_path = Path(__file__).parent.parent / 'config.json'

        # Load existing config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {'profiles': {}}

        # Add new profile
        config['profiles'][name] = {
            'provider': provider,
            'model': model,
            'api_key': api_key,
            'base_url': base_url
        }

        # Set as default if first profile
        if len(config['profiles']) == 1:
            config['default_profile'] = name

        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({
            'success': True,
            'data': {
                'message': f'Profile {name} created successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/profiles/<profile_name>', methods=['DELETE'])
def delete_profile(profile_name):
    """Delete API profile"""
    try:
        import json
        config_path = Path(__file__).parent.parent / 'config.json'

        if not config_path.exists():
            return jsonify({
                'success': False,
                'error': 'No config file found'
            }), 404

        with open(config_path, 'r') as f:
            config = json.load(f)

        if profile_name not in config.get('profiles', {}):
            return jsonify({
                'success': False,
                'error': f'Profile {profile_name} not found'
            }), 404

        del config['profiles'][profile_name]

        # Update default if deleted
        if config.get('default_profile') == profile_name:
            if config['profiles']:
                config['default_profile'] = list(config['profiles'].keys())[0]
            else:
                config['default_profile'] = None

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return jsonify({
            'success': True,
            'data': {
                'message': f'Profile {profile_name} deleted successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Payload Generation APIs
# ============================================================================

@app.route('/api/generate-variants', methods=['POST'])
def generate_variants():
    """Generate prompt variants"""
    try:
        data = request.get_json()

        base_payload = data.get('payload')
        strategies = data.get('strategies', ['base64', 'rot13', 'leet', 'unicode'])

        if not base_payload:
            return jsonify({
                'success': False,
                'error': 'payload is required'
            }), 400

        # Import payload utils
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from payload_utils import PayloadGenerator

        generator = PayloadGenerator()
        variants = generator.generate_variants(base_payload, strategies)

        return jsonify({
            'success': True,
            'data': {
                'variants': variants,
                'count': len(variants)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/media/<path:filepath>')
def serve_media(filepath):
    """Serve media files"""
    import os
    media_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'media')
    return send_from_directory(media_dir, filepath)


@app.route('/generated_images/<path:filepath>')
def serve_generated_images(filepath):
    """Serve generated images from campaigns"""
    import os
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generated_images')
    return send_from_directory(images_dir, filepath)


@app.route('/api/scans', methods=['GET'])
def get_scans():
    """Get security scans"""
    limit = int(request.args.get('limit', 50))
    scan_type = request.args.get('type')  # static, llm, all

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    query = """
        SELECT id, target, scan_type, mode, profile_name,
               total_findings, critical_count, high_count, medium_count, low_count,
               scan_duration, started_at, completed_at
        FROM security_scans
    """

    params = []
    if scan_type and scan_type != 'all':
        query += " WHERE scan_type = ?"
        params.append(scan_type)

    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    scans = []
    for row in rows:
        scans.append({
            'id': row[0],
            'target': row[1],
            'scan_type': row[2],
            'mode': row[3],
            'profile_name': row[4],
            'total_findings': row[5],
            'critical_count': row[6],
            'high_count': row[7],
            'medium_count': row[8],
            'low_count': row[9],
            'scan_duration': row[10],
            'started_at': row[11],
            'completed_at': row[12]
        })

    conn.close()

    return jsonify({
        'scans': scans,
        'total': len(scans)
    })


@app.route('/api/scans/<int:scan_id>', methods=['GET'])
def get_scan_detail(scan_id):
    """Get scan details with findings"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Get scan info
    cursor.execute("""
        SELECT id, target, scan_type, mode, profile_name,
               total_findings, critical_count, high_count, medium_count, low_count,
               scan_duration, started_at, completed_at
        FROM security_scans
        WHERE id = ?
    """, (scan_id,))

    scan_row = cursor.fetchone()
    if not scan_row:
        conn.close()
        return jsonify({'error': 'Scan not found'}), 404

    scan = {
        'id': scan_row[0],
        'target': scan_row[1],
        'scan_type': scan_row[2],
        'mode': scan_row[3],
        'profile_name': scan_row[4],
        'total_findings': scan_row[5],
        'critical_count': scan_row[6],
        'high_count': scan_row[7],
        'medium_count': scan_row[8],
        'low_count': scan_row[9],
        'scan_duration': scan_row[10],
        'started_at': scan_row[11],
        'completed_at': scan_row[12]
    }

    # Get findings
    cursor.execute("""
        SELECT id, cwe_id, cwe_name, severity, confidence,
               file_path, line_number, column_number, function_name,
               code_snippet, message, llm_reasoning, created_at
        FROM security_findings
        WHERE scan_id = ?
        ORDER BY
            CASE severity
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
                ELSE 5
            END,
            confidence DESC
    """, (scan_id,))

    findings_rows = cursor.fetchall()
    findings = []
    for row in findings_rows:
        findings.append({
            'id': row[0],
            'cwe_id': row[1],
            'cwe_name': row[2],
            'severity': row[3],
            'confidence': row[4],
            'file_path': row[5],
            'line_number': row[6],
            'column_number': row[7],
            'function_name': row[8],
            'code_snippet': row[9],
            'message': row[10],
            'llm_reasoning': row[11],
            'created_at': row[12]
        })

    scan['findings'] = findings
    conn.close()

    return jsonify(scan)


@app.route('/api/scans/stats', methods=['GET'])
def get_scan_stats():
    """Get scan statistics"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    # Total scans
    cursor.execute("SELECT COUNT(*) FROM security_scans")
    total_scans = cursor.fetchone()[0]

    # Scans by type
    cursor.execute("""
        SELECT scan_type, COUNT(*) as count
        FROM security_scans
        GROUP BY scan_type
    """)
    scans_by_type = {row[0]: row[1] for row in cursor.fetchall()}

    # Total findings by severity
    cursor.execute("""
        SELECT
            SUM(critical_count) as critical,
            SUM(high_count) as high,
            SUM(medium_count) as medium,
            SUM(low_count) as low
        FROM security_scans
    """)
    severity_row = cursor.fetchone()
    findings_by_severity = {
        'critical': severity_row[0] or 0,
        'high': severity_row[1] or 0,
        'medium': severity_row[2] or 0,
        'low': severity_row[3] or 0
    }

    # Most common CWEs
    cursor.execute("""
        SELECT cwe_id, cwe_name, COUNT(*) as count
        FROM security_findings
        GROUP BY cwe_id, cwe_name
        ORDER BY count DESC
        LIMIT 10
    """)
    top_cwes = [
        {'cwe_id': row[0], 'cwe_name': row[1], 'count': row[2]}
        for row in cursor.fetchall()
    ]

    # Recent scans
    cursor.execute("""
        SELECT id, target, scan_type, total_findings, started_at
        FROM security_scans
        ORDER BY started_at DESC
        LIMIT 5
    """)
    recent_scans = [
        {
            'id': row[0],
            'target': row[1],
            'scan_type': row[2],
            'total_findings': row[3],
            'started_at': row[4]
        }
        for row in cursor.fetchall()
    ]

    conn.close()

    return jsonify({
        'total_scans': total_scans,
        'scans_by_type': scans_by_type,
        'findings_by_severity': findings_by_severity,
        'top_cwes': top_cwes,
        'recent_scans': recent_scans
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
