"""
Prompt Manager - 프롬프트 관리 유틸리티
"""

import re
from typing import Optional, Dict


def has_template_variable(prompt: str) -> bool:
    """프롬프트에 템플릿 변수가 있는지 확인"""
    patterns = [
        r'\{INPUT\}',
        r'\{USER_INPUT\}',
        r'\{\}',
        r'\[INSERT PROMPT HERE\]',
        r'\[INSERT.*?\]',
        r'\<INSERT.*?\>',
        r'\{INSERT.*?\}',
        r'\[\.\.\.\]',
        r'\<\.\.\.\>',
        r'\{.*?here.*?\}',
        r'\[.*?here.*?\]',
    ]
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns)


def fill_template(prompt: str, user_input: str) -> str:
    """템플릿 변수를 사용자 입력으로 채우기"""
    # 기본 패턴
    prompt = prompt.replace('{INPUT}', user_input)
    prompt = prompt.replace('{USER_INPUT}', user_input)
    prompt = prompt.replace('{}', user_input)

    # [INSERT PROMPT HERE] 패턴
    prompt = re.sub(r'\[INSERT PROMPT HERE\]', user_input, prompt, flags=re.IGNORECASE)

    # [INSERT ...] 패턴
    prompt = re.sub(r'\[INSERT[^\]]*?\]', user_input, prompt, flags=re.IGNORECASE)

    # <INSERT ...> 패턴
    prompt = re.sub(r'\<INSERT[^\>]*?\>', user_input, prompt, flags=re.IGNORECASE)

    # {INSERT ...} 패턴
    prompt = re.sub(r'\{INSERT[^\}]*?\}', user_input, prompt, flags=re.IGNORECASE)

    # [...] 패턴
    prompt = re.sub(r'\[\.\.\.\]', user_input, prompt)

    # <...> 패턴
    prompt = re.sub(r'\<\.\.\.\>', user_input, prompt)

    # {...here...} 패턴
    prompt = re.sub(r'\{[^\}]*?here[^\}]*?\}', user_input, prompt, flags=re.IGNORECASE)

    # [...here...] 패턴
    prompt = re.sub(r'\[[^\]]*?here[^\]]*?\]', user_input, prompt, flags=re.IGNORECASE)

    return prompt


def extract_template_variables(prompt: str) -> list:
    """템플릿에서 변수 추출"""
    variables = []

    # {INPUT} 형태
    if '{INPUT}' in prompt:
        variables.append('INPUT')

    # {USER_INPUT} 형태
    if '{USER_INPUT}' in prompt:
        variables.append('USER_INPUT')

    # {} 형태
    if '{}' in prompt:
        variables.append('PLACEHOLDER')

    # [INSERT PROMPT HERE] 형태
    if re.search(r'\[INSERT PROMPT HERE\]', prompt, re.IGNORECASE):
        variables.append('INSERT_PROMPT_HERE')

    # [INSERT ...] 형태
    inserts = re.findall(r'\[INSERT[^\]]*?\]', prompt, re.IGNORECASE)
    if inserts:
        variables.extend([f'BRACKET_INSERT_{i+1}' for i in range(len(inserts))])

    # <INSERT ...> 형태
    angle_inserts = re.findall(r'\<INSERT[^\>]*?\>', prompt, re.IGNORECASE)
    if angle_inserts:
        variables.extend([f'ANGLE_INSERT_{i+1}' for i in range(len(angle_inserts))])

    # {...here...} 형태
    here_vars = re.findall(r'\{[^\}]*?here[^\}]*?\}', prompt, re.IGNORECASE)
    if here_vars:
        variables.extend([f'HERE_VAR_{i+1}' for i in range(len(here_vars))])

    # [...] 형태
    if re.search(r'\[\.\.\.\]', prompt):
        variables.append('ELLIPSIS')

    return variables


def validate_prompt(prompt: str) -> Dict:
    """프롬프트 유효성 검사"""
    result = {
        'valid': True,
        'is_template': False,
        'variables': [],
        'warnings': []
    }

    if not prompt or len(prompt.strip()) == 0:
        result['valid'] = False
        result['warnings'].append('빈 프롬프트입니다')
        return result

    if len(prompt) > 10000:
        result['warnings'].append('프롬프트가 너무 깁니다 (10000자 초과)')

    if has_template_variable(prompt):
        result['is_template'] = True
        result['variables'] = extract_template_variables(prompt)

    return result
