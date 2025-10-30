"""
Custom Endpoint Client
HTTP 요청 기반 커스텀 엔드포인트 지원 (CTF, 웹서비스 테스트용)
"""

import aiohttp
import json
from typing import Optional, Dict, Any, List


class CustomEndpointClient:
    """
    커스텀 HTTP 엔드포인트용 클라이언트
    CTF 챌린지나 실제 웹 서비스의 LLM 기반 API 테스트용
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'url': 엔드포인트 URL,
                'method': 'POST' | 'GET',
                'body_template': POST body 템플릿 (선택),
                'query_template': GET query 템플릿 (선택),
                'headers': 커스텀 헤더 (선택),
                'response_path': 응답에서 텍스트 추출 경로
            }
        """
        self.url = config['url']
        self.method = config['method'].upper()
        self.body_template = config.get('body_template', '{"message": "{prompt}"}')
        self.query_template = config.get('query_template', 'prompt={prompt}')
        self.headers = config.get('headers', {})
        self.response_path = config.get('response_path', 'response')

        # Default headers
        if 'Content-Type' not in self.headers and self.method == 'POST':
            self.headers['Content-Type'] = 'application/json'

    async def generate(self, prompt: str, image_url: Optional[str] = None, turn: int = 1, **kwargs) -> str:
        """
        커스텀 엔드포인트로 요청 전송

        Args:
            prompt: 텍스트 프롬프트
            image_url: 이미지 URL (선택)
            turn: 턴 번호
            **kwargs: 추가 파라미터

        Returns:
            응답 텍스트
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.method == 'POST':
                    # Body 템플릿을 먼저 JSON으로 파싱
                    try:
                        body_template_obj = json.loads(self.body_template)
                    except json.JSONDecodeError as e:
                        return f"[ERROR] Invalid body template: {e}"

                    # 템플릿의 값들을 실제 값으로 치환
                    body = self._replace_template_values(
                        body_template_obj,
                        prompt=prompt,
                        image_url=image_url or '',
                        turn=turn
                    )

                    async with session.post(self.url, json=body, headers=self.headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return self._extract_response(data)
                        else:
                            error_text = await resp.text()
                            return f"[ERROR] HTTP {resp.status}: {error_text}"

                elif self.method == 'GET':
                    # Query 템플릿에 값 채우기
                    query_str = self.query_template.format(
                        prompt=prompt,
                        image_url=image_url or '',
                        turn=turn
                    )
                    url = f"{self.url}?{query_str}"

                    async with session.get(url, headers=self.headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return self._extract_response(data)
                        else:
                            error_text = await resp.text()
                            return f"[ERROR] HTTP {resp.status}: {error_text}"

        except Exception as e:
            return f"[ERROR] Request failed: {str(e)}"

    def _replace_template_values(self, obj: Any, **kwargs) -> Any:
        """
        템플릿 객체의 플레이스홀더를 실제 값으로 치환

        Args:
            obj: 템플릿 객체 (dict, list, str 등)
            **kwargs: 치환할 값들

        Returns:
            치환된 객체
        """
        if isinstance(obj, dict):
            return {k: self._replace_template_values(v, **kwargs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_template_values(item, **kwargs) for item in obj]
        elif isinstance(obj, str):
            # 플레이스홀더 치환
            result = obj
            for key, value in kwargs.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            return result
        else:
            return obj

    def _extract_response(self, data: Dict[str, Any]) -> str:
        """
        응답에서 텍스트 추출

        Args:
            data: JSON 응답

        Returns:
            추출된 텍스트
        """
        # response_path를 따라 데이터 탐색 (예: "data.message" → data['message'])
        keys = self.response_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return str(current)

            return str(current) if current is not None else "[ERROR] No response found"

        except (KeyError, IndexError, TypeError) as e:
            # Fallback: 전체 응답 반환
            return json.dumps(data, ensure_ascii=False, indent=2)


class CustomMultimodalEndpointClient(CustomEndpointClient):
    """
    멀티모달 지원 커스텀 엔드포인트 클라이언트
    """

    async def generate_multimodal(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        image_data: Optional[str] = None,
        turn: int = 1,
        **kwargs
    ) -> str:
        """
        멀티모달 요청 전송

        Args:
            prompt: 텍스트 프롬프트
            image_url: 이미지 URL
            image_data: Base64 인코딩된 이미지 데이터
            turn: 턴 번호
            **kwargs: 추가 파라미터

        Returns:
            응답 텍스트
        """
        # 이미지가 있으면 템플릿에 image_url 또는 image_data 추가
        return await self.generate(
            prompt=prompt,
            image_url=image_url or image_data,
            turn=turn,
            **kwargs
        )
