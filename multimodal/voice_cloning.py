"""
음성 복제 (Voice Cloning) 시스템
- TTS (Text-to-Speech) 기반 음성 복제
- Speaker Embedding 추출
- Voice Conversion
"""

import numpy as np
import soundfile as sf
import warnings
from typing import Optional, Dict, List
import os

# Coqui TTS 선택적 임포트
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    warnings.warn("Coqui TTS not installed. Install with: pip install TTS")

# Resemblyzer (Speaker Embedding) 선택적 임포트
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    warnings.warn("Resemblyzer not installed. Install with: pip install resemblyzer")


class VoiceCloner:
    """
    음성 복제 시스템

    기능:
    - Zero-shot Voice Cloning: 짧은 샘플로 음성 복제
    - Speaker Embedding: 화자 특징 벡터 추출
    - Multi-language Support: 다국어 지원
    """

    def __init__(self, model_name: str = None, device: str = 'cpu'):
        """
        Args:
            model_name: TTS 모델명 (None이면 자동 선택)
            device: 'cpu' 또는 'cuda'
        """
        self.device = device

        # TTS 모델 초기화
        if TTS_AVAILABLE:
            if model_name is None:
                # 기본 다국어 모델
                model_name = "tts_models/multilingual/multi-dataset/your_tts"

            try:
                self.tts = TTS(model_name=model_name, progress_bar=False).to(device)
                self.model_name = model_name
            except Exception as e:
                warnings.warn(f"Failed to load TTS model: {e}")
                self.tts = None
        else:
            self.tts = None

        # Speaker Encoder 초기화
        if RESEMBLYZER_AVAILABLE:
            self.voice_encoder = VoiceEncoder(device=device)
        else:
            self.voice_encoder = None

    def clone_voice(self, reference_audio: str, target_text: str,
                   output_path: str, language: str = 'en') -> Dict:
        """
        음성 복제 (Zero-shot Voice Cloning)

        Args:
            reference_audio: 복제할 목소리 샘플 (3-10초 권장)
            target_text: 생성할 텍스트
            output_path: 출력 오디오 경로
            language: 언어 코드 ('en', 'ko', 'ja', 'zh' 등)

        Returns:
            {
                'output_path': 출력 경로,
                'text': 생성된 텍스트,
                'duration': 오디오 길이 (초),
                'sample_rate': 샘플레이트,
                'success': 성공 여부
            }
        """
        if not TTS_AVAILABLE:
            return {
                'error': 'TTS not installed',
                'message': 'Install with: pip install TTS',
                'success': False
            }

        if self.tts is None:
            return {
                'error': 'TTS model not loaded',
                'success': False
            }

        try:
            # 음성 생성
            print(f"Cloning voice from {reference_audio}...")
            print(f"Generating: {target_text[:50]}...")

            self.tts.tts_to_file(
                text=target_text,
                speaker_wav=reference_audio,
                language=language,
                file_path=output_path
            )

            # 오디오 정보 읽기
            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate

            return {
                'output_path': output_path,
                'text': target_text,
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'language': language,
                'success': True
            }

        except Exception as e:
            return {
                'error': 'Voice cloning failed',
                'message': str(e),
                'success': False
            }

    def extract_speaker_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Speaker Embedding 추출 (256-dim vector)

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            Speaker embedding (numpy array) 또는 None
        """
        if not RESEMBLYZER_AVAILABLE:
            warnings.warn("Resemblyzer not available")
            return None

        try:
            # 오디오 전처리
            wav = preprocess_wav(audio_path)

            # Embedding 추출
            embedding = self.voice_encoder.embed_utterance(wav)

            return embedding

        except Exception as e:
            warnings.warn(f"Failed to extract embedding: {e}")
            return None

    def compare_voices(self, audio1_path: str, audio2_path: str) -> float:
        """
        두 음성 비교 (코사인 유사도)

        Args:
            audio1_path: 첫 번째 오디오
            audio2_path: 두 번째 오디오

        Returns:
            유사도 (0~1, 높을수록 유사)
        """
        emb1 = self.extract_speaker_embedding(audio1_path)
        emb2 = self.extract_speaker_embedding(audio2_path)

        if emb1 is None or emb2 is None:
            return 0.0

        # 코사인 유사도
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def list_available_models(self) -> List[str]:
        """사용 가능한 TTS 모델 리스트"""
        if not TTS_AVAILABLE:
            return []

        return TTS.list_models()

    def change_model(self, model_name: str) -> bool:
        """TTS 모델 변경"""
        if not TTS_AVAILABLE:
            return False

        try:
            self.tts = TTS(model_name=model_name, progress_bar=False).to(self.device)
            self.model_name = model_name
            return True
        except Exception as e:
            warnings.warn(f"Failed to change model: {e}")
            return False

    def synthesize_speech(self, text: str, output_path: str,
                         speaker: str = None, language: str = 'en') -> Dict:
        """
        일반 TTS (음성 복제 없이)

        Args:
            text: 생성할 텍스트
            output_path: 출력 경로
            speaker: 화자 ID (모델에 따라 다름)
            language: 언어 코드

        Returns:
            결과 딕셔너리
        """
        if not TTS_AVAILABLE or self.tts is None:
            return {
                'error': 'TTS not available',
                'success': False
            }

        try:
            if speaker:
                self.tts.tts_to_file(
                    text=text,
                    speaker=speaker,
                    language=language,
                    file_path=output_path
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    language=language,
                    file_path=output_path
                )

            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate

            return {
                'output_path': output_path,
                'text': text,
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'success': True
            }

        except Exception as e:
            return {
                'error': 'TTS synthesis failed',
                'message': str(e),
                'success': False
            }


# ========================================
# 고수준 API
# ========================================

def clone_voice_simple(reference_audio: str, target_text: str,
                      output_path: str = 'media/cloned_voice.wav',
                      language: str = 'en') -> Dict:
    """
    간편한 음성 복제 API

    Args:
        reference_audio: 참조 음성 파일
        target_text: 생성할 텍스트
        output_path: 출력 경로
        language: 언어 코드

    Returns:
        결과 딕셔너리
    """
    if not TTS_AVAILABLE:
        return {
            'error': 'TTS not installed',
            'message': 'Install with: pip install TTS',
            'success': False
        }

    cloner = VoiceCloner()
    return cloner.clone_voice(reference_audio, target_text, output_path, language)


def compare_speaker_similarity(audio1: str, audio2: str) -> float:
    """
    두 음성의 화자 유사도 계산

    Args:
        audio1: 첫 번째 오디오
        audio2: 두 번째 오디오

    Returns:
        유사도 (0.0 ~ 1.0)
    """
    if not RESEMBLYZER_AVAILABLE:
        warnings.warn("Resemblyzer not available. Returning 0.0")
        return 0.0

    cloner = VoiceCloner()
    return cloner.compare_voices(audio1, audio2)


class VoiceConverter:
    """
    Voice Conversion (음성 변환)

    실시간 음성 변환 및 RVC (Retrieval-based Voice Conversion) 지원
    """

    def __init__(self):
        """Voice Converter 초기화"""
        # RVC는 별도 리포지토리 필요
        # https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
        self.rvc_available = False

    def convert_voice(self, source_audio: str, target_speaker: str,
                     output_path: str) -> Dict:
        """
        음성 변환 (RVC 기반)

        Args:
            source_audio: 원본 오디오
            target_speaker: 타겟 화자 모델
            output_path: 출력 경로

        Returns:
            결과 딕셔너리
        """
        return {
            'error': 'RVC not implemented',
            'message': 'Clone RVC repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion',
            'success': False
        }


if __name__ == "__main__":
    # 테스트
    if TTS_AVAILABLE:
        print("Testing Voice Cloning...")

        cloner = VoiceCloner()

        # 사용 가능한 모델 리스트
        print("\nAvailable TTS models:")
        models = cloner.list_available_models()
        for i, model in enumerate(models[:5]):  # 처음 5개만 표시
            print(f"  {i+1}. {model}")

        print("\nVoice Cloner ready!")
        print("Usage:")
        print("  cloner.clone_voice(reference_audio, target_text, output_path)")
        print("  cloner.extract_speaker_embedding(audio_path)")
        print("  cloner.compare_voices(audio1, audio2)")

    else:
        print("Coqui TTS not installed. Install with:")
        print("  pip install TTS")

    if RESEMBLYZER_AVAILABLE:
        print("\n✓ Resemblyzer available for speaker embedding")
    else:
        print("\n✗ Resemblyzer not installed. Install with:")
        print("  pip install resemblyzer")
