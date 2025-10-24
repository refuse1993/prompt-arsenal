"""
API Profile Manager
config.json에서 API 프로필을 로드하고 LLM을 초기화하는 유틸리티
"""

import json
import os
from typing import Dict, Optional, Tuple


class ProfileManager:
    """API 프로필 관리 클래스"""

    def __init__(self, config_path: str = "config.json"):
        """
        Args:
            config_path: config.json 파일 경로
        """
        self.config_path = config_path
        self.profiles = {}
        self.default_profile = None
        self.load_config()

    def load_config(self) -> None:
        """config.json 로드"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  설정 파일을 찾을 수 없습니다: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.profiles = config.get('profiles', {})
                self.default_profile = config.get('default_profile', None)
        except Exception as e:
            print(f"⚠️  설정 파일 로드 실패: {e}")

    def get_profile(self, profile_name: Optional[str] = None) -> Optional[Dict]:
        """
        프로필 가져오기

        Args:
            profile_name: 프로필 이름 (None이면 기본 프로필)

        Returns:
            프로필 딕셔너리 또는 None
        """
        if profile_name is None:
            profile_name = self.default_profile

        if profile_name is None:
            print("⚠️  프로필 이름이 지정되지 않았고 기본 프로필도 없습니다.")
            return None

        profile = self.profiles.get(profile_name)
        if profile is None:
            print(f"⚠️  프로필을 찾을 수 없습니다: {profile_name}")
            return None

        return profile

    def get_llm_profile(self, profile_name: Optional[str] = None) -> Optional[Dict]:
        """
        LLM 프로필만 가져오기 (type='llm')

        Args:
            profile_name: 프로필 이름 (None이면 기본 프로필)

        Returns:
            LLM 프로필 또는 None
        """
        profile = self.get_profile(profile_name)

        if profile and profile.get('type') != 'llm':
            print(f"⚠️  '{profile_name}' 프로필은 LLM 프로필이 아닙니다 (type={profile.get('type')})")
            return None

        return profile

    def list_llm_profiles(self) -> Dict[str, Dict]:
        """
        모든 LLM 프로필 목록 반환

        Returns:
            {profile_name: profile_dict}
        """
        llm_profiles = {}
        for name, profile in self.profiles.items():
            if profile.get('type') == 'llm':
                llm_profiles[name] = profile

        return llm_profiles

    def get_llm_config(self, profile_name: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
        """
        LLM 설정 반환 (provider, model, api_key)

        Args:
            profile_name: 프로필 이름

        Returns:
            (provider, model, api_key) 튜플 또는 None
        """
        profile = self.get_llm_profile(profile_name)

        if profile is None:
            return None

        provider = profile.get('provider')
        model = profile.get('model')
        api_key = profile.get('api_key')

        if not all([provider, model, api_key]):
            print(f"⚠️  프로필에 필수 정보가 없습니다: provider={provider}, model={model}, api_key={'***' if api_key else None}")
            return None

        return (provider, model, api_key)

    def select_llm_profile_interactive(self) -> Optional[str]:
        """
        대화형으로 LLM 프로필 선택

        Returns:
            선택된 프로필 이름 또는 None
        """
        llm_profiles = self.list_llm_profiles()

        if not llm_profiles:
            print("⚠️  사용 가능한 LLM 프로필이 없습니다.")
            print("💡  interactive_cli.py에서 API 프로필을 먼저 설정하세요.")
            return None

        # 프로필 목록 출력
        print("\n📋 사용 가능한 LLM 프로필:")
        profile_list = list(llm_profiles.keys())

        for idx, name in enumerate(profile_list, 1):
            profile = llm_profiles[name]
            default_marker = " [기본]" if name == self.default_profile else ""
            print(f"  {idx}. {name}{default_marker}")
            print(f"     Provider: {profile.get('provider')}, Model: {profile.get('model')}")

        # 사용자 입력
        try:
            choice = input(f"\n프로필 선택 (1-{len(profile_list)}, 기본값: 1): ").strip()

            if not choice:
                choice = "1"

            idx = int(choice) - 1

            if 0 <= idx < len(profile_list):
                selected = profile_list[idx]
                print(f"✅ '{selected}' 프로필 선택됨")
                return selected
            else:
                print("⚠️  잘못된 선택입니다.")
                return None

        except ValueError:
            print("⚠️  숫자를 입력하세요.")
            return None
        except KeyboardInterrupt:
            print("\n❌ 취소됨")
            return None


# 전역 싱글톤 인스턴스
_profile_manager = None


def get_profile_manager(config_path: str = "config.json") -> ProfileManager:
    """
    ProfileManager 싱글톤 인스턴스 반환

    Args:
        config_path: config.json 경로

    Returns:
        ProfileManager 인스턴스
    """
    global _profile_manager

    if _profile_manager is None:
        _profile_manager = ProfileManager(config_path)

    return _profile_manager
