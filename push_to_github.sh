#!/bin/bash
# GitHub에 Push하는 스크립트
# 사용법: ./push_to_github.sh YOUR_USERNAME

if [ -z "$1" ]; then
    echo "❌ 사용법: ./push_to_github.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

USERNAME=$1
REPO_NAME="prompt-arsenal"

echo "🔗 Remote 추가 중..."
git remote add origin https://github.com/$USERNAME/$REPO_NAME.git

echo "📤 Push 중..."
git branch -M main
git push -u origin main

echo "✅ 완료! https://github.com/$USERNAME/$REPO_NAME"
