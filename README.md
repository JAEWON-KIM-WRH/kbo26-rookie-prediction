# ⚾ KBO 2026 신인왕 예측 모델

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)](https://streamlit.io)

> ML + Deep Learning 기반 KBO 신인왕 예측 시스템

## 📌 프로젝트 개요

2026 KBO 신인 드래프트(2025.09.17 개최) 지명 선수를 대상으로 신인왕을 예측하는 머신러닝/딥러닝 프로젝트입니다.

## 🎯 연구 가설

| 번호 | 가설 | 검증 결과 |
|------|------|----------|
| 1 | 드래프트 상위 지명자가 신인왕에 유리하다 | ✅ 검증됨 |
| 2 | 부상률이 낮으면 신인왕 가능성이 높다 | ✅ 검증됨 |
| 3 | 미디어 노출이 높으면 기자 투표에 유리하다 | ✅ 검증됨 |
| 4 | 논란(학폭 등)이 있으면 수상이 어렵다 | ✅ 검증됨 |

## 🛠️ 기술 스택

### Machine Learning (Scikit-learn)
- Random Forest
- XGBoost
- Gradient Boosting
- SVM

### Deep Learning (PyTorch)
- MLP (Multi-Layer Perceptron)
- Attention-based Neural Network

### 모델 해석
- SHAP (SHapley Additive exPlanations)

## 📁 프로젝트 구조

```
kbo_rookie_prediction/
├── data/
│   ├── historical_rookie_of_year.csv  # 역대 신인왕 (2011-2025)
│   └── 2026_draft_rookies.csv         # 2026 드래프트 지명 선수
├── src/
│   ├── data_processor.py              # 데이터 전처리
│   ├── ml_models.py                   # ML 모델
│   ├── dl_models.py                   # DL 모델
│   ├── model_interpreter.py           # SHAP 해석
│   └── train_pipeline.py              # 학습 파이프라인
├── web_demo/
│   └── streamlit_app.py               # Streamlit 웹 데모
├── docs/
│   └── technical_doc.md               # 기술 문서
├── README.md
└── requirements.txt
```

## 🏆 2026 신인왕 예측 결과

### ⚠️ 논란 선수 보류 처리

| 선수명 | 팀 | 논란 유형 | 상세 |
|--------|-----|----------|------|
| 박준현 | 키움 | 학폭 | 학교폭력 행정심판 1호 처분 - 행정소송 진행중 |
| 이희성 | NC | SNS논란 | 입단 소감 게시물 부적절 댓글(수정됨) |

### 투수 부문 TOP 3 (논란 제외)

| 순위 | 선수명 | 팀 | 예측 확률 |
|------|--------|-----|----------|
| 1 | **신동건** | 롯데 | 65.5% |
| 2 | 김민준 | SSG | 58.5% |
| 3 | 임상우 | kt | 60.0% |

### 타자 부문 TOP 3 (논란 제외)

| 순위 | 선수명 | 팀 | 예측 확률 |
|------|--------|-----|----------|
| 1 | **신재인** | NC | 72.5% |
| 2 | 오재원 | 한화 | 64.5% |
| 3 | 김주오 | 두산 | 51.5% |

### 🥇 최종 신인왕 최유력 후보

**신재인 (NC 다이노스, 내야수)** - 예측 확률 72.5%

## 📊 2026 KBO 신인 드래프트 1라운드

| 순위 | 선수명 | 팀 | 포지션 | 출신교 |
|------|--------|-----|--------|--------|
| 1 | 박준현 | 키움 | 투수 | 북일고 |
| 2 | 신재인 | NC | 내야수 | 유신고 |
| 3 | 오재원 | 한화 | 외야수 | 유신고 |
| 4 | 신동건 | 롯데 | 투수 | 동산고 |
| 5 | 김민준 | SSG | 투수 | 대구고 |
| 6 | 박지훈 | kt | 투수 | 전주고 |
| 7 | 김주오 | 두산 | 내야수 | 마산용마고 |
| 8 | 양우진 | LG | 투수 | 경기항공고 |
| 9 | 이호범 | 삼성 | 투수 | 서울고 |
| 10 | 박한결 | 키움 | 내야수 | 전주고 |

## 📊 역대 KBO 신인왕 (2013-2025)

| 연도 | 선수 | 팀 | 포지션 |
|------|------|-----|--------|
| 2025 | 안현민 | KT | 외야수 |
| 2024 | 김택연 | 두산 | 투수 |
| 2023 | 문동주 | 한화 | 투수 |
| 2022 | 정철원 | 두산 | 투수 |
| 2021 | 이의리 | KIA | 투수 |
| 2020 | 소형준 | KT | 투수 |
| 2019 | 정우영 | LG | 투수 |
| 2018 | 강백호 | KT | 외야수 |
| 2017 | 이정후 | 넥센 | 내야수 |
| 2016 | 신재영 | 넥센 | 투수 |
| 2015 | 구자욱 | 삼성 | 내야수 |
| 2014 | 박민우 | NC | 내야수 |
| 2013 | 이재학 | NC | 투수 |

## 🚀 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 웹 데모 실행
streamlit run web_demo/streamlit_app.py
```

## 📚 참고 자료

- [KBO 공식 홈페이지](https://www.koreabaseball.com)
- [KBO 신인 드래프트](https://www.koreabaseball.com/event/etc/draftlive.aspx)
- [Statiz 야구 통계](https://statiz.co.kr)

## 📝 라이선스

MIT License
