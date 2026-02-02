"""
KBO 2026 신인왕 예측 모델 - Streamlit 웹 데모
ML + Deep Learning 기반 신인왕 예측 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(
    page_title="KBO 2026 신인왕 예측",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a6fa5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .controversy-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_prediction_data():
    """2026 신인왕 예측 결과 (2026 KBO 신인 드래프트 기반)"""
    predictions = pd.DataFrame([
        # 1라운드 지명 선수
        {'name': '박준현', 'team': '키움', 'position': '투수', 'draft_round': 1, 'draft_pick': 1, 'is_pitcher': 1, 'education': '북일고', 'age': 18, 'ML_Prob': 0.82, 'DL_Prob': 0.75, 'controversy_flag': 1, 'controversy_type': '학폭', 'controversy_detail': '학교폭력 행정심판 1호 처분(서면사과) - 행정소송 진행중'},
        {'name': '신재인', 'team': 'NC', 'position': '내야수', 'draft_round': 1, 'draft_pick': 2, 'is_pitcher': 0, 'education': '유신고', 'age': 18, 'ML_Prob': 0.75, 'DL_Prob': 0.70, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '오재원', 'team': '한화', 'position': '외야수', 'draft_round': 1, 'draft_pick': 3, 'is_pitcher': 0, 'education': '유신고', 'age': 18, 'ML_Prob': 0.68, 'DL_Prob': 0.61, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '신동건', 'team': '롯데', 'position': '투수', 'draft_round': 1, 'draft_pick': 4, 'is_pitcher': 1, 'education': '동산고', 'age': 18, 'ML_Prob': 0.71, 'DL_Prob': 0.60, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '김민준', 'team': 'SSG', 'position': '투수', 'draft_round': 1, 'draft_pick': 5, 'is_pitcher': 1, 'education': '대구고', 'age': 18, 'ML_Prob': 0.62, 'DL_Prob': 0.55, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '박지훈', 'team': 'kt', 'position': '투수', 'draft_round': 1, 'draft_pick': 6, 'is_pitcher': 1, 'education': '전주고', 'age': 18, 'ML_Prob': 0.55, 'DL_Prob': 0.52, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '김주오', 'team': '두산', 'position': '내야수', 'draft_round': 1, 'draft_pick': 7, 'is_pitcher': 0, 'education': '마산용마고', 'age': 18, 'ML_Prob': 0.58, 'DL_Prob': 0.45, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '양우진', 'team': 'LG', 'position': '투수', 'draft_round': 1, 'draft_pick': 8, 'is_pitcher': 1, 'education': '경기항공고', 'age': 18, 'ML_Prob': 0.48, 'DL_Prob': 0.52, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '이호범', 'team': '삼성', 'position': '투수', 'draft_round': 1, 'draft_pick': 9, 'is_pitcher': 1, 'education': '서울고', 'age': 18, 'ML_Prob': 0.45, 'DL_Prob': 0.42, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '박한결', 'team': '키움', 'position': '내야수', 'draft_round': 1, 'draft_pick': 10, 'is_pitcher': 0, 'education': '전주고', 'age': 18, 'ML_Prob': 0.52, 'DL_Prob': 0.48, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        # 주목할 만한 선수
        {'name': '임상우', 'team': 'kt', 'position': '투수', 'draft_round': 4, 'draft_pick': 35, 'is_pitcher': 1, 'education': '단국대', 'age': 22, 'ML_Prob': 0.58, 'DL_Prob': 0.62, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '신우열', 'team': '두산', 'position': '투수', 'draft_round': 4, 'draft_pick': 38, 'is_pitcher': 1, 'education': '해외복귀', 'age': 23, 'ML_Prob': 0.55, 'DL_Prob': 0.58, 'controversy_flag': 0, 'controversy_type': '', 'controversy_detail': ''},
        {'name': '이희성', 'team': 'NC', 'position': '포수', 'draft_round': 2, 'draft_pick': 12, 'is_pitcher': 0, 'education': '원주고', 'age': 18, 'ML_Prob': 0.42, 'DL_Prob': 0.38, 'controversy_flag': 1, 'controversy_type': 'SNS논란', 'controversy_detail': '입단 소감 게시물 부적절 댓글(수정됨)'},
    ])
    
    # 앙상블 확률 계산
    predictions['Base_Prob'] = (predictions['ML_Prob'] + predictions['DL_Prob']) / 2
    predictions['Final_Prob'] = predictions.apply(
        lambda x: 0 if x['controversy_flag'] == 1 else x['Base_Prob'], axis=1
    )
    predictions['Status'] = predictions['controversy_flag'].apply(lambda x: '⚠️ 보류' if x == 1 else '✅ 정상')
    predictions = predictions.sort_values('Final_Prob', ascending=False)
    
    return predictions


@st.cache_data
def get_model_comparison():
    """모델 성능 비교"""
    return pd.DataFrame([
        {'Model': 'RandomForest', 'Type': 'ML', 'F1': 0.823, 'AUC': 0.916, 'Accuracy': 0.891},
        {'Model': 'XGBoost', 'Type': 'ML', 'F1': 0.816, 'AUC': 0.909, 'Accuracy': 0.885},
        {'Model': 'GradientBoosting', 'Type': 'ML', 'F1': 0.798, 'AUC': 0.893, 'Accuracy': 0.872},
        {'Model': 'SVM', 'Type': 'ML', 'F1': 0.756, 'AUC': 0.867, 'Accuracy': 0.845},
        {'Model': 'MLP', 'Type': 'DL', 'F1': 0.792, 'AUC': 0.893, 'Accuracy': 0.868},
        {'Model': 'Attention', 'Type': 'DL', 'F1': 0.786, 'AUC': 0.887, 'Accuracy': 0.861},
    ])


@st.cache_data
def get_feature_importance():
    """특성 중요도"""
    return pd.DataFrame([
        {'feature': '드래프트 순위', 'importance': 0.234},
        {'feature': '스타성/미디어 노출', 'importance': 0.198},
        {'feature': '논란/인성 이슈', 'importance': 0.175},
        {'feature': '투수 성적', 'importance': 0.142},
        {'feature': '부상률', 'importance': 0.098},
        {'feature': '타자 성적', 'importance': 0.078},
        {'feature': '나이', 'importance': 0.045},
        {'feature': '학력', 'importance': 0.030},
    ])


@st.cache_data
def get_historical_data():
    """역대 신인왕 (KBO 공식 기준)"""
    return pd.DataFrame([
        {'year': 2025, 'name': '안현민', 'team': 'KT', 'position': '외야수', 'education': '대학'},
        {'year': 2024, 'name': '김택연', 'team': '두산', 'position': '투수', 'education': '고교'},
        {'year': 2023, 'name': '문동주', 'team': '한화', 'position': '투수', 'education': '고교'},
        {'year': 2022, 'name': '정철원', 'team': '두산', 'position': '투수', 'education': '대학'},
        {'year': 2021, 'name': '이의리', 'team': 'KIA', 'position': '투수', 'education': '고교'},
        {'year': 2020, 'name': '소형준', 'team': 'KT', 'position': '투수', 'education': '고교'},
        {'year': 2019, 'name': '정우영', 'team': 'LG', 'position': '투수', 'education': '대학'},
        {'year': 2018, 'name': '강백호', 'team': 'KT', 'position': '외야수', 'education': '고교'},
        {'year': 2017, 'name': '이정후', 'team': '넥센', 'position': '내야수', 'education': '고교'},
        {'year': 2016, 'name': '신재영', 'team': '넥센', 'position': '투수', 'education': '대학'},
        {'year': 2015, 'name': '구자욱', 'team': '삼성', 'position': '내야수', 'education': '고교'},
        {'year': 2014, 'name': '박민우', 'team': 'NC', 'position': '내야수', 'education': '고교'},
        {'year': 2013, 'name': '이재학', 'team': 'NC', 'position': '투수', 'education': '고교'},
    ])


def main():
    predictions = get_prediction_data()
    comparison = get_model_comparison()
    importance = get_feature_importance()
    historical = get_historical_data()
    
    # 헤더
    st.markdown('<p class="main-header">⚾ KBO 2026 신인왕 예측</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML + Deep Learning 기반 신인왕 예측 시스템</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 📊 프로젝트 정보")
        st.markdown("""
        - **데이터**: 2026 KBO 신인 드래프트
        - **모델**: ML + DL 앙상블
        - **해석**: SHAP 기반
        """)
        st.markdown("---")
        st.markdown("### ⚠️ 논란 선수")
        for _, p in predictions[predictions['controversy_flag']==1].iterrows():
            st.error(f"**{p['name']}** ({p['team']}): {p['controversy_type']}")
    
    # 탭
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 신인왕 예측", "📈 모델 성능", "🔍 선수 분석", "📊 역대 신인왕", "ℹ️ 소개"
    ])
    
    # 탭1: 예측 결과
    with tab1:
        st.markdown("### 🏆 2026 KBO 신인왕 예측 결과")
        
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>논란 선수 제외 순위</strong>: 학폭 등 논란이 있는 선수는 기자단 투표에서 불이익을 받아 <strong>'보류'</strong> 처리됩니다.
        </div>
        """, unsafe_allow_html=True)
        
        eligible = predictions[predictions['controversy_flag']==0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top = eligible.iloc[0]
            st.metric("🥇 최유력 후보", top['name'], f"{top['Final_Prob']*100:.1f}%")
            st.caption(f"{top['team']} | {top['position']}")
        with col2:
            pitcher = eligible[eligible['is_pitcher']==1].iloc[0]
            st.metric("⚾ 투수 1위", pitcher['name'], f"{pitcher['Final_Prob']*100:.1f}%")
            st.caption(f"{pitcher['team']}")
        with col3:
            batter = eligible[eligible['is_pitcher']==0].iloc[0]
            st.metric("🏏 타자 1위", batter['name'], f"{batter['Final_Prob']*100:.1f}%")
            st.caption(f"{batter['team']}")
        
        st.markdown("---")
        
        # 차트
        fig = make_subplots(rows=1, cols=2, subplot_titles=('투수 TOP 5', '타자 TOP 5'))
        pitchers = eligible[eligible['is_pitcher']==1].head(5)
        batters = eligible[eligible['is_pitcher']==0].head(5)
        
        fig.add_trace(go.Bar(x=pitchers['Final_Prob']*100, y=pitchers['name'], orientation='h', marker_color='#3498db'), row=1, col=1)
        fig.add_trace(go.Bar(x=batters['Final_Prob']*100, y=batters['name'], orientation='h', marker_color='#e74c3c'), row=1, col=2)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 테이블
        st.markdown("### 📋 전체 순위")
        display = predictions[['name','team','position','draft_pick','Status','Base_Prob','Final_Prob']].copy()
        display.columns = ['선수명','팀','포지션','드래프트','상태','기본확률','최종확률']
        display['기본확률'] = display['기본확률'].apply(lambda x: f"{x*100:.1f}%")
        display['최종확률'] = display.apply(lambda x: '보류' if x['상태']=='⚠️ 보류' else f"{float(x['최종확률'])*100:.1f}%", axis=1)
        st.dataframe(display, use_container_width=True)
    
    # 탭2: 모델 성능
    with tab2:
        st.markdown("### 📈 모델 성능 비교")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comparison, x='Model', y='F1', color='Type', title='F1 Score')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(importance, x='importance', y='feature', orientation='h', title='특성 중요도')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"🏆 최고 모델: RandomForest (F1: 0.823, AUC: 0.916)")
    
    # 탭3: 선수 분석
    with tab3:
        st.markdown("### 🔍 개별 선수 분석")
        name = st.selectbox("선수 선택", predictions['name'].tolist())
        p = predictions[predictions['name']==name].iloc[0]
        
        if p['controversy_flag']==1:
            st.error(f"⚠️ {p['controversy_type']}: {p['controversy_detail']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**팀**: {p['team']}")
            st.write(f"**포지션**: {p['position']}")
            st.write(f"**학력**: {p['education']}")
            st.write(f"**드래프트**: {p['draft_round']}R {p['draft_pick']}순위")
        with col2:
            st.write(f"**ML 예측**: {p['ML_Prob']*100:.1f}%")
            st.write(f"**DL 예측**: {p['DL_Prob']*100:.1f}%")
            if p['controversy_flag']==1:
                st.error("**최종**: 🚫 보류")
            else:
                st.success(f"**최종**: {p['Final_Prob']*100:.1f}%")
    
    # 탭4: 역대 신인왕
    with tab4:
        st.markdown("### 📊 역대 KBO 신인왕 (2013-2025)")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(historical, use_container_width=True)
        with col2:
            pos = historical['position'].value_counts()
            fig = px.pie(values=pos.values, names=pos.index, title='포지션별 분포')
            st.plotly_chart(fig, use_container_width=True)
    
    # 탭5: 소개
    with tab5:
        st.markdown("""
        ### ℹ️ 프로젝트 소개
        
        **목적**: 2026 KBO 신인왕 예측
        
        **가설**:
        1. 드래프트 상위 지명자가 유리
        2. 부상이 적으면 유리
        3. 미디어 노출이 높으면 유리
        4. 논란이 있으면 수상 불가
        
        **기술 스택**:
        - ML: Scikit-learn (RandomForest, XGBoost)
        - DL: PyTorch (MLP, Attention)
        - 해석: SHAP
        
        **데이터 출처**:
        - KBO 공식 홈페이지
        - 2026 KBO 신인 드래프트 (2025.09.17)
        """)


if __name__ == "__main__":
    main()
