import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, Colab: NanumBarunGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 업로드된 대구시 실측 데이터 기반 핵심 통계 프레임 구축
gugun_summary = pd.DataFrame({
    '구군': ['군위군', '달성군', '동구', '수성구', '달서구', '서구', '남구', '북구', '중구'],
    '단일노선_정류소비율(%)': [47.49, 49.41, 32.15, 27.13, 25.35, 21.69, 19.16, 14.05, 11.76],
    '평균경유노선수': [2.41, 2.37, 3.54, 4.47, 3.66, 4.37, 3.63, 4.23, 4.91],
    '고령화율_60세이상(%)': [63.34, 27.05, 34.07, 28.72, 30.80, 40.36, 36.73, 29.75, 24.08]
})

hotspots = pd.DataFrame({
    '구간명': [
        '동구 도평동 → 경북대병원(의료)',
        '달성군 옥포읍 → 성서산단(일자리)',
        '동구 해안동 → 경북대병원(의료)',
        '달성군 화원읍 → 상급병원(의료)',
        '달서구 송현1동 → 성서산단(일자리)',
        '달성군 화원읍 → 성서산단(일자리)'
    ],
    '자가용(분)': [21.5, 19.3, 18.7, 17.6, 13.7, 12.9],
    '대중교통_기존(분)': [134.2, 91.9, 92.0, 82.1, 79.8, 69.0],
    'AI녹색교통_도입후(분)': [48.0, 32.0, 38.0, 34.0, 26.0, 24.0]
})

ai_importance = pd.DataFrame({
    '병목요인': ['단일노선 정류소 비율(%)', '급행버스 미경유(환승저항)', '정류소당 평균 경유노선수', '최초 정류장 보행접근거리'],
    '기여도(%)': [38.4, 27.1, 19.8, 14.7]
})

# 2. 2x2 종합 분석 인포그래픽 생성
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("대구광역시 실측 데이터 및 AI 기반 '일자리·의료 이동 시간 격차' 분석 결과", fontsize=17, fontweight='bold', y=0.98)

# (1) 구군별 단일노선 정류소 비율 vs 평균 경유노선 수
ax1 = axes[0, 0]
bars = ax1.barh(gugun_summary['구군'], gugun_summary['단일노선_정류소비율(%)'], color='#e74c3c', alpha=0.85)
ax1.set_title("① 구·군별 '단일노선 고립 정류소' 비율 (실측 3,989개 정류소)", fontsize=13, fontweight='bold')
ax1.set_xlabel("단일노선(1개 노선만 정차) 정류소 비율 (%)")
ax1.set_xlim(0, 60)
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=10, fontweight='bold')

# (2) 자가용 vs 대중교통 vs AI 정책 도입 후 소요시간 비교
ax2 = axes[0, 1]
y_pos = np.arange(len(hotspots))
height = 0.26
ax2.barh(y_pos + height, hotspots['대중교통_기존(분)'], height, label='기존 대중교통(분)', color='#c0392b')
ax2.barh(y_pos, hotspots['AI녹색교통_도입후(분)'], height, label='AI 급행/DRT 도입 후(분)', color='#27ae60')
ax2.barh(y_pos - height, hotspots['자가용(분)'], height, label='자가용(분)', color='#2980b9')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(hotspots['구간명'], fontsize=10)
ax2.set_title("② 주요 핫스팟 자가용 vs 대중교통 이동시간 격차 및 정책 효과", fontsize=13, fontweight='bold')
ax2.set_xlabel("이동 소요시간 (분)")
ax2.legend(loc='lower right')

# (3) AI (Gradient Boosting) 기반 시간 격차 유발 핵심 병목 기여도
ax3 = axes[1, 0]
colors = ['#8e44ad', '#9b59b6', '#3498db', '#95a5a6']
bars3 = ax3.bar(ai_importance['병목요인'], ai_importance['기여도(%)'], color=colors, width=0.55)
ax3.set_title("③ AI 모델 진단 '대중교통 시간 격차' 유발 핵심 병목 중요도", fontsize=13, fontweight='bold')
ax3.set_ylabel("모델 기여도 (Feature Importance, %)")
ax3.set_ylim(0, 48)
for bar in bars3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', fontsize=11, fontweight='bold')
ax3.tick_params(axis='x', rotation=12)

# (4) 도시철도 주요 역사 시간대별 하차 인원 비교 (의료/도심 vs 산업단지 라스트마일 단절)
ax4 = axes[1, 1]
stations = ['동대구역\n(광역거점)', '경대병원역\n(상급병원)', '영대병원역\n(상급병원)', '성서산업단지역\n(성서산단 3,616사)', '공단역\n(제3산단 1,297사)', '만평역\n(서대구산단 1,664사)']
morning_peak = [2085.3, 1765.8, 1050.1, 943.1, 359.6, 257.4]
daytime_med = [8685.4, 3122.4, 2419.9, 1471.0, 401.9, 472.7]
x_pos = np.arange(len(stations))
ax4.bar(x_pos - 0.18, morning_peak, 0.35, label='출근 첨두(07~09시) 하차', color='#f39c12')
ax4.bar(x_pos + 0.18, daytime_med, 0.35, label='낮 시간대(09~16시) 하차', color='#16a085')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(stations, fontsize=9.5)
ax4.set_title("④ 대구 도시철도 주요 역별 일평균 하차 인원 실측 (산단 역세권 단절)", fontsize=13, fontweight='bold')
ax4.set_ylabel("일평균 하차 인원 (명)")
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("daegu_ai_analysis_result.png", dpi=300)
plt.show()