import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os
import platform
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 페이지 설정
st.set_page_config(page_title="E-commerce 전략 대시보드", page_icon="🛍️", layout="wide")

# Matplotlib 기본 설정
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'Data')

    try:
        customer = pd.read_csv(os.path.join(data_dir, "Customer_info.csv"))
        discount = pd.read_csv(os.path.join(data_dir, "Discount_info.csv"))
        marketing = pd.read_csv(os.path.join(data_dir, "Marketing_info.csv"))
        onlinesales = pd.read_csv(os.path.join(data_dir, "Onlinesales_info.csv"))
        tax = pd.read_csv(os.path.join(data_dir, "Tax_info.csv"))
    except FileNotFoundError as e:
        st.error(f"❌ 데이터 파일 없음. 경로 확인 필요: {e}")
        st.stop()

    # 전처리
    month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    if discount['월'].dtype == 'object':
        discount['월'] = discount['월'].map(month_mapping)
        
    onlinesales['거래날짜'] = pd.to_datetime(onlinesales['거래날짜'])
    onlinesales['월'] = onlinesales['거래날짜'].dt.month

    # 병합
    df = pd.merge(onlinesales, customer, on='고객ID', how='left')
    df = pd.merge(df, discount, on=['월', '제품카테고리'], how='left')
    df = pd.merge(df, tax, on='제품카테고리', how='left')

    df['쿠폰코드'].fillna('unknown', inplace=True)
    df['할인율'].fillna(0, inplace=True)

    # 파생변수
    df['전체금액'] = df['수량'] * df['평균금액']
    def calculate_total(row):
        price = row['전체금액']
        gst = row['GST']
        discount_rate = row['할인율'] if row['쿠폰상태'] == 'Used' else 0
        return price * (1 - discount_rate/100) + (price * (1 - discount_rate/100) * gst)

    df['지불금액'] = df.apply(calculate_total, axis=1)

    first_delivery_fee = df.groupby(['고객ID', '거래ID'])['배송료'].first()
    customer_delivery_fee_sum = first_delivery_fee.groupby('고객ID').sum()

    # RFM 계산
    last_date = df['거래날짜'].max() + pd.DateOffset(days=1)
    rfm_df = df.groupby(['고객ID']).agg({
        '거래날짜': lambda x: (last_date - x.max()).days,
        '거래ID': lambda x: x.nunique(),
        '지불금액': 'sum'
    })
    rfm_df.rename(columns={'거래날짜': 'Recency', '거래ID': 'Frequency', '지불금액': 'Monetary'}, inplace=True)
    
    for customer_id, delivery_fee in customer_delivery_fee_sum.items():
        if customer_id in rfm_df.index:
            rfm_df.loc[customer_id, 'Monetary'] += delivery_fee
            
    rfm_df.reset_index(inplace=True)
    df = df.merge(rfm_df, on='고객ID')

    # 가중치 Recency
    product_category_values = {
        'Office': 9, 'Apparel': 6, 'Nest-USA': 5, 'Drinkware': 13,
        'Lifestyle': 17, 'Nest': 4, 'Bags': 18, 'Headgear': 27,
        'Notebooks & Journals': 20, 'Waze': 23
    }
    df_temp = df.copy()
    for category, value in product_category_values.items():
        df_temp.loc[df_temp['제품카테고리'] == category, '거래날짜'] += pd.Timedelta(days=value)
    
    df_temp['거래날짜'] = pd.to_datetime(df_temp['거래날짜'])
    last_weighted = df_temp['거래날짜'].max() + pd.DateOffset(days=27)
    weighted_r = df_temp.groupby(['고객ID']).agg({'거래날짜': lambda x: (last_weighted - x.max()).days})
    df['Recency'] = df['고객ID'].map(weighted_r['거래날짜'])
    
    # 고객 데이터 생성
    customer_df = df.groupby('고객ID')[['Recency', 'Frequency', 'Monetary']].first().reset_index()

    # 등급 부여
    customer_df['R'] = customer_df['Recency'].apply(lambda x: 5 if x<=50 else (4 if x<=100 else (3 if x<=150 else (2 if x<=200 else (1 if x<=300 else 0)))))
    customer_df['F'] = customer_df['Frequency'].apply(lambda x: 0 if x<=8 else (1 if x<=20 else (2 if x<=50 else (3 if x<=100 else (4 if x<=300 else 5)))))
    customer_df['M'] = customer_df['Monetary'].apply(lambda x: 0 if x<=1676 else (1 if x<=2500 else (2 if x<=4000 else (3 if x<=6000 else (4 if x<=10000 else 5)))))

    # 세그먼트 분류
    def classify_segment(row):
        R, F, M = row['R'], row['F'], row['M']
        if R==5 and F==5 and M==5: return 'VIP고객'
        elif R>=3 and F>=3 and M>=3: return '충성고객'
        elif R>=2 and F>=2 and M>=1: return '잠재충성고객'
        elif R>=0 and F>=2 and M>=2: return '놓치면안될고객'
        elif R>=3 and F>=0 and M>=0: return '최근신규방문고객'
        elif R>=0 and F>=1 and M>=1: return '이탈우려고객'
        else: return '기타'

    customer_df['segment'] = customer_df.apply(classify_segment, axis=1)
    
    # 영문 매핑
    seg_map = {'VIP고객': 'VIP', '충성고객': 'Loyal', '잠재충성고객': 'Potential Loyal',
               '놓치면안될고객': "Can't Lose", '최근신규방문고객': 'New Customers',
               '이탈우려고객': 'At Risk', '기타': 'Others'}
    customer_df['segment_en'] = customer_df['segment'].map(seg_map)

    # 최종 병합
    df_final = df.merge(customer_df[['고객ID', 'R', 'F', 'M', 'segment', 'segment_en']], on='고객ID')
    df_final['최초거래월'] = df_final.groupby('고객ID')['월'].transform('min')
    df_final['경과월'] = df_final['월'] - df_final['최초거래월']

    return df_final, customer_df, marketing

with st.spinner('데이터 분석 수행 중...'):
    df, customer_df, marketing = load_and_process_data()

# 사이드바
st.sidebar.title("이커머스 분석 메뉴")
menu = st.sidebar.radio("페이지 이동", ["1. 대시보드 개요", "2. RFM 고객 세분화", "3. 리텐션 & 코호트", "4. 연관 분석", "5. 마케팅 전략"])
st.sidebar.markdown("---")

# 지역 필터
all_regions = sorted(df['고객지역'].unique())
selected_regions = st.sidebar.multiselect("지역 선택", all_regions, default=all_regions)

if selected_regions:
    df_filtered = df[df['고객지역'].isin(selected_regions)]
    target_ids = df_filtered['고객ID'].unique()
    customer_df_filtered = customer_df[customer_df['고객ID'].isin(target_ids)]
else:
    df_filtered = df
    customer_df_filtered = customer_df

st.sidebar.info(f"선택 고객 수: {customer_df_filtered['고객ID'].nunique():,}명")

# 1. 개요
if menu == "1. 대시보드 개요":
    st.title("📊 대시보드 개요")
    c1, c2, c3, c4 = st.columns(4)
    total_cust = customer_df_filtered['고객ID'].nunique()
    total_rev = df_filtered['지불금액'].sum()
    c1.metric("총 고객 수", f"{total_cust:,} 명")
    c2.metric("총 매출액", f"${total_rev:,.0f}")
    c3.metric("객단가", f"${total_rev/total_cust:,.0f}" if total_cust else 0)
    c4.metric("총 거래수", f"{df_filtered['거래ID'].nunique():,} 건")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        monthly = df_filtered.groupby('월')['지불금액'].sum().reset_index()
        st.plotly_chart(px.line(monthly, x='월', y='지불금액', markers=True, title="Monthly Revenue"), use_container_width=True)
    with col2:
        reg_cnt = df_filtered.groupby('고객지역')['고객ID'].nunique().reset_index()
        st.plotly_chart(px.pie(reg_cnt, values='고객ID', names='고객지역', hole=0.4, title="Customers by Region"), use_container_width=True)

# 2. RFM
elif menu == "2. RFM 고객 세분화":
    st.title("👥 RFM 고객 세분화 분석")
    seg_cnt = customer_df_filtered['segment'].value_counts().reset_index()
    seg_cnt.columns = ['Segment', 'Count']
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(seg_cnt, values='Count', names='Segment', title="세그먼트 비율"), use_container_width=True)
    with c2: st.plotly_chart(px.bar(seg_cnt, x='Segment', y='Count', color='Segment', title="세그먼트별 고객 수"), use_container_width=True)
    st.plotly_chart(px.scatter_3d(customer_df_filtered, x='Recency', y='Frequency', z='Monetary', color='segment', opacity=0.7), use_container_width=True)

# 3. 리텐션
elif menu == "3. 리텐션 & 코호트":
    st.title("🔄 세그먼트별 리텐션 분석")
    
    def get_cohort(d):
        if d.empty: return None
        g = d.groupby(['최초거래월', '경과월'])['고객ID'].nunique().reset_index()
        p = g.pivot(index='최초거래월', columns='경과월', values='고객ID')
        return p.divide(p.iloc[:,0], axis=0) if not p.empty else None

    seg_list = ["전체 고객"] + sorted(df_filtered['segment'].unique().tolist())
    seg_eng_map = {"전체 고객": "All Customers", "VIP고객": "VIP", "충성고객": "Loyal", "잠재충성고객": "Potential Loyal", 
                   "놓치면안될고객": "Can't Lose", "최근신규방문고객": "New Customers", "이탈우려고객": "At Risk", "기타": "Others"}
    
    col1, col2 = st.columns([1, 3])
    with col1: sel_seg = st.selectbox("분석할 세그먼트 선택:", seg_list)
    
    target = df_filtered if sel_seg == "전체 고객" else df_filtered[df_filtered['segment'] == sel_seg]
    mat = get_cohort(target)
    
    if mat is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(mat, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=0.5, ax=ax)
        ax.set_title(f"{seg_eng_map.get(sel_seg, sel_seg)} Cohort Analysis", fontsize=15)
        ax.set_ylabel("First Transaction Month", fontsize=12)
        ax.set_xlabel("Months Passed", fontsize=12)
        st.pyplot(fig)
    else: st.warning("데이터 부족")

# 4. 연관 분석
elif menu == "4. 연관 분석":
    st.title("🛒 장바구니 연관 분석")
    
    opts = ["전체"] + sorted(df_filtered['segment'].dropna().unique().tolist())
    tgt_seg = st.selectbox("분석 대상 세그먼트", opts)
    sup = st.slider("최소 지지도", 0.005, 0.1, 0.01)
    
    if st.button("분석 실행"):
        with st.spinner("계산 중..."):
            d = df_filtered[['고객ID', '제품카테고리']] if tgt_seg == "전체" else df_filtered[df_filtered['segment'] == tgt_seg][['고객ID', '제품카테고리']]
            ds = [list(set(x)) for x in d.groupby('고객ID')['제품카테고리'].apply(list)]
            te = TransactionEncoder()
            te_ary = te.fit(ds).transform(ds)
            res = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=sup, use_colnames=True)
            
            if not res.empty:
                rule = association_rules(res, metric="lift", min_threshold=1).sort_values('lift', ascending=False).head(15)
                rule['antecedents'] = rule['antecedents'].apply(lambda x: ', '.join(list(x)))
                rule['consequents'] = rule['consequents'].apply(lambda x: ', '.join(list(x)))
                st.dataframe(rule[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                st.plotly_chart(px.scatter(rule, x="support", y="confidence", size="lift", color="lift", title=f"지지도 vs 신뢰도 ({tgt_seg})"), use_container_width=True)
            else: st.warning("결과 없음")

# 5. 마케팅 전략
elif menu == "5. 마케팅 전략":
    st.title("💡 최종 마케팅 전략 제안 (Action Plan)")
    
    tab1, tab2, tab3 = st.tabs(["🎯 전략 1: 타겟 마케팅", "🔄 전략 2: 리텐션 & 유입", "🗺️ 지역 연계 (MLB)"])

    # 탭 1: 타겟 마케팅
    with tab1:
        st.header("데이터 기반 세그먼트별 공략")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("잠재충성고객")
            st.info("**Office + Bags ➡ Lifestyle**")
            st.markdown("- **Insight:** Office, Bags 동시 구매 시 Lifestyle 제품 구매 확률 높음\n- **Action:** 해당 고객군에게 Lifestyle 카테고리 할인 쿠폰 발송 (Cross-selling)")
        
        with c2:
            st.subheader("VIP / 충성고객")
            st.info("**대량 구매 이력 활용**")
            st.markdown("- **Insight:** Headgear 등 특정 품목 대량 구매 패턴\n- **Action:** 신상품 출시 시 우선 알림 및 얼리버드 혜택 제공으로 Lock-in 강화")
            
        with c3:
            st.subheader("이탈우려고객")
            st.info("**새로운 카테고리 제안**")
            st.markdown("- **Insight:** 기존 구매 품목에 대한 반응률 저조\n- **Action:** 기존 품목 대신 베스트셀러나 신규 카테고리 위주의 환기성 메일링 시도")

    # 탭 2: 리텐션 & 유입
    with tab2:
        st.header("리텐션 및 첫 구매 유도 전략")
        
        # 1. 의류 재구매
        st.subheader("1. 의류(Apparel) Next-Day 전략")
        st.write("의류 구매 고객 중 약 **41%**가 구매 **바로 다음 날** 재구매하는 패턴 발견")
        
        # 재구매 데이터 시각화
        re_data = pd.DataFrame({'Cycle': ['Same Day', 'Next Day (D+1)', '2~7 Days', '8+ Days'], 'Count': [739, 153, 20, 411]})
        fig_apparel = px.bar(re_data, x='Cycle', y='Count', text='Count', title="Apparel Repurchase Cycle", color='Count')
        st.plotly_chart(fig_apparel, use_container_width=True)
        st.success("🚀 **Action:** 의류 구매 익일, 어울리는 액세서리나 하의를 추천하는 푸시 알림 발송")
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        
        # 2. 비회원/신규
        with col_a:
            st.subheader("2. 비회원 및 신규 가입자")
            ghost_members = 72 # 분석 텍스트 기반 하드코딩
            st.metric("가입 후 미거래 고객 (Ghost)", f"{ghost_members}명")
            st.markdown("""
            - **Insight:** 마케팅 집중 기간(8월~) 유입되었으나 거래 없음
            - **Action:** '첫 구매 전용 15% 쿠폰' 및 단계별 혜택(첫달 30%, 익월 10%) 제공
            """)
            
        # 3. 요일별 프로모션
        with col_b:
            st.subheader("3. 요일별 게릴라 프로모션")
            # 요일별 거래량 계산
            df_filtered['day_name'] = df_filtered['거래날짜'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df_filtered['day_name'].value_counts().reindex(day_order).reset_index()
            day_counts.columns = ['Day', 'Transactions']
            
            fig_day = px.bar(day_counts, x='Day', y='Transactions', color='Transactions', title="Transaction Volume by Day")
            st.plotly_chart(fig_day, use_container_width=True)
            st.success("🚀 **Action:** 거래량이 가장 낮은 월/화요일에 '게릴라 쿠폰' 배포로 매출 방어")

    # 탭 3: 지역 연계 (MLB)
    with tab3:
        st.header("지역 특성 활용 (MLB 연고지)")
        st.markdown("**전략:** 야구장이 위치한 주요 도시를 타겟으로 7~8월(올스타전/하반기) Headgear 프로모션 진행")
        
        locations = {'Chicago': (41.8781, -87.6298), 'California': (36.7783, -119.4179), 'New York': (40.7128, -74.0060), 'New Jersey': (40.0583, -74.4057), 'Washington DC': (38.9072, -77.0369)}
        stadiums = {'Wrigley Field': (41.9484, -87.6550), 'Yankee Stadium': (40.8296, -73.9261), 'Dodger Stadium': (34.0738, -118.2400), 'Nationals Park': (38.8972, -77.0211)}

        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        for city, coord in locations.items():
            folium.Circle(location=coord, radius=60000, color='blue', fill=True, opacity=0.2, popup=city).add_to(m)
        for stad, coord in stadiums.items():
            folium.Marker(location=coord, popup=stad, icon=folium.Icon(color='red', icon='star')).add_to(m)

        st_folium(m, width=800, height=500)
