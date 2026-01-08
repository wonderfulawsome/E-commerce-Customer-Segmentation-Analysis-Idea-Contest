import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import platform
import os

# 페이지 설정
st.set_page_config(page_title="E-commerce 고객 세분화 대시보드", page_icon="🛍️", layout="wide")

# 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        try:
            plt.rc('font', family='NanumGothic')
        except:
            pass
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'Data')

    try:
        # CSV 파일 읽기
        customer = pd.read_csv(os.path.join(data_dir, "Customer_info.csv"))
        discount = pd.read_csv(os.path.join(data_dir, "Discount_info.csv"))
        marketing = pd.read_csv(os.path.join(data_dir, "Marketing_info.csv"))
        onlinesales = pd.read_csv(os.path.join(data_dir, "Onlinesales_info.csv"))
        tax = pd.read_csv(os.path.join(data_dir, "Tax_info.csv"))
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        st.stop()

    # 월 매핑
    month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    if discount['월'].dtype == 'object':
        discount['월'] = discount['월'].map(month_mapping)
        
    onlinesales['거래날짜'] = pd.to_datetime(onlinesales['거래날짜'])
    onlinesales['월'] = onlinesales['거래날짜'].dt.month

    # 데이터 병합
    df = pd.merge(onlinesales, customer, on='고객ID', how='left')
    df = pd.merge(df, discount, on=['월', '제품카테고리'], how='left')
    df = pd.merge(df, tax, on='제품카테고리', how='left')

    # 결측치 처리
    df['쿠폰코드'].fillna('unknown', inplace=True)
    df['할인율'].fillna(0, inplace=True)

    # 지불금액 계산
    df['전체금액'] = df['수량'] * df['평균금액']
    
    def calculate_total(row):
        price = row['전체금액']
        gst = row['GST']
        discount_rate = row['할인율'] if row['쿠폰상태'] == 'Used' else 0
        subtotal = price * (1 - discount_rate/100)
        return subtotal + (subtotal * gst)

    df['지불금액'] = df.apply(calculate_total, axis=1)

    # 배송료 합산
    first_delivery_fee = df.groupby(['고객ID', '거래ID'])['배송료'].first()
    customer_delivery_fee_sum = first_delivery_fee.groupby('고객ID').sum()

    # RFM 기본 계산
    last_date = df['거래날짜'].max() + pd.DateOffset(days=1)

    rfm_df = df.groupby(['고객ID']).agg({
        '거래날짜': lambda x: (last_date - x.max()).days,
        '거래ID': lambda x: x.nunique(),
        '지불금액': 'sum'
    })
    rfm_df.rename(columns={'거래날짜': 'Recency', '거래ID': 'Frequency', '지불금액': 'Monetary'}, inplace=True)
    
    # Monetary 업데이트
    for customer_id, delivery_fee in customer_delivery_fee_sum.items():
        if customer_id in rfm_df.index:
            rfm_df.loc[customer_id, 'Monetary'] += delivery_fee
            
    rfm_df.reset_index(inplace=True)
    df = df.merge(rfm_df, on='고객ID')

    # Recency 가중치 적용
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
    weighted_r.rename(columns={'거래날짜': 'Weighted_Recency'}, inplace=True)
    
    df['Recency'] = df['고객ID'].map(weighted_r['Weighted_Recency'])
    
    # 고객 데이터프레임 생성
    customer_df = df.groupby('고객ID')[['Recency', 'Frequency', 'Monetary']].first().reset_index()

    # R 등급 부여
    def assign_R(recency):
        if recency <= 50: return 5
        elif recency <= 100: return 4
        elif recency <= 150: return 3
        elif recency <= 200: return 2
        elif recency <= 300: return 1
        else: return 0
    customer_df['R'] = customer_df['Recency'].apply(assign_R)

    # F 등급 부여
    def assign_f_score(freq):
        if freq <= 8: return 0
        elif freq <= 20: return 1
        elif freq <= 50: return 2
        elif freq <= 100: return 3
        elif freq <= 300: return 4
        else: return 5
    customer_df['F'] = customer_df['Frequency'].apply(assign_f_score)

    # M 등급 부여
    def assign_m_score(money):
        if money <= 1676: return 0
        elif money <= 2500: return 1
        elif money <= 4000: return 2
        elif money <= 6000: return 3
        elif money <= 10000: return 4
        else: return 5
    customer_df['M'] = customer_df['Monetary'].apply(assign_m_score)

    # 세그먼트 분류
    def classify_customer_segment(row):
        R, F, M = row['R'], row['F'], row['M']
        if R == 5 and F == 5 and M == 5: return 'VIP고객'
        elif R >= 3 and F >= 3 and M >= 3: return '충성고객'
        elif R >= 2 and F >= 2 and M >= 1: return '잠재충성고객'
        elif R >= 0 and F >= 2 and M >= 2: return '놓치면안될고객'
        elif R >= 3 and F >= 0 and M >= 0: return '최근신규방문고객'
        elif R >= 0 and F >= 1 and M >= 1: return '이탈우려고객'
        else: return '기타'

    customer_df['segment'] = customer_df.apply(classify_customer_segment, axis=1)
    
    # 최종 병합
    df_final = df.merge(customer_df[['고객ID', 'R', 'F', 'M', 'segment']], on='고객ID')
    
    # 코호트 변수 생성
    df_final['최초거래월'] = df_final.groupby('고객ID')['월'].transform('min')
    df_final['경과월'] = df_final['월'] - df_final['최초거래월']

    return df_final, customer_df, marketing

# 데이터 로딩
with st.spinner('데이터 분석 중...'):
    df, customer_df, marketing = load_and_process_data()

# 사이드바 메뉴
st.sidebar.title("분석 메뉴")
menu = st.sidebar.radio(
    "페이지 선택",
    ["1. 대시보드 개요", "2. RFM 고객 세분화", "3. 리텐션 & 코호트", "4. 연관 분석", "5. 마케팅 전략"]
)
st.sidebar.markdown("---")

# 지역 필터
st.sidebar.subheader("지역 필터")
all_regions = sorted(df['고객지역'].unique())
selected_regions = st.sidebar.multiselect("지역 선택", all_regions, default=all_regions)

if selected_regions:
    df_filtered = df[df['고객지역'].isin(selected_regions)]
    target_ids = df_filtered['고객ID'].unique()
    customer_df_filtered = customer_df[customer_df['고객ID'].isin(target_ids)]
else:
    df_filtered = df
    customer_df_filtered = customer_df

st.sidebar.info(f"고객 수: {customer_df_filtered['고객ID'].nunique():,}명")

# Page 1: 개요
if menu == "1. 대시보드 개요":
    st.title("📊 대시보드 개요")
    
    # KPI 지표
    col1, col2, col3, col4 = st.columns(4)
    total_customers = customer_df_filtered['고객ID'].nunique()
    total_revenue = df_filtered['지불금액'].sum()
    avg_ticket = total_revenue / total_customers if total_customers > 0 else 0
    total_tx = df_filtered['거래ID'].nunique()

    col1.metric("총 고객 수", f"{total_customers:,} 명")
    col2.metric("총 매출액", f"${total_revenue:,.0f}")
    col3.metric("고객당 평균매출", f"${avg_ticket:,.0f}")
    col4.metric("총 거래수", f"{total_tx:,} 건")

    st.markdown("---")
    
    # 차트 영역
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("월별 매출")
        monthly = df_filtered.groupby('월')['지불금액'].sum().reset_index()
        fig1 = px.line(monthly, x='월', y='지불금액', markers=True)
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("지역별 분포")
        region_cnt = df_filtered.groupby('고객지역')['고객ID'].nunique().reset_index()
        fig2 = px.pie(region_cnt, values='고객ID', names='고객지역', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# Page 2: RFM
elif menu == "2. RFM 고객 세분화":
    st.title("👥 RFM 고객 세분화")
    
    # 세그먼트 차트
    seg_counts = customer_df_filtered['segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("비율")
        fig_pie = px.pie(seg_counts, values='Count', names='Segment', color='Segment')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("고객 수")
        fig_bar = px.bar(seg_counts, x='Segment', y='Count', color='Segment', text='Count')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("3D 시각화")
    fig_3d = px.scatter_3d(customer_df_filtered, x='Recency', y='Frequency', z='Monetary',
                           color='segment', opacity=0.7, size_max=10)
    st.plotly_chart(fig_3d, use_container_width=True)

# Page 3: 리텐션 (Selectbox 적용)
elif menu == "3. 리텐션 & 코호트":
    st.title("🔄 세그먼트별 리텐션 분석")

    # 코호트 함수
    def get_retention_matrix(data):
        if data.empty: return None
        grouping = data.groupby(['최초거래월', '경과월'])
        cohort_data = grouping['고객ID'].apply(pd.Series.nunique).reset_index()
        cohort_counts = cohort_data.pivot(index='최초거래월', columns='경과월', values='고객ID')
        if cohort_counts.empty: return None
        retention = cohort_counts.divide(cohort_counts.iloc[:, 0], axis=0)
        return retention

    # 세그먼트 선택
    segments_list = ["전체 고객"] + sorted(df_filtered['segment'].unique().tolist())
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_seg = st.selectbox("세그먼트 선택:", segments_list)

    # 데이터 필터링
    if selected_seg == "전체 고객":
        cohort_data = df_filtered
        st.info("전체 고객 리텐션")
    else:
        cohort_data = df_filtered[df_filtered['segment'] == selected_seg]
        st.info(f"**{selected_seg}** 리텐션")

    # 히트맵 출력
    retention_matrix = get_retention_matrix(cohort_data)

    if retention_matrix is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("표시할 데이터가 부족합니다.")

# Page 4: 연관 분석
elif menu == "4. 연관 분석":
    st.title("🛒 연관 분석")

    target_seg = st.selectbox("분석 대상", ["전체", "잠재충성고객", "이탈우려고객"])
    min_sup = st.slider("최소 지지도", 0.005, 0.1, 0.01)

    if st.button("분석 실행"):
        with st.spinner("계산 중..."):
            if target_seg == "전체":
                data_sub = df_filtered[['고객ID', '제품카테고리']]
            else:
                data_sub = df_filtered[df_filtered['segment'] == target_seg][['고객ID', '제품카테고리']]
            
            # 전처리
            dataset = [list(set(x)) for x in data_sub.groupby('고객ID')['제품카테고리'].apply(list).values.tolist()]

            te = TransactionEncoder()
            te_ary = te.fit(dataset).transform(dataset)
            df_te = pd.DataFrame(te_ary, columns=te.columns_)

            frequent = apriori(df_te, min_support=min_sup, use_colnames=True)
            
            if frequent.empty:
                st.warning("결과가 없습니다.")
            else:
                rules = association_rules(frequent, metric="lift", min_threshold=1)
                rules = rules.sort_values(by='lift', ascending=False).head(15)
                
                # 출력
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                
                fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                                 title="Support vs Confidence")
                st.plotly_chart(fig, use_container_width=True)

# Page 5: 마케팅 전략
elif menu == "5. 마케팅 전략":
    st.title("💡 마케팅 전략")

    st.header("1. 지역 MLB 마케팅")
    
    # 지도 시각화
    locations = {
        'Chicago': (41.8781, -87.6298), 'California': (36.7783, -119.4179),
        'New York': (40.7128, -74.0060), 'New Jersey': (40.0583, -74.4057),
        'Washington DC': (38.9072, -77.0369)
    }
    stadiums = {
        'Wrigley Field': (41.9484, -87.6550), 'Yankee Stadium': (40.8296, -73.9261),
        'Dodger Stadium': (34.0738, -118.2400), 'Nationals Park': (38.8972, -77.0211)
    }

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    for city, coord in locations.items():
        folium.Circle(location=coord, radius=60000, color='blue', fill=True, opacity=0.2, popup=city).add_to(m)
    
    for stad, coord in stadiums.items():
        folium.Marker(location=coord, popup=stad, icon=folium.Icon(color='red', icon='star')).add_to(m)

    st_folium(m, width=800, height=500)

    st.markdown("---")
    st.header("2. 의류 재구매 유도")
    
    # 재구매 차트
    re_data = pd.DataFrame({'주기': ['당일', '익일', '2~7일', '8일+'], '고객수': [739, 153, 20, 411]})
    fig_bar = px.bar(re_data, x='주기', y='고객수', title="재구매 소요기간", text='고객수')
    st.plotly_chart(fig_bar, use_container_width=True)
