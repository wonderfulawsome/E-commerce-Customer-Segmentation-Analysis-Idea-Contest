import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 페이지 기본 설정
st.set_page_config(page_title="E-commerce Dashboard", page_icon="🛍️", layout="wide")

# 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    # 경로 설정 (app.py 위치 기준 Data 폴더)
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
        st.error(f"데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요: {e}")
        st.stop()

    # 월 매핑 (영문 월 -> 숫자)
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

    # 지불금액 계산 (파생변수)
    df['전체금액'] = df['수량'] * df['평균금액']
    
    def calculate_total(row):
        price = row['전체금액']
        gst = row['GST']
        discount_rate = row['할인율'] if row['쿠폰상태'] == 'Used' else 0
        subtotal = price * (1 - discount_rate/100)
        return subtotal + (subtotal * gst)

    df['지불금액'] = df.apply(calculate_total, axis=1)

    # 배송료 합산 처리
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
    
    # Monetary에 배송비 추가
    for customer_id, delivery_fee in customer_delivery_fee_sum.items():
        if customer_id in rfm_df.index:
            rfm_df.loc[customer_id, 'Monetary'] += delivery_fee
            
    rfm_df.reset_index(inplace=True)
    df = df.merge(rfm_df, on='고객ID')

    # Recency 가중치 적용 (카테고리별 재구매 주기 반영)
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
    
    # 고객 레벨 데이터프레임
    customer_df = df.groupby('고객ID')[['Recency', 'Frequency', 'Monetary']].first().reset_index()

    # 등급 부여 (R, F, M)
    customer_df['R'] = customer_df['Recency'].apply(lambda x: 5 if x<=50 else (4 if x<=100 else (3 if x<=150 else (2 if x<=200 else (1 if x<=300 else 0)))))
    customer_df['F'] = customer_df['Frequency'].apply(lambda x: 0 if x<=8 else (1 if x<=20 else (2 if x<=50 else (3 if x<=100 else (4 if x<=300 else 5)))))
    customer_df['M'] = customer_df['Monetary'].apply(lambda x: 0 if x<=1676 else (1 if x<=2500 else (2 if x<=4000 else (3 if x<=6000 else (4 if x<=10000 else 5)))))

    # 세그먼트 분류 (한글)
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
    
    # 그래프용 영문 세그먼트명 매핑 (그래프 깨짐 방지)
    seg_map = {
        'VIP고객': 'VIP', '충성고객': 'Loyal', '잠재충성고객': 'Potential Loyal',
        '놓치면안될고객': "Can't Lose", '최근신규방문고객': 'New Customers',
        '이탈우려고객': 'At Risk', '기타': 'Others'
    }
    customer_df['segment_en'] = customer_df['segment'].map(seg_map)

    # 최종 병합
    df_final = df.merge(customer_df[['고객ID', 'R', 'F', 'M', 'segment', 'segment_en']], on='고객ID')
    
    # 코호트 분석용 변수
    df_final['최초거래월'] = df_final.groupby('고객ID')['월'].transform('min')
    df_final['경과월'] = df_final['월'] - df_final['최초거래월']

    return df_final, customer_df, marketing

# 데이터 로딩 실행
with st.spinner('데이터를 불러오고 분석 로직을 수행 중입니다...'):
    df, customer_df, marketing = load_and_process_data()

# 사이드바 메뉴 설정
st.sidebar.title("Analyze Menu")
menu = st.sidebar.radio(
    "Go to",
    ["1. 대시보드 개요", "2. RFM 고객 세분화", "3. 리텐션 & 코호트", "4. 연관 분석", "5. 마케팅 전략"]
)
st.sidebar.markdown("---")

# 지역 필터링
st.sidebar.subheader("Region Filter")
all_regions = sorted(df['고객지역'].unique())
selected_regions = st.sidebar.multiselect("Select Region", all_regions, default=all_regions)

if selected_regions:
    df_filtered = df[df['고객지역'].isin(selected_regions)]
    target_ids = df_filtered['고객ID'].unique()
    customer_df_filtered = customer_df[customer_df['고객ID'].isin(target_ids)]
else:
    df_filtered = df
    customer_df_filtered = customer_df

st.sidebar.info(f"Selected Customers: {customer_df_filtered['고객ID'].nunique():,}명")

# -----------------------------------------------------------------------------
# 1. 대시보드 개요
# -----------------------------------------------------------------------------
if menu == "1. 대시보드 개요":
    st.title("📊 Dashboard Overview")
    
    # KPI 지표
    col1, col2, col3, col4 = st.columns(4)
    total_customers = customer_df_filtered['고객ID'].nunique()
    total_revenue = df_filtered['지불금액'].sum()
    avg_ticket = total_revenue / total_customers if total_customers > 0 else 0
    total_tx = df_filtered['거래ID'].nunique()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Avg Revenue per User", f"${avg_ticket:,.0f}")
    col4.metric("Total Transactions", f"{total_tx:,}")

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Monthly Revenue Trend")
        monthly = df_filtered.groupby('월')['지불금액'].sum().reset_index()
        fig1 = px.line(monthly, x='월', y='지불금액', markers=True, 
                       labels={'월': 'Month', '지불금액': 'Revenue'}, title="Monthly Revenue")
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Customer Distribution by Region")
        region_cnt = df_filtered.groupby('고객지역')['고객ID'].nunique().reset_index()
        fig2 = px.pie(region_cnt, values='고객ID', names='고객지역', hole=0.4, 
                      title="Customers by Region")
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. RFM 고객 세분화
# -----------------------------------------------------------------------------
elif menu == "2. RFM 고객 세분화":
    st.title("👥 RFM Segmentation")
    
    # 세그먼트 분포 (영문 라벨 사용)
    seg_counts = customer_df_filtered['segment_en'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Segment Ratio")
        fig_pie = px.pie(seg_counts, values='Count', names='Segment', 
                         color='Segment', title="Customer Segmentation Ratio")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("Customer Count by Segment")
        fig_bar = px.bar(seg_counts, x='Segment', y='Count', color='Segment', 
                         text='Count', title="Customer Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("3D RFM Analysis")
    fig_3d = px.scatter_3d(customer_df_filtered, x='Recency', y='Frequency', z='Monetary',
                           color='segment_en', opacity=0.7, size_max=10,
                           labels={'segment_en': 'Segment'})
    st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. 리텐션 & 코호트
# -----------------------------------------------------------------------------
elif menu == "3. 리텐션 & 코호트":
    st.title("🔄 Retention & Cohort Analysis")

    def get_retention_matrix(data):
        if data.empty: return None
        grouping = data.groupby(['최초거래월', '경과월'])
        cohort_data = grouping['고객ID'].apply(pd.Series.nunique).reset_index()
        cohort_counts = cohort_data.pivot(index='최초거래월', columns='경과월', values='고객ID')
        if cohort_counts.empty: return None
        retention = cohort_counts.divide(cohort_counts.iloc[:, 0], axis=0)
        return retention

    # 한글 세그먼트 리스트 (UI용)
    segments_list_kr = ["전체 고객"] + sorted(df_filtered['segment'].unique().tolist())
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_seg_kr = st.selectbox("Select Segment:", segments_list_kr)

    # 필터링
    if selected_seg_kr == "전체 고객":
        cohort_data = df_filtered
        st.info("Showing Retention for **All Customers**")
        plot_title = "All Customers Cohort"
    else:
        cohort_data = df_filtered[df_filtered['segment'] == selected_seg_kr]
        # UI에 영문 매핑된 이름도 보여주면 좋음
        seg_en_name = df[df['segment'] == selected_seg_kr]['segment_en'].iloc[0]
        st.info(f"Showing Retention for **{seg_en_name} ({selected_seg_kr})**")
        plot_title = f"{seg_en_name} Cohort"

    # 히트맵
    retention_matrix = get_retention_matrix(cohort_data)

    if retention_matrix is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=0.5, ax=ax)
        
        # 그래프 라벨 영어로 설정
        ax.set_title(plot_title, fontsize=15)
        ax.set_ylabel("First Transaction Month", fontsize=12)
        ax.set_xlabel("Month Passed", fontsize=12)
        
        st.pyplot(fig)
    else:
        st.warning("Not enough data to display cohort.")

# -----------------------------------------------------------------------------
# 4. 연관 분석
# -----------------------------------------------------------------------------
elif menu == "4. 연관 분석":
    st.title("🛒 Market Basket Analysis")

    target_seg = st.selectbox("Target Segment", ["All", "Potential Loyal", "At Risk"])
    min_sup = st.slider("Min Support", 0.005, 0.1, 0.01)

    # 선택에 따른 한글 세그먼트 매핑
    seg_map_reverse = {"All": "전체", "Potential Loyal": "잠재충성고객", "At Risk": "이탈우려고객"}
    target_seg_kr = seg_map_reverse[target_seg]

    if st.button("Run Analysis"):
        with st.spinner("Calculating..."):
            if target_seg == "All":
                data_sub = df_filtered[['고객ID', '제품카테고리']]
            else:
                data_sub = df_filtered[df_filtered['segment'] == target_seg_kr][['고객ID', '제품카테고리']]
            
            # 리스트 변환 및 중복 제거
            dataset = [list(set(x)) for x in data_sub.groupby('고객ID')['제품카테고리'].apply(list).values.tolist()]

            te = TransactionEncoder()
            te_ary = te.fit(dataset).transform(dataset)
            df_te = pd.DataFrame(te_ary, columns=te.columns_)

            frequent = apriori(df_te, min_support=min_sup, use_colnames=True)
            
            if frequent.empty:
                st.warning("No rules found. Try lowering Min Support.")
            else:
                rules = association_rules(frequent, metric="lift", min_threshold=1)
                rules = rules.sort_values(by='lift', ascending=False).head(15)
                
                # 가공
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.subheader(f"Top 15 Association Rules ({target_seg})")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                
                fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                                 title=f"Support vs Confidence ({target_seg})",
                                 labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 마케팅 전략
# -----------------------------------------------------------------------------
elif menu == "5. 마케팅 전략":
    st.title("💡 Marketing Strategy")

    st.header("1. MLB Location Marketing")
    st.markdown("Targeting MLB Stadium locations for **Headgear** sales.")
    
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
    st.header("2. Repurchase Strategy (Apparel)")
    st.info("26% of Apparel customers repurchase the **next day**.")
    
    # 영문 라벨 데이터
    re_data = pd.DataFrame({
        'Cycle': ['Same Day', 'Next Day (D+1)', '2~7 Days', '8+ Days'], 
        'Count': [739, 153, 20, 411]
    })
    
    fig_bar = px.bar(re_data, x='Cycle', y='Count', text='Count', 
                     title="Apparel Repurchase Cycle",
                     labels={'Cycle': 'Days', 'Count': 'Customer Count'})
    st.plotly_chart(fig_bar, use_container_width=True)
