import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from streamlit_folium import st_folium
import plotly.express as px
import platform
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 페이지 기본 설정
st.set_page_config(page_title="E-commerce 대시보드", page_icon="🛍️", layout="wide")

# 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else: # Linux (Streamlit Cloud)
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
        else:
            print("⚠️ 폰트가 없습니다. packages.txt를 확인하세요.")
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    # 경로 설정
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_dir, 'Data')

    try:
        # 파일 읽기
        cust = pd.read_csv(os.path.join(data_dir, "Customer_info.csv"))
        disc = pd.read_csv(os.path.join(data_dir, "Discount_info.csv"))
        mkt = pd.read_csv(os.path.join(data_dir, "Marketing_info.csv"))
        sale = pd.read_csv(os.path.join(data_dir, "Onlinesales_info.csv"))
        tax = pd.read_csv(os.path.join(data_dir, "Tax_info.csv"))
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

    # 전처리
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    if disc['월'].dtype == 'object':
        disc['월'] = disc['월'].map(month_map)

    sale['거래날짜'] = pd.to_datetime(sale['거래날짜'])
    sale['월'] = sale['거래날짜'].dt.month

    # 병합
    df = pd.merge(sale, cust, on='고객ID', how='left')
    df = pd.merge(df, disc, on=['월', '제품카테고리'], how='left')
    df = pd.merge(df, tax, on='제품카테고리', how='left')

    df['쿠폰코드'].fillna('unknown', inplace=True)
    df['할인율'].fillna(0, inplace=True)

    # 금액 계산
    df['전체금액'] = df['수량'] * df['평균금액']
    
    def calc_total(row):
        amt = row['전체금액'] * (1 - (row['할인율'] if row['쿠폰상태'] == 'Used' else 0)/100)
        return amt * (1 + row['GST'])

    df['지불금액'] = df.apply(calc_total, axis=1)

    # 배송비 처리
    del_fee = df.groupby(['고객ID', '거래ID'])['배송료'].first().groupby('고객ID').sum()

    # RFM 계산
    last_day = df['거래날짜'].max() + pd.DateOffset(days=1)
    rfm = df.groupby('고객ID').agg({
        '거래날짜': lambda x: (last_day - x.max()).days,
        '거래ID': lambda x: x.nunique(),
        '지불금액': 'sum'
    })
    rfm.rename(columns={'거래날짜':'Recency', '거래ID':'Frequency', '지불금액':'Monetary'}, inplace=True)
    
    # 배송비 합산
    for cid, fee in del_fee.items():
        if cid in rfm.index: rfm.loc[cid, 'Monetary'] += fee
    
    rfm = rfm.reset_index()
    df = df.merge(rfm, on='고객ID')

    # Recency 가중치
    cat_w = {'Office':9, 'Apparel':6, 'Nest-USA':5, 'Drinkware':13, 'Lifestyle':17,
             'Nest':4, 'Bags':18, 'Headgear':27, 'Notebooks & Journals':20, 'Waze':23}
    
    temp = df.copy()
    for c, v in cat_w.items():
        temp.loc[temp['제품카테고리']==c, '거래날짜'] += pd.Timedelta(days=v)
    
    temp['거래날짜'] = pd.to_datetime(temp['거래날짜'])
    w_last = temp['거래날짜'].max() + pd.DateOffset(days=27)
    w_r = temp.groupby('고객ID')['거래날짜'].apply(lambda x: (w_last - x.max()).days)
    df['Recency'] = df['고객ID'].map(w_r)

    # 고객 DF 생성
    cdf = df.groupby('고객ID')[['Recency', 'Frequency', 'Monetary']].first().reset_index()

    # 등급 부여
    cdf['R'] = cdf['Recency'].apply(lambda x: 5 if x<=50 else (4 if x<=100 else (3 if x<=150 else (2 if x<=200 else (1 if x<=300 else 0)))))
    cdf['F'] = cdf['Frequency'].apply(lambda x: 0 if x<=8 else (1 if x<=20 else (2 if x<=50 else (3 if x<=100 else (4 if x<=300 else 5)))))
    cdf['M'] = cdf['Monetary'].apply(lambda x: 0 if x<=1676 else (1 if x<=2500 else (2 if x<=4000 else (3 if x<=6000 else (4 if x<=10000 else 5)))))

    # 세그먼트
    def get_seg(r):
        R, F, M = r['R'], r['F'], r['M']
        if R==5 and F==5 and M==5: return 'VIP고객'
        elif R>=3 and F>=3 and M>=3: return '충성고객'
        elif R>=2 and F>=2 and M>=1: return '잠재충성고객'
        elif R>=0 and F>=2 and M>=2: return '놓치면안될고객'
        elif R>=3 and F>=0 and M>=0: return '최근신규방문고객'
        elif R>=0 and F>=1 and M>=1: return '이탈우려고객'
        else: return '기타'

    cdf['segment'] = cdf.apply(get_seg, axis=1)
    df = df.merge(cdf[['고객ID', 'segment']], on='고객ID')

    # 코호트 변수
    df['최초거래월'] = df.groupby('고객ID')['월'].transform('min')
    df['경과월'] = df['월'] - df['최초거래월']

    return df, cdf

with st.spinner('데이터 처리 중...'):
    df, cdf = load_data()

# 사이드바
st.sidebar.title("메뉴")
menu = st.sidebar.radio("이동", ["1. 개요", "2. RFM 세분화", "3. 리텐션 분석", "4. 연관 분석", "5. 마케팅 전략"])
st.sidebar.markdown("---")

# 지역 필터
regions = sorted(df['고객지역'].unique())
sel_regions = st.sidebar.multiselect("지역 필터", regions, default=regions)

if sel_regions:
    df_f = df[df['고객지역'].isin(sel_regions)]
    cdf_f = cdf[cdf['고객ID'].isin(df_f['고객ID'])]
else:
    df_f = df
    cdf_f = cdf

st.sidebar.info(f"선택 고객: {cdf_f['고객ID'].nunique():,}명")

# 1. 개요
if menu == "1. 개요":
    st.title("📊 대시보드 개요")
    
    c1, c2, c3, c4 = st.columns(4)
    cust_cnt = cdf_f['고객ID'].nunique()
    sales = df_f['지불금액'].sum()
    
    c1.metric("고객 수", f"{cust_cnt:,}")
    c2.metric("매출액", f"${sales:,.0f}")
    c3.metric("객단가", f"${sales/cust_cnt:,.0f}" if cust_cnt else 0)
    c4.metric("거래건수", f"{df_f['거래ID'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        mon_sales = df_f.groupby('월')['지불금액'].sum().reset_index()
        st.plotly_chart(px.line(mon_sales, x='월', y='지불금액', title="월별 매출"), use_container_width=True)
    with col2:
        reg_cnt = df_f.groupby('고객지역')['고객ID'].nunique().reset_index()
        st.plotly_chart(px.pie(reg_cnt, values='고객ID', names='고객지역', hole=0.4, title="지역별 분포"), use_container_width=True)

# 2. RFM
elif menu == "2. RFM 세분화":
    st.title("👥 RFM 세분화")
    
    seg_cnt = cdf_f['segment'].value_counts().reset_index()
    seg_cnt.columns = ['Seg', 'Cnt']
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(seg_cnt, values='Cnt', names='Seg', title="비율"), use_container_width=True)
    with c2: st.plotly_chart(px.bar(seg_cnt, x='Seg', y='Cnt', color='Seg', title="고객 수"), use_container_width=True)

    st.subheader("RFM 3D 분포")
    st.plotly_chart(px.scatter_3d(cdf_f, x='Recency', y='Frequency', z='Monetary', color='segment', opacity=0.7), use_container_width=True)

# 3. 리텐션
elif menu == "3. 리텐션 분석":
    st.title("🔄 세그먼트별 리텐션")

    def get_cohort(d):
        if d.empty: return None
        g = d.groupby(['최초거래월', '경과월'])['고객ID'].nunique().reset_index()
        p = g.pivot(index='최초거래월', columns='경과월', values='고객ID')
        if p.empty: return None
        return p.divide(p.iloc[:,0], axis=0)

    # 세그먼트 선택
    seg_list = ["전체"] + sorted(df_f['segment'].unique().tolist())
    sel_seg = st.selectbox("세그먼트 선택", seg_list)

    target = df_f if sel_seg == "전체" else df_f[df_f['segment'] == sel_seg]
    mat = get_cohort(target)

    if mat is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(mat, annot=True, fmt='.0%', cmap='Blues', vmin=0, vmax=0.5, ax=ax)
        ax.set_title(f"{sel_seg} 코호트")
        st.pyplot(fig)
    else:
        st.warning("데이터 부족")

# 4. 연관 분석
elif menu == "4. 연관 분석":
    st.title("🛒 연관 분석")
    
    tgt = st.selectbox("대상", ["전체", "잠재충성고객", "이탈우려고객"])
    sup = st.slider("최소 지지도", 0.005, 0.1, 0.01)

    if st.button("실행"):
        with st.spinner("계산 중..."):
            d = df_f if tgt=="전체" else df_f[df_f['segment']==tgt]
            ds = [list(set(x)) for x in d.groupby('고객ID')['제품카테고리'].apply(list)]
            
            te = TransactionEncoder()
            te_ary = te.fit(ds).transform(ds)
            res = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=sup, use_colnames=True)
            
            if not res.empty:
                rule = association_rules(res, metric="lift", min_threshold=1).sort_values('lift', ascending=False).head(15)
                rule['antecedents'] = rule['antecedents'].apply(lambda x: ', '.join(list(x)))
                rule['consequents'] = rule['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.dataframe(rule[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                st.plotly_chart(px.scatter(rule, x="support", y="confidence", size="lift", color="lift"), use_container_width=True)
            else:
                st.warning("결과 없음")

# 5. 마케팅
elif menu == "5. 마케팅 전략":
    st.title("💡 마케팅 전략")
    
    st.header("1. 지역 마케팅 (MLB)")
    locs = {'Chicago':(41.87,-87.62), 'California':(36.77,-119.41), 'New York':(40.71,-74.00), 'New Jersey':(40.05,-74.40), 'Washington DC':(38.90,-77.03)}
    stads = {'Wrigley':(41.94,-87.65), 'Yankee':(40.82,-73.92), 'Dodger':(34.07,-118.24), 'Nationals':(38.89,-77.02)}
    
    m = folium.Map([39.82, -98.57], zoom_start=4)
    for c, pos in locs.items(): folium.Circle(pos, radius=60000, color='blue', fill=True, opacity=0.2, popup=c).add_to(m)
    for s, pos in stads.items(): folium.Marker(pos, popup=s, icon=folium.Icon(color='red', icon='star')).add_to(m)
    st_folium(m, width=800, height=500)

    st.header("2. 재구매 유도")
    st.plotly_chart(px.bar(pd.DataFrame({'D':['당일','익일','2~7일','8일+'], 'Cnt':[739,153,20,411]}), x='D', y='Cnt', title="의류 재구매 주기"), use_container_width=True)
