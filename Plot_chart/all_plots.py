import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.corpus import stopwords
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import string

# ==================== FILTER FUNCTION ====================

def filter_data(df, verified_filter, selected_years, month_fly_num_filter, seat_type_filter, aircraft_filter, country_filter, experience_filter, origin_filter, destination_filter, transit_filter):
    """
    Hàm lọc dữ liệu theo tất cả các bộ lọc, bao gồm cả giá trị 'Unknown'.
    """
    df = df.copy()

    # Ép kiểu an toàn cho các cột số để đảm bảo so sánh chính xác
    df['year_fly'] = pd.to_numeric(df['year_fly'], errors='coerce')
    df['month_fly_num'] = pd.to_numeric(df['month_fly_num'], errors='coerce')

    # Hàm trợ giúp để xử lý các bộ lọc có thể có giá trị 'Unknown'
    def apply_categorical_filter(data, column, value):
        if value != "All":
            if value == "Unknown":
                data = data[data[column].isnull()]
            else:
                data = data[data[column] == value]
        return data

    # Bộ lọc Verified
    if verified_filter == "True":
        df = df[df['verified'] == True]
    elif verified_filter == "False":
        df = df[df['verified'] != True]

    # Bộ lọc Thời gian (so sánh kiểu số)
    if selected_years != "All":
        df = df[df['year_fly'] == int(selected_years)]
    if month_fly_num_filter != "All":
        df = df[df['month_fly_num'] == int(month_fly_num_filter)]

    # Áp dụng các bộ lọc phân loại khác
    df = apply_categorical_filter(df, 'seat_type', seat_type_filter)
    df = apply_categorical_filter(df, 'aircraft_1', aircraft_filter)
    df = apply_categorical_filter(df, 'country', country_filter)
    df = apply_categorical_filter(df, 'experience', experience_filter)
    df = apply_categorical_filter(df, 'origin', origin_filter)
    df = apply_categorical_filter(df, 'destination', destination_filter)
    
    # Bộ lọc Quá cảnh
    if transit_filter != "All":
        df = df[df['transit'] == transit_filter]

    return df

# ==================== OVERVIEW PAGE FUNCTIONS ====================

def calculate_primary_metrics(df):
    """
    Tính toán các metrics chính không cần so sánh delta.
    1. Tổng số reviews.
    2. Tỷ lệ verified reviews.
    3. Tổng số dòng máy bay duy nhất.
    4. Tổng số quốc gia.
    """
    if df.empty:
        return 0, 0, 0, 0

    total_reviews = len(df)
    
    # Tính tỷ lệ verified reviews
    verified_reviews = df['verified'].sum() # Giả sử True=1, False=0
    verified_percentage = (verified_reviews / total_reviews * 100) if total_reviews > 0 else 0
    
    # Đếm các dòng máy bay duy nhất từ 3 cột
    aircraft_cols = ['aircraft_1', 'aircraft_2', 'aircraft_3']
    existing_ac_cols = [col for col in aircraft_cols if col in df.columns]
    unique_aircraft = pd.concat([df[col] for col in existing_ac_cols]).dropna().nunique() if existing_ac_cols else 0
    
    # Đếm số quốc gia duy nhất
    total_countries = df['country'].nunique() if 'country' in df.columns else 0
    
    return total_reviews, verified_percentage, unique_aircraft, total_countries


def calculate_secondary_metrics_change(df, selected_years, month_fly_num_filter, verified_filter, seat_type_filter, aircraft_filter, country_filter, experience_filter, origin_filter, destination_filter, transit_filter):
    """
    Tính toán các metrics phụ và sự thay đổi so với năm trước.
    1. Recommendation Percentage
    2. Trung bình money_value
    3. Trung bình score
    4. Tổng số reviews
    """
    
    # Hàm helper để tính các metrics cho một dataframe cụ thể
    def _calculate_metrics(df_period):
        if df_period.empty:
            return 0, 0, 0, 0
        
        total_reviews = len(df_period)
        
        # Recommendation Percentage
        recommended_count = df_period['recommended'].sum() # Giả sử True=1
        rec_perc = (recommended_count / total_reviews * 100) if total_reviews > 0 else 0
        
        # Avg money_value và score
        avg_money_value = df_period['money_value'].mean()
        avg_score = df_period['score'].mean()
        
        return total_reviews, rec_perc, avg_money_value, avg_score

    # Lọc dữ liệu cho giai đoạn hiện tại
    df_current = filter_data(df, verified_filter, selected_years, month_fly_num_filter, seat_type_filter, aircraft_filter, country_filter, experience_filter, origin_filter, destination_filter, transit_filter)
    current_reviews, current_rec_perc, current_money_value, current_score = _calculate_metrics(df_current)

    # Nếu không chọn năm cụ thể, không tính delta
    if selected_years == "All":
        return current_reviews, current_rec_perc, current_money_value, current_score, None, None, None, None

    # Tính toán cho năm trước
    try:
        previous_year = int(selected_years) - 1
        df_previous = filter_data(df, verified_filter, str(previous_year), month_fly_num_filter, seat_type_filter, aircraft_filter, country_filter, experience_filter, origin_filter, destination_filter, transit_filter)
        prev_reviews, prev_rec_perc, prev_money_value, prev_score = _calculate_metrics(df_previous)
        
        if prev_reviews == 0: # Không có dữ liệu năm trước để so sánh
             return current_reviews, current_rec_perc, current_money_value, current_score, None, None, None, None

        # Tính toán delta
        delta_reviews = ((current_reviews - prev_reviews) / prev_reviews * 100)
        delta_rec_perc = ((current_rec_perc - prev_rec_perc) / prev_rec_perc * 100) if pd.notna(prev_rec_perc) and prev_rec_perc != 0 else None
        
        delta_money_value = ((current_money_value - prev_money_value) / prev_money_value * 100) if pd.notna(prev_money_value) and prev_money_value != 0 else None
        delta_score = ((current_score - prev_score) / prev_score * 100) if pd.notna(prev_score) and prev_score != 0 else None
        
        return current_reviews, current_rec_perc, current_money_value, current_score, delta_reviews, delta_rec_perc, delta_money_value, delta_score

    except (ValueError, TypeError):
        return current_reviews, current_rec_perc, current_money_value, current_score, None, None, None, None

# ==================== DETAILED ANALYSIS FUNCTIONS ====================

def plot_reviews_and_recommendation_trends(df, time_granularity='Yearly'):
    """
    Vẽ biểu đồ đường kép cho tổng số reviews và tỷ lệ khuyến nghị trung bình theo thời gian bằng Plotly.
    """
    if df.empty or 'year_fly' not in df.columns:
        return None
    
    df_copy = df.copy()
    
    if time_granularity == 'Monthly':
        if 'month_fly_num' not in df.columns:
            return None
        # FIX: Bỏ các dòng không có dữ liệu năm/tháng và chuyển sang kiểu int
        df_copy.dropna(subset=['year_fly', 'month_fly_num'], inplace=True)
        df_copy['time_axis'] = pd.to_datetime(
            df_copy['year_fly'].astype(int).astype(str) + '-' + df_copy['month_fly_num'].astype(int).astype(str)
        )
        time_col = 'time_axis'
    else:  # Yearly
        df_copy.dropna(subset=['year_fly'], inplace=True)
        time_col = 'year_fly'
        df_copy[time_col] = df_copy[time_col].astype(int)

    # Nhóm dữ liệu
    time_based_data = df_copy.groupby(time_col).agg(
        total_reviews=('year_fly', 'size'),
        avg_recommendation=('recommended', 'mean')
    ).reset_index()

    # FIX: Tạo một dải thời gian đầy đủ để đảm bảo không thiếu năm/tháng
    if not time_based_data.empty:
        if time_granularity == 'Monthly':
            full_range_df = pd.DataFrame({
                time_col: pd.date_range(start=time_based_data[time_col].min(), end=time_based_data[time_col].max(), freq='MS')
            })
        else:  # Yearly
            full_range_df = pd.DataFrame({
                time_col: range(time_based_data[time_col].min(), time_based_data[time_col].max() + 1)
            })
        
        # Hợp nhất để lấp đầy các khoảng trống
        time_based_data = pd.merge(full_range_df, time_based_data, on=time_col, how='left')
        time_based_data['total_reviews'] = time_based_data['total_reviews'].fillna(0)
    else:
        return None # Không có dữ liệu để vẽ

    if time_based_data.empty:
        return None

    time_based_data['avg_recommendation'] *= 100

    # Tạo biểu đồ với 2 trục y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Thêm đường Total Reviews
    fig.add_trace(
        go.Scatter(
            x=time_based_data[time_col],
            y=time_based_data['total_reviews'],
            name="Total Reviews",
            line=dict(color='royalblue'),
            mode='lines+markers',
            hovertemplate='Total Reviews: <b style="font-size:16px;">%{y:,.0f}</b><extra></extra>'
        ),
        secondary_y=False,
    )

    # Thêm đường Avg Recommendation
    fig.add_trace(
        go.Scatter(
            x=time_based_data[time_col],
            y=time_based_data['avg_recommendation'],
            name="Recommendation Percentage",
            line=dict(color='tomato'),
            mode='lines+markers',
            hovertemplate='Recommendation Percentage: <b style="font-size:16px;">%{y:.1f}%</b><extra></extra>'
        ),
        secondary_y=True,
    )

    # Cấu hình layout
    fig.update_layout(
        title_text="Trends in Reviews and Recommendation Percentage Over Time",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',  # Hiển thị tooltip cho cả 2 đường cùng lúc
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )

    # Cấu hình trục x
    fig.update_xaxes(title_text="Time")

    # Cấu hình trục y
    fig.update_yaxes(title_text="Count of Reviews", secondary_y=False, color='royalblue')
    fig.update_yaxes(title_text="Recommendation Percentage (%)", secondary_y=True, color='tomato', range=[0, 100])
        
    return fig


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def plot_avg_value_and_score_trends(df, time_granularity='Yearly'):
    """
    Vẽ biểu đồ đường cho trung bình money_value và score theo thời gian bằng Plotly.
    """
    if df.empty or 'year_fly' not in df.columns or 'money_value' not in df.columns or 'score' not in df.columns:
        return None

    df_copy = df.copy()

    if time_granularity == 'Monthly':
        if 'month_fly_num' not in df.columns:
            return None
        df_copy.dropna(subset=['year_fly', 'month_fly_num'], inplace=True)
        df_copy['time_axis'] = pd.to_datetime(
            df_copy['year_fly'].astype(int).astype(str) + '-' + df_copy['month_fly_num'].astype(int).astype(str)
        )
        time_col = 'time_axis'
    else:  # Yearly
        df_copy.dropna(subset=['year_fly'], inplace=True)
        time_col = 'year_fly'
        df_copy[time_col] = df_copy[time_col].astype(int)

    time_based_data = df_copy.groupby(time_col).agg(
        avg_money_value=('money_value', 'mean'),
        avg_score=('score', 'mean')
    ).reset_index()

    if not time_based_data.empty:
        if time_granularity == 'Monthly':
            full_range_df = pd.DataFrame({
                time_col: pd.date_range(start=time_based_data[time_col].min(), end=time_based_data[time_col].max(), freq='MS')
            })
        else:  # Yearly
            full_range_df = pd.DataFrame({
                time_col: range(time_based_data[time_col].min(), time_based_data[time_col].max() + 1)
            })
        time_based_data = pd.merge(full_range_df, time_based_data, on=time_col, how='left')
    else:
        return None

    if time_based_data.empty:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_based_data[time_col],
            y=time_based_data['avg_money_value'],
            name="Average Money Value",
            line=dict(color='mediumseagreen'),
            mode='lines+markers',
            hovertemplate='Average Money Value: <b style="font-size:16px;">%{y:.2f}</b><extra></extra>'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_based_data[time_col],
            y=time_based_data['avg_score'],
            name="Average Score",
            line=dict(color='darkorange'),
            mode='lines+markers',
            hovertemplate='Average Score: <b style="font-size:16px;">%{y:.2f}</b><extra></extra>'
        )
    )

    fig.update_layout(
        title_text="Trends in Average Money Value and Score Over Time",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        ),
        yaxis=dict(range=[0, 6])
    )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Average Score")

    return fig

def plot_service_rating_distribution_with_avg(df, service_col):
    if df.empty or service_col not in df.columns or 'year_fly' not in df.columns:
        return None

    # Calculate overall average for the selected service
    overall_avg_score = df[service_col].mean()

    # Group by year and count ratings
    yearly_data = df.groupby('year_fly')[service_col].value_counts().unstack(fill_value=0)
    yearly_avg_score = df.groupby('year_fly')[service_col].mean()

    # Ensure all score columns (1-5) exist
    for i in range(1, 6):
        if i not in yearly_data.columns:
            yearly_data[i] = 0
    yearly_data = yearly_data[[1, 2, 3, 4, 5]]

    if yearly_data.empty:
        return None

    service_name_map = {
        'seat_comfort': 'Seat Comfort',
        'cabin_serv': 'Cabin Service',
        'food': 'Food',
        'ground_service': 'Ground Service',
        'wifi': 'Wifi'
    }
    service_name = service_name_map.get(service_col, service_col)

    colors = {1: '#F08080', 2: '#FFA07A', 3: '#FFEE6B', 4: '#98FB98', 5: '#3CB371'}
    
    fig = go.Figure()

    # Add stacked bar chart for rating distribution
    for i in sorted(yearly_data.columns):
        fig.add_trace(go.Bar(
            x=yearly_data.index,
            y=yearly_data[i],
            name=f'Score {i}',
            marker_color=colors[i],
            hovertemplate='Score {}: <b>%{{y}}</b><extra></extra>'.format(i)
        ))

    # Add line chart for yearly average score
    fig.add_trace(go.Scatter(
        x=yearly_avg_score.index,
        y=yearly_avg_score,
        name='Yearly Average Score',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='blue'),
        hovertemplate='Yearly Average Score: <b>%{y:.2f}</b><extra></extra>'
    ))

    # Add dashed line for overall average score
    fig.add_trace(go.Scatter(
        x=yearly_avg_score.index,
        y=[overall_avg_score] * len(yearly_avg_score.index),
        name='Overall Average Score',
        mode='lines',
        yaxis='y2',
        line=dict(color='purple', dash='dash'),
        hovertemplate='Overall Average Score: <b>%{y:.2f}</b><extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f'Rating Distribution and Average Score of {service_name} Over Time',
        xaxis_title='Time',
        yaxis_title='Count of Reviews',
        yaxis2=dict(
            title='Average Score',
            overlaying='y',
            side='right',
            range=[0, 5.5]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.08,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )

    return fig

def plot_aircraft_manufacturers_composition(df):
    """
    Vẽ biểu đồ tròn thể hiện tỷ lệ các nhà sản xuất máy bay.
    """
    if df.empty:
        return None

    # Combine all aircraft columns into a single Series
    all_aircrafts = pd.concat([
        df['aircraft_1'].dropna(),
        df['aircraft_2'].dropna(),
        df['aircraft_3'].dropna()
    ])

    # Count 'Unknown' for aircraft_1 where it's null
    unknown_count = df['aircraft_1'].isnull().sum()

    # Function to map aircraft code to manufacturer
    def get_manufacturer(code):
        if not isinstance(code, str) or not code:
            return 'Unknown' # Treat non-string or empty as Unknown
        if code.startswith('B'): return 'Boeing'
        if code.startswith('A'): return 'Airbus'
        if code.startswith('E'): return 'Embraer'
        if code.startswith(('C', 'Q')): return 'Bombardier'
        return 'Unknown'

    manufacturer_counts = all_aircrafts.apply(get_manufacturer).value_counts()
    
    # Add the 'Unknown' count from nulls in aircraft_1
    if unknown_count > 0:
        manufacturer_counts['Unknown'] = manufacturer_counts.get('Unknown', 0) + unknown_count
    
    # Sort the values in descending order
    manufacturer_counts = manufacturer_counts.sort_values(ascending=False)

    if manufacturer_counts.empty:
        return None

    # Calculate the total for the center text
    total_reviews = manufacturer_counts.sum()

    fig = go.Figure(go.Pie(
        labels=manufacturer_counts.index,
        values=manufacturer_counts.values,
        hole=.4,
        textinfo='percent',
        hovertemplate='%{label}<br>Reviews: <b>%{value:,}</b><br>Percentage: <b>%{percent}</b><extra></extra>',
        sort=False # Use the sorting from the dataframe
    ))
    
    fig.update_layout(
        title_text='Top Aircraft Manufacturers Composition',
        annotations=[dict(text=f'{total_reviews:,.0f}', x=0.5, y=0.5, font_size=24, showarrow=False)],
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            align='left'
        ),
        legend=dict(
            font=dict(
                size=12
            )
        ),
        height=450
    )
    
    return fig

def plot_top_aircraft_models(df):
    if df.empty:
        return None

    # ... (rest of the function is the same, just removing the title from layout)
    all_aircrafts = []
    # Xử lý aircraft_1, các giá trị rỗng được tính là "Unknown"
    all_aircrafts.extend(df['aircraft_1'].fillna('Unknown'))
    # Xử lý aircraft_2 và aircraft_3, bỏ qua giá trị rỗng
    all_aircrafts.extend(df['aircraft_2'].dropna())
    all_aircrafts.extend(df['aircraft_3'].dropna())

    # Đếm số lần xuất hiện
    model_counts = pd.Series(all_aircrafts).value_counts().nlargest(10)

    def format_model_name(model):
        if model == 'Unknown':
            return 'Unknown'
        
        name_map = {
            'B737': 'Boeing 737', 'B777': 'Boeing 777', 'B787': 'Boeing 787',
            'B767': 'Boeing 767', 'B757': 'Boeing 757', 'B747': 'Boeing 747',
            'A320': 'Airbus 320', 'A319': 'Airbus 319', 'A330': 'Airbus 330',
            'A340': 'Airbus 340', 'A380': 'Airbus 380', 'A350': 'Airbus 350',
            'EMB 175': 'EMB 175', 'EMB 145': 'EMB 145', 'EMB 170': 'EMB 170',
            'EMB 190': 'EMB 190',
            'CRJ 200': 'CRJ 200', 'CRJ 700': 'CRJ 700', 'CRJ 900': 'CRJ 900',
            'Q400': 'Bombardier Q400'
        }
        return name_map.get(model, model)

    model_counts.index = model_counts.index.map(format_model_name)
    
    if model_counts.empty:
        return None

    fig = go.Figure(go.Bar(
        y=model_counts.index,
        x=model_counts.values,
        orientation='h',
        text=model_counts.values,
        textposition='auto',
        hovertemplate='%{y}<br>Number of Reviews: <b>%{x}</b><extra></extra>'
    ))

    fig.update_layout(
        title_text='Top 10 Aircraft Models',
        xaxis_title='Number of Reviews',
        yaxis_title='Aircraft Model',
        yaxis=dict(autorange="reversed"),
        height=450,
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    
    return fig

def get_service_averages(df):
    """
    Trả về điểm trung bình của 5 dịch vụ chính: seat_comfort, cabin_serv, food, ground_service, wifi
    """
    service_columns = {
        'seat_comfort': 'Seat Comfort',
        'cabin_serv': 'Cabin Service',
        'food': 'Food',
        'ground_service': 'Ground Service',
        'wifi': 'Wifi'
    }
    result = {}
    for col, name in service_columns.items():
        if col in df.columns:
            avg = df[col].mean()
            result[name] = round(avg, 2) if not pd.isnull(avg) else None
        else:
            result[name] = None
    return result

def plot_top_origin_cities(df):
    if df.empty or 'origin' not in df.columns:
        return None
    top_origins = df['origin'].value_counts().nlargest(5)
    fig = go.Figure(go.Bar(
        y=top_origins.index[::-1],
        x=top_origins.values[::-1],
        orientation='h',
        marker_color='mediumpurple',
        text=top_origins.values[::-1],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Number of Passengers: <b>%{x}</b><extra></extra>'
    ))
    fig.update_layout(
        title_text='Top Origin Cities',
        xaxis_title='Number of Reviews',
        yaxis_title='',
        height=350,
        margin=dict(l=80, r=20, t=50, b=40),
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    return fig

def plot_top_destination_cities(df):
    if df.empty or 'destination' not in df.columns:
        return None
    top_dest = df['destination'].value_counts().nlargest(5)
    fig = go.Figure(go.Bar(
        y=top_dest.index[::-1],
        x=top_dest.values[::-1],
        orientation='h',
        marker_color='gold',
        text=top_dest.values[::-1],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Number of Passengers: <b>%{x}</b><extra></extra>'
    ))
    fig.update_layout(
        title_text='Top Destination Cities',
        xaxis_title='Number of Reviews',
        yaxis_title='',
        height=350,
        margin=dict(l=80, r=20, t=50, b=40),
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    return fig

def get_popular_routes_table(df, top_n=10):
    if df.empty or 'origin' not in df.columns or 'destination' not in df.columns:
        return None
    route_counts = df.groupby(['origin', 'destination']).size().reset_index(name='count')
    avg_ratings = df.groupby(['origin', 'destination'])['score'].mean().reset_index(name='avg_rating')
    merged = route_counts.merge(avg_ratings, on=['origin', 'destination'])
    merged = merged.sort_values(by='count', ascending=False).head(top_n)
    # Format stars
    def render_stars(score):
        if pd.isnull(score):
            return ''
        full = int(score)
        half = 1 if score - full >= 0.5 else 0
        empty = 5 - full - half
        return '★' * full + '☆' * (5 - full)
    merged['stars'] = merged['avg_rating'].apply(render_stars)
    merged['avg_rating'] = merged['avg_rating'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
    return merged[['origin', 'destination', 'count', 'stars', 'avg_rating']]

def plot_top15_countries_by_review_count(df):
    if df.empty or 'country' not in df.columns:
        return None
    country_counts = df['country'].value_counts().nlargest(15)
    fig = go.Figure(go.Bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        marker_color='teal',
        hovertemplate='%{y}<br>Total Reviews: <b>%{x}</b><extra></extra>'
    ))
    fig.update_layout(
        title='Top 15 Countries with the Most Reviewed Passengers',
        xaxis_title='Total Reviews',
        yaxis_title='Country',
        yaxis=dict(autorange='reversed'),
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    return fig

def plot_seat_type_bar_line(df):
    if df.empty or 'seat_type' not in df.columns or 'score' not in df.columns:
        return None
    seat_stats = df.groupby('seat_type').agg(
        total_reviews=('seat_type', 'size'),
        avg_score=('score', 'mean')
    ).sort_values('total_reviews', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=seat_stats.index,
        y=seat_stats['total_reviews'],
        name='Total Reviews',
        marker_color='indianred',
        hovertemplate='Total Reviews: <b>%{y}</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=seat_stats.index,
        y=seat_stats['avg_score'],
        name='Average Score',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='royalblue', width=3),
        marker=dict(size=10),
        hovertemplate='Average Score: <b>%{y:.2f}</b><extra></extra>'
    ))
    fig.update_layout(
        title='Total Reviews and Average Score by Seat Type',
        xaxis_title='Seat Type',
        yaxis=dict(title='Total Reviews'),
        yaxis2=dict(title='Average Score', overlaying='y', side='right', range=[0,5]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    return fig

def plot_traveller_type_bar_line(df):
    if df.empty or 'type' not in df.columns or 'score' not in df.columns:
        return None
    type_stats = df.groupby('type').agg(
        total_reviews=('type', 'size'),
        avg_score=('score', 'mean')
    ).sort_values('total_reviews', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=type_stats.index,
        y=type_stats['total_reviews'],
        name='Total Reviews',
        marker_color='seagreen',
        hovertemplate='Total Reviews: <b>%{y}</b><extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=type_stats.index,
        y=type_stats['avg_score'],
        name='Average Score',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='darkorange', width=3),
        marker=dict(size=10),
        hovertemplate='Average Score: <b>%{y:.2f}</b><extra></extra>'
    ))
    fig.update_layout(
        title='Total Reviews and Average Score by Traveller Type',
        xaxis_title='Traveller Type',
        yaxis=dict(title='Total Reviews'),
        yaxis2=dict(title='Average Score', overlaying='y', side='right', range=[0,5]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='right'
        )
    )
    return fig 

def get_top_keywords(df, top_n=15):
    if df.empty or 'review' not in df.columns:
        return []
    text = ' '.join(df['review'].dropna().astype(str)).lower()
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stopwords = set(STOPWORDS)
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    freq = pd.Series(filtered).value_counts().head(top_n)
    return freq.index.tolist()

def plot_experience_donut_chart(df):
    if df.empty or 'experience' not in df.columns:
        return None
    exp_counts = df['experience'].value_counts().reindex(['Good', 'Fair', 'Poor'], fill_value=0)
    labels = exp_counts.index.tolist()
    values = exp_counts.values.tolist()
    colors = ['#6dd3ce', '#ffb86b', '#ff6b8a']
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # giống chart nhà sản xuất
        marker=dict(colors=colors),
        textinfo='percent',
        hoverinfo='label+percent+value',
        showlegend=True,
        hovertemplate='%{label}<br>Total Reviews: <b>%{value}</b><br>Percentage: <b>%{percent}</b><extra></extra>',
        sort=False
    ))
    fig.update_layout(
        title_text='Sentiment Analysis Composition',
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.1),
        margin=dict(l=40, r=40, t=40, b=40),  # tăng margin như chart nhà sản xuất
        height=300,
        width=400,
        hoverlabel=dict(
            font_size=14,
            font_family='Arial',
            align='left'
        )
    )
    return fig

def get_sample_reviews_by_experience(df, type_review='Latest'):
    # Lấy 1 review cho mỗi experience: Good, Fair, Poor, theo lựa chọn type_review
    if df.empty or 'experience' not in df.columns or 'review' not in df.columns:
        return {}
    result = {}
    for exp in ['Good', 'Fair', 'Poor']:
        sub = df[df['experience'] == exp]
        if not sub.empty:
            if 'date_review' in sub.columns:
                sub = sub.copy()
                sub['date_review'] = pd.to_datetime(sub['date_review'], errors='coerce')
                sub = sub.dropna(subset=['date_review'])
                if not sub.empty:
                    if type_review == 'Latest':
                        row = sub.sort_values('date_review', ascending=False).iloc[0]
                    elif type_review == 'Oldest':
                        row = sub.sort_values('date_review', ascending=True).iloc[0]
                    elif type_review == 'Random':
                        row = sub.sample(1).iloc[0]
                    else:
                        row = sub.iloc[0]
                else:
                    row = sub.iloc[0]
            else:
                row = sub.iloc[0]
            result[exp] = {
                'review': row['review'],
                'aircraft': row['aircraft'] if 'aircraft' in row else '',
                'score': row['score'] if 'score' in row else '',
                'seat_type': row['seat_type'] if 'seat_type' in row else '',
                'route': row['route'] if 'route' in row else '',
                'date_review': row['date_review'] if 'date_review' in row else ''
            }
    return result 