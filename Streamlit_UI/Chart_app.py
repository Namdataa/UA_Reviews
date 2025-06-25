import streamlit as st
import pandas as pd
from Plot_chart.all_plots import (
    filter_data, calculate_primary_metrics, calculate_secondary_metrics_change, 
    plot_reviews_and_recommendation_trends, plot_avg_value_and_score_trends,
    plot_service_rating_distribution_with_avg,
    plot_aircraft_manufacturers_composition, plot_top_aircraft_models,
    get_service_averages,
    plot_top_origin_cities, plot_top_destination_cities, get_popular_routes_table,
    plot_top15_countries_by_review_count, plot_seat_type_bar_line, plot_traveller_type_bar_line,
    get_top_keywords, plot_experience_donut_chart, get_sample_reviews_by_experience
)
from Streamlit_UI.request_data import load_data
import base64
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def dashboard(): 
    # Tải dữ liệu từ S3
    df = load_data()

    # Header gradient với logo base64, màu nền mới, logo lớn hơn
    def get_base64_logo(path=r"Image/united_logo.png"):
        with open(path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        return b64

    logo_b64 = get_base64_logo(r"Image/united_logo.png")

    st.markdown("""
        <style>
        /* Nền tổng thể */
        body, .stApp {
            background: #f6f8fc !important;
        }
        /* Header gradient đổi sang xanh nhạt */
        .header-gradient {
            background: linear-gradient(90deg, #e3f0ff 0%, #f6eaff 100%);
            border-radius: 16px;
            padding: 24px 0 18px 0;
            margin-bottom: 18px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(44,62,80,0.08);
        }
        .header-title {
            color: #1a237e;
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.2em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
        }
        .header-desc {
            color: #374151;
            font-size: 1.25rem;
            font-weight: 400;
            margin-top: 0.2em;
        }
        .logo-img {
            height: 64px;
            width: 64px;
            object-fit: contain;
            margin-right: 8px;
            vertical-align: middle;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #fff !important;
            border-radius: 0 18px 18px 0;
            box-shadow: 2px 0 8px rgba(44,62,80,0.06);
        }
        /* Card, box, filter, metrics, plot, bảng... padding đều */
        .stMetric, .stDataFrame, .stPlotlyChart, .stSelectbox, .stRadio, .stButton, .stTextInput, .stNumberInput, .stSlider, .stMultiSelect {
            background: #fff !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
            padding: 18px 28px !important;
            margin-bottom: 18px !important;
        }
        /* Tag màu nhạt */
        span[style*="background"] {
            filter: brightness(1.15);
        }
        /* Chỉnh màu text các subheader, label */
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stSubheader, .stLabel {
            color: #1a237e !important;
        }
        /* Bảng, table */
        table {
            background: #fff !important;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
        }
        /* Giảm bóng cho toàn bộ */
        .stApp {
            box-shadow: none !important;
        }
        /* Giảm khoảng trống phía trên header */
        section.main, .block-container {
            padding-top: 45px !important;
            margin-top: 0 !important;
        }
        /* Loại bỏ nền trắng rộng hơn ở các container ngoài */
        .block-container {
            background: transparent !important;
            box-shadow: none !important;
        }
        /* Xóa nền không bo góc của các container ngoài */
        .block-container, .stContainer, .element-container {
            background: transparent !important;
            box-shadow: none !important;
        }
        /* Chỉ giữ nền trắng bo góc cho từng card/box nhỏ */
        .stMetric, .stDataFrame, .stPlotlyChart, .stTable {
            background: #fff !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(44,62,80,0.04);
            padding: 18px 28px !important;
            margin-bottom: 18px !important;
        }
        /* Chỉ sửa phần nền của biểu đồ Plotly */
        .stPlotlyChart {
            background: #fff !important;
            border-radius: 18px !important;
            box-shadow: 0 3px 8px rgba(44,62,80,0.12) !important;
            border: 1px solid #e1e5e9 !important;
            padding: 0px 0px !important;
            margin-bottom: 18px !important;
            overflow: visible !important;
            max-width: 100% !important;
        }
        /* Kiểm soát iframe bên trong chart */
        .stPlotlyChart iframe {
            border-radius: 18px !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        /* Chỉnh sửa sidebar filters */
        .stSelectbox, .stRadio {
            background: #ffffff !important;
            border: 1px solid #e1e5e9 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08) !important;
            padding: 12px 16px !important;
            margin-bottom: 8px !important;
        }
        /* Giảm khoảng cách giữa các filter */
        .stSelectbox + .stSelectbox, .stRadio + .stRadio {
            margin-top: 4px !important;
        }
        /* Chỉnh màu nền sidebar rõ ràng hơn */
        section[data-testid="stSidebar"] {
            background: #f8fafc !important;
            border-radius: 0 18px 18px 0;
            box-shadow: 2px 0 8px rgba(44,62,80,0.06);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="header-gradient">
            <div class="header-title">
                <img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="Logo" />
                United Airlines Flight Reviews Dashboard
            </div>
            <div class="header-desc">
                Interactive analysis of customer flight experiences
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(44,62,80,0.08); padding: 18px 28px 18px 28px; margin-bottom: 22px; max-width: 100%; margin-left: auto; margin-right: auto;'>
    <div style='font-size:18px; font-weight:700; color:#232526; text-align:left; margin-bottom:4px;'>Self-Sampling Bias</div>
    <div style='font-size:17px; color:#374151; line-height:1.6; text-align:left;'>
        Our analysis of United Airlines reviews is subject to self-selection sampling bias, as reviewers may have had extreme experiences or specific motivations for providing feedback. Rather than generalizing findings, we focus on identifying actionable areas for improvement based on the available reviews.
    </div>
    </div>
    """, unsafe_allow_html=True)

    # --- BỘ LỌC ---
    st.sidebar.header("General Filters")

    def get_unique_values(column_name):
        if column_name in df.columns:
            # Thay thế các giá trị rỗng (NaN) bằng 'Unknown' để người dùng có thể chọn
            unique_list = df[column_name].fillna('Unknown').unique().tolist()
            return sorted(unique_list)
        return []

    def get_numeric_values(column_name):
        """Hàm riêng để lấy danh sách giá trị số, loại bỏ các giá trị rỗng và sắp xếp"""
        if column_name in df.columns:
            # Chỉ lấy các giá trị số hợp lệ, loại bỏ NaN
            value_list = df[column_name].dropna().unique().tolist()
            return sorted(value_list)
        return []

    def get_transit_values():
        """Hàm riêng để lấy danh sách quá cảnh, chỉ lấy các giá trị thực sự có"""
        if 'transit' in df.columns:
            # Chỉ lấy các giá trị thực sự có, loại bỏ NaN (vì không có quá cảnh = direct flight)
            transit_list = df['transit'].dropna().unique().tolist()
            return sorted(transit_list)
        return []

    # Sắp xếp lại bộ lọc theo logic người dùng
    selected_years = st.sidebar.selectbox("Review Year", ["All"] + get_numeric_values('year_fly'), len(get_numeric_values('year_fly'))   )
    country_filter = st.sidebar.selectbox("Country", ["All"] + get_unique_values('country'))

    st.sidebar.markdown("---")
    origin_filter = st.sidebar.selectbox("Origin", ["All"] + get_unique_values('origin'))
    destination_filter = st.sidebar.selectbox("Destination", ["All"] + get_unique_values('destination'))
    transit_filter = st.sidebar.selectbox("Transit", ["All"] + get_transit_values())

    st.sidebar.markdown("---")
    month_fly_num_filter = st.sidebar.selectbox("Review Month", ["All"] + get_numeric_values('month_fly_num'))
    aircraft_filter = st.sidebar.selectbox("Aircraft", ["All"] + get_unique_values('aircraft_1'))
    seat_type_filter = st.sidebar.selectbox("Seat Type", ["All"] + get_unique_values('seat_type'))
    st.sidebar.markdown("---")
    verified_filter = st.sidebar.selectbox("Verified Status", ["All", "True", "False"])
    experience_filter = st.sidebar.selectbox("Experience", ["All"] + get_unique_values('experience'))

    # --- LỌC DỮ LIỆU ---
    filtered_df = filter_data(
        df, verified_filter, selected_years, month_fly_num_filter, seat_type_filter, 
        aircraft_filter, country_filter, experience_filter, 
        origin_filter, destination_filter, transit_filter
    )

    # --- HIỂN THỊ DASHBOARD ---

    # ==================== PHẦN 1: TỔNG QUAN ====================

    # --- Metrics Nhóm 1 (Không có Delta) ---
    total_reviews_p1, verified_percentage, unique_aircraft, total_countries = calculate_primary_metrics(filtered_df)

    # Card chỉ số tổng quan đẹp với emoji icon, màu xanh nhạt/trắng, số liệu động (English)
    st.markdown(f"""
    <div style='background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(44,62,80,0.08); padding: 18px 28px 18px 28px; margin-bottom: 22px;'>
    <div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Data Summary</div>
    <div style='display: flex; gap: 24px;'>
        <div style='flex:1; background:#e8f0fe; border-radius:14px; padding:18px 24px; display:flex; align-items:center; justify-content:space-between;'>
        <div>
            <div style='font-size:15px; color:#232526;'>Total Reviews</div>
            <div style='font-size:2rem; font-weight:700; color:#1976d2;'>{total_reviews_p1:,}</div>
        </div>
        <div style='background:#d2e3fc; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center;'>
            <span style='font-size:1.5rem; color:#1976d2;'>💬</span>
        </div>
        </div>
        <div style='flex:1; background:#f4faff; border-radius:14px; padding:18px 24px; display:flex; align-items:center; justify-content:space-between;'>
        <div>
            <div style='font-size:15px; color:#232526;'>Verified Percentage</div>
            <div style='font-size:2rem; font-weight:700; color:#00b96b;'>{verified_percentage:.1f}%</div>
        </div>
        <div style='background:#e0f2f1; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center;'>
            <span style='font-size:1.5rem; color:#00b96b;'>✅</span>
        </div>
        </div>
        <div style='flex:1; background:#f4faff; border-radius:14px; padding:18px 24px; display:flex; align-items:center; justify-content:space-between;'>
        <div>
            <div style='font-size:15px; color:#232526;'>Aircraft Types</div>
            <div style='font-size:2rem; font-weight:700; color:#1976d2;'>{unique_aircraft:,}</div>
        </div>
        <div style='background:#e0f2f1; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center;'>
            <span style='font-size:1.5rem; color:#1976d2;'>✈️</span>
        </div>
        </div>
        <div style='flex:1; background:#f4faff; border-radius:14px; padding:18px 24px; display:flex; align-items:center; justify-content:space-between;'>
        <div>
            <div style='font-size:15px; color:#232526;'>Countries</div>
            <div style='font-size:2rem; font-weight:700; color:#e6a100;'>{total_countries:,}</div>
        </div>
        <div style='background:#fffde7; border-radius:50%; width:40px; height:40px; display:flex; align-items:center; justify-content:center;'>
            <span style='font-size:1.5rem; color:#e6a100;'>🌐</span>
        </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Metrics Nhóm 2 (Có Delta) ---
    (current_reviews, current_rec_perc, current_money_value, current_score, 
    delta_reviews, delta_rec_perc, delta_money_value, delta_score) = calculate_secondary_metrics_change(
        df, selected_years, month_fly_num_filter, verified_filter, seat_type_filter, 
        aircraft_filter, country_filter, experience_filter, 
        origin_filter, destination_filter, transit_filter
    )

    # Tạo biến string cho delta và giá trị động
    if selected_years != 'All' and delta_reviews is not None:
        if delta_reviews >= 0:
            delta_reviews_str = f"<span style='color:green;'>▲ {delta_reviews:.1f}%</span>"
        else:
            delta_reviews_str = f"<span style='color:red;'>▼ {abs(delta_reviews):.1f}%</span>"
    else:
        delta_reviews_str = "None"

    if selected_years != 'All' and delta_rec_perc is not None:
        if delta_rec_perc >= 0:
            delta_rec_perc_str = f"<span style='color:green;'>▲ {delta_rec_perc:.1f}%</span>"
        else:
            delta_rec_perc_str = f"<span style='color:red;'>▼ {abs(delta_rec_perc):.1f}%</span>"
    else:
        delta_rec_perc_str = "None"

    if selected_years != 'All' and delta_money_value is not None:
        if delta_money_value >= 0:
            delta_money_value_str = f"<span style='color:green;'>▲ {delta_money_value:.1f}%</span>"
        else:
            delta_money_value_str = f"<span style='color:red;'>▼ {abs(delta_money_value):.1f}%</span>"
    else:
        delta_money_value_str = "None"

    if selected_years != 'All' and delta_score is not None:
        if delta_score >= 0:
            delta_score_str = f"<span style='color:green;'>▲ {delta_score:.1f}%</span>"
        else:
            delta_score_str = f"<span style='color:red;'>▼ {abs(delta_score):.1f}%</span>"
    else:
        delta_score_str = "None"

    current_money_value_str = f"{current_money_value:.2f}" if current_money_value is not None else "N/A"
    current_score_str = f"{current_score:.2f}" if current_score is not None else "N/A"

    st.markdown(f"""
    <div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Performance Metrics ({selected_years})</div>
    <div style='display: flex; gap: 24px;'>
    <div style='flex:1; background:#fff; border-radius:14px; padding:18px 24px; box-shadow:0 1px 4px rgba(44,62,80,0.04);'>
        <div style='font-size:15px; color:#232526; font-weight:600;'>Total Reviews</div>
        <div style='font-size:2rem; font-weight:700; color:#232526;'>{current_reviews:,}</div>
        <div style='font-size:1.1rem; color:#888; margin-top:2px;'>
        {delta_reviews_str}
        <span style='font-size:12px; color:#aaa; display:block;'>compared to previous year</span>
        </div>
        <div style='font-size:13px; color:#555; margin-top:8px;'>Total number of customer reviews submitted during this period.</div>
    </div>
    <div style='flex:1; background:#fff; border-radius:14px; padding:18px 24px; box-shadow:0 1px 4px rgba(44,62,80,0.04);'>
        <div style='font-size:15px; color:#232526; font-weight:600;'>Recommendation Percentage</div>
        <div style='font-size:2rem; font-weight:700; color:#232526;'>{current_rec_perc:.1f}%</div>
        <div style='font-size:1.1rem; color:#888; margin-top:2px;'>
        {delta_rec_perc_str}
        <span style='font-size:12px; color:#aaa; display:block;'>compared to previous year</span>
        </div>
        <div style='font-size:13px; color:#555; margin-top:8px;'>Percentage of customers who would recommend United Airlines to others.</div>
    </div>
    <div style='flex:1; background:#fff; border-radius:14px; padding:18px 24px; box-shadow:0 1px 4px rgba(44,62,80,0.04);'>
        <div style='font-size:15px; color:#232526; font-weight:600;'>Money Value Score</div>
        <div style='font-size:2rem; font-weight:700; color:#232526;'>{current_money_value_str}/5</div>
        <div style='font-size:1.1rem; color:#888; margin-top:2px;'>
        {delta_money_value_str}
        <span style='font-size:12px; color:#aaa; display:block;'>compared to previous year</span>
        </div>
        <div style='font-size:13px; color:#555; margin-top:8px;'>Value for money score based on customer feedback and ratings.</div>
    </div>
    <div style='flex:1; background:#fff; border-radius:14px; padding:18px 24px; box-shadow:0 1px 4px rgba(44,62,80,0.04);'>
        <div style='font-size:15px; color:#232526; font-weight:600;'>Overall Score</div>
        <div style='font-size:2rem; font-weight:700; color:#232526;'>{current_score_str}/5</div>
        <div style='font-size:1.1rem; color:#888; margin-top:2px;'>
        {delta_score_str}
        <span style='font-size:12px; color:#aaa; display:block;'>compared to previous year</span>
        </div>
        <div style='font-size:13px; color:#555; margin-top:8px;'>Overall customer satisfaction rating across all review categories.</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Biểu đồ Time-based Analysis mới
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Time-based Analysis</div>", unsafe_allow_html=True)
    time_granularity = st.radio(
        "Select time granularity",
        ('Yearly', 'Monthly'),
        horizontal=True
    )
    granularity_param = time_granularity

    # Biểu đồ 1: Xu hướng đánh giá và khuyến nghị
    fig_trends = plot_reviews_and_recommendation_trends(filtered_df, granularity_param)
    if fig_trends:
        with st.container():
            st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.warning("Not enough data to display review and recommendation trends chart.")

    # Biểu đồ 2: Xu hướng Money Value và Score
    fig_value_score = plot_avg_value_and_score_trends(filtered_df, granularity_param)
    if fig_value_score:
        with st.container():
            st.plotly_chart(fig_value_score, use_container_width=True)
    else:
        st.warning("Not enough data to display Money Value and Score trends chart.")

    st.markdown("---")

    # Biểu đồ 3: Phân phối điểm dịch vụ theo năm
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Detailed Service Analysis by Year</div>", unsafe_allow_html=True)

    # Hiển thị 5 card metrics điểm trung bình dịch vụ
    service_averages = get_service_averages(filtered_df)
    metric_cols = st.columns(5)
    for i, (service, avg) in enumerate(service_averages.items()):
        with metric_cols[i]:
            st.metric(label=service, value=f"{avg if avg is not None else 'N/A'}/5")

    service_options = {
        'Seat Comfort': 'seat_comfort',
        'Cabin Service': 'cabin_serv',
        'Food': 'food',
        'Ground Service': 'ground_service',
        'Wifi': 'wifi'
    }
    selected_service_name = st.selectbox("Select a service to analyze", list(service_options.keys()))
    selected_service_col = service_options[selected_service_name]

    fig_service_dist = plot_service_rating_distribution_with_avg(filtered_df, service_col=selected_service_col)
    if fig_service_dist:
        with st.container():
            st.plotly_chart(fig_service_dist, use_container_width=True)
    else:
        st.warning(f"Not enough data to display distribution chart for '{selected_service_name}'.")

    st.markdown("---")

    # ==================== PHẦN 4: PHÂN TÍCH MÁY BAY ====================
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'> Aircraft Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_manufacturer = plot_aircraft_manufacturers_composition(filtered_df)
        if fig_manufacturer:
            with st.container():
                st.plotly_chart(fig_manufacturer, use_container_width=True)
        else:
            st.warning("Not enough data to display aircraft manufacturer chart.")

    with col2:
        fig_models = plot_top_aircraft_models(filtered_df)
        if fig_models:
            with st.container():
                st.plotly_chart(fig_models, use_container_width=True)
        else:
            st.warning("Not enough data to display aircraft models chart.")

    st.markdown("---")
    # ==================== PHÂN TÍCH TUYẾN BAY ====================
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Route Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_origin = plot_top_origin_cities(filtered_df)
        if fig_origin:
            with st.container():
                st.plotly_chart(fig_origin, use_container_width=True)
        else:
            st.warning("Not enough data to display origin cities chart.")
    with col2:
        fig_dest = plot_top_destination_cities(filtered_df)
        if fig_dest:
            with st.container():
                st.plotly_chart(fig_dest, use_container_width=True)
        else:
            st.warning("Not enough data to display destination cities chart.")

    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Popular Routes</div>", unsafe_allow_html=True)
    popular_routes = get_popular_routes_table(filtered_df, top_n=10)
    if popular_routes is not None and not popular_routes.empty:
        def render_row(row):
            return f"<tr><td>{row['origin']}</td><td>{row['destination']}</td><td>{row['count']}</td><td style='font-size:18px;color:#FFD700'>{row['stars']}</td><td><b>{row['avg_rating']}</b></td></tr>"
        table_html = """
        <table style='width:100%;border-collapse:collapse;'>
            <thead>
                <tr style='background:#f7f7f7;'>
                    <th style='padding:8px 4px;text-align:left;'>ORIGIN</th>
                    <th style='padding:8px 4px;text-align:left;'>DESTINATION</th>
                    <th style='padding:8px 4px;text-align:center;'>COUNT</th>
                    <th style='padding:8px 4px;text-align:center;'> </th>
                    <th style='padding:8px 4px;text-align:center;'>AVG. RATING</th>
                </tr>
            </thead>
            <tbody>
        """
        for _, row in popular_routes.iterrows():
            table_html += render_row(row)
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Not enough data to display popular routes table.")

    st.markdown("---")


    # ==================== PHÂN TÍCH KHÁCH HÀNG ====================
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'> Customer Analysis</div>", unsafe_allow_html=True)

    # Biểu đồ 1: Top 15 quốc gia có nhiều khách hàng đánh giá nhất
    fig_cust_country = plot_top15_countries_by_review_count(filtered_df)
    if fig_cust_country:
        with st.container():
            st.plotly_chart(fig_cust_country, use_container_width=True)
    else:
        st.warning("Not enough data to display customer countries chart.")

    # Hai biểu đồ dưới: seat_type (trái), traveller type (phải)
    col1, col2 = st.columns(2)
    with col1:
        fig_seat = plot_seat_type_bar_line(filtered_df)
        if fig_seat:
            with st.container():
                st.plotly_chart(fig_seat, use_container_width=True)
        else:
            st.warning("Not enough data to display seat type chart.")
    with col2:
        fig_traveller = plot_traveller_type_bar_line(filtered_df)
        if fig_traveller:
            with st.container():
                st.plotly_chart(fig_traveller, use_container_width=True)
        else:
            st.warning("Not enough data to display traveller type chart.")

    st.markdown("---")



    # ==================== REVIEW TEXT ANALYSIS ====================
    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Review Text Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div style='font-weight:600; color:#374151; margin-bottom: 0.5em;'>Common Keywords</div>", unsafe_allow_html=True)
        keywords = get_top_keywords(filtered_df, 15)
        if keywords:
            color_map = ['#4fc3f7', '#81c784', '#ffd54f', '#ff8a65', '#ba68c8', '#e57373', '#64b5f6', '#aed581', '#ffb74d', '#9575cd', '#4db6ac', '#f06292', '#7986cb', '#dce775', '#ffb300']
            tag_html = ''
            for i, word in enumerate(keywords):
                tag_html += f"<span style='background:{color_map[i%len(color_map)]};color:#fff;padding:4px 12px;border-radius:8px;font-size:16px;font-weight:500;display:inline-block;margin-right:10px;margin-bottom:8px;'>{word}</span>"
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.write('No keyword data available.')

    with col2:
        fig_exp = plot_experience_donut_chart(filtered_df)
        if fig_exp:
            with st.container():
                st.plotly_chart(fig_exp, use_container_width=True, height=160)
        else:
            st.write("Not enough data to display sentiment chart.")

    st.markdown("<div style='font-size:22px; font-weight:700; color:#232526; margin-bottom:12px;'>Sample Reviews</div>", unsafe_allow_html=True)

    # Thêm lựa chọn cách lấy review mẫu
    type_review = st.selectbox('Sample review selection method', ['Latest', 'Oldest', 'Random'], key='sample_review_type')

    # Sample Reviews
    sample_reviews = get_sample_reviews_by_experience(filtered_df, type_review)
    exp_map = {'Good': ('#e8fbee', 'Good Review', '#1b8c3b'), 'Fair': ('#fffbe7', 'Medium Review', '#bfa100'), 'Poor': ('#fff2f2', 'Bad Review', '#d32f2f')}
    star_color = {'Good': '#ffc107', 'Fair': '#bfa100', 'Poor': '#d32f2f'}
    for exp in ['Good', 'Fair', 'Poor']:
        if exp in sample_reviews:
            info = sample_reviews[exp]
            bg, title, color = exp_map[exp]
            stars = int(round(float(info['score']))) if info['score'] else 0
            star_html = ''.join([f"<span style='color:{star_color[exp]};font-size:22px'>&#9733;</span>" for _ in range(stars)])
            date_html = ''
            if info.get('date_review'):
                try:
                    date_val = pd.to_datetime(info['date_review'])
                    date_html = f"<span style='font-size:14px;color:#888;float:right;'>{date_val.strftime('%d/%m/%Y')}</span>"
                except:
                    date_html = f"<span style='font-size:14px;color:#888;float:right;'>{info['date_review']}</span>"
            st.markdown(f"""
    <div style='background:{bg};border-radius:10px;padding:18px 24px 12px 24px;margin-bottom:18px;border-left:5px solid {color};position:relative;'>
    <div style='font-weight:700;font-size:18px;color:{color};margin-bottom:4px'>{title}{date_html}</div>
    <div style='font-size:16px;color:#374151;margin-bottom:10px'>{info['review']}</div>
    <div style='display:flex;align-items:center;justify-content:space-between;font-size:15px;color:#555;'>
        <div>
        {info['aircraft']} &bull; {info['seat_type']}
        </div>
        <div>{star_html}</div>
    </div>
    <div style='font-size:14px;color:#7b7b7b;margin-top:4px;text-align:right'>{info['route']}</div>
    </div>
    """, unsafe_allow_html=True)



 
