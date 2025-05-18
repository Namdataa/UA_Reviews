import pandas as pd
import numpy as np
import os
from typing import Dict, List
import logging
import re
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
import time

def clean_country(df: pd.DataFrame) -> pd.DataFrame:
    df['countries'] = df['countries'].str.replace(r'[()]', '', regex=True)
    return df

def clean_review(df: pd.DataFrame) -> pd.DataFrame:
    if 'review_bodies' not in df.columns:
        return df

    split_df = df['review_bodies'].str.split('|', expand=True)

    if len(split_df.columns) == 1:
        df['review'] = split_df[0]
        df['verified'] = pd.NA
    else:
        df['verified'], df['review'] = split_df[0], split_df[1]

    mask = df['review'].isnull() & df['verified'].notnull()
    df.loc[mask, ['review', 'verified']] = df.loc[mask, ['verified', 'review']].values

    df['verified'] = df['verified'].str.contains('Trip Verified', case=False, na=False)

    df.drop(columns=['review_bodies'], inplace=True)
    return df

def clean_date_review(df: pd.DataFrame) -> pd.DataFrame:
    df[['Day Review', 'Month Review', 'Year Review']] = df['dates'].str.split(expand=True)
    df['Day Review'] = df['Day Review'].str[:-2]

    df['Dates Review'] = pd.to_datetime(
        df['Day Review'] + ' ' + df['Month Review'] + ' ' + df['Year Review'],
        format='%d %B %Y'
    )

    return df

def clean_date_flown(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={'Date Flown': 'Month Flown'}, inplace=True)
    df[['Month Flown', 'Year Flown']] = df['Month Flown'].str.split(' ', expand=True)

    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    df['Month Flown Number'] = df['Month Flown'].map(month_mapping)
    df['Month Review Number'] = df['Month Review'].map(month_mapping)

    df['Month Flown Number'] = pd.to_numeric(df['Month Flown Number'], errors='coerce').astype('Int64')

    df['Month Year Flown'] = pd.to_datetime(
        df['Year Flown'].astype(str) + '-' + df['Month Flown Number'].astype(str).str.zfill(2) + '-01',
        format='%Y-%m-%d',
        errors='coerce'
    ).dt.strftime('%m-%Y')

    return df

def clean_space(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda x: x.strip() if isinstance(x, str) else x)

def create_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='Dates Review', ascending=False)
    df['id'] = range(len(df))
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_column_names: Dict[str, str] = {
        'Dates Review': 'date_review',
        'Day Review': 'day_review',
        'Month Review': 'month_review',
        'Month Review Number': 'month_review_num',
        'Year Review': 'year_review',
        'customer_names': 'name',
        'Month Flown': 'month_fly',
        'Month Flown Number': 'month_fly_num',
        'Year Flown': 'year_fly',
        'Month Year Flown': 'month_year_fly',
        'countries': 'country',
        'Aircraft': 'aircraft',
        'Type Of Traveller': 'type',
        'Seat Type': 'seat_type',
        'Route': 'route',
        'Seat Comfort': 'seat_comfort',
        'Cabin Staff Service': 'cabin_serv',
        'Food & Beverages': 'food',
        'Ground Service': 'ground_service',
        'Wifi & Connectivity': 'wifi',
        'Value For Money': 'money_value',
        'Recommended': 'recommended'
    }
    return df.rename(columns=new_column_names)

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_order = [
    'id', 'verified', 'date_review', 'day_review', 'month_review', 'month_review_num', 'year_review',
    'name', 'month_fly', 'month_fly_num', 'year_fly', 'month_year_fly', 'country', 'aircraft',
    'aircraft_combined', 'aircraft_1', 'aircraft_2', 'aircraft_3', 'type',
    'seat_type', 'route', 'is_return', 'mapped_route', 'origin', 'destination', 'transit', 'multi_leg',
    'seat_comfort', 'cabin_serv', 'food', 'ground_service', 'wifi', 'money_value',
    'recommended', 'review', 'score', 'experience'
]
    return df[column_order]

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = (df.pipe(clean_country)
            .pipe(clean_review)
            .pipe(clean_date_review)
            .pipe(clean_date_flown)
            .pipe(clean_space)
            .pipe(create_id)
            .pipe(rename_columns))
    return df

def standardize_aircraft_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # 1. Thêm khoảng trắng giữa chữ và số để xử lý chính xác hơn
    df[column_name] = df[column_name].astype(str).apply(lambda x: re.sub(r'([a-zA-Z])(\d)', r'\1 \2', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'(\d)([a-zA-Z])', r'\1 \2', x))

    # 2. Viết hoa chữ cái đầu mỗi từ
    df[column_name] = df[column_name].str.title()

    # 3. Thay "and", ",", "&" bằng "/"
    df[column_name] = df[column_name].str.replace(r'(?i)\band\b|&|,', '/', regex=True)

    # 4. Xóa khoảng trắng giữa chữ và số (nối lại)
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'([a-zA-Z])\s+(\d)', r'\1\2', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'(\d)\s+([a-zA-Z])', r'\1\2', x))

    # 5. Gán None nếu không chứa chữ số
    df[column_name] = df[column_name].apply(
        lambda x: x if pd.notna(x) and any(char.isdigit() for char in str(x)) else None
    )
        # 6. Chuẩn hóa các giá trị "trống" thành None
    df[column_name] = df[column_name].replace(
        to_replace=['', 'None', 'none', 'NaN', 'nan', 'null', 'NULL', 'N/A', 'n/a'],
        value=None
    )

    return df

def strip_whitespace_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    def process_value(x):
        if pd.isna(x) or not any(char.isdigit() for char in str(x)):
            return None

        x = str(x).strip().upper()
        x = re.sub(r'\s+', '', x)
        x = re.sub(r'[^A-Z0-9]', '', x)

        # Lấy phần chữ cái đầu tiên + tối đa 3 chữ số sau đó
        match = re.match(r'([A-Z]+)(\d{1,3})?', x)
        if match:
            letters = match.group(1)
            digits = match.group(2) if match.group(2) else ''
            result = letters + digits
        else:
            return None

        # Sau khi trích xuất, nếu giống mã chuyến bay hoặc chỉ có 1-2 số → loại
        if re.fullmatch(r'[A-Z]{2}\d{1,3}', result) or re.fullmatch(r'\d{1,2}', result):
            return None

        return result

    df[column_name] = df[column_name].apply(process_value)
    return df

def split_aircraft_column(df, column_name, prefix='aircraft'):
    """
    Tách một cột trong DataFrame thành nhiều cột dựa trên dấu '/' và nối vào DataFrame gốc.

    Parameters:
        df (pd.DataFrame): DataFrame đầu vào.
        column_name (str): Tên cột cần tách.
        prefix (str): Tiền tố cho các cột mới. Mặc định là 'aircraft'.

    Returns:
        pd.DataFrame: DataFrame với các cột mới được thêm vào.
    """
    split_cols = df[column_name].str.split('/', expand=True)
    split_cols.columns = [f'{prefix}_{i+1}' for i in range(split_cols.shape[1])]
    return pd.concat([df, split_cols], axis=1)


def normalize_aircraft_name(name: str) -> str:
    if not isinstance(name, str) or name.strip() == "":
        return None

    name = re.sub(r'[^a-zA-Z0-9]', '', name).title()  # Xóa dấu, viết hoa đầu từ
    digits = re.findall(r'\d{2,3}', name)
    if digits:
      last = digits[-1]
    # 8. Nếu bắt đầu bằng B → BA + số
    if (name.startswith('B') or name.startswith('Boe') or name.startswith('Bo')) and not last.endswith('00'):
        digits = re.findall(r'\d+', name)
        if digits:
            return f'B{digits[0]}'

    # 2. Nếu bắt đầu bằng E → E + số
    if name.startswith('E'):
        digits = re.findall(r'\d+', name)
        if digits:
            return f'E{digits[0]}'

    # 3. Nếu bắt đầu bằng C → C + số
    if name.startswith('C'):
        digits = re.findall(r'\d+', name)
        if digits:
            return f'C{digits[0]}'

    # 4. Nếu có "j" mà không bắt đầu bằng E hoặc C
    if 'j' in name.lower() and not name.startswith(('E', 'C')):
        digits = re.findall(r'\d{2,3}', name)
        if digits:
            last = digits[-1]
            if last.endswith('00'):
                return f'C{last}'
            else:
                return f'E{last}'

    # 5. Nếu chỉ có 2 chữ số → Atr + 2 số đó
    if re.fullmatch(r'\d{2}', name) and name.startswith('A'):
        return f'Atr{name}'

    # 6. Nếu bắt đầu bằng A → A + số
    if name.startswith('A'):
        digits = re.findall(r'\d+', name)
        if digits:
            return f'A{digits[0]}'

    # 7. Nếu có chữ q → Q + số sau Q
    if 'q' in name.lower():
        digits = re.findall(r'\d+', name)
        if digits:
            return f'Q{digits[0]}'

    return name  # Nếu không khớp điều kiện nào thì giữ nguyên

def recombine_aircraft_columns(df, cols, new_col='aircraft_combined'):
    df[new_col] = df[cols].apply(
        lambda row: '/'.join(
            [str(x) for x in row if pd.notna(x) and str(x).strip().lower() != 'none']
        ),
        axis=1
    )
    # Nếu chuỗi kết quả là rỗng thì thay bằng None (hoặc bạn có thể để là "")
    df[new_col] = df[new_col].replace('', None)
    return df

def apply_aircraft_standardization(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = df[column_name].astype(str).apply(lambda x: normalize_aircraft_name(x.strip()))
    return df

def clean_route_text(text):
    if not isinstance(text, str):
        return text

    # Xử lý các từ dính liền "via" như "UnitedviaLAX" → "United via LAX"
    text = re.sub(r'(?i)(\w)(via)(\w)', r'\1 \2 \3', text)

    # Thay "via" thành "/"
    text = re.sub(r'\s*via\s*', '-', text, flags=re.IGNORECASE)

    # Thay "to" thành "/" nếu có khoảng trắng quanh nó
    text = re.sub(r'\s+to\s+', '-', text, flags=re.IGNORECASE)

    return text.strip()
def standardize_route_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # 1. Thêm khoảng trắng giữa chữ và số để xử lý chính xác hơn
    df[column_name] = df[column_name].astype(str).apply(lambda x: re.sub(r'([a-zA-Z])(\d)', r'\1 \2', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'(\d)([a-zA-Z])', r'\1 \2', x))

    # 2. Viết hoa chữ cái đầu mỗi từ
    df[column_name] = df[column_name].str.lower()

    # 3. Thay "and", "&" bằng "/"
    df[column_name] = df[column_name].str.replace(r'(?i)\band\b|&', '/', regex=True)
    # 4. Thay thế via , to thành  dấu "-"
    df[column_name] = df[column_name].apply(clean_route_text)

    # Giả sử cột chứa thông tin chuyến bay là 'route'
    df['is_return'] = df[column_name].str.contains('return', case=False, na=False)


        # 6. Chuẩn hóa các giá trị "trống" thành None
    df[column_name] = df[column_name].replace(
        to_replace=['', 'None', 'none', 'NaN', 'nan', 'null', 'NULL', 'N/A', 'n/a'],
        value=None
    )

    return df

def process_routes(df, column):
    new_rows = []

    for _, row in df.iterrows():
        route_text = row[column]

        # Kiểm tra nếu route_text không có ý nghĩa gì
        if pd.isna(route_text) or not str(route_text).strip():
            new_rows.append({
                **row,
                column: None,
                'origin': None,
                'destination': None,
                'transit': None,
                'multi_leg': False
            })
            continue

        # Tách và chuẩn hóa từng phần, chuyển các phần trống thành None
        parts = [part.strip() if part.strip() else None for part in str(route_text).split('-')]

        # Nếu tất cả các phần đều không có giá trị (vd: "-", "--", "- -")
        if all(p is None for p in parts):
            new_rows.append({
                **row,
                column: None,
                'origin': None,
                'destination': None,
                'transit': None,
                'multi_leg': False
            })
            continue

        num_parts = len(parts)
        multi_leg = num_parts > 3

        origin = parts[0] if num_parts > 0 else None
        destination = parts[1] if num_parts > 1 else None
        transit = parts[2] if num_parts > 2 else None

        base_row = {
            **row,
            column: route_text.strip(),  # Có thể giữ nguyên hoặc làm sạch
            'origin': origin,
            'destination': destination,
            'transit': transit,
            'multi_leg': multi_leg
        }
        new_rows.append(base_row)

        # Nếu có tuyến bay phụ (đa chặng)
        if multi_leg and num_parts > 3:
            next_origin = destination
            next_destination = parts[3] if parts[3] else None
            next_transit = parts[4] if num_parts > 4 and parts[4] else None

            # Kiểm tra có chặng phụ có gì không
            if any([next_origin, next_destination, next_transit]):
                extra_row = {
                    **row,
                    column: route_text.strip(),
                    'origin': next_origin,
                    'destination': next_destination,
                    'transit': next_transit,
                    'multi_leg': False
                }
                new_rows.append(extra_row)

    return pd.DataFrame(new_rows)

def remove_duplicates(route_str):
    parts = [p.strip() for p in route_str.split(' - ')]
    unique_parts = list(dict.fromkeys(parts))
    return ' - '.join(unique_parts)

def scrape_iata_code(key, retries=3, threshold=90):
    url = f"https://www.iata.org/en/publications/directories/code-search/?airport.search={key}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    for attempt in range(retries):
      try:
          response = requests.get(url, headers=headers, timeout=15)
          soup = BeautifulSoup(response.content, "html.parser")
          table = soup.find("table", class_="datatable")
          if table:
              for row in table.find_all("tr"):
                  cells = row.find_all("td")
                  if cells:
                      city, airport, code = [c.text.strip() for c in cells]
                      if code.lower() == key.lower():
                          return code  # Trả về kết quả đầu tiên khớp
                      score = fuzz.partial_ratio(key.lower(), city.lower())
                      if key.lower() in airport.lower() and any(keyword in airport.lower() for keyword in ['international', 'intl', 'municipal']):
                          return code
                      if score >= threshold:
                          return code
      except Exception as e:
          print(f"Thử lần {attempt+1} bị lỗi với {key}: {e}")
          time.sleep(2)
    return ""

# Hàm ánh xạ với 3 cấp độ
def match_airport(location, df, threshold=70):
    original_location = location.strip()
    location = location.lower().strip()

    if len(original_location) == 3:
        # So khớp với Airport Code
        for _, row in df.iterrows():
            if location == row['Airport Code'].lower():
                return row['Airport Code']

        # Nếu không tìm thấy thì scrape
        return scrape_iata_code(original_location)

    else:
        # So khớp với Airport Name
        for _, row in df.iterrows():
            if location in row['Airport Name'].lower():
                return row['Airport Code']

        # Fuzzy matching với Airport Name
        best_match = None
        best_score = 0
        for _, row in df.iterrows():
            score = fuzz.partial_ratio(location, row['Airport Name'].lower())
            if score > best_score:
                best_score = score
                best_match = row['Airport Code']

        if best_score >= threshold:
            return best_match

        # Nếu không có gì hợp lý thì scrape
        return scrape_iata_code(original_location)

# Xử lý tuyến bay và trả ra tuple (location, code
def process_route_preserve_format(route_str, airport_df):
    if not isinstance(route_str, str) or not route_str.strip():
        return ""

    result = []

    # Bước 1: Tách theo dấu "-"
    parts = [part.strip() for part in route_str.split("-") if part.strip()]

    for part in parts:
        sub_parts = [p.strip() for p in part.split("/") if p.strip()]  # Tách theo dấu "/"
        sub_result = []

        for sub in sub_parts:
            cleaned = clean_location(sub)
            code = match_airport(cleaned, airport_df)
            sub_result.append(code)

        # Ghép lại các sub_parts bằng dấu " / " nếu có
        result.append(" / ".join(sub_result))

    # Ghép lại các parts bằng dấu " - "
    return " - ".join(result).upper()
def clean_location(text):
    if not isinstance(text, str) or text.strip() == "":
        return None
    return text.split('/')[0].strip().lower()

def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    df['score'] = df[['seat_comfort', 'cabin_serv', 'food', 'ground_service', 'wifi']].mean(axis=1)
    return df

def calculate_experience(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (df['money_value'] <= 2),
        (df['money_value'] == 3),
        (df['money_value'] >= 4)
    ]
    choices = ['Poor', 'Fair', 'Good']
    df['experience'] = np.select(conditions, choices, default='unknown')
    return df

def calculate_service_score(df: pd.DataFrame) -> pd.DataFrame:
    df['score'] = df[['seat_comfort', 'cabin_serv', 'food', 'ground_service', 'wifi']].mean(axis=1)
    return df

def replace_yes_no_with_bool(df: pd.DataFrame, column: str) -> pd.DataFrame:
    with pd.option_context('future.no_silent_downcasting', True):
        df[column] = df[column].replace({'yes': True, 'no': False})
    df[column] = df[column].astype('boolean')
    return df

def feature_engineer(df, df_airport):
    df = calculate_score(df)
    df = standardize_aircraft_column(df, "aircraft")
    df = split_aircraft_column(df,"aircraft", prefix='aircraft')
    aircraft_cols = sorted([col for col in df.columns if col.startswith('aircraft_')])
    for col in aircraft_cols:
        df = strip_whitespace_column(df, col)
        df = apply_aircraft_standardization(df, col)
    df = recombine_aircraft_columns(df, aircraft_cols)
    df.drop(columns=aircraft_cols, inplace=True)
    df = split_aircraft_column(df,'aircraft_combined', prefix='aircraft')
    df = standardize_route_column(df, 'route')
    df['mapped_route'] = df['route'].apply(lambda x: process_route_preserve_format(x, df_airport))
    df['mapped_route'] = df['mapped_route'].apply(remove_duplicates)
    df = process_routes(df, 'mapped_route')
    df = calculate_experience(df)
    df = calculate_service_score(df)
    df = replace_yes_no_with_bool(df, 'recommended')
    df=reorder_columns(df)
    return df