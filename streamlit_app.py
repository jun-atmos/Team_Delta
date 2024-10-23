import streamlit as st
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
from geopy.geocoders import Nominatim

import matplotlib.image as mpimg
import matplotlib.colorbar as colorbar
import matplotlib.colors as clr
import matplotlib.ticker as ticker

from datetime import datetime,timedelta
from pytz import timezone

tourist_spot = st.text_input("관광지 검색","우도")
lat = 33.46423019
lon = 126.935006
address = "address"

#########################################################################
########################### download_file ###############################
#########################################################################

def download_file(tm2,tm2_10):

    # URL 및 파일 경로 설정
    wea_url = f"https://apihub-pub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm2={tm2}&stn=0&disp=1&help=1&authKey=zbvBntvMSSK7wZ7bzDkiLg"
    cloud_url = f"https://apihub-pub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min_cloud?tm2={tm2}&stn=0&disp=1&help=1&authKey=zbvBntvMSSK7wZ7bzDkiLg"
    vi_url = f"https://apihub-pub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min_vis?tm2={tm2}&stn=0&disp=1&help=1&authKey=zbvBntvMSSK7wZ7bzDkiLg"
    url_rain = f"https://apihub.kma.go.kr/api/typ03/cgi/dfs/nph-qpf_ana_img?eva=1&tm={tm2_10}&qpf=B&ef=120&map=HB&grid=0.1&legend=1&size=6000&zoom_level=1000&zoom_x=3000&zoom_y=1000&x1=1000&y1=1000&authKey=829vQlOcRAuvb0JTnFQLrQ"

    wea_save_file_path = f'wea.csv'
    cloud_save_file_path = f'cloud.csv'
    vi_save_file_path = f'vi.csv'
    rain_save_file_path = 'output_file.jpg'

    # 저장 경로와 파일 URL을 리스트로 설정
    save_paths = [wea_save_file_path, cloud_save_file_path, vi_save_file_path, rain_save_file_path]
    file_urls = [wea_url, cloud_url, vi_url, url_rain]

    # 파일 다운로드 반복문
    for save_path, file_url in zip(save_paths, file_urls):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"File saved to {save_path}")
        else:
            print(f"Failed to download from {file_url}. Status code: {response.status_code}")

# 현재 시간에서 1분 전의 시간을 계산
tm2 = str(int(datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d%H%M')) - 1)

# 현재 시간에서 10분 전 시간 계산
ten_minute_ago = datetime.now(timezone('Asia/Seoul')) - timedelta(minutes=10)
tm2_10 = ten_minute_ago.strftime('%Y%m%d%H%M')
tm2_10 = str(int(round(float(tm2_10) * 0.1) * 10))
#download
download_file(tm2,tm2_10)

#########################################################################
############################### jeju_aws ################################
#########################################################################

def jeju_aws():
  stn_name = ['진달래밭', '한라산남벽', '영실', '사제비', '윗세오름', '삼각봉', '성판악', '어리목', '강정', '제주남원', '지귀도', '한남', '마라도', '가파도', '대정', '중문', '서귀포', '서호', '성산', '서광', '안덕화순', '제주가시리', '표선', '제주(공)', '제주김녕', '송당', '구좌', '산천단', '새별오름', '애월', '유수암', '오등', '외도', '제주', '우도', '대흘', '와산', '추자도', '고산', '낙천', '제주금악', '한림']
  stn_num = ['870', '965', '869', '868', '871', '867', '782', '753', '980', '780', '960', '885', '726', '855', '793', '328', '189', '884', '188', '752', '989', '890', '792', '184', '861', '862', '781', '329', '883', '893', '727', '865', '863', '182', '725', '330', '751', '724', '185', '990', '993', '779']

  # Create the DataFrame
  jeju_aws = pd.DataFrame({
      'STN_NAME': stn_name,
      'STN_ID': stn_num
  })

  jeju_aws['STN_ID'] = pd.to_numeric(jeju_aws['STN_ID'], errors='coerce').astype('Int64')
  return jeju_aws

jeju_aws = jeju_aws()

#########################################################################
############################# preprocessing #############################
#########################################################################

def preprocessing(n,tm2):
  filename = [f'wea.csv',f'cloud.csv',f'vi.csv']
  df_coloums=[
      ['TIME', 'STN_ID', 'WD1', 'WS1', 'WDS', 'WSS', 'WD10', 'WS10', 'TA','RE', 'RN-15m', 'RN-60m', 'RN-12H', 'RN-DAY', 'HM', 'PA', 'PS', 'TD'],
      ['TIME', 'STN_ID', 'LON', 'LAT', 'CH_LOW', 'CH_MID', 'CH_TOP', 'CA_TOP'],
      ['TIME', 'STN_ID', 'LON', 'LAT', 'S', 'VIS1', 'VIS10', 'WW1', 'WW15']
  ]
  df = pd.read_csv(f'{filename[n]}',encoding='cp949',skiprows=22,header=None)
  df = df.iloc[:-1, :-1]
  df.columns = df_coloums[n]
  df['STN_ID'] = df['STN_ID'].astype(int)
  df_jeju = df[df['STN_ID'].isin(jeju_aws['STN_ID'])]
  df_jeju = df_jeju.reset_index(drop=True)
  df_jeju = df_jeju.apply(pd.to_numeric, errors='coerce')
  df_jeju[df_jeju < -50] = np.nan
  df_jeju['TIME']= df_jeju['TIME'].astype('str')
  df_jeju['TIME']= pd.to_datetime(df_jeju['TIME'])
  return df_jeju

wea_api_jeju = preprocessing(0,tm2)
cloud_api_jeju = preprocessing(1,tm2)
vi_api_jeju = preprocessing(2,tm2)

#########################################################################
################################ map ####################################
#########################################################################

##### ------- find lat lon ------ #####

geolocator = Nominatim(user_agent="location_finder")

def get_location_info(place_name):
    try:
        # 장소 이름을 기반으로 위치 정보 검색
        location = geolocator.geocode(place_name)
        if location.address:
        # 주소와 위도/경도 출력
          st.success(f"주소: {location.address}, 위도: {location.latitude}, 경도: {location.longitude}")
          return location.latitude, location.longitude, location.address
        else:
          st.error(f"{place_name}에 대한 정보를 찾을 수 없습니다.")
          return None, None, None
    except Exception as e:
        st.error(f"오류 발생: {e}")
        return None, None, None

#########################################################################
############################### web code ################################
#########################################################################

# 관광지 검색

if tourist_spot is None:
  lat = 33.46423019
  lon = 126.935006
  address = "우도"
else:
    lat, lon, address = get_location_info(tourist_spot)

config = {
    "version": "v1",
    "config": {
        "mapState": {
            "bearing": 0,
            "latitude": lat,
            "longitude": lon,
            "pitch": 0,
            "zoom": 10,
        }
    },
}
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

if lat is None and lon is None:
  lat = 33.46423019
  lon = 126.935006
  address = "address"

left, right = st.columns(2)
left_r = left.button("정보 지도",type="primary", use_container_width=True)
left_r = True
right_r = right.button("초단기 강수 예측", type="primary", use_container_width=True)

if right_r:
  left_r = False
  st.subheader("초단기 강수 예측")
  image_path = 'output_file.jpg'
  img = mpimg.imread(image_path)

  cropped_img = img[3000:5000, 2300:4300, :]
  fig , ax = plt.subplots(figsize=(8, 8),dpi=1000)

  ax.imshow(cropped_img)

  ax.spines['top'].set_color('black')
  ax.spines['top'].set_linewidth(1)
  ax.spines['right'].set_color('black')
  ax.spines['right'].set_linewidth(1)
  ax.spines['bottom'].set_color('black')
  ax.spines['bottom'].set_linewidth(1)
  ax.spines['left'].set_color('black')
  ax.spines['left'].set_linewidth(1)

  ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

  colors = [
      (51/255, 51/255, 51/255),
      (0/255, 3/255, 144/255),
      (76/255, 78/255, 177/255),
      (179/255, 180/255, 222/255),
      (147/255, 0/255, 228/255),
      (179/255, 41/255, 255/255),
      (201/255, 105/255, 255/255),
      (244/255, 169/255, 255/255),
      (180/255, 0/255, 0/255),
      (210/255, 0/255, 0/255),
      (255/255, 50/255, 0/255),
      (255/255, 102/255, 0/255),
      (204/255, 170/255, 0/255),
      (224/255, 185/255, 0/255),
      (249/255, 205/255, 0/255),
      (255/255, 220/255, 31/255),
      (255/255, 225/255, 0/255),
      (0/255, 90/255, 0/255),
      (0/255, 140/255, 0/255),
      (0/255, 190/255, 0/255),
      (0/255, 255/255, 0/255),
      (0/255, 51/255, 245/255),
      (0/255, 155/255, 245/255),
      (0/255, 200/255, 255/255)
  ]

  num_colors = len(colors)
  colors.reverse()
  cmap = clr.ListedColormap(colors)

  cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
  cb = colorbar.ColorbarBase(cbar_ax, orientation='vertical', cmap=cmap, norm=plt.Normalize(-0.5, num_colors - 0.5),label='mm/hr')

  cb.set_ticks(range(num_colors))
  bounds = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 110]
  cb.ax.set_yticklabels(bounds)

  #######################################################
  # 검색 위치 별 표시
  x1, y1 = 1440, 840  # lat1 = 33.517, lon1 = 126.950
  x2, y2 = 1148, 797  # lat2 = 33.555, lon2 = 126.6445

  lat1, lon1 = 33.5181, 126.9490
  lat2, lon2 = 33.5553, 126.6446

  # 위경도 -> x, y 변환을 위한 선형 보간 함수
  def latlon_to_xy(lat, lon):
      x = x1 + (x2 - x1) * (lon - lon1) / (lon2 - lon1)
      y = y1 + (y2 - y1) * (lat - lat1) / (lat2 - lat1)
      return x, y

  x, y = latlon_to_xy(lat, lon)
  ##################################################

  ax.scatter(x,y,s=20,marker='*',color='darkred')

  st.pyplot(fig)

if left_r:
    map_1 = KeplerGl()
    map_1.config = config
    keplergl_static(map_1)
