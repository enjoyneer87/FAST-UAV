import os
import sys
import streamlit.web.cli as stcli

# 정적 분석기를 속여서 PyInstaller가 대시보드에 필요한 모든 라이브러리를 포함하게 합니다.
if False:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import fastoad.api as oad
    from fastuav.models.supply_chain.model import run_supply_chain_scenario

if __name__ == '__main__':
    # PyInstaller 빌드 후 실행 시 생성되는 임시 폴더(MEIPASS) 경로를 찾습니다.
    if getattr(sys, 'frozen', False):
        app_dir = sys._MEIPASS
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
    # 앱으로 실행할 원본 Streamlit 스크립트 경로 지정
    script_path = os.path.join(app_dir, 'supply_chain_app.py')
    
    # argv를 덮어씌워 Streamlit 서버 실행 명령어를 대신 호출
    sys.argv = [
        "streamlit", 
        "run", 
        script_path, 
        "--global.developmentMode=false"
    ]
    
    sys.exit(stcli.main())
