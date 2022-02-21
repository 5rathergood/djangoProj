# yolov5 object detective 및 tracking 적용 영상 출력 로컬 웹서버

1. 레포지토리를 복사한다.
> ```git clone https://github.com/5rathergood/djangoProj.git```

2. 레포지토리 다운로드 후 <Anaconda prompt 혹은 pycharm terminal에서 다운받은 경로로 진입


3. djangoProj 폴더에서 다음 명령어 실행
> ``` pip install -r requirements.txt ```


3. djangoProj 폴더 내부에 있는 Pythonuser 폴더로 접근 후 다음 명령어 실행
> ``` python manage.py runserver ```


4. 정상적으로 실행에 성공하면, 자체 생성된 로컬 ip 주소가 출력됨.<br/>
주소를 웹에 입력하면 서버 접속 가능
> ``` http://127.0.0.1:8000/ ```


**★ 3번 항목 실행시 실행에 필요한 모듈들이 없을경우 설치가 필요하다고 뜨는데, <br/>
출력된 이름의 모듈들을 그대로 pip install ~을 통해 모두 설치하면 됩니다.<br/>**
> ``` pip install [해당 모듈] ```

