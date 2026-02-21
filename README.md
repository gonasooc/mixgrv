# mixgrv 사용 가이드 (초기 세팅 → GUI 업로드/트레이닝 → 음악 생성)

`mixgrv`는 ACE-Step 1.5 기반으로 LoRA 트레이닝과 AI 음악 생성을 한 화면에서 진행할 수 있도록 구성된 프로젝트입니다.  
이 문서는 **처음 GitHub에서 저장소를 클론한 직후**부터, **현재 GUI(`index.html`)에서 파일 업로드/트레이닝 후 음악 생성**까지의 전체 흐름을 한국어로 정리합니다.

## 작업 구조 원칙

- 루트(`mixgrv`): 웹 클라이언트(`index.html`) 개발/수정/테스트/커밋
- `ACE-Step-1.5`: Python 백엔드 의존성 설치(`uv sync`), 서버 실행, 모델/학습 자산 관리
- 원칙: 백엔드 명령은 `ACE-Step-1.5` 기준으로 실행하고, 클라이언트 변경은 루트에서 진행

## 목차

- [작업 구조 원칙](#작업-구조-원칙)
- [1. 권장 환경 및 사전 요구사항](#1-권장-환경-및-사전-요구사항)
- [2. 최초 설치 (GitHub 클론 이후)](#2-최초-설치-github-클론-이후)
- [3. 서버 실행 (ACE-Step Gradio + API)](#3-서버-실행-ace-step-gradio--api)
- [4. GUI 실행 및 연결](#4-gui-실행-및-연결)
- [5. GUI 기준 전체 워크플로우](#5-gui-기준-전체-워크플로우)
- [6. 선택: Python/REST API로 실행](#6-선택-pythonrest-api로-실행)
- [7. 트러블슈팅](#7-트러블슈팅)
- [8. 변경 요약 (통합 문서 부록)](#8-변경-요약-통합-문서-부록)
- [9. 검증 체크리스트](#9-검증-체크리스트)

## 1. 권장 환경 및 사전 요구사항

- OS: macOS Apple Silicon (M1/M2/M3/M4)
- 필수 도구:
  - `uv` (Python/패키지 관리)
  - `ffmpeg` (전처리 필수)
- 권장: 메모리 여유 확보 (트레이닝 중 다른 무거운 앱 종료)

설치 예시:

```bash
# uv 설치 (둘 중 하나)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 또는
brew install uv

# FFmpeg 설치
brew install ffmpeg
```

## 2. 최초 설치 (GitHub 클론 이후)

### 2-1. 저장소 클론

```bash
git clone https://github.com/gonasooc/mixgrv.git
cd mixgrv
```

### 2-2. ACE-Step 의존성 설치

루트에서 작업 중이어도 백엔드 의존성 설치는 `ACE-Step-1.5`로 이동해서 실행합니다.

```bash
cd ACE-Step-1.5
uv sync
```

### 2-3. `.env` 생성 및 핵심 설정

```bash
cp .env.example .env
```

`.env`에 아래 항목을 반영하세요.

```env
# Apple Silicon 권장
ACESTEP_LM_BACKEND=mlx

# index.html에서 API 호출하려면 필수
ENABLE_API=--enable-api

# 선택
# PORT=7860
# SERVER_NAME=127.0.0.1
# ACESTEP_API_KEY=sk-your-secret-key
```

### 2-4. MPS NaN 패치 확인 (중요)

`ACE-Step-1.5/acestep/training/trainer.py`에서 MPS 설정이 아래와 같은지 확인하세요.

- `_select_compute_dtype("mps") -> torch.float32`
- `_select_fabric_precision("mps") -> "32-true"`

현재 저장소 기준으로는 해당 패치가 반영되어 있습니다.

## 3. 서버 실행 (ACE-Step Gradio + API)

백엔드 실행 작업은 `ACE-Step-1.5` 디렉토리에서만 진행합니다.

### 방법 A: macOS 런치 스크립트 (권장)

```bash
bash start_gradio_ui_macos.sh
```

### 방법 B: 직접 실행

```bash
uv run acestep --backend mlx --enable-api
```

기본 접속 주소:

- `http://127.0.0.1:7860`

참고:

- 첫 실행 시 모델 다운로드가 자동으로 진행되어 시간이 걸릴 수 있습니다.
- 종료는 서버 터미널에서 `Ctrl+C`.

## 4. GUI 실행 및 연결

프로젝트 루트(`mixgrv`)로 이동 후:

```bash
open index.html
```

화면 상단 서버 입력값은 기본적으로 `http://127.0.0.1:7860`이며, `연결` 버튼으로 상태를 확인합니다.

## 5. GUI 기준 전체 워크플로우

### 5-1. 트레이닝 탭 Step 1: 데이터셋 준비 (파일 업로드)

1. `데이터셋 이름` 입력
2. 필요 시 `커스텀 태그`, `인스트루멘탈 전용` 설정
3. 오디오 파일 드래그 앤 드롭 또는 `파일 선택`
4. `업로드 & 스캔` 클릭
   - 내부적으로 `POST /v1/dataset/upload` 후 `POST /v1/dataset/scan`이 자동 실행됩니다.
5. 스캔 결과 확인 후 `데이터셋 저장`

지원 확장자:

- `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.wma`, `.opus`

업로드 저장 위치:

- `ACE-Step-1.5/datasets/uploads/{dataset_name}/`

### 5-2. 트레이닝 탭 Step 2: 전처리

1. `출력 디렉토리` 확인 (기본: `./datasets/preprocessed_tensors`)
2. `전처리 시작` 클릭
3. 진행률/상태 완료까지 대기

### 5-3. 트레이닝 탭 Step 3: LoRA 트레이닝

1. `텐서 디렉토리` 입력 후 `불러오기`
2. 하이퍼파라미터 설정
   - 예: 에포크, LoRA Rank/Alpha, 학습률, 배치 크기, 체크포인트 간격
3. 출력 디렉토리 확인 (기본: `./lora_output`)
4. `트레이닝 시작`
5. 로그/진행률 확인, 필요 시 `트레이닝 중지`

학습 산출물 예시:

- `ACE-Step-1.5/lora_output/<실험명>/final/adapter/`

### 5-4. 음악 생성 탭: LoRA 적용 생성

1. (선택) `LoRA 경로`에 어댑터 경로 입력 후 `로드`
   - 예: `./lora_output/<실험명>/final/adapter`
2. `LoRA 활성화` 체크, `강도` 조정 (예: 0.8)
3. `캡션/설명`, `가사`, BPM/Key/박자/길이 등 입력
4. `음악 생성` 클릭
5. 결과 오디오 확인 및 다운로드

가사 힌트:

- `[verse]`, `[chorus]`, `[bridge]`, `[outro]` 태그 사용 가능
- 비워두면 인스트루멘탈 생성 가능

## 6. 선택: Python/REST API로 실행

### 6-1. Python `gradio_client` 음악 생성 예시

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
    "upbeat electronic pop, bright synth melody",
    "[verse]\nHello world\n[chorus]\nLa la la",
    120,
    "C major",
    "4/4",
    "en",
    8,
    7.0,
    True, -1,
    None,
    30,
    1,
    api_name="/generation_wrapper"
)
```

### 6-2. REST 업로드 예시

```bash
curl -X POST http://127.0.0.1:7860/v1/dataset/upload \
  -F "dataset_name=my_dataset" \
  -F "files=@track01.mp3" \
  -F "files=@track02.wav"
```

## 7. 트러블슈팅

### 7-1. `index.html` 연결 시 404

- 원인: `.env`에 `ENABLE_API=--enable-api` 미설정
- 해결: 값 추가 후 서버 재시작

### 7-2. 전처리 실패 (FFmpeg 미설치)

```bash
brew install ffmpeg
```

### 7-3. 포트 충돌 (`7860` 사용 중)

```bash
lsof -i :7860
kill <PID>
```

또는:

```bash
PORT=7861 bash start_gradio_ui_macos.sh
```

### 7-4. MPS 트레이닝 Loss가 NaN

- `trainer.py`의 MPS dtype/precision이 `float32` / `"32-true"`인지 재확인
- 전처리 텐서에 NaN이 섞인 경우 `torch.nan_to_num` 방식으로 치환

### 7-5. `Rejected unsafe directory path` 에러

- Gradio 보안 정책으로 프로젝트 외부 경로 접근이 차단될 수 있습니다.
- `index.html` 업로드 기능을 사용하면 파일이 `datasets/uploads/`로 들어가 이 문제를 피할 수 있습니다.

### 7-6. 무시 가능한 경고

- `Skipping import of cpp extensions due to incompatible torch version`
- `bitsandbytes not installed. Using standard AdamW.`
- `Standard MLX load failed ... retrying with 'model.' prefix remapping`

## 8. 변경 요약 (통합 문서 부록)

이 저장소의 현재 GUI/백엔드 기준 핵심 변경점:

1. `ACE-Step-1.5/acestep/ui/gradio/api/api_routes.py`
   - `POST /v1/dataset/upload` 엔드포인트 추가
   - 멀티파트 오디오 파일 업로드, 확장자 필터링, 업로드 디렉토리 저장
2. `index.html`
   - 전면 한국어화
   - Step 1을 경로 입력 방식에서 파일 업로드(드래그 앤 드롭) 방식으로 전환
   - `업로드 → 스캔` 자동 체인 및 업로드 진행 상태 표시

## 9. 검증 체크리스트

- [ ] 서버가 `http://127.0.0.1:7860`에서 정상 기동됨
- [ ] `index.html`에서 서버 연결 성공
- [ ] 오디오 파일 업로드 및 스캔 성공
- [ ] 데이터셋 저장 후 전처리 완료
- [ ] 텐서 로드 후 LoRA 트레이닝 시작/완료 확인
- [ ] LoRA 로드 후 음악 생성 및 결과 다운로드 확인
