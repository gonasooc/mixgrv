# mixgrv — 실행 가이드 & 작업 내역

## 목차

- [실행 방법](#실행-방법)
  - [1. 서버 실행 (ACE-Step Gradio + API)](#1-서버-실행-ace-step-gradio--api)
  - [2. 클라이언트 실행 (index.html)](#2-클라이언트-실행-indexhtml)
  - [3. Python 클라이언트 실행](#3-python-클라이언트-실행)
- [알려진 이슈](#알려진-이슈)
- [작업 내역: 한국어화 + 파일 업로드 UI](#작업-내역-한국어화--파일-업로드-ui)

---

## 실행 방법

### 사전 요구사항

| 항목 | 필수 여부 | 설치 방법 |
|------|-----------|-----------|
| macOS Apple Silicon (M1/M2/M3/M4) | 필수 | - |
| Python 3.12 | 자동 (uv가 관리) | - |
| uv (패키지 매니저) | 필수 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` 또는 `brew install uv` |
| FFmpeg | 필수 (전처리) | `brew install ffmpeg` |

### 1. 서버 실행 (ACE-Step Gradio + API)

> **중요**: `index.html`의 트레이닝/음악 생성 기능을 사용하려면 `.env`에 `ENABLE_API=--enable-api`가 설정되어 있어야 합니다.

#### 방법 A: macOS 런치 스크립트 (권장)

```bash
cd ACE-Step-1.5
bash start_gradio_ui_macos.sh
```

- 자동으로 MLX 백엔드 설정, 의존성 설치, 모델 다운로드 처리
- 기본 주소: `http://127.0.0.1:7860`
- 첫 실행 시 HuggingFace에서 모델 자동 다운로드 (~2분 40초)

#### 방법 B: 직접 실행

```bash
cd ACE-Step-1.5
export ACESTEP_LM_BACKEND=mlx
uv run acestep --backend mlx --enable-api
```

#### 서버 설정 변경

`.env` 파일에서 설정 가능 (런치 스크립트가 자동으로 읽음):

```env
# 모델 설정
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_LM_BACKEND=mlx

# 서버 설정
PORT=7860
SERVER_NAME=127.0.0.1

# API 활성화 (mixgrv index.html 사용 시 필수)
ENABLE_API=--enable-api

# API 키 (선택)
# ACESTEP_API_KEY=sk-your-secret-key
```

#### 서버 종료

```bash
# 터미널에서
Ctrl+C

# 또는 포트로 프로세스 찾아서 종료
lsof -i :7860
kill <PID>
```

### 2. 클라이언트 실행 (index.html)

서버가 실행된 상태에서 `index.html`을 브라우저에서 열면 됩니다.

```bash
# 프로젝트 루트에서
open index.html
```

또는 브라우저에서 직접 파일 열기:
```
file:///Users/joel/Desktop/gonasoo.dev/repositories/mixgrv/index.html
```

> **참고**: `index.html`은 정적 파일이므로 별도 웹 서버가 필요 없습니다. `file://` 프로토콜로 직접 열 수 있습니다. CORS 설정이 `null` origin을 허용하도록 되어 있습니다.

#### 사용 흐름

```
1. 서버 연결: 서버 주소 입력 → [연결] 클릭 (기본값: http://127.0.0.1:7860)

2. 트레이닝 탭:
   Step 1: 오디오 파일 드래그 앤 드롭 → [업로드 & 스캔] → [데이터셋 저장]
   Step 2: [전처리 시작] → 텐서 변환 완료 대기
   Step 3: 텐서 디렉토리 로드 → 하이퍼파라미터 설정 → [트레이닝 시작]

3. 음악 생성 탭:
   LoRA 로드 (선택) → 캡션/가사/설정 입력 → [음악 생성]
```

### 3. Python 클라이언트 실행

`gradio_client`를 사용한 프로그래밍 방식의 접근:

#### 음악 생성

```python
from gradio_client import Client

client = Client('http://127.0.0.1:7860')

# 음악 생성
result = client.predict(
    'upbeat electronic pop, bright synth melody',  # Caption
    '[verse]\nHello world\n[chorus]\nLa la la',     # Lyrics
    120,        # BPM
    'C major',  # Key
    '4/4',      # Time Signature
    'en',       # Language
    8,          # DiT Steps
    7.0,        # Guidance Scale
    True, -1,   # Random Seed
    None,       # Reference Audio
    30,         # Duration (seconds)
    1,          # Batch Size
    api_name='/generation_wrapper'
)
```

#### LoRA 트레이닝 (Python)

```python
from gradio_client import Client
client = Client('http://127.0.0.1:7860')

# 1. 데이터셋 로드
client.predict(
    tensor_dir='./datasets/preprocessed_tensors',
    api_name='/load_training_dataset'
)

# 2. 트레이닝 시작
client.predict(
    tensor_dir='./datasets/preprocessed_tensors',
    r=64, a=128, d=0.1, lr=0.0003,
    ep=100, bs=1, ga=1, se=50, sh=3.0, sd=42,
    od='./lora_output/my_model',
    rc=None,
    api_name='/training_wrapper'
)
```

#### LoRA 로드 및 적용

```python
# LoRA 로드 & 활성화
client.predict(lora_path='./lora_output/my_model/final/adapter', api_name='/load_lora')
client.predict(True, api_name='/set_use_lora')    # 활성화
client.predict(0.8, api_name='/set_lora_scale')   # Scale 0.8

# LoRA 해제
client.predict(api_name='/unload_lora')
```

#### REST API 직접 호출 (curl)

```bash
# 서버 상태 확인
curl http://127.0.0.1:7860/health

# 음악 생성
curl -X POST http://127.0.0.1:7860/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "dreamy lo-fi beat with soft piano",
    "lyrics": "",
    "bpm": 80,
    "audio_duration": 30,
    "batch_size": 1
  }'

# 파일 업로드 (multipart)
curl -X POST http://127.0.0.1:7860/v1/dataset/upload \
  -F "dataset_name=my_dataset" \
  -F "files=@track01.mp3" \
  -F "files=@track02.wav"
```

---

## 알려진 이슈

### MPS 트레이닝 NaN 그래디언트 (Apple Silicon 필수 패치)

**증상**: 트레이닝 시작 후 loss가 `NaN`으로 발산

**원인 1**: `trainer.py`의 MPS float16 mixed precision

**해결**: `ACE-Step-1.5/acestep/training/trainer.py`에서 MPS 관련 함수 수정:

```python
# 변경 전 (float16 → NaN 발생)
def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type == "mps":
        return torch.float16

def _select_fabric_precision(device_type: str) -> str:
    if device_type == "mps":
        return "16-mixed"

# 변경 후 (float32 → NaN 해결)
def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type == "mps":
        return torch.float32

def _select_fabric_precision(device_type: str) -> str:
    if device_type == "mps":
        return "32-true"
```

**원인 2**: 전처리 텐서의 `encoder_hidden_states`에 float16 오버플로우로 NaN 포함

**해결**: 전처리 후 NaN 값 치환 스크립트 실행:

```python
import torch, glob

for f in glob.glob('./datasets/preprocessed_tensors/*.pt'):
    pt = torch.load(f, map_location='cpu', weights_only=True)
    for k, v in pt.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            pt[k] = torch.nan_to_num(v, nan=0.0)
    torch.save(pt, f)
```

### Gradio 보안 경로 제한

**증상**: Gradio UI에서 프로젝트 외부 경로 접근 시 `Rejected unsafe directory path` 에러

**해결**: 데이터셋 파일을 반드시 `ACE-Step-1.5/datasets/` 내부로 복사. `index.html`의 파일 업로드 기능은 이 문제를 자동으로 해결 — 업로드 파일이 `datasets/uploads/` 디렉토리에 저장됨.

### 포트 충돌 (7860 이미 사용 중)

```bash
lsof -i :7860          # 사용 중인 프로세스 확인
kill <PID>             # 해당 PID 종료

# 또는 다른 포트로 실행
PORT=7861 bash start_gradio_ui_macos.sh
```

### FFmpeg 미설치 시 전처리 실패

**증상**: 전처리(Step 2)가 오디오 파일을 읽지 못함

**해결**:
```bash
brew install ffmpeg
```

### 무시해도 되는 경고

| 경고 메시지 | 설명 |
|-------------|------|
| `Skipping import of cpp extensions due to incompatible torch version` | torchao 호환성 경고, 무시 가능 |
| `bitsandbytes not installed. Using standard AdamW.` | 표준 옵티마이저 사용, 정상 동작 |
| `Standard MLX load failed ... retrying with 'model.' prefix remapping` | 자동 복구됨, 무시 가능 |

### `ENABLE_API` 미설정 시 index.html 연결 실패

**증상**: `index.html`에서 서버 연결 시 404 에러

**원인**: `.env`에 `ENABLE_API=--enable-api` 설정이 빠져 있으면 REST API 엔드포인트가 등록되지 않음

**해결**: `.env` 파일에 추가:
```env
ENABLE_API=--enable-api
```

---

## 작업 내역: 한국어화 + 파일 업로드 UI

### 작업 일시

2026-02-20

### 수정 파일 목록

| 파일 | 변경 유형 |
|------|-----------|
| `ACE-Step-1.5/acestep/ui/gradio/api/api_routes.py` | import 추가 + 엔드포인트 추가 |
| `index.html` | 전면 재작성 (한국어화 + UI 변경) |

---

### 1. `api_routes.py` — 파일 업로드 엔드포인트 추가

#### 변경 사항

##### import 추가

```python
# 기존
from fastapi import APIRouter, HTTPException, Request, Depends, Header

# 변경 후
from fastapi import APIRouter, HTTPException, Request, Depends, Header, UploadFile, File, Form
```

##### `POST /v1/dataset/upload` 엔드포인트 추가

`_register_extended_endpoints()` 함수 내에 LoRA 엔드포인트 앞에 삽입.

```python
@app.post("/v1/dataset/upload")
async def upload_dataset_files(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(default="uploaded_dataset"),
    _: None = Depends(verify_api_key),
):
```

- **수신 방식**: `multipart/form-data`
- **저장 경로**: `{프로젝트 루트}/datasets/uploads/{dataset_name}/`
- **허용 확장자**: `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.m4a`, `.wma`, `.opus`
- **응답 형식**:
  ```json
  {
    "data": {
      "upload_dir": "/absolute/path/to/datasets/uploads/my_dataset",
      "saved_files": ["track01.mp3", "track02.wav"],
      "skipped_files": ["readme.txt"],
      "count": 2
    },
    "code": 200,
    "error": null,
    "timestamp": 1740000000000,
    "extra": null
  }
  ```

---

### 2. `index.html` — 전면 한국어화 + 파일 업로드 UI

#### 2-1. 한국어화 범위

##### 페이지 헤더

| 항목 | 기존 (영문) | 변경 (한국어) |
|------|-------------|---------------|
| 타이틀 | `mixgrv - LoRA Training & Music Generation` | `mixgrv - LoRA 트레이닝 & AI 음악 생성` |
| 서브타이틀 | (없음) | `ACE-Step 1.5 기반의 LoRA 학습과 AI 음악 생성을 위한 웹 인터페이스입니다.` |
| 서버 라벨 | `Server:` | `서버:` |
| 연결 버튼 | `Connect` | `연결` |

##### 탭

| 기존 | 변경 |
|------|------|
| `Training` | `트레이닝` |
| `Music Generation` | `음악 생성` |

##### 카드 제목 + 설명 추가

| Step | 기존 제목 | 변경 제목 | 추가 설명 |
|------|-----------|-----------|-----------|
| Step 1 | Dataset Scan | 데이터셋 준비 | 학습에 사용할 오디오 파일을 업로드하고 데이터셋을 구성합니다. |
| Step 2 | Preprocessing | 전처리 | 오디오 파일을 모델 학습에 적합한 텐서 형태로 변환합니다. |
| Step 3 | Training | 트레이닝 | 전처리된 데이터로 LoRA 모델을 학습합니다. |
| - | LoRA Settings | LoRA 설정 | 학습된 LoRA 어댑터를 로드하여 음악 생성 스타일을 변경합니다. |
| - | Generation Settings | 생성 설정 | 생성할 음악의 장르, 분위기, 가사, BPM 등을 설정합니다. |

##### 라벨/버튼/기타 텍스트

| 영문 | 한국어 |
|------|--------|
| Dataset name | 데이터셋 이름 |
| Custom tag | 커스텀 태그 |
| Instrumental only (no vocals) | 인스트루멘탈 전용 (보컬 없음) |
| Scan | 업로드 & 스캔 |
| Save Dataset | 데이터셋 저장 |
| Output directory | 출력 디렉토리 |
| Start Preprocessing | 전처리 시작 |
| Tensor directory | 텐서 디렉토리 |
| Load | 불러오기 |
| Epochs | 에포크 수 |
| LoRA Rank | LoRA 랭크 |
| LoRA Alpha | LoRA 알파 |
| Learning Rate | 학습률 |
| Batch Size | 배치 크기 |
| Dropout | 드롭아웃 |
| Checkpoint interval (epochs) | 체크포인트 간격 (에포크) |
| Start Training | 트레이닝 시작 |
| Stop Training | 트레이닝 중지 |
| Training Log | 트레이닝 로그 |
| LoRA path | LoRA 경로 |
| Load / Unload | 로드 / 해제 |
| Enable LoRA | LoRA 활성화 |
| Strength | 강도 |
| Caption / Description | 캡션 / 설명 |
| Lyrics | 가사 |
| BPM (Auto) | BPM (자동) |
| Key (Auto) | 키 (자동) |
| Time Signature | 박자 |
| Duration (seconds) | 재생 시간 (초) |
| Language | 언어 |
| Generate Music | 음악 생성 |
| Download | 다운로드 |

##### 토스트 메시지 한국어화

| 영문 | 한국어 |
|------|--------|
| Server connected | 서버에 연결되었습니다 |
| Connection failed | 연결 실패 |
| Scan complete | 업로드 및 스캔이 완료되었습니다 |
| Dataset saved | 데이터셋이 저장되었습니다 |
| Preprocessing started | 전처리가 시작되었습니다 |
| Preprocessing complete! | 전처리가 완료되었습니다! |
| Training started | 트레이닝이 시작되었습니다 |
| Training complete! | 트레이닝이 완료되었습니다! |
| Music generated! | 음악이 생성되었습니다! |
| Enter a caption/description | 캡션/설명을 입력하세요 |

##### 가사 힌트

```
[verse], [chorus], [bridge], [outro] 태그로 곡의 구조를 지정할 수 있습니다.
비워두면 인스트루멘탈로 생성됩니다.
```

##### 시간 포맷

```
기존: 5s / 2m 30s / 1h 15m
변경: 5초 / 2분 30초 / 1시간 15분
```

---

#### 2-2. 파일 업로드 UI (Step 1 변경)

##### 기존 구조

```
[디렉토리 경로 텍스트 입력] + [Scan 버튼]
```

##### 변경 구조

```
┌──────────────────────────────────────────────┐
│  데이터셋 이름: [my_dataset]                    │
│  커스텀 태그:   [lo-fi, ambient]               │
│  ☐ 인스트루멘탈 전용                             │
│                                              │
│  ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐  │
│  │   오디오 파일을 여기에 드래그하거나       │  │
│  │         [파일 선택] 을 클릭하세요        │  │
│  │   (mp3, wav, flac, ogg, aac, m4a,    │  │
│  │    wma, opus)                         │  │
│  └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘  │
│  선택된 파일: 8개 (총 45.2 MB)                  │
│  ├ track01.mp3 (5.6 MB)                      │
│  ├ track02.wav (12.1 MB)                     │
│  └ ...                                       │
│                                              │
│  [업로드 & 스캔]  [데이터셋 저장]                │
└──────────────────────────────────────────────┘
```

##### HTML 요소

- `<input type="file" multiple accept=".mp3,.wav,.flac,.ogg,.aac,.m4a,.wma,.opus">` (숨김)
- 드래그 앤 드롭 영역 (`.drop-zone`, dashed border, hover 효과)
- 선택된 파일 요약 (`#selectedFilesSummary`) — "선택된 파일: N개 (총 X MB)"
- 선택된 파일 목록 (`#selectedFilesList`) — 파일명 + 크기
- 업로드 프로그레스 바 (`#uploadProgress`, `#uploadBar`, `#uploadPercent`)

##### CSS 추가

```css
.drop-zone           /* 드래그 앤 드롭 영역 */
.drop-zone:hover     /* 호버 효과 */
.drop-zone.drag-over /* 드래그 오버 효과 (border-color: primary) */
.drop-zone-text      /* 안내 텍스트 */
.browse-link         /* "파일 선택" 링크 스타일 */
.drop-zone-ext       /* 허용 확장자 안내 */
.selected-files-summary  /* 파일 요약 */
.selected-files-list     /* 파일 목록 */
.upload-progress-wrap    /* 업로드 프로그레스 래퍼 */
.card-desc               /* 카드 설명 텍스트 */
.header .subtitle        /* 헤더 서브타이틀 */
```

##### JavaScript 변경

| 함수/속성 | 설명 |
|-----------|------|
| `State.uploadedDir` | 업로드된 디렉토리 경로 저장 |
| `State.selectedFiles` | 선택된 파일 객체 배열 |
| `API.uploadDataset(formData)` | XHR 기반 업로드 (progress 이벤트 활용) |
| `_initDropZone()` | 드래그 앤 드롭 + 파일 선택 이벤트 바인딩 |
| `_handleFiles(files)` | 파일 유효성 검사 + 목록 렌더링 |
| `uploadAndScan()` | 업로드 → 스캔 자동 체인 |
| `formatFileSize(bytes)` | 파일 크기 포맷 (B/KB/MB) |

##### 업로드 & 스캔 흐름

```
1. 사용자가 파일 선택 (드래그 또는 클릭)
   → _handleFiles()로 오디오 파일 필터링 + 목록 표시

2. [업로드 & 스캔] 클릭
   → uploadAndScan() 실행

3. FormData 생성 (files + dataset_name)
   → POST /v1/dataset/upload (XHR, progress 표시)

4. 업로드 완료 → upload_dir 수신

5. upload_dir로 POST /v1/dataset/scan 자동 호출

6. 스캔 결과 표시 + [데이터셋 저장] 버튼 활성화
```

##### 제거된 요소

- `#datasetDir` 텍스트 입력 (디렉토리 경로 직접 입력)
- `scanDataset()` 함수 (uploadAndScan()으로 대체)
- `_saveInputs`에서 `datasetDir` 키 제거

---

### 검증 체크리스트

- [ ] 서버 재시작 후 `POST /v1/dataset/upload` 엔드포인트 동작 확인
- [ ] `index.html`에서 오디오 파일 드래그 앤 드롭 → 업로드 → 스캔 동작 확인
- [ ] 모든 UI 텍스트가 한국어로 표시되는지 확인
- [ ] 기존 전처리/트레이닝/음악 생성 흐름이 정상 동작하는지 확인
- [ ] 업로드 프로그레스 바가 정상적으로 표시되는지 확인
