# ACE-Step 1.5 환경 설정 및 LoRA 트레이닝 가이드

## Context

ACE-Step 1.5는 소비자급 하드웨어에서 상용 수준의 음악을 생성하는 오픈소스 음악 파운데이션 모델이다. 이 문서는 macOS Apple Silicon 환경에서 환경 설정 → 음악 생성 → LoRA 트레이닝 → LoRA 적용 생성까지의 전체 워크플로우를 정리한다.

### 실행 환경

- **macOS** Darwin 25.1.0, Apple M4 Pro 16코어 GPU
- **통합 메모리**: 48GB (MPS로 37.4GB 감지, GPU와 공유)
- **Python**: 3.12.11 (uv가 자동 설치)
- **uv**: 0.8.22
- **FFmpeg**: 8.0.1 (Homebrew)

---

## Part 1: 환경 설정

### Step 1: 사전 요구사항 설치

```bash
# FFmpeg (트레이닝 전처리에 필요)
brew install ffmpeg
```

### Step 2: 프로젝트 클론

```bash
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
```

### Step 3: 의존성 설치

```bash
uv sync
```

- uv가 `pyproject.toml`을 읽고 **Python 3.12.11** 및 **123개 패키지**를 자동 설치
- `.venv` 가상환경이 프로젝트 내에 생성됨
- 주요 패키지: `mlx 0.30.6`, `mlx-lm 0.29.1`, `torch 2.10.0`, `gradio 6.2.0`, `transformers 4.57.6`

### Step 4: 환경 설정 (.env)

```bash
cp .env.example .env
```

**변경 사항** — `.env`에서 `ACESTEP_LM_BACKEND`만 수정:

```env
# 기본값 (변경 불필요)
ACESTEP_CONFIG_PATH=acestep-v15-turbo
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_DEVICE=auto
ACESTEP_INIT_LLM=auto

# vllm → mlx로 변경 (Apple Silicon 전용)
ACESTEP_LM_BACKEND=mlx
```

- **1.7B 모델**: 48GB 통합 메모리에서 정상 구동 확인됨
- **mlx 백엔드**: macOS 런치 스크립트가 강제 설정하지만, `.env`에도 명시하면 `uv run acestep` 직접 실행 시에도 적용됨
- 만약 1.7B가 불안정하면 `acestep-5Hz-lm-0.6B`로 다운그레이드

### Step 5: MPS 트레이닝 패치 (필수)

Apple Silicon MPS에서 `float16 mixed precision`이 NaN 그래디언트를 발생시키므로, `acestep/training/trainer.py`를 수정:

```python
# 변경 전 (69~86행)
def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type == "mps":
        return torch.float16         # ← NaN 발생

def _select_fabric_precision(device_type: str) -> str:
    if device_type == "mps":
        return "16-mixed"             # ← NaN 발생

# 변경 후
def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type == "mps":
        return torch.float32          # ← NaN 해결

def _select_fabric_precision(device_type: str) -> str:
    if device_type == "mps":
        return "32-true"              # ← NaN 해결
```

### Step 6: Gradio UI 실행

```bash
# macOS 전용 런치 스크립트 (권장)
bash start_gradio_ui_macos.sh

# 또는 직접 실행
uv run acestep --backend mlx
```

### 첫 실행 시 자동 처리 사항

1. **모델 다운로드**: HuggingFace에서 `ACE-Step/Ace-Step1.5` 28개 파일 자동 다운로드 (~2분 40초)
   - 저장 위치: `checkpoints/` 디렉토리
2. **DiT 모델 초기화**: MLX-DiT + MLX-VAE (compiled)
3. **5Hz LM 토크나이저 로드**: ~8초
4. **MLX LM 모델 로드**: `model.` prefix 자동 리맵 후 ~0.2초

### 확인된 GPU 설정

```
GPU Memory: 37.44 GB
Configuration Tier: unlimited
Max Duration (with LM): 600s (10 min)
Max Duration (without LM): 600s (10 min)
Max Batch Size (with LM): 8
Max Batch Size (without LM): 8
Available LM Models: [0.6B, 1.7B, 4B]
```

### 접속

- 브라우저에서 `http://127.0.0.1:7860` 접속

---

## Part 2: 음악 생성 테스트

### Gradio API를 통한 생성

```python
from gradio_client import Client

client = Client('http://127.0.0.1:7860')
result = client.predict(
    'upbeat electronic pop, bright synth melody, energetic drums',  # Caption
    '[verse]\nHello world\n[chorus]\nLa la la',                     # Lyrics
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
    # ... (나머지 파라미터는 기본값 사용)
    api_name='/generation_wrapper'
)
```

### 생성 결과 (LoRA 없이, 30초)

| 항목 | 값 |
|------|-----|
| 총 생성 시간 | **13.82초** |
| LM 단계 | 5.33초 |
| DiT 단계 | 8.49초 |
| MP3 변환 | 2.40초 |
| 파일 크기 | 555KB |
| 저장 위치 | `gradio_outputs/batch_*/` |

---

## Part 3: LoRA 트레이닝

### 트레이닝 파이프라인 개요

```
오디오 파일 준비 → 데이터셋 스캔 → 텐서 전처리 → LoRA 트레이닝 → 가중치 저장
```

### Step 1: 트레이닝 데이터 준비

프로젝트 내 `datasets/` 디렉토리에 오디오 + 메타데이터 파일 배치:

```
datasets/kim_kyungho/
├── Song1.mp3                   # 오디오 파일
├── Song1.caption.txt           # 캡션 (장르, 악기, 스타일 설명)
├── Song1.lyrics.txt            # 가사 (없으면 [Instrumental])
├── Song2.mp3
├── Song2.caption.txt
├── Song2.lyrics.txt
└── ...
```

**caption.txt 예시:**
```
kim kyung ho, Korean power rock ballad, soaring male vocals, electric guitar solo, dramatic orchestral arrangement, 1990s Korean rock
```

**lyrics.txt 예시 (가사가 없는 경우):**
```
[Instrumental]
```

> 주의: Gradio UI에서 `datasets/` 외부 경로는 보안 정책으로 차단됨. 반드시 프로젝트 내부로 복사해야 함.

### Step 2: 데이터셋 스캔 및 JSON 저장

```python
from acestep.training.dataset_builder import DatasetBuilder

builder = DatasetBuilder()
samples, status = builder.scan_directory('./datasets/kim_kyungho')
# ✅ Found 3 audio files, 3 captions, 3 lyrics

builder.set_custom_tag('kim kyung ho', tag_position='prepend')
builder.set_all_instrumental(True)
builder.save_dataset('./datasets/kim_kyungho/dataset.json', dataset_name='kim_kyungho')
```

- `labeled=True`여야 전처리 가능 → `.caption.txt` 파일 필수
- 스캔 시 `.json`, `.csv`, `.caption.txt` 등 자동 인식

### Step 3: 텐서 전처리

```python
from acestep.training_v2.preprocess import preprocess_audio_files

result = preprocess_audio_files(
    audio_dir=None,
    output_dir='./datasets/preprocessed_tensors',
    checkpoint_dir='./checkpoints',
    variant='turbo',
    max_duration=240.0,
    dataset_json='./datasets/kim_kyungho/dataset.json',
    device='auto',
    precision='auto',
)
# Processed: 3, Failed: 0
```

- **2-pass 방식**: Pass 1 (VAE + 텍스트 인코딩) → Pass 2 (DiT 조건 인코딩)
- 출력: `.pt` 텐서 파일 (각 ~5MB)

#### 전처리 NaN 수정 (MPS 필수)

float16 전처리 시 `encoder_hidden_states`에 NaN 발생. 반드시 수정해야 트레이닝 가능:

```python
import torch, glob

for f in glob.glob('./datasets/preprocessed_tensors/*.pt'):
    pt = torch.load(f, map_location='cpu', weights_only=True)
    for k, v in pt.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            pt[k] = torch.nan_to_num(v, nan=0.0)
    torch.save(pt, f)
```

### Step 4: LoRA 트레이닝 실행

Gradio API를 통해 실행:

```python
from gradio_client import Client
client = Client('http://127.0.0.1:7860')

# 데이터셋 로드
client.predict(tensor_dir='./datasets/preprocessed_tensors', api_name='/load_training_dataset')

# 트레이닝 시작
client.predict(
    tensor_dir='./datasets/preprocessed_tensors',
    r=64,           # LoRA Rank
    a=128,          # LoRA Alpha
    d=0.1,          # Dropout
    lr=0.0003,      # Learning Rate
    ep=100,         # Max Epochs (최소 100)
    bs=1,           # Batch Size
    ga=1,           # Gradient Accumulation
    se=50,          # Save Every N Epochs
    sh=3.0,         # Shift
    sd=42,          # Seed
    od='./lora_output/kim_kyungho',
    rc=None,        # Resume Checkpoint
    api_name='/training_wrapper'
)
```

### 트레이닝 결과 (김경호 3곡, 100 에폭)

| 항목 | 값 |
|------|-----|
| 트레이닝 데이터 | 3곡 (Elise, Exodus, Forbidden Love) |
| Trainable 파라미터 | 44,040,192 (1.81%) |
| 에폭당 시간 | ~37초 |
| 총 소요 시간 | ~60분 |
| Loss 추이 | 1.33 → 0.75 → 0.67 (수렴) |
| 체크포인트 | epoch_50, epoch_100, final |
| LoRA 가중치 크기 | 168MB (`adapter_model.safetensors`) |

```
lora_output/kim_kyungho/
├── checkpoints/
│   ├── epoch_50/adapter/     # 중간 체크포인트
│   └── epoch_100/adapter/    # 최종 체크포인트
├── final/adapter/            # 최종 가중치
│   ├── adapter_config.json
│   ├── adapter_model.safetensors (168MB)
│   └── README.md
└── logs/                     # TensorBoard 로그
```

---

## Part 4: LoRA 적용 음악 생성

### LoRA 로드

```python
from gradio_client import Client
client = Client('http://127.0.0.1:7860')

# LoRA 로드 & 활성화
client.predict(lora_path='./lora_output/kim_kyungho/final/adapter', api_name='/load_lora')
client.predict(True, api_name='/set_use_lora')    # 활성화
client.predict(0.8, api_name='/set_lora_scale')   # Scale 0.8

# LoRA 해제
client.predict(api_name='/unload_lora')
```

### LoRA 적용 생성 결과 (김경호 스타일, 60초)

| 항목 | 값 |
|------|-----|
| 프롬프트 | Korean power rock ballad, soaring high-pitched male vocals, electric guitar solo |
| LoRA | kim_kyungho (scale 0.8) |
| 설정 | 130 BPM, E minor, 4/4, 60초 |
| 총 생성 시간 | **66.15초** |
| LM 단계 | 27.47초 |
| DiT 단계 | 38.68초 |
| MP3 변환 | 1.65초 |
| 파일 크기 | 938KB |

### 생성 성능 비교

| 조건 | 길이 | 생성 시간 | 초당 시간 |
|------|------|-----------|-----------|
| LoRA 없음 | 30초 | 13.82초 | 0.46초/초 |
| LoRA 적용 (0.8) | 60초 | 66.15초 | 1.10초/초 |

---

## 트러블슈팅

### 포트 충돌 (7860 이미 사용 중)

```bash
lsof -i :7860          # 사용 중인 프로세스 확인
kill <PID>             # 해당 PID 종료
# 또는
PORT=7861 bash start_gradio_ui_macos.sh
```

### 업데이트 체크 건너뛰기

```bash
CHECK_UPDATE=false bash start_gradio_ui_macos.sh
```

### MPS 트레이닝 NaN 그래디언트

**원인 1**: `trainer.py`의 MPS float16 mixed precision
**해결**: `_select_compute_dtype`과 `_select_fabric_precision`에서 MPS를 `float32` / `"32-true"`로 변경

**원인 2**: 전처리 텐서의 `encoder_hidden_states`에 float16 오버플로우로 NaN 포함
**해결**: `torch.nan_to_num(tensor, nan=0.0)` 적용

### FFmpeg 미설치 (전처리 실패)

```bash
brew install ffmpeg
```

`torchcodec`가 FFmpeg을 필요로 함. 없으면 `soundfile` 폴백으로 duration만 읽히고 전처리 시 실패.

### Gradio 보안 경로 제한

Gradio UI에서 프로젝트 외부 경로 접근 시 `Rejected unsafe directory path` 에러 발생.
반드시 프로젝트 내부 (`datasets/` 등)로 파일을 복사해야 함.

### 무시해도 되는 경고

- `Skipping import of cpp extensions due to incompatible torch version` — torchao 호환성 경고
- `bitsandbytes not installed. Using standard AdamW.` — 표준 옵티마이저 사용, 정상
- `Standard MLX load failed ... retrying with 'model.' prefix remapping` — 자동 복구됨

---

## 디렉토리 구조 요약

```
ACE-Step-1.5/
├── .env                          # 환경 설정 (mlx 백엔드)
├── .venv/                        # Python 가상환경
├── checkpoints/                  # 다운로드된 모델 가중치
│   ├── acestep-v15-turbo/        # DiT 모델
│   └── acestep-5Hz-lm-1.7B/     # LM 모델
├── datasets/
│   ├── kim_kyungho/              # 트레이닝 원본 데이터
│   │   ├── *.mp3                 # 오디오 파일
│   │   ├── *.caption.txt         # 캡션
│   │   ├── *.lyrics.txt          # 가사
│   │   └── dataset.json          # 스캔 결과 JSON
│   └── preprocessed_tensors/     # 전처리된 텐서 파일
│       └── *.pt
├── lora_output/
│   └── kim_kyungho/
│       ├── final/adapter/        # 최종 LoRA 가중치
│       ├── checkpoints/          # 중간 체크포인트
│       └── logs/                 # TensorBoard 로그
├── gradio_outputs/               # 생성된 음악 파일
│   └── batch_*/
│       ├── *.mp3                 # 생성된 오디오
│       └── *.json                # 생성 메타데이터
└── start_gradio_ui_macos.sh      # macOS 런치 스크립트
```

---

## 주의사항

- **첫 실행 시 모델 다운로드**: HuggingFace에서 자동 다운로드하므로 인터넷 연결 필요
- **메모리**: 48GB이므로 충분하지만, 트레이닝 중에는 다른 무거운 앱 종료 권장
- **MLX 백엔드**: Apple Silicon 전용 최적화. vllm은 macOS에서 지원되지 않음
- **서버 종료**: 터미널에서 `Ctrl+C` 또는 `kill <PID>`
- **LoRA 트레이닝 최소 에폭**: Gradio UI에서 최소 100 에폭으로 설정됨
- **트레이닝 권장 에폭**: 3곡 기준 800 에폭, 100곡 기준 500 에폭 (참고값)
