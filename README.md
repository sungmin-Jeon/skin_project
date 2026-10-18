# Skin Project

F/L/R 스마트폰 얼굴 사진에서 부위별 피부 상태를 예측하는 멀티뷰 학습 코드다.
하나의 공유 EfficientNet-B0와 부위별 출력 head를 사용하며 세 가지 범위를 지원한다.

| task | 출력 |
|---|---|
| `forehead_1` | 이마 주름 전문가 등급 1개 |
| `grades_11` | 부위별 전문가 등급 11개 |
| `core_15` | 전문가 등급 11개 + 이마/양 볼/턱 수분 4개 |

## 필요한 파일

학습 서버에는 다음 세 가지가 있어야 한다.

1. 원본 이미지 루트: `train/images/...`, `val/images/...`
2. `subject_manifest_with_split.csv`: 한 사람당 한 행, F/L/R 경로와 90개 라벨
3. `region_manifest.csv`: 한 JSON당 한 행, `facepart`와 `bbox` 포함

원본 이미지와 manifest, crop 캐시, 모델 가중치는 Git에 올리지 않는다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

GPU 서버에서는 먼저 서버 CUDA 환경에 맞는 PyTorch 설치가 정상인지 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 실행

아래 경로는 서버의 실제 위치로 변경한다.

```bash
skin-train \
  --task forehead_1 \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --region-manifest /path/to/region_manifest.csv \
  --data-root /path/to/skin_data \
  --cache-dir /path/to/skin_cache \
  --output-dir /path/to/outputs
```

전문가 등급 11개:

```bash
skin-train \
  --task grades_11 \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --region-manifest /path/to/region_manifest.csv \
  --data-root /path/to/skin_data \
  --cache-dir /path/to/skin_cache \
  --output-dir /path/to/outputs \
  --batch-size 4
```

핵심 15개:

```bash
skin-train \
  --task core_15 \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --region-manifest /path/to/region_manifest.csv \
  --data-root /path/to/skin_data \
  --cache-dir /path/to/skin_cache \
  --output-dir /path/to/outputs \
  --batch-size 4
```

첫 실행에서는 필요한 부위 crop을 원본에서 한 번 생성한다. 이후 실행은 같은 캐시를
재사용한다. 캐시만 미리 생성하려면 `--cache-only`를 추가한다.

## 모델과 loss

- 입력 shape: `[batch, region, view=3, RGB, 224, 224]`
- F/L/R은 동일한 EfficientNet-B0로 인코딩하고 특징을 평균한다.
- 공유 인코더 뒤에 부위별 head를 둔다.
- 각 target의 Huber loss를 계산한 뒤 target 간 평균을 total loss로 사용한다.
- 수분값은 train split의 평균과 표준편차로 표준화한 뒤 loss를 계산한다.
- grade와 수분 target에 동일한 가중치를 주므로 라벨 수가 많은 부위가 loss를 독점하지 않는다.

`grades_11`과 `core_15`는 한 사람당 8개 부위 × 3개 뷰를 사용하므로 GPU 메모리가
부족하면 `--batch-size 2`로 낮춘다. CUDA가 없는 CPU에서도 실행은 가능하지만 매우 느리다.

## 출력

각 task별 출력 폴더에 다음 파일이 생긴다.

- `best.pt`: validation loss 기준 최고 checkpoint
- `history.csv`: epoch별 train/validation 성능
- `test_metrics.csv`: 라벨별 test 지표
- `test_predictions.csv`: 피험자별 실제값과 예측값
- `summary.json`: 전체 test 요약
