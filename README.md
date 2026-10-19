# Skin Project

F/L/R 스마트폰 얼굴 사진에서 부위별 피부 상태를 예측하는 멀티뷰 학습 코드다.
하나의 공유 EfficientNet-B0와 부위별 출력 head를 사용하며 세 가지 범위를 지원한다.

| task | 출력 |
|---|---|
| `forehead_1` | 이마 주름 전문가 등급 1개 |
| `grades_11` | 부위별 전문가 등급 11개 |
| `core_15` | 전문가 등급 11개 + 이마/양 볼/턱 수분 4개 |

## 필요한 파일

학습 서버에는 다음 파일이 있어야 한다.

1. 원본 이미지 루트: `train/images/...`, `val/images/...`
2. `subject_manifest_with_split.csv`: 한 사람당 한 행, F/L/R 경로와 90개 라벨
3. `region_manifest.csv`: 정답 bbox manifest
4. `landmark_region_manifest.csv`: 서비스와 동일한 MediaPipe bbox manifest

원본 이미지와 manifest, crop 캐시, 모델 가중치는 Git에 올리지 않는다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

랜드마크 manifest와 랩미팅 그래프까지 만들려면 다음 선택 의존성을 설치한다.

```bash
pip install -e '.[landmark,report]'
```

GPU 서버에서는 먼저 서버 CUDA 환경에 맞는 PyTorch 설치가 정상인지 확인한다.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 서비스용 랜드마크 bbox 생성

정답 bbox는 성능 상한선 확인용으로 보존하고, 실제 서비스 모델은 학습과 추론 모두
동일한 MediaPipe Face Mesh bbox를 사용한다.

```bash
curl -L \
  -o /path/to/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

skin-landmark-manifest \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --data-root /path/to/skin_data \
  --face-landmarker-model /path/to/face_landmarker.task \
  --output /path/to/landmark_region_manifest.csv
```

MediaPipe 구버전의 `solutions.face_mesh`가 설치된 환경에서는 모델 경로 없이도
동작하고, 최신 Tasks API에서는 `face_landmarker.task` 경로가 필요하다.

서비스 추론 코드에서는 `skin_project.landmarks.crop_regions_from_landmarks`를
그대로 호출한다. 이 함수는 위 manifest 생성과 같은 랜드마크 규칙 및 학습 crop의
기본 여백 10%를 사용하므로, 별도의 서비스 bbox 규칙을 다시 만들지 않는다.
미간은 양쪽 눈썹의 안쪽 끝과 코뿌리만 사용하며, 눈썹 전체를 포함하지 않는다.

생성 직후에는 랜드마크 규칙을 고정하기 전에 정답 bbox와 IoU 및 최악 사례를 확인한다.

```bash
python scripts/compare_landmark_boxes.py \
  --ground-truth-manifest /path/to/region_manifest.csv \
  --landmark-manifest /path/to/landmark_region_manifest.csv \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --data-root /path/to/skin_data \
  --output-dir /path/to/landmark_qc \
  --view-policy service
```

`landmark_iou_summary.csv`, 부위별 IoU/coverage boxplot, 취약 사례 overlay 이미지가 생성된다.
정답 bbox가 작은 측정 지점인 경우 IoU보다 `mean_ground_truth_coverage`와
`center_hit_rate`를 우선 확인한다. 기본 QC는 실제 서비스에서 사용하는 부위-view
조합만 평가한다.
랜드마크 ROI가 육안으로 부적절하면 `REGION_LANDMARKS`와 `REGION_PADDING`을 수정한 뒤
manifest와 landmark crop cache를 다시 만든다.

랜드마크 crop은 정답 bbox crop과 다른 캐시 디렉터리를 사용해야 한다.

```text
skin_cache_ground_truth/  # 기존 JSON bbox
skin_cache_landmark/      # MediaPipe bbox
```

## 학습 실행

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

서비스용 핵심 15개 기본 실험:

```bash
skin-train \
  --task core_15 \
  --run-name landmark_service_softmax_mean \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --region-manifest /path/to/landmark_region_manifest.csv \
  --data-root /path/to/skin_data \
  --cache-dir /path/to/skin_cache_landmark \
  --output-dir /path/to/outputs \
  --batch-size 4 \
  --view-policy service \
  --selection-metric grade_qwk \
  --grade-objective softmax \
  --fusion mean
```

첫 실행에서는 필요한 부위 crop을 원본에서 한 번 생성한다. 이후 실행은 같은 캐시를
재사용한다. 캐시만 미리 생성하려면 `--cache-only`를 추가한다.

## 실험 축

### 부위별 view routing

`--view-policy service`는 실제로 해당 부위가 보이는 방향만 사용한다.

| region | views |
|---|---|
| forehead, glabella, lips | F |
| right_eye, right_cheek | F, R |
| left_eye, left_cheek | F, L |
| chin | F, L, R |

비교 실험은 `--view-policy all`, `front`, `service`로 실행한다.

### 등급 objective

- `regression`: 기존 가중 Huber 회귀
- `softmax`: effective-number class-balanced Cross-Entropy
- `ordinal`: 순서가 고정된 cutpoint 기반 cumulative ordinal loss

수분 4개는 모든 설정에서 표준화 Huber 회귀를 유지한다.

### view fusion

- `mean`: 유효 view feature 평균
- `concat`: F/L/R 위치를 보존하여 이어 붙이고 mask도 입력
- `attention`: view embedding과 mask를 사용한 학습형 가중 평균

## 최종 비교 실험

`scripts/run_colab_experiments.sh`는 같은 split과 랜드마크 bbox로 여섯 실험을
순서대로 실행한다.

```bash
bash scripts/run_colab_experiments.sh \
  /path/to/subject_manifest_with_split.csv \
  /path/to/landmark_region_manifest.csv \
  /path/to/skin_data \
  /path/to/skin_cache_landmark \
  /path/to/outputs
```

| run | view policy | grade objective | fusion |
|---|---|---|---|
| `landmark_all_regression_mean` | all | regression | mean |
| `landmark_service_regression_mean` | service | regression | mean |
| `landmark_service_softmax_mean` | service | softmax | mean |
| `landmark_service_ordinal_mean` | service | ordinal | mean |
| `landmark_service_softmax_concat` | service | softmax | concat |
| `landmark_service_softmax_attention` | service | softmax | attention |

첫 두 실행으로 view routing 효과를 분리하고, 다음 두 실행으로 회귀·Softmax·ordinal을
비교하며, 마지막 두 실행으로 concat과 attention을 mean과 비교한다. 모든 실행은
validation macro QWK를 기준으로 checkpoint를 선택한다.

## 모델과 loss

- 입력 shape: `[batch, region, view=3, RGB, 224, 224]`
- 모든 유효 crop은 동일한 EfficientNet-B0로 인코딩한다.
- bbox가 없거나 view policy에서 제외된 입력은 encoder와 fusion에서 제외한다.
- 공유 인코더 뒤에 target별 head를 둔다.
- 각 target loss를 계산한 뒤 target 간 평균을 total loss로 사용한다.
- 수분값은 train split의 평균과 표준편차로 표준화한 뒤 loss를 계산한다.
- grade와 수분 target에 동일한 가중치를 주므로 라벨 수가 많은 부위가 loss를 독점하지 않는다.
- `regression` 등급 target은 train split의 등급 빈도에 반비례하는 완만한 표본 가중치를
  사용한다. 기본값은 제곱근 역빈도(`--grade-balance-power 0.5`)이며 최대 3배로 제한한다.
- `softmax`는 effective-number class weight, `ordinal`은 threshold별 positive weight를
  train split에서 계산한다. 연속형 수분 target에는 이 가중치를 적용하지 않는다.
- 최종 비교 실험은 validation macro QWK로 checkpoint를 선택한다. test의 라벨별 MAE,
  정확도, balanced accuracy, QWK 계산에는 학습 가중치를 적용하지 않는다.

`grades_11`과 `core_15`는 한 사람당 8개 부위 × 3개 뷰를 사용하므로 GPU 메모리가
부족하면 `--batch-size 2`로 낮춘다. CUDA가 없는 CPU에서도 실행은 가능하지만 매우 느리다.

## 랩미팅 시각화

실험이 끝난 뒤 다음 명령으로 비교 표와 그림을 만든다.

```bash
python scripts/make_lab_meeting_figures.py \
  --outputs-dir /path/to/outputs \
  --runs \
    landmark_all_regression_mean \
    landmark_service_regression_mean \
    landmark_service_softmax_mean \
    landmark_service_ordinal_mean \
    landmark_service_softmax_concat \
    landmark_service_softmax_attention \
  --figure-dir /path/to/lab_meeting_figures \
  --subject-manifest /path/to/subject_manifest_with_split.csv \
  --region-manifest /path/to/landmark_region_manifest.csv
```

생성물:

- `experiment_summary.csv`: 실행별 핵심 test 지표
- `01_model_comparison.png`: MAE, 정확도, balanced accuracy, QWK 비교
- `02_target_qwk_heatmap.png`: 라벨별 QWK heatmap
- `03_training_curves.png`: validation loss와 QWK 학습 곡선
- `04_service_pipeline.png`: end-to-end 서비스 파이프라인
- `05_grade_distributions.png`: 등급 불균형 분포
- `06_bbox_availability.png`: 부위·방향별 landmark bbox 검출률
- `07_grade_confusion_matrices.png`: 최고 QWK 실험의 취약 등급 라벨 4개 혼동행렬
- `08_moisture_predictions.png`: 최고 QWK 실험의 수분 실제값-예측값 산점도

## 출력

각 task별 출력 폴더에 다음 파일이 생긴다.

- `best.pt`: 선택한 validation 지표(기본 loss, 비교 실험은 macro QWK) 기준 최고 checkpoint
- `history.csv`: epoch별 train/validation 성능
- `test_metrics.csv`: 라벨별 test 지표
- `test_predictions.csv`: 피험자별 실제값과 예측값
- `summary.json`: 전체 test 요약
