# 실습 과제 1: 처음 보는 환경에서도 잘 걷는 Ant

**한눈에 보기**

- 제공된 PPO baseline은 처음 보는 지형에서 거의 걷지 못합니다(T1~T4 평균 21.2). 진단해 보니 원인은 **"높은 곳에서 출발하면 얼어붙는 것"** 이었습니다. 관측의 몸통 높이가 월드 z 절대값이라 학습 때 본 적 없는 입력이 되기 때문입니다.
- 원래 가설인 "명령-반응 이력으로 보정"(조건 C, D)은 같은 학습량에서 추가 이득이 없었습니다.
- **사후 탐색 실험 E:** 몸통 높이를 지면 기준으로 바꾸고 주변 지형 높이맵을 관측에 넣었습니다. T1~T4 평균이 58.8이 됐습니다.
- **사후 탐색 실험 E2:** 두 번째 진단으로 남은 손실이 "거친 지형에서 느린 것"임을 찾았습니다. 원인은 세 가지였습니다.
  - 탐색 잡음이 1000 it 무렵 사라져 학습이 멈췄습니다.
  - 높이맵이 발 디딜 곳을 11~16%만 덮었습니다.
  - 학습 지형의 높이 범위가 좁았습니다.
- 셋을 고친 E2는 **T1~T4 평균 83.2**(baseline의 3.9배)입니다. 학습 전에 새로 정한 시험 지형 **T5에서도 74.7**(E 51.2)입니다.
- **제출 모델: E2 (seed 42)**, T1~T4 평균 93.9.

| 조건 | Flat | T1 형태 | T2 μ 0.2 | T3 형태+μ 0.4 | T4 새 형태 | T5 새 형태 2 | T1~T4 평균 | T1~T5 평균 |
|---|---|---|---|---|---|---|---|---|
| A_ref: 제공 baseline (1000 it) | 134.9 | 7.3 | 60.8 | 7.7 | 9.2 | 11.4 | 21.2 | 19.3 |
| A: 평지 학습 | 141.0 ± 4.3 | 9.8 ± 1.7 | 62.8 ± 13.4 | 10.7 ± 2.2 | 19.0 ± 8.2 | 22.5 ± 8.8 | 25.6 ± 4.4 | 24.9 ± 5.3 |
| B: + 지형·마찰 DR | 110.1 ± 12.7 | 22.5 ± 3.7 | 111.4 ± 11.9 | 26.7 ± 4.7 | 46.1 ± 7.2 | 53.0 ± 1.8 | 51.7 ± 5.0 | 51.9 ± 4.3 |
| C: B + 명령-반응 이력 입력 | 84.1 ± 20.4 | 19.1 ± 3.3 | 84.2 ± 22.8 | 26.9 ± 1.9 | 30.5 ± 6.7 | 38.0 ± 7.1 | 40.2 ± 5.8 | 39.8 ± 6.0 |
| D: B@2000 + 예측 오차 기반 residual 보정 | 110.5 ± 11.3 | 22.9 ± 3.0 | 110.3 ± 12.9 | 26.9 ± 5.8 | 46.5 ± 9.3 | 53.3 ± 0.8 | 51.7 ± 4.8 | 52.0 ± 4.0 |
| E0: B + 지면 기준 높이 (사후 탐색) | 102.3 ± 2.4 | 35.6 ± 0.9 | 104.1 ± 4.6 | 46.9 ± 2.7 | 41.7 ± 3.3 | 50.6 ± 2.3 | 57.1 ± 1.5 | 55.8 ± 1.2 |
| E: E0 + 지형 높이맵 (사후 탐색) | 116.5 ± 5.5 | 39.6 ± 2.7 | 101.3 ± 10.7 | 48.1 ± 2.7 | 46.4 ± 4.3 | 51.2 ± 1.9 | 58.8 ± 3.6 | 57.3 ± 3.2 |
| **E2: 넓은 높이맵 + 엔트로피 + 2단계 지형 (사후 탐색 2차)** | **139.2 ± 16.4** | **61.8 ± 5.4** | 115.4 ± 25.7 | **75.6 ± 10.2** | **79.9 ± 2.1** | **74.7 ± 5.4** | **83.2 ± 10.7** | **81.5 ± 9.6** |
| └ E2 seed 42 (제출) | 157.3 | 65.9 | 142.2 | 85.4 | 82.3 | 79.2 | 93.9 | 91.0 |
| *참고: Oracle (테스트 형태로 학습한 천장, seed 42, 제출 불가)* | *102.7* | *75.8* | *121.3* | *85.9* | *88.2* | *70.9* | *92.8* | *88.4* |

- 값은 평가 reward(공식 `play_one_episode.py`와 같은 누적 규칙)입니다. 각 정책을 100 env로 평가해 평균을 내고, seed 3개(42/43/44)의 평균 ± 표준편차를 적었습니다.
- 평가 조건: seed 24, 잡음 없는 정책입니다.
- 굵은 글씨는 미리 정한 비교 규칙(평균 차 > 큰 쪽 표준편차, 그리고 seed 3쌍 모두 같은 방향)으로 E2 > E인 칸입니다. T2는 규칙상 차이가 없습니다.
- 전체 표(본 계획의 T1~T3 평균 포함), 넘어짐 비율, 전진 거리, 사전 등록 비교는 [results/summary.md](results/summary.md)에 있습니다.

## 1. 평가 방법 (조교님용)

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-WideScan-v0 \
  --seed 24 \
  --num_envs 100 \
  --checkpoint assignments/hw1_ant/checkpoints/E2_seed42/model.pt \
  --video \
  --video_length 960
```

- 같은 내용이 [EVAL_COMMAND.txt](EVAL_COMMAND.txt)에 있습니다. 원본 평지에서 이 명령의 결과는 157.30 ± 31.01입니다.
- **task와 지형:** `Isaac-Ant-WideScan-v0`는 원본 `Isaac-Ant-v0` 설정을 상속하고 레이캐스트 센서만 추가한 task입니다.
  - 지형, reward, 종료 조건, 에피소드 길이, 로봇은 원본과 같습니다.
  - 원본 `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`의 지형(`MySceneCfg.terrain`)과 지형 물성만 바꾸면 이 task도 그 지형에서 평가됩니다.
  - 센서는 Isaac Lab 지형의 기본 경로인 `/World/ground`를 읽습니다. 평면 지형과 생성 지형 모두 동작합니다.
- **저장소 위치:** conda env `lerobot-arena`의 Isaac Lab은 `~/IsaacLab_RS/source`를 가리킵니다. 이 저장소를 `~/IsaacLab_RS`에 받아 주세요.
  - 다른 위치에 받았다면 명령 앞에 `PYTHONPATH=<저장소>/source/isaaclab_tasks:$PYTHONPATH`를 붙이면 그 위치의 task가 쓰입니다.
- **원본 task 이름(`Isaac-Ant-v0`)으로만 평가해야 하는 경우:** 원본 관측 그대로인 조건 B를 쓰면 됩니다.
  - `--task Isaac-Ant-v0 --checkpoint assignments/hw1_ant/checkpoints/B_seed44/model.pt`
  - T1~T4 평균은 51.7로 E2(83.2)보다 낮습니다.

## 2. 문제와 가설

- 평가 reward의 대부분은 목표 방향 전진 거리(m)입니다. 생존 관련 항은 모두 합쳐도 최대 17.6점입니다.
- 넘어지거나 멈추면 그 뒤의 점수를 잃습니다. 몸통 높이 0.31 m 미만(월드 z 기준)이면 넘어진 것으로 보고 에피소드가 끝납니다.

> **원래 가설 (사전 등록).** PPO baseline이 처음 보는 환경에서 무너지는 이유는 지금 어떤 환경인지 알 수 없기 때문이다. 최근의 명령과 실제 반응을 보고 걸음을 보정하면 일반화가 개선된다.

| 조건 | 바꾼 한 가지 | 답하는 질문 |
|---|---|---|
| A | (원본) 평지, 마찰 1.0에서 3000 it | 출발점 |
| B | A + 지형·마찰 domain randomization | 다양한 환경 경험만으로 충분한가 (H1) |
| C | B + 최근 15 step의 명령·반응 이력 입력 (510차원) | 이력 정보가 도움이 되는가 (H2) |
| D | B@2000 고정 + 보정 Δa를 1000 it 학습, 보정 강도 α = 동역학 예측 오차 | 기본 걸음과 보정을 분리하면 나은가 (H3, H4) |

- **D의 구조:**
  - `a = π_base(o) + α·Δa`이고, 토크는 7.5·a(원본과 같은 scale)입니다.
  - α는 평지에서 학습한 동역학 모델의 최근 1초 예측 오차로 정하며, 학습하지 않습니다.
  - 라이브러리를 고치지 않고 커스텀 action term으로 구현했습니다. 최종 행동을 action 버퍼에 되써서 원본 reward의 행동·에너지 벌점이 실제 적용 행동으로 계산됩니다.
- **공정성:** 모든 조건은 PPO 3000 iteration입니다(D는 B의 2000 + 1000, E2는 1단계 1500 + 2단계 1500). reward와 관측 정규화(끔)가 같고, B~E2는 같은 학습 지형에서 출발합니다.
- **사전 등록:** 실험 전에 가설, 수치 예측, 비교 규칙("평균 차 > 표준편차이면서 seed 3쌍 모두 같은 방향")을 정하고 동결했습니다.
  - E와 E2는 결과를 본 뒤의 **사후 탐색 실험**입니다.
  - 그래서 각각 학습 전에 별도 계획(가설, 예측, 새 테스트 지형 T4·T5, 제출 규칙)을 다시 동결했습니다.

## 3. 학습 환경과 자체 "처음 보는 환경"

**학습 환경 (B, C, D, E0, E, E2 1단계 공통):**

- 8 m 타일 24행 × 8열(+x 192 m)이고 타일마다 종류와 난이도가 랜덤입니다.
- 구성은 평지 20%, 요철 30%(0.01~0.08 m), 파형 20%(0~0.08 m), 낮은 장애물 15%(0.02~0.10 m), 완만한 경사 15%입니다.
- 마찰은 env마다 0.3~1.2이고, 학습 env에서만 로봇 재질로 랜덤화했습니다.
- 넘어짐 판정이 월드 z 기준이라 모든 지형 높이를 0 이상으로 만들었습니다(학습된 Ant의 몸통은 0.38~0.50 m에서 걷습니다).
- **E2의 2단계(DR-Hard):** 같은 다섯 종류에서 높이 범위만 넓혔습니다. 요철·파형 최대 0.12 m, 장애물 0.20 m, 경사 0.30입니다. 요철은 T1(0.13~0.16 m)보다 낮게 두었고, 테스트 지형의 형태는 넣지 않았습니다.

**처음 보는 환경 (학습에 사용하지 않음):**

| 환경 | 내용 |
|---|---|
| Flat | 원본 평지 (마찰 1.0) |
| T1 | 학습에 없던 피라미드 계단, 박스 격자, 더 큰 요철. 출발점의 60%가 0.5 m 안팎의 높은 곳 |
| T2 | 평지, 마찰 0.2 (학습 범위 밖) |
| T3 | T1 지형, 마찰 0.4 |
| T4 | (E 학습 전에 정의) 솟은 상자(0.22~0.30 m) 위 출발, 레일, 원기둥 장애물 |
| T5 | (E2 학습 전에 정의) 원뿔, 기울어진 상자, 긴 파형(최대 0.2 m). 로봇 100대가 서로 다른 100개 타일에서 출발 |

- T1~T3는 A_ref만 보고 "평지의 50% 이하"가 되도록 정했고, 이후에는 바꾸지 않았습니다.
- T1~T4는 출발점이 10곳(열 10개 × 로봇 10대)입니다. T5는 출발점을 100곳으로 늘렸습니다.
- 분석용 환경도 있습니다. Grid(요철 높이 × 마찰 heatmap)와 Switch(5초에 마찰 1.0 → 0.25)입니다.

## 4. 결과와 분석

![main](figures/main_comparison.png)

1. **H1 지지.** B는 A보다 T1~T5에서 모두 좋습니다(사전 규칙 충족). 대가로 평지 점수가 22% 낮아지는 "타협형 걸음"이 됩니다.
2. **H2 기각.** C는 B보다 낮습니다. 입력이 8.5배(510)로 커졌는데 학습량은 같아 학습이 느리고 seed 편차가 컸습니다.
3. **H3, H4: D는 B와 차이가 없습니다**(D와 출발점 B@2000도 차이 없음).
   - α는 평지에서 0.02로 꺼지고, 처음 보는 형태(T1·T3)에서 0.2, T4에서 0.36으로 켜집니다.
   - 그러나 마찰 변화(T2, 마찰 전환)에는 거의 반응하지 않았습니다. 동역학 예측 오차가 미끄러짐보다 지형 충격에 반응했기 때문입니다.
   - 낯선 지형에서 실제 적용된 보정량은 기본 행동의 12~16%였지만 성능을 바꾸지 못했습니다.
4. **진단 1: 높은 곳에서 출발하면 얼어붙습니다.**
   - T1에서 출발 높이와 전진 거리의 상관은 모든 정책에서 −0.73 ~ −0.84입니다.
   - 계단 꼭대기(0.5~0.6 m)에서 출발한 B는 16초 동안 0 m를 갑니다.
   - D의 α는 얼어붙은 env에서 오히려 낮습니다(0.16~0.22, 걷는 env는 0.36~0.41). 멈춘 몸은 예측하기 쉬워서 보정이 꺼집니다.

   ![spawn](figures/t1_spawn_height.png)

5. **E (사후 탐색): 지면 기준 높이와 지형 높이맵.**
   - B와 비교해 T1 +76%, T3 +80%이고, 평지는 오히려 회복했습니다(110 → 117).
   - 지면 기준 높이만 바꾼 E0도 T1·T3에서 크게 좋아집니다. 개선의 대부분은 "얼어붙음 해소"에서 나왔다는 뜻입니다.
   - 한계도 있습니다. T4는 B와 같고, Grid의 가장 거친 열(0.15 m)은 B보다 낮습니다(47.4 vs 55.9, 사전 규칙상 E < B).
6. **진단 2: E의 남은 손실은 "느림"입니다.**
   - E는 거친 지형(T1, T3~T5)에서 평균 2.4~3.2 m/s로 걷습니다(평지 7.3 m/s). 넘어짐을 모두 없애도 1.5~5점만 오릅니다.
   - **탐색이 일찍 멈췄습니다.** Ant 기본 PPO 설정은 `entropy_coef = 0`입니다. 잡음 std가 0.46 → 0.05(1000 it) → 0.02(3000 it)로 줄고 학습률도 하한(1e-5)에 붙어, 1000 it 이후 학습 reward가 거의 오르지 않았습니다.
   - **높이맵이 발을 못 봅니다.** Ant의 발은 몸통에서 좌우 0.3~0.95 m에 닿는데, E의 높이맵(앞뒤 1.6 m × 좌우 1.0 m)은 접지 위치의 11~16%만 덮었습니다.
   - **학습 지형이 낮습니다.** 학습은 최대 0.08~0.10 m, 테스트는 0.13~0.30 m입니다.
7. **E2 (사후 탐색 2차): 세 가지를 함께 고쳤습니다.**
   - **넓은 높이맵:** 몸통 뒤 1 m ~ 앞 3 m, 좌우 ±1.25 m, 0.25 m 간격 187개입니다.
   - **엔트로피 보너스:** `entropy_coef` 0.005로 잡음이 0.4 안팎으로 유지됩니다.
   - **2단계 지형:** DR 지형에서 1500 it 학습한 뒤, 높이 범위를 넓힌 DR-Hard 지형에서 이어서 1500 it를 학습합니다.
   - **결과:**
     - E2 > E(사전 규칙)가 T1, T3, T4, T5와 모든 평균에서 성립합니다.
     - 거친 지형 속도가 4.2~5.4 m/s로 E의 약 1.7배입니다.
     - T5는 51.2 → 74.7, Grid 0.15 m 열은 47.4 → 70.9입니다. E의 한계였던 거친 지형이 크게 좋아졌습니다.
     - 평지도 116.5 → 139.2로 올랐습니다.
   - **무엇이 얼마나 기여했나** (seed 42 하나라 참고용, T1~T4 평균):

     ![e2](figures/e2_decomposition.png)

     | 단계 | T1~T4 평균 | T5 |
     |---|---|---|
     | E @1000 it | 62.9 | 53.5 |
     | + 엔트로피 @1000 it | 71.7 (+8.8) | 60.3 |
     | + 넓은 높이맵 @1000 it | 71.7 (+0.0) | 62.9 |
     | E2 1단계 끝 @1500 it | 76.6 | 69.7 |
     | 같은 지형으로 계속 @3000 it (E2c) | 79.0 | 67.2 |
     | **높은 지형으로 계속 @3000 it (E2)** | **93.9** (E2c 대비 +14.9) | **79.2** |
     | 참고: 테스트 형태로 계속 @3000 it (Oracle) | 92.8 | 70.9 |

     - 가장 큰 기여는 엔트로피(+8.8)와 2단계 높은 지형(+14.9)입니다.
     - 넓은 높이맵은 1000 it 시점의 쉬운 지형에서는 효과가 없었습니다. 높은 지형에서의 효과는 따로 나누지 못했습니다.
     - E2(seed 42)는 테스트 형태로 직접 학습한 Oracle과 T1~T4에서 같은 수준이고, 처음 보는 T5에서는 더 좋습니다(79.2 vs 70.9).
   - **영상:**
     - 계단 꼭대기 출발 (이전 영상과 같은 출발점): [A_ref](videos/T1_A_ref.mp4), [B](videos/T1_B_seed44.mp4)는 그대로 서 있습니다. [E](videos/T1_E_seed44.mp4)와 [E2](videos/T1_E2_seed42.mp4)는 1초 안에 계단을 내려가 끝까지 걷습니다.
     - T5 같은 출발점: [E](videos/T5_E_seed44.mp4)는 43 m 지점에서 넘어지고, [E2](videos/T5_E2_seed42.mp4)는 끝까지 116 m를 갑니다.
     - 평지: [E](videos/Flat_E_seed44.mp4), [E2](videos/Flat_E2_seed42.mp4)

![heatmap](figures/heatmap_friction_roughness.png)

![training](figures/training_curves.png)

**한계**

- E와 E2는 결과를 본 뒤 설계한 사후 실험입니다. 학습 전에 새 테스트 지형(T4, T5)을 정해 따로 검증했지만, 본 가설의 판정과는 구분해야 합니다.
- E2는 seed 편차가 큽니다(T1~T4 평균 93.9 / 72.5 / 83.2). 평지에서 넘어짐도 늘었습니다(5% → 14%). 더 빠른 걸음의 대가로 보입니다.
- T2(마찰 0.2)는 E2도 규칙상 E와 차이가 없습니다. 마찰에 민감한 감지(예: 발 미끄러짐 속도)는 다루지 못했습니다.
- E2 기여 분해는 seed 하나의 결과입니다.
- seed 3개라 통계적 검정력이 낮습니다.
- 넘어짐 판정 자체가 월드 z 기준이라, 지면보다 낮은 곳(구덩이)이 있는 지형에서는 잘 걸어도 종료될 수 있습니다.

## 5. 저장소에 추가한 것

원본 Isaac Lab 파일은 수정하지 않았습니다(루트 README 맨 위 안내 블록과 `.gitignore` 제외).

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust/   [신규] 과제 1 환경 패키지
├── __init__.py              task 등록 (학습 8, 평가 5, 테스트·분석 42)
├── env_cfgs.py              원본 AntEnvCfg 상속: 관측(C, D, E0, E, E2), 행동(D), 레이캐스트 센서, DR 학습, 테스트 지형
├── terrains.py              DR 학습 지형, DR-Hard(E2 2단계), T1~T5, Oracle(E2 천장 참고값), Grid, Switch
├── agents/rsl_rl_ppo_cfg.py PPO 설정 (3000 it, D는 1000 it·noise 0.2·lr 1e-4, E2는 단계마다 1500 it·entropy 0.005)
├── mdp/observations.py      반응 상태, 원본 관측 재계산, 기본 정책 행동, 지면 기준 높이
├── mdp/events.py            로봇 마찰 랜덤화 (학습 env 전용), T5 출발점 분산
├── mdp/models.py            고정 MLP, 동역학 모델, 이력·오차 버퍼
├── mdp/residual_action.py   D의 action term
└── weights/                 D용 기본 정책(B@2000 actor)과 동역학 모델 (seed별)
assignments/hw1_ant/                                                       [신규]
├── README.md, EVAL_COMMAND.txt
├── checkpoints/<조건>_seed<S>/model.pt (+ params/)    A, B, C, D, E0, E, E2 × seed 3개와 Oracle_seed42의 최종 가중치
├── scripts/                 실행·평가·분석 스크립트 (아래 6장)
├── results/                 summary.md, main_runs.csv, grid_runs.csv, submission.json, dynamics/, diagnosis/, raw/
└── figures/, videos/
```

## 6. 재현 방법

모든 Isaac 실행은 `scripts/isaac_run.sh`로 감쌌습니다. 이 래퍼는 conda 활성화, GPU 선택(`GPU=<번호>`), 출력 버퍼 해제, RAM 기반 동시 실행 제한을 처리합니다. 래퍼 없이 `./isaaclab.sh -p ...`로 실행해도 됩니다.

```bash
T=scripts/reinforcement_learning/rsl_rl/train.py
./isaaclab.sh -p $T --task Isaac-Ant-v0 --headless --seed 42 --max_iterations 3000 --run_name A_seed42   # A
./isaaclab.sh -p $T --task Isaac-Ant-DR-v0 --headless --seed 42 --run_name B_seed42                       # B
./isaaclab.sh -p $T --task Isaac-Ant-Hist-DR-v0 --headless --seed 42 --run_name C_seed42                  # C
bash assignments/hw1_ant/scripts/pipeline_d.sh 42                                                         # D (B@2000 필요)
./isaaclab.sh -p $T --task Isaac-Ant-RelHeight-DR-v0 --headless --seed 42 --run_name E0_seed42            # E0
./isaaclab.sh -p $T --task Isaac-Ant-Scan-DR-v0 --headless --seed 42 --run_name E_seed42                  # E
python assignments/hw1_ant/scripts/pipeline_e2.py      # E2 1단계 → 2단계(재개 학습), 진단 run, Oracle, 평가까지 한 번에
python assignments/hw1_ant/scripts/sweep_eval.py jobs --out jobs.jsonl                                    # 평가 목록
python assignments/hw1_ant/scripts/run_queue.py jobs.jsonl --gpus 0                                       # 평가 실행
python assignments/hw1_ant/scripts/sweep_eval.py aggregate && python assignments/hw1_ant/scripts/plots.py # 표·그림
python assignments/hw1_ant/scripts/select_submission.py                                                   # 제출 선택
```

E2를 손으로 돌릴 때는 1단계(`Isaac-Ant-WideScan-DR-v0`, 1500 it) 뒤에 2단계를 재개 학습으로 실행합니다.

```bash
./isaaclab.sh -p $T --task Isaac-Ant-WideScan-DR-v0 --headless --seed 42 --run_name E2s1_seed42
./isaaclab.sh -p $T --task Isaac-Ant-WideScan-DRHard-v0 --headless --seed 42 --run_name E2_seed42 \
  --resume --load_run <E2s1_seed42 폴더 이름 전체> --checkpoint model_1499.pt
```

## 7. 규칙 준수

- **평가 reward 식:** 바꾸지 않았습니다. 원본 7개 항을 그대로 상속합니다.
- **로봇 USD:** 수정하지 않았습니다.
- **물성:** 학습 환경에서만 로봇 마찰을 랜덤화했습니다. 평가 task는 로봇 물성을 바꾸지 않습니다.
- **추가한 센서:** 레이캐스트로 지형 형태만 측정합니다(관측 구성은 과제에서 허용).
- **제출 모델 선택:** 결과를 보기 전에 정한 규칙을 따랐습니다. E2는 "T1~T4 평균에서 규칙상 E2 > E이고 T5에서 규칙상 E2 < E가 아님"을 만족해 E를 대체했습니다. seed는 학습 reward로 골랐습니다(테스트 지형 미사용).
- **Oracle:** 테스트 지형과 같은 형태로 학습한 "천장 참고값"입니다. 점수 해석에만 쓰고 제출 후보에서 뺐습니다.
- **코드 작성:** 논문과 공개 프로젝트는 아이디어만 참고했고 코드는 직접 작성했습니다.

## 8. 참고 문헌 (아이디어만 참고)

- J. Lee et al., "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, 2020. DR 지형 학습
- N. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL," *CoRL*, 2021. 병렬 PPO, 지형 생성기, 높이맵 관측, 지형 커리큘럼
- J. He et al., "Attention-based map encoding for learning generalized legged locomotion," *Science Robotics*, 2025. 발 디딜 곳을 보는 지형 지도
- A. Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots," *RSS*, 2021. 이력으로 숨은 환경 정보 추정
- I. M. Aswin Nahrendra et al., "DreamWaQ," *ICRA*, 2023. 이력 기반 추정과 신뢰도 조절
- J. Long et al., "Hybrid Internal Model (HIM)," *ICLR*, 2024. 명령-반응 이력
- T. Silver et al., "Residual Policy Learning," arXiv:1812.06298, 2018. 기본 제어기 위 residual 보정
- T. Johannink et al., "Residual Reinforcement Learning for Robot Control," *ICRA*, 2019.
