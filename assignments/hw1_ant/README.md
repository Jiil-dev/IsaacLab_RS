# 실습 과제 1: 처음 보는 환경에서도 잘 걷는 Ant

**한눈에 보기**

- 제공된 PPO baseline은 처음 보는 지형에서 거의 걷지 못합니다(unseen 평균 25.3). 진단해 보니 원인은 **"높은 곳에서 출발하면 얼어붙는 것"** 이었습니다. 관측의 몸통 높이가 월드 z 절대값이라 학습 때 본 적 없는 입력이 되기 때문입니다.
- 원래 가설인 "명령-반응 이력으로 보정"(조건 C, D)은 같은 학습량에서 추가 이득이 없었습니다.
- 진단을 바탕으로 사후 탐색 실험 **E**를 했습니다. 몸통 높이를 지면 기준으로 바꾸고 주변 지형 높이맵(레이캐스트)을 관측에 넣었습니다.
- 그 결과 unseen 평균이 **25.3 → 63.0**(2.5배)이 됐습니다. 처음 보는 형태 T1은 **7.3 → 39.6**, T3는 **7.7 → 48.1**입니다.
- **제출 모델: E (seed 44).**

| 조건 | Flat | T1 형태 | T2 μ 0.2 | T3 형태+μ 0.4 | T4 새 형태 | unseen (T1~T3) | unseen4 (T1~T4) |
|---|---|---|---|---|---|---|---|
| A_ref: 제공 baseline (1000 it) | 134.9 | 7.3 | 60.8 | 7.7 | 9.2 | 25.3 | 21.2 |
| A: 평지 학습 | 141.0 ± 4.3 | 9.8 ± 1.7 | 62.8 ± 13.4 | 10.7 ± 2.2 | 19.0 ± 8.2 | 27.8 ± 4.5 | 25.6 ± 4.4 |
| B: + 지형·마찰 DR | 110.1 ± 12.7 | 22.5 ± 3.7 | 111.4 ± 11.9 | 26.7 ± 4.7 | 46.1 ± 7.2 | 53.5 ± 4.5 | 51.7 ± 5.0 |
| C: B + 명령-반응 이력 입력 | 84.1 ± 20.4 | 19.1 ± 3.3 | 84.2 ± 22.8 | 26.9 ± 1.9 | 30.5 ± 6.7 | 43.4 ± 6.8 | 40.2 ± 5.8 |
| D: B@2000 + 예측 오차 기반 residual 보정 | 110.5 ± 11.3 | 22.9 ± 3.0 | 110.3 ± 12.9 | 26.9 ± 5.8 | 46.5 ± 9.3 | 53.4 ± 5.3 | 51.7 ± 4.8 |
| E0: B + 지면 기준 높이 (사후 탐색) | 102.3 ± 2.4 | 35.6 ± 0.9 | 104.1 ± 4.6 | 46.9 ± 2.7 | 41.7 ± 3.3 | 62.2 ± 1.2 | 57.1 ± 1.5 |
| **E: E0 + 지형 높이맵 (사후 탐색, 제출)** | **116.5 ± 5.5** | **39.6 ± 2.7** | 101.3 ± 10.7 | **48.1 ± 2.7** | 46.4 ± 4.3 | **63.0 ± 3.5** | **58.8 ± 3.6** |

- 값은 평가 reward(공식 `play_one_episode.py`와 같은 누적 규칙)입니다. 각 정책을 100 env로 평가해 평균을 내고, seed 3개(42/43/44)의 평균 ± 표준편차를 적었습니다.
- 평가 조건: seed 24, 잡음 없는 정책입니다.
- 전체 표와 넘어짐 비율, 전진 거리, 사전 등록 비교는 [results/summary.md](results/summary.md)에 있습니다.

## 1. 평가 방법 (조교님용)

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-Scan-v0 \
  --seed 24 \
  --num_envs 100 \
  --checkpoint assignments/hw1_ant/checkpoints/E_seed44/model.pt \
  --video \
  --video_length 960
```

- 같은 내용이 [EVAL_COMMAND.txt](EVAL_COMMAND.txt)에 있습니다.
- **task와 지형:** `Isaac-Ant-Scan-v0`는 원본 `Isaac-Ant-v0` 설정을 상속하고 레이캐스트 센서만 추가한 task입니다.
  - 지형, reward, 종료 조건, 에피소드 길이, 로봇은 원본과 같습니다.
  - 원본 `source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_env_cfg.py`의 지형(`MySceneCfg.terrain`)과 지형 물성만 바꾸면 이 task도 그 지형에서 평가됩니다.
  - 센서는 Isaac Lab 지형의 기본 경로인 `/World/ground`를 읽습니다. 평면 지형과 생성 지형 모두 동작합니다.
- **저장소 위치:** conda env `lerobot-arena`의 Isaac Lab은 `~/IsaacLab_RS/source`를 가리킵니다. 이 저장소를 `~/IsaacLab_RS`에 받아 주세요.
  - 다른 위치에 받았다면 명령 앞에 `PYTHONPATH=<저장소>/source/isaaclab_tasks:$PYTHONPATH`를 붙이면 그 위치의 task가 쓰입니다.
- **원본 task 이름(`Isaac-Ant-v0`)으로만 평가해야 하는 경우:** 원본 관측 그대로인 조건 B를 쓰면 됩니다.
  - `--task Isaac-Ant-v0 --checkpoint assignments/hw1_ant/checkpoints/B_seed44/model.pt`
  - unseen 평균은 53.5로 E(63.0)보다 낮습니다.

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
- **공정성:** 모든 조건은 PPO 3000 iteration입니다(D는 B의 2000 + 1000). reward, PPO 설정, 관측 정규화(끔)가 같고, B·C·D·E는 같은 학습 지형을 씁니다.
- **사전 등록:** 실험 전에 가설, 수치 예측, 비교 규칙("평균 차 > 표준편차이면서 seed 3쌍 모두 같은 방향")을 정하고 동결했습니다. E는 결과를 본 뒤의 **사후 탐색 실험**이라, 학습 전에 별도 계획(가설 H5, 예측, 새 테스트 지형 T4, 확장 제출 규칙)을 다시 동결했습니다.

## 3. 학습 환경과 자체 "처음 보는 환경"

**학습 환경 (B, C, D, E 공통):**

- 8 m 타일 24행 × 8열(+x 192 m)이고 타일마다 종류와 난이도가 랜덤입니다.
- 구성은 평지 20%, 요철 30%(0.01~0.08 m), 파형 20%(0~0.08 m), 낮은 장애물 15%(0.02~0.10 m), 완만한 경사 15%입니다.
- 마찰은 env마다 0.3~1.2이고, 학습 env에서만 로봇 재질로 랜덤화했습니다.
- 넘어짐 판정이 월드 z 기준이라 모든 지형 높이를 0 이상으로 만들었습니다(학습된 Ant의 몸통은 0.38~0.50 m에서 걷습니다).

**처음 보는 환경 (학습에 사용하지 않음):**

| 환경 | 내용 |
|---|---|
| Flat | 원본 평지 (마찰 1.0) |
| T1 | 학습에 없던 피라미드 계단, 박스 격자, 더 큰 요철. 출발점의 60%가 0.5 m 안팎의 높은 곳 |
| T2 | 평지, 마찰 0.2 (학습 범위 밖) |
| T3 | T1 지형, 마찰 0.4 |
| T4 | (E 학습 전에 정의) 솟은 상자(0.22~0.30 m) 위 출발, 레일, 원기둥 장애물 |

- T1~T3는 A_ref만 보고 "평지의 50% 이하"가 되도록 정했고, 이후에는 바꾸지 않았습니다.
- 분석용 환경도 있습니다. Grid(요철 높이 × 마찰 heatmap)와 Switch(5초에 마찰 1.0 → 0.25)입니다.

## 4. 결과와 분석

![main](figures/main_comparison.png)

1. **H1 지지.** B는 A보다 T1·T2·T3·T4에서 모두 좋습니다(사전 규칙 충족). 대가로 평지 점수가 22% 낮아지는 "타협형 걸음"이 됩니다.
2. **H2 기각.** C는 B보다 낮습니다. 입력이 8.5배(510)로 커졌는데 학습량은 같아 학습이 느리고 seed 편차가 컸습니다.
3. **H3, H4: D는 B와 차이가 없습니다**(D와 출발점 B@2000도 차이 없음).
   - α는 평지에서 0.02로 꺼지고, 처음 보는 형태(T1·T3)에서 0.2, T4에서 0.36으로 켜집니다.
   - 그러나 마찰 변화(T2, 마찰 전환)에는 거의 반응하지 않았습니다. 동역학 예측 오차가 미끄러짐보다 지형 충격에 반응했기 때문입니다.
   - 낯선 지형에서 실제 적용된 보정량은 기본 행동의 12~16%였지만 성능을 바꾸지 못했습니다.
4. **진단: 높은 곳에서 출발하면 얼어붙습니다.**
   - T1에서 출발 높이와 전진 거리의 상관은 모든 정책에서 −0.73 ~ −0.83입니다.
   - 계단 꼭대기(0.5~0.6 m)에서 출발한 B는 16초 동안 0 m를 갑니다.
   - D의 α는 얼어붙은 env에서 오히려 낮습니다(0.16~0.22, 걷는 env는 0.36~0.41). 멈춘 몸은 예측하기 쉬워서 보정이 꺼집니다.

   ![spawn](figures/t1_spawn_height.png)

5. **E (사후 탐색): 지면 기준 높이와 지형 높이맵.**
   - B와 비교해 T1 +76%, T3 +80%이고, 평지는 오히려 회복했습니다(110 → 117).
   - T4는 B와 같고, 마찰 0.2(T2)는 약간 낮습니다.
   - Grid(아래 heatmap)에서 가장 거친 열(요철 0.15 m, 학습 최대 0.10 m 초과)은 E가 B보다 낮습니다(마찰 5개 평균 47.4 vs 55.9, 사전 규칙상 E < B). 나머지 열은 규칙상 차이가 없습니다.
   - 지면 기준 높이만 바꾼 E0도 T1·T3에서 크게 좋아집니다. 개선의 대부분은 "얼어붙음 해소"에서 나왔다는 뜻입니다.
   - 같은 출발점(T1 계단 꼭대기) 영상입니다.
     - [A_ref](videos/T1_A_ref.mp4), [B](videos/T1_B_seed44.mp4): 그대로 서 있습니다.
     - [E](videos/T1_E_seed44.mp4): 계단을 내려가 끝까지 걷습니다.
     - [E 평지](videos/Flat_E_seed44.mp4)

![heatmap](figures/heatmap_friction_roughness.png)

**한계**

- E는 결과를 본 뒤 설계한 사후 실험입니다. T4로 따로 검증했지만, 본 가설의 판정과는 구분해야 합니다.
- E의 이득은 주로 "얼어붙음 해소"에서 왔습니다. 학습 범위를 넘는 거친 지형(T4의 0.22~0.30 m 상자, Grid 0.15 m 열)에서는 B보다 낫지 않습니다.
- seed 3개라 통계적 검정력이 낮습니다.
- 넘어짐 판정 자체가 월드 z 기준이라, 지면보다 낮은 곳(구덩이)이 있는 지형에서는 잘 걸어도 종료될 수 있습니다.
- 마찰에 민감한 감지(예: 발 미끄러짐 속도)는 다루지 못했습니다.

## 5. 저장소에 추가한 것

원본 Isaac Lab 파일은 수정하지 않았습니다(루트 README 맨 위 안내 블록과 `.gitignore` 제외).

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant_robust/   [신규] 과제 1 환경 패키지
├── __init__.py              task 등록 (학습 5, 평가 4, 테스트·분석 30)
├── env_cfgs.py              원본 AntEnvCfg 상속: 관측(C, D, E0, E), 행동(D), 레이캐스트 센서, DR 학습, 테스트 지형
├── terrains.py              DR 학습 지형, T1~T4, Grid, Switch
├── agents/rsl_rl_ppo_cfg.py PPO 설정 (3000 it, D는 1000 it·noise 0.2·lr 1e-4)
├── mdp/observations.py      반응 상태, 원본 관측 재계산, 기본 정책 행동, 지면 기준 높이
├── mdp/events.py            로봇 마찰 랜덤화 (학습 env 전용)
├── mdp/models.py            고정 MLP, 동역학 모델, 이력·오차 버퍼
├── mdp/residual_action.py   D의 action term
└── weights/                 D용 기본 정책(B@2000 actor)과 동역학 모델 (seed별)
assignments/hw1_ant/                                                       [신규]
├── README.md, EVAL_COMMAND.txt
├── checkpoints/<조건>_seed<S>/model.pt (+ params/)    A, B, C, D, E0, E × seed 3개의 최종 가중치
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
python assignments/hw1_ant/scripts/sweep_eval.py jobs --out jobs.jsonl                                    # 평가 목록
python assignments/hw1_ant/scripts/run_queue.py jobs.jsonl --gpus 0                                       # 평가 실행
python assignments/hw1_ant/scripts/sweep_eval.py aggregate && python assignments/hw1_ant/scripts/plots.py # 표·그림
python assignments/hw1_ant/scripts/select_submission.py                                                   # 제출 선택
```

## 7. 규칙 준수

- **평가 reward 식:** 바꾸지 않았습니다. 원본 7개 항을 그대로 상속합니다.
- **로봇 USD:** 수정하지 않았습니다.
- **물성:** 학습 환경에서만 로봇 마찰을 랜덤화했습니다. 평가 task는 로봇 물성을 바꾸지 않습니다.
- **추가한 센서:** 레이캐스트로 지형 형태만 측정합니다(관측 구성은 과제에서 허용).
- **코드 작성:** 논문과 공개 프로젝트는 아이디어만 참고했고 코드는 직접 작성했습니다.

## 8. 참고 문헌 (아이디어만 참고)

- J. Lee et al., "Learning quadrupedal locomotion over challenging terrain," *Science Robotics*, 2020. DR 지형 학습
- N. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL," *CoRL*, 2021. 병렬 PPO, 지형 생성기, 높이맵 관측
- A. Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots," *RSS*, 2021. 이력으로 숨은 환경 정보 추정
- I. M. Aswin Nahrendra et al., "DreamWaQ," *ICRA*, 2023. 이력 기반 추정과 신뢰도 조절
- J. Long et al., "Hybrid Internal Model (HIM)," *ICLR*, 2024. 명령-반응 이력
- T. Silver et al., "Residual Policy Learning," arXiv:1812.06298, 2018. 기본 제어기 위 residual 보정
- T. Johannink et al., "Residual Reinforcement Learning for Robot Control," *ICRA*, 2019.
