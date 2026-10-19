# vla-success-vs-ppo

**성공 중심 온라인 학습은 언제 VLA 강화학습을 대체할 수 있는가** — 데이터 갱신, 음의 정책 업데이트, 회복 행동에 대한 통제 연구

같은 출발 정책과 같은 온라인 상호작용 예산에서, 로봇이 스스로 굴린 롤아웃 중 성공한 것만 따라 하는 학습(A2⁺)이 terminal PPO(A3⁰)의 실행 OOD 성능을 어디까지 재현하는지 사전 고정된 비열등성 검정으로 판정한다. 출발 연구는 *What Can RL Bring to VLA Generalization?* (NeurIPS 2025, [gen-robot/RL4VLA](https://github.com/gen-robot/RL4VLA))이다.

## 현재 상태

- 연구 명세: [docs/vla_rl_decomposition_package_v0_12.md](docs/vla_rl_decomposition_package_v0_12.md) (최신, 6차 외부 검토 반영). 이전 판 v0.8~v0.11도 같은 폴더에 있다.
- 외부 검토: 요청서 [docs/review_request_v0_11.md](docs/review_request_v0_11.md), 대응표는 명세 부록 E-8. 신규성 확인 [pilot/G0_check_2026-10-19.md](pilot/G0_check_2026-10-19.md).
- 단계: G1 통과 (조건부, 2026-10-19) = G2 진입 승인 (본실험 착수 승인은 G-PRE). 종합 보고 [pilot/G1_report.md](pilot/G1_report.md), 이탈 기록 [pilot/deviations.md](pilot/deviations.md).
- 다음: G2 (H200 실측, 64 환경 전체 규모 확인, 생성–학습 잔여 불일치·엄격 결정성·S_rec 타당성 확인, Linux 경로의 공개 체크포인트 재평가) → G-PRE → E1.

## 사전 고정 기록

판정 기준과 임계값은 결과를 보기 전에 이 저장소의 커밋으로 고정한다. 사전 고정 시점마다 태그를 남기고, 이후 변경은 문서의 이탈 기록에 날짜·사유와 함께 적는다.

## 저장소에 넣지 않는 것

모델 가중치, 체크포인트, 롤아웃·평가 원자료, 실행 로그는 용량 때문에 `.gitignore`로 제외한다. 대신 커밋 해시, 설정, 해시값을 기록한다 (문서 9.3절).
