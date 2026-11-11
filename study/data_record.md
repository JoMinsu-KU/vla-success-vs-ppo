# 본실험 데이터 기록 (A1·W2 시연)

- 생성: 2026-11-10~11, H200 인스턴스, `study/data_pipeline.sh` (코드 `e1f0d9b`). 로그 `~/vla_main_data/data_pipeline.log`.
- 설계: [A1_design.md](A1_design.md) 1절, [W2_design.md](W2_design.md) 1절.
- 시연 생성: A1 = `collect_simpler --num_traj 16400 --seed=100 --num_procs 4` (16,400개), W2 = `--num_traj 2048 --seed=200 --num_procs 4` (2,048개, 5,138초). 성공 궤적만 저장된다.
- A1 시연 생성은 처음에 프로세스 16개로 시작했다가 컨테이너 메모리 상한(26 GiB) 문제로 지우고 4개로 다시 만들었다 (2026-11-10 12:12 → 13:11, 운영 기록 `public/progress.log`). 아래는 4개로 만든 집합이다.
- 작은 행동 제거 비율 38% (A1 학습: 196,906 / 513,262). 명세 C3의 "약 1/3"과 맞다.
- 위치: `~/vla_main_data/` (A1 310 GB, W2 66 GB). 지우지 않는다.

| 집합 | 폴더 | 에피소드 | 프레임 | 제거 / 전체 행동 | SHA-256 파일 목록 | SHA-256 토큰 | SHA-256 프레임 |
|---|---|---|---|---|---|---|---|
| A1 학습 | `a1/train` | 16,384 | 315,892 | 196,906 / 513,262 | `0999b5189991aaacbf4169f16fec2f44ff3f5a108545b0f50fd2e3ff53437c94` | `ca0c4e28e3a7286587f45c75854228b15a6d518c404ba9d1a5a9942d8c7757dd` | `560e6d08582b66d6fc87b89d69b83257bc0f9a01e0068720f6096bb8f144fea5` |
| A1 검증 | `a1/val` | 16 | 306 | 190 / 496 | `8dd3dbd006ffa0b7d90f3694f13107d3d7972e39372fee199176f1eb3db25810` | `b5761a168cb5814510753170ad6193eef80ce72012d477f52d1dce12ab8c59e2` | `2d39ed59be42ae7a7f4f401cd1b85bac6868a8a31a7c42727d78072d1fc76aa9` |
| W2 n256 | `w2/prep/n256` | 256 | 4,933 | 3,052 / 8,009 | `42f9ffd5b213fd4f19920773edfaa029c662a824f7d1cd7aec0df1f1c1ad9254` | `8ab2b34410400399a5ce6c949553d984c49272ef8130631c6bed3311bcb15e90` | `ab10e7670bd0da8fef1b6a5d71678afc2f3f0eb5b645f47806191a6b1a50d241` |
| W2 n512 | `w2/prep/n512` | 512 | 9,868 | 6,137 / 16,039 | `540bfd4d52e3c0ef2c86460f09fd29e376be904237481ec3615bdfccd7731133` | `bf0d5794a8b0abc470abca61a16be3f646d9f8ed3c4a7676f807dda07f3ce656` | `b0ae6100d6c2f90815dc3cd931e87d7ba6e6b140b726504dbcd7d8ed84bb3c69` |
| W2 n1024 | `w2/prep/n1024` | 1,024 | 19,624 | 12,272 / 31,950 | `ec4c9a41ea1f5bb4450ff77da542a3b3e2671d6f6dd352e5af1e782b950343a5` | `cb7201347423101c386bc4a0e99360a9bd2bb773cfc8b47c97a13ab8bfdb7804` | `e594a7da18821317ddfba29e9748c3f4f34b719b7f4d077d357f8d06022a6f44` |
| W2 n2048 | `w2/prep/n2048` | 2,048 | 39,290 | 24,535 / 63,879 | `104a87ffda2bf36e899c5a6bb84588882a60c4fb924ce5fcfe99b49b8cf0ef60` | `323d4a56785fb901f49ba5cf0dc4ffd7050f45c05a957529c39303d8af9358b7` | `97cecf85e2afdc20faa90ce3c617352dfdbf3e7346007de50389ad69ddf1d18c` |
