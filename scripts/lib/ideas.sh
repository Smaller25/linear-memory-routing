# shellcheck shell=bash
# 아이디어 이름 → config 이름(configs/<이름>_<SIZE>.env)과 한 줄 설명. run_idea.sh·run_pack.sh 가 같이 쓴다.
declare -A CFG=(
  [vanilla]=vanilla
  [mc]=mc
  [ce]=mc_ce
  [sepgate]=mc_ce_sepgate
  [maxsim]=mc_ce_sepgate_maxsim
  [share]=mc_ce_sepgate_maxsim_share
  [detach]=mc_ce_detach
  [detach_maxsim]=mc_ce_detach_maxsim
  [maxsim_share_nolabel]=mc_maxsim_share
  [chained]=mc_chained
  [chunk64]=mc_chunk64
  [dense_chained]=mc_dense_chained
)
declare -A DESC=(
  [vanilla]="설정 1. 기준선: 과거 state 저장 없는 GDN-2"
  [mc]="설정 2. MC 원본(256토큰 구간, 평균 요약, top-4, 게이트 하나)"
  [ce]="설정 3. 아이디어 a: span 복사 정답으로 라우터 CE(후보 전부에 기울기)"
  [sepgate]="설정 4. a + 아이디어 c: 고르는 점수와 섞는 비율 분리"
  [maxsim]="설정 5. a + c + 아이디어 b 일부: 32토큰 하위 블록 최댓값으로 점수"
  [share]="설정 5 + 아이디어 b 나머지: 0층이 고른 구간을 위층 전부가 읽음"
  [detach]="설정 4 의 라우터 분리판: 전용 W_q/W_k, detach 입력, cosine x 온도(CE 가 backbone 에 안 닿음)"
  [detach_maxsim]="detach + 32토큰 하위 블록 최댓값으로 점수(설정 5 의 라우터 분리판)"
  [maxsim_share_nolabel]="아이디어 b 단독: max-sim + 층 공유, 라벨 없이 LM 손실만"
  [chained]="아이디어 d: MC 원본 + 끊지 않는 압축기(구간 state 가 앞에서 이어짐)"
  [chunk64]="아이디어 e: MC 원본 + 64토큰 구간"
  [dense_chained]="설정 6(MARCH식): 끊지 않는 압축기 + 과거 구간 전부 섞기"
)
ORDER="vanilla mc ce sepgate maxsim share detach detach_maxsim maxsim_share_nolabel chained chunk64 dense_chained"
