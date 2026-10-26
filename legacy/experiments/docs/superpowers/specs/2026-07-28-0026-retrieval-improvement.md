GDN2 State-Retrieved Linear Attention (SRLA) Pilot System

> **rev2 정정 (2026-07-28, 사용자 지시). 아래 두 항목이 rev1 본문보다 우선한다.**
> 1. **`L_c = 2048`은 chunk 크기가 아니라 파일럿의 전체 context 길이다.** chunk 크기는
>    별도 하이퍼파라미터 `L_chunk`(설정값, 기본 256 — 학습된 GDN2 state 입도와 일치)이며,
>    context 2048 → chunk 8개. 평가는 §5대로 1k~16k로 확장한다. 코드에 2048을 chunk로
>    하드코딩 금지.
> 2. **Router query는 chunk 평균이 아니라 입력 토큰별(per-token)로 계산한다.**
>    `q_t = W_q · q_t^{proj}` (매 입력 토큰 t마다), 선택·가중·융합도 토큰별로 수행한다.
>    §2B의 `\bar q_n` (chunk 평균 query) 정의는 **폐기**한다. 아래 §2B-rev2 참조.

## 2B-rev2. Per-token router (rev1 §2B를 대체)

Descriptor는 chunk 단위로 유지한다 (chunk당 1개):

$$D_m = W_{desc} \cdot \bar k_m, \qquad \bar k_m = \frac{1}{|C_m|}\sum_{t \in C_m} k_t$$

Query는 **모든 입력 토큰 t**에서:

$$q_t = W_q \cdot q_t^{proj}, \qquad
\text{Sim}(q_t, D_m) = \frac{\langle q_t, D_m\rangle}{\lVert q_t\rVert_2 \lVert D_m\rVert_2 \cdot \tau}$$

$$\mathcal{K}_t = \mathrm{Top\text{-}K}_{\,m \,<\, \mathrm{chunk}(t)}(\text{Sim}(q_t, D_m)), \qquad
\alpha_{t,m} = \mathrm{Softmax}_{m \in \mathcal{K}_t}(\text{Sim}(q_t, D_m))$$

$$S^{(t)}_{\text{fused}} = S^{(t)}_{\text{curr}} + \sum_{m \in \mathcal{K}_t} \alpha_{t,m}\,\Phi_{\text{align}}(S_m)$$

인과성: 토큰 t는 **자기 chunk보다 앞선 완결 chunk만** 선택할 수 있다
(`m < chunk(t)`; 현재/미래 chunk 금지).

**구현 결정 사항 (사용자 확인 필요)**: state 가산 융합을 토큰별로 하면 융합 결과는
그 토큰의 읽기에만 쓰이고 backbone의 진행 중인 recurrent state는 오염시키지 않는 것이
자연스럽다(= 토큰별 가산 read term). 이를 기본으로 하고, chunk 경계에서 융합 결과를
실제 state에 영구 반영하는 변형은 플래그(`--persist-fusion`)로 둔다.

1. Project Overview & ObjectivesGoal: Build an efficient State-Retrieved Linear Attention (SRLA) module for long-context 30B GDN2/Linear Attention models.Core Hypothesis: Instead of expanding context length via text concatenation (Standard RAG), retrieving and fusing past chunk states ($S_m$) directly into the current recurrent state ($S_{curr}$) will dramatically reduce Time-to-First-Token (TTFT) and solve context forgetting without quadratic attention costs.Pilot Strategy: Freeze the 30B backbone LLM completely. First, implement and fine-tune a baseline using Average Pooling Key Descriptors, then build an extensible interface for Non-average Hybrid Descriptors and Contrastive State Routing (CSR).2. Mathematical FormulationsA. Context Segmentation & State RepresentationSequence is chunked into non-overlapping segments $C_1, C_2, \dots, C_M$ of size $L_c$ (default: 2048 tokens).Each chunk $m$ generates a GDN2 Recurrent State $S_m \in \mathbb{R}^{d_k \times d_v}$.B. Pilot Baseline Router (Average Pooling)Key Descriptor ($D_m$):$$D_m = W_{desc} \cdot \bar{k}_m \quad \text{where } \bar{k}_m = \frac{1}{L_c} \sum_{t \in C_m} k_t \in \mathbb{R}^{d_k}$$Query Vector ($q_n$):$$q_n = W_q \cdot \bar{q}_n \quad \text{where } \bar{q}_n = \frac{1}{L_c} \sum_{t \in C_n} q_t \in \mathbb{R}^{d_k}$$Similarity & Selection:$$\text{Sim}(q_n, D_m) = \frac{\langle q_n, D_m \rangle}{\Vert{}q_n\Vert{}_2 \Vert{}D_m\Vert{}_2 \cdot \tau}$$$$\mathcal{K}_n = \mathrm{Top\text{-}K}_{m < n}(\text{Sim}(q_n, D_m)), \quad \alpha_{n, m} = \mathrm{Softmax}_{m \in \mathcal{K}_n}(\text{Sim}(q_n, D_m))$$State Fusion:$$S_{\text{fused}} = S_{\text{curr}} + \sum_{m \in \mathcal{K}_n} \alpha_{n, m} \cdot \Phi_{\text{align}}(S_m)$$C. Advanced Upgrade Path (Phase 2 Interface)Hybrid Non-average Descriptor: $D_m = W_{desc} \cdot \left[ k_m^{\text{energy}} \;\mathbin{\Vert}\; k_m^{\text{max}} \;\mathbin{\Vert}\; h_m^{\text{state}} \right]$Contrastive State Loss ($\mathcal{L}_{\text{CSR}}$): InfoNCE Loss between Query $q_n$ and target state $S_{m^+}$.3. Required Module ArchitectureImplement the following modular PyTorch components:StateCacheManager:Stores historical chunk states $S_1, S_2, \dots, S_{M}$ and key descriptors $D_1, D_2, \dots, D_M$.Provides CPU/GPU offloading and fast top-$K$ indexing.AverageKeyDescriptor / HybridKeyDescriptor:Base class BaseStateDescriptor.Computes chunk-level descriptors with $O(L_c)$ complexity.StateRouter:Contains trainable projections $W_{desc}$, $W_q$, and alignment module $\Phi_{\text{align}}$.Computes top-$K$ cosine similarity and returns fused state $S_{\text{fused}}$.GDN2StateFusionWrapper:Wraps the target 30B model's linear attention layers.Intercepts layer states during prefill/decoding and injects $S_{\text{fused}}$.4. Training Pipeline & Loss FunctionsFrozen-Backbone 2-Stage TrainingBackbone: 30B Pretrained Model (FROZEN).Trainable Parameters: $W_{desc}$, $W_q$, $\Phi_{\text{align}}$ (and optional LoRA weights on Key/Value projections).Python# Stage 1: Router Alignment (Loss Options)
# Option A: Unsupervised LM Loss
loss_lm = cross_entropy(logits, targets)

# Option B: Supervised Needle-in-a-Haystack Router Loss
loss_router = F.cross_entropy(sim_scores / tau, target_chunk_indices)

# Total Loss
loss = loss_lm + lambda_csr * loss_router
5. Evaluation BenchmarksPasskey / Needle-in-a-Haystack (1k~16k): Measure retrieval accuracy vs. context length.Full-Window Perplexity: Evaluate language modeling performance across entire sequence (avoiding last-100 token bias).Latency & TTFT Benchmark: Measure Time-To-First-Token comparison between Standard Text RAG vs. State Retrieval.