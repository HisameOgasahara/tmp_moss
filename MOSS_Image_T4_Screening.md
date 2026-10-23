# MOSS-Image: 모델 정보와 Colab T4 테스트 후보

2026-10-23 기준. **무료 Colab T4에서 후보별 스모크 테스트 → 통과한 구성만 이미지 생성.** 목표는 1024×1024, batch 1이다.

## 모델

`프롬프트 → Qwen3.5-9B → 텍스트 어댑터 → 이미지·텍스트 joint DiT → FLUX.2 VAE → 이미지`

| 항목 | 사양 |
|---|---|
| 공개·라이선스 | OpenMOSS, 2026-10-16, Apache-2.0 |
| 체크포인트 | Base: 사전학습 모델. SFT: 정제 데이터로 미세조정한 생성용 후보 |
| 본체 | 5.08B, single-stream DiT 42블록, hidden 3072, FFN 8192 |
| Attention·FFN | 24 heads × 128차원, 4축 RoPE, SiTU-GLU |
| 텍스트 | Qwen3.5-9B의 마지막 층 4096차원 특징, 최대 4096토큰 |
| 텍스트 어댑터 | 4096→3072 투영 + 텍스트 블록 2개 |
| 잠재공간 | FLUX.2 VAE의 32채널·8배 축소를 2×2 packing하여 128채널·16배 축소 |
| 1024² 입력 | DiT `[1,128,64,64]`, 이미지 토큰 4096개 |
| 조건 | 자연어·태그·Layout JSON. Layout은 중심 원점 절대 좌표 `yxyx` |
| 기본 생성 | Euler, 50 step, CFG 3, shift 5, 음성 분기 block 22 skip |

학습은 **DiT를 처음부터 학습하고 인코더·VAE를 고정**한다. ReLAION·OmniCorpus·DataComp 등 약 50억 이미지를 중복·AI 이미지 제거 후 약 30억으로 구성한다. Gemma 4의 짧은·중간·긴 재캡션과 문자·박스·줄바꿈 조건을 사용하며 원본/재캡션 비율은 20:80이다. 해상도·문맥 길이를 3단계로 늘려 최종 1024²·4096토큰, 310,000 step EMA를 사용한다. 손실은 flow matching MSE, 옵티마이저는 Muon–AdamW, 대표 설정은 LR 5e-5·EMA 0.9999다.

[제작팀](https://github.com/OpenMOSS/MOSS-Image) · [기술보고서](https://arxiv.org/html/2610.23374v1) · [HF SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)

## 코드에서 알아둘 점

- **ComfyUI 내장 경로:** `UNETLoader` + `CLIPLoader(type=moss_image)` + `VAELoader`, 잠재값은 `EmptyFlux2LatentImage`. Comfy-Org 가중치의 fused QKV를 지원한다.
- **제작팀 HF 플러그인:** 별도 로더·모델 구현이다. 내장 모델은 timestep multiplier 1·shift 5이므로 구형 플러그인의 `ModelSamplingSD3`를 그대로 섞지 않는다.
- **T4 dtype:** SM75. INT8 layout 최소 SM75, W4A8 빠른 경로 최소 SM80. MOSS 기본 inference dtype 목록은 BF16·FP32이므로 자동 선택과 `--fp16-unet`을 별도 후보로 시험한다.
- **DiffSynth-Studio:** FlowMatchScheduler, full/LoRA 학습과 disk offload·EntroPack 구현 제공. LoRA 예제는 rank 32·LR 1e-4·gradient checkpointing이다.

[ComfyUI 본체](https://github.com/Comfy-Org/ComfyUI/blob/bc5f880dbdd43e426077f42ccec53bb577ffee9f/comfy/ldm/moss_image/model.py) · [통합 안내](https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui) · [양자화 커널](https://github.com/Comfy-Org/comfy-kitchen/tree/f8b7e97a1418d2e529bd5405fbf86312be198ae6/comfy_kitchen/tensor) · [DiffSynth](https://github.com/modelscope/DiffSynth-Studio/tree/60b709c4dab8eeabce06ee9bc168328ed4c780e8/examples/moss_image)

## 커뮤니티 실행 사례

| GPU | 조합·작성자 보고 |
|---|---|
| 3060 6GB / RAM 16GB | Spectrum + 8-step LoRA·CFG 1: 2MP 약 90초. 25-step·CFG 2.5는 캐시 적용 342초 / 미적용 600초 |
| 3060 12GB / RAM 16GB | 초기 INT8 ConvRot + 제작팀 플러그인: 1MP·50-step 약 137초, peak 약 9.7GB |
| 5090D | 8-step LoRA·CFG 1: 4MP 약 18초 |

[Reddit·Civitai 동일 작성자](https://www.reddit.com/r/comfyui/comments/1xcfzxo/optimized_moss_an_image_generation_model_in/) · [12GB 사례](https://www.reddit.com/r/StableDiffusion/comments/1x7kc3o/mossimage_5b_apache_20_on_a_12_gb_rtx_3060_int8/) · [Bilibili](https://www.bilibili.com/video/BV1wxJa65EvE) · [중국어권 FP16-fixed 공유](https://bbs.monster/thread-5322-1-1.html)

Civitai 원문: 접근 실패, 동일 작성자의 Reddit에서 수치 확보. 아카라이브: 검색 사례 미확보·직접 접근 실패.

## T4 테스트 후보

SFT와 `flux2-vae_small.safetensors`를 공통으로 사용한다. 메모리 관리·attention은 ComfyUI 기본값으로 시작한다.

| ID | 본체 / 인코더 | 본체 연산 설정 |
|---|---|---|
| N1 | INT8 ConvRot / W4A8 GPTQ | 자동 |
| N2 | INT8 ConvRot / INT8 ConvRot | 자동 |
| N3 | INT8 ConvRot / INT8 ConvRot | `--fp16-unet` |
| N4 | FP16-fixed / INT8 ConvRot | `--fp16-unet` |

다운로드 크기: 본체 INT8 **5.414GB**, FP16-fixed **10.165GB**; 인코더 INT8 **9.748GB**, W4A8 **7.344GB**; small VAE **0.250GB**. [Comfy-Org 파일](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main) · [FP16-fixed](https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed)

추가 옵션은 기준구성에 하나씩 붙인 독립 후보로 등록한다. 메모리 절감 후보도 기준구성의 성공 여부와 관계없이 시험한다.

| 후보 옵션 | 스모크에서 실행할 부분 |
|---|---|
| [lodestones 8-step LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) | 현재 카드의 일반 LoRA 로더로 patch 적용 후 forward |
| [native Spectrum](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) | warmup과 forecast를 각각 실행 |
| compile | 같은 shape의 첫 호출·재호출 |
| PyTorch attention | 해당 attention으로 전체 본체 forward |
| tiled VAE | 합성 잠재값 디코드 |
| lowvram | `--disable-dynamic-vram --lowvram`으로 전체 스모크 |
| [MOSS GGUF](https://github.com/RealRebelAI/ComfyUI-MOSS-GGUF)·FP8 | 해당 저장 형식·로더로 전체 스모크 |
| [rapid-moss 4-step](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora) | 전용 adapter·time path·sampler 경로 |

## 스모크와 노트북 구성

**후보마다 새 프로세스**에서 실제 ComfyUI 로더·model patcher·메모리 관리 경로를 사용한다.

| 단계 | 실행·통과 조건 |
|---|---|
| 로드 | 실제 후보 파일 로딩 완료 |
| 텍스트 | 고정 양성 프롬프트·빈 음성 프롬프트 인코딩. 마지막 차원 4096, 유한값 |
| 본체 | 노이즈 `[1,128,64,64]`·sigma 0.5로 양성/음성 각각 forward 1회. shape 일치·유한값 |
| VAE | 합성 잠재값 디코드. 1024² RGB·유한값 |

전 단계 통과 시 생성 후보에 남긴다. 실패 시 해당 단계에서 종료하고 다음 후보로 넘어간다. 스모크에는 샘플러 반복·완성 이미지 저장을 넣지 않는다.

노트북은 **① T4·RAM·torch/CUDA 확인 → ② ComfyUI 설치 → ③ 후보 정의·필요 파일 다운로드 → ④ 후보별 스모크 → ⑤ 결과표 → ⑥ 통과 후보의 이미지 생성** 순서로 구성한다. 실행은 Colab 노트북 셀에서 직접 제어한다.

결과표에는 `후보·설정 / 통과·실패·시간초과 / 실패 단계·오류 / 소요시간 / peak VRAM·RAM`만 기록한다. 시작 제한시간은 로드 120초·인코딩 60초·본체 1회 60초·VAE 30초이며 compile 첫 호출은 180초다.

구현 기준 ComfyUI: `bc5f880dbdd43e426077f42ccec53bb577ffee9f` (`comfy-kitchen 0.2.40`, `comfy-aimdo 0.6.0`). 연결 지점은 `nodes.py` 로더, `comfy/model_management.py`, `comfy/model_base.py`의 `MOSSImage`다. 자동 dtype·실제 설치 버전은 결과와 함께 저장한다.
