# MOSS-Image T4 후보 비교

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_moss/blob/f5d3041a5fea49e3ab9ab2682f3431e88fc3951b/MOSS_T4_Smoke.ipynb)

Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택하고 1~5번 셀을 실행한다.
3번 셀에서 기본 후보·단일 옵션·옵션 조합·프롬프트·생성 조건을 함께 선택한다.
기본 비교 후보는 `N3-lora8`, `N1-lora8`, `N1-lora8+spectrum`이다.

| 측정 | 시작과 끝 |
|---|---|
| 개별 검사 | 로딩·패치·인코딩·각 시험 연산·정리·VAE 각각의 실행 시간 |
| 프로세스 시작 → 첫 이미지 | 새 프로세스 시작부터 첫 PNG 저장 완료 |
| 로딩 시작 → 첫 이미지 | 모델 로딩 시작부터 첫 PNG 저장 완료 |
| 모델 유지 재생성 | 동일 프로세스에서 시드 변경 → 프롬프트 변경 → 스텝·CFG 변경; 요청마다 인코딩·샘플링·VAE·저장 시간을 분리 |

검사와 실사용 측정은 별도 프로세스에서 실행한다. 실사용 측정은 모델을 한 번 읽고 이미지 4장을 순서대로 생성한다.
같은 텍스트의 인코딩 결과는 재사용하며 캐시 사용 여부를 기록한다. VRAM 관리는 ComfyUI가 수행한다.
첫 생성은 모델 파일 다운로드가 완료된 상태에서 새 프로세스로 측정한다. 파일·컴파일 디스크 캐시는 유지한다.
기본 후보는 50스텝·CFG 3, LoRA 후보는 8스텝·CFG 1을 사용하며 실제 조건은 결과표에 표시한다.

5번 셀은 개별 검사 표와 실사용 비교 표를 출력한다. 동일 조건에서 통과한 검사는 재사용하며 실사용 측정은 실행할 때마다 새로 수행한다.
6번 셀은 `smoke_stages.csv`, `usage.csv`, JSON·로그·PNG를 ZIP으로 내려받는다. 7번 셀에서 생성 이미지를 볼 수 있다.

[모델 정보와 후보 선정 근거](MOSS_Image_T4_Screening.md)
