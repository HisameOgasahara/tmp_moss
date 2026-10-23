# MOSS-Image T4 스모크 테스트

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_moss/blob/b6b6a2662a38a991d4e6137d5dce0fd1b0768d5c/MOSS_T4_Smoke.ipynb)

Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택하고 위에서부터 실행한다.

- INT8/W4A8·INT8/INT8·FP16-fixed 구성과 선택한 최적화 옵션을 각각 시험한다.
- 로딩 → 텍스트 인코딩 → 본체 forward → VAE 디코드 결과를 표·CSV·JSON으로 저장한다.
- 통과한 후보의 이미지 생성은 마지막 셀에서 선택한다.

[모델 정보와 후보 선정 근거](MOSS_Image_T4_Screening.md)
