# Kalone (칼로네)

> **LLM에게 Ctrl+C / Ctrl+V를 쥐여주면 어떻게 될까?**
> 모델이 원문을 토큰으로 다시 "타이핑"하는 대신, 입력의 `[시작, 끝]` 구간을 **좌표 두 개로 가리켜 통째로 붙여넣는** 아주 작은 연구용 프로토타입입니다.

[English summary](#english-summary) · [설계 문서](docs/DESIGN.md) · [로드맵](ROADMAP.md) · [기여 가이드](CONTRIBUTING.md)

---

## 왜 만들었나

- LLM은 **입력(읽기)은 병렬로 처리돼 싸고, 출력(쓰기)은 토큰을 하나씩 만들어 비쌉니다.**
- 그런데 코드 수정, 문서 편집, 정확한 인용처럼 **출력의 대부분이 원문을 그대로 옮기는 일**인 작업이 많습니다.
- 그 부분을 "생성"하지 않고 "복사"하면 비싼 출력 토큰을 줄이고, 옮기는 도중 생기는 오탈자·누락도 없앨 수 있지 않을까? 를 가장 작은 규모로 확인해 보는 프로젝트입니다.

## 동작 방식

디코더는 한 스텝마다 아래 셋 중 하나를 내보냅니다.

| 행동 | 의미 | 비용 |
|---|---|---|
| `GEN(c)` | 문자 하나를 새로 생성 | 스텝 1번 |
| `COPY[s:e]` | 입력의 `s`~`e` 구간을 **통째로 붙여넣기** | 구간이 얼마나 길든 스텝 1번 |
| `EOS` | 종료 | |

```
입력 : penq ozg qmyjjk mpyutb ztesyz ofiwje|a
행동 : COPY[0:35]  GEN(' ') GEN('d') GEN('o') GEN('n') GEN('e')
출력 : penq ozg qmyjjk mpyutb ztesyz ofiwje done     (출력 41자 / 스텝 6번)
```

붙여넣은 구간은 인코더 상태의 평균 벡터 **하나**로 디코더 기억에 들어갑니다. 그래서 붙여넣은 뒤에도 이어서 문맥에 맞게 쓸 수 있습니다.

## 빠른 시작

```bash
git clone https://github.com/navytriange/Kalone.git
cd Kalone
pip install -e ".[dev]"

# 1) 동봉된 체크포인트로 바로 실행
python -m kalone.demo --ckpt checkpoints/kalone-v0.1.pt "abc def ghi|r1|xyz" "abc def ghi|d0" "abc def ghi|a"

# 2) 직접 학습 (CPU 1코어 기준 약 9분)
python -m kalone.train --steps 2000 --bs 64

# 3) 테스트
pytest
```

입력 형식 (장난감 과제): `문장|r<번호>|<새 단어>` 교체, `문장|d<번호>` 삭제, `문장|a` 뒤에 ` done` 추가.

## 현재 결과 (v0.1)

| 과제 | exact-match |
|---|---|
| replace | 100% |
| delete | 98% |
| append | 100% |
| **overall** | **99.33%** (300문항, 고정 시드 1234) |

⚠️ **이 숫자를 과대해석하지 마세요.** 문자 단위 소형 GRU가, 직접 만든 3가지 장난감 과제에서, 학습 때와 같은 분포의 문제를 푼 결과입니다. "복사 좌표를 정확히 고를 수 있다"는 것만 보여 줄 뿐, 실제 언어에서 *무엇을 복사할지 고르는 판단*이나 긴 문서에서의 비용 절감은 아직 **측정되지 않았습니다.** 그것을 확인하는 것이 이 프로젝트의 다음 과제입니다.

## 저장소 구조

```
kalone/
  vocab.py    문자 사전, 행동 타입 상수
  data.py     합성 과제 레지스트리(@register_task), 배치 만들기
  model.py    CopyLM (인코더 + 복사/생성 디코더)
  train.py    학습 + 과제별 평가
  demo.py     체크포인트 실행
tests/        데이터/모델 테스트 (pytest)
docs/DESIGN.md   설계와 확장 방법
ROADMAP.md       해야 할 일, 좋은 첫 이슈
```

## 함께 개선해 주세요

이 저장소는 **"다른 사람이 가져가서 개선하기 쉬운 것"** 을 목표로 만들었습니다.

- 새 과제 추가는 함수 하나 + 데코레이터 한 줄입니다 → [설계 문서](docs/DESIGN.md#새-과제-추가하기)
- 처음이라면 [ROADMAP](ROADMAP.md)의 `good first issue` 후보부터 보세요.
- 결과를 올릴 때는 **시드, 평가 문항 수, 과제 구성**을 같이 적어 주세요 ([CONTRIBUTING](CONTRIBUTING.md#결과-보고-규칙)).

## 알려진 한계

- 문자 단위라 긴 문서·실제 토큰 단위로 확장해 보지 않았습니다.
- 평가가 학습과 같은 분포라서 **일반화(더 긴 입력, 처음 보는 형식)는 검증되지 않았습니다.**
- "엉뚱한 구간을 정확하게 복사하는" 새로운 오류 유형이 생길 수 있습니다.
- 외부 저작물을 복사하는 기능의 **저작권·프롬프트 인젝션** 문제는 다루지 않았습니다 (연구 범위 밖, 로드맵에는 포함).

## English summary

**Kalone** is a tiny research prototype of an LLM decoder that can *copy-paste* instead of retyping. At each step it emits `GEN(char)`, `COPY[start:end]` (paste an input span with just two coordinates), or `EOS`. A pasted span costs one decoding step regardless of length and is fed back as a single pooled vector. On three synthetic editing tasks (replace / delete / append) a small char-level GRU reaches 99.33% exact match, but this only shows that span pointers are learnable. It says nothing yet about real-language span selection or generalization. See [ROADMAP](ROADMAP.md) for open problems and [CONTRIBUTING](CONTRIBUTING.md) to get involved.

## License

[MIT](LICENSE)
