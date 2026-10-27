# 설계 문서

## 1. 행동 공간 (action space)

한 디코더 스텝 = 행동 하나.

- `GEN`: `char_head`가 문자 분포를 내고 하나를 고릅니다.
- `COPY`: `q_start`, `q_end` 두 포인터가 입력 위치 분포를 내고 각각 argmax. 추론 시 `end >= start`를 강제합니다 (`model.py::generate`).
- `EOS`: 종료.

타입 선택(`type_head`), 문자(`char_head`), 시작/끝 포인터는 모두 **디코더 상태 + 입력에 대한 어텐션 컨텍스트**를 보고 결정합니다.

## 2. 붙여넣은 내용을 디코더가 아는 방법

`COPY[s:e]` 다음 스텝의 입력은 `type_emb(COPY) + span_proj(mean(enc[s:e]))` 입니다. 구간 전체를 문자 단위로 다시 먹이지 않고 **풀링된 벡터 하나**로 요약하므로, 긴 붙여넣기도 디코더는 한 스텝만 돕니다. 학습 때는 `cumsum` 트릭으로 구간 평균을 배치 연산으로 계산합니다 (`CopyLM.span_mean`).

## 3. 학습

- 정답 행동열을 만들어 놓고 teacher forcing 으로 학습합니다 (이전 정답 행동을 입력으로 넣음).
- 손실 = 타입 CE + (GEN인 스텝의) 문자 CE + (COPY인 스텝의) 시작 CE + 끝 CE. 해당 없는 항목은 `-100`으로 무시합니다.

## 4. 평가

`train.evaluate`: 고정 시드의 새 문제를 풀게 하고 **출력 문자열이 정답과 완전히 일치하는 비율**을 과제별/전체로 냅니다. 과제를 번갈아 뽑으므로 과제별 문항 수가 균등합니다.

## 새 과제 추가하기

`kalone/data.py`에 함수를 하나 추가하면 학습/평가에 자동으로 들어갑니다.

```python
@register_task("swap")
def swap_task(rng):
    words, src, starts, ends = _sentence(rng)
    i = rng.randrange(len(words) - 1)
    inp = f"{src}|s{i}"                      # 모델이 읽는 입력 (전부 복사 가능)
    acts = []                                # 정답 행동열
    _cp(acts, 0, starts[i])                  # [0, starts[i]) 복사
    _cp(acts, starts[i + 1], ends[i + 1])    # 다음 단어
    _cp(acts, ends[i], starts[i + 1])        # 사이 공백
    _cp(acts, starts[i], ends[i])            # 앞 단어
    _cp(acts, ends[i + 1], len(src))         # 나머지
    target = " ".join(words[:i] + [words[i + 1], words[i]] + words[i + 2:])
    return Example(inp, acts, target)
```

체크리스트:

1. `Example.target` 은 **행동열을 실행한 결과와 정확히 같아야** 합니다. `tests/test_data.py::test_gold_actions_reproduce_target` 이 모든 등록 과제에 대해 이를 검사합니다.
2. 입력에 쓰는 문자는 `vocab.py::CHARS` 안에 있어야 합니다.
3. 복사 구간은 `("copy", start, end_inclusive)` (끝 **포함**) 입니다. `_cp` 는 반열린 구간 `[start, end)` 을 받아 변환해 줍니다.
4. 과제를 추가한 PR에는 `python -m kalone.train` 결과(과제별 정확도)를 첨부해 주세요.

## 다른 모델로 바꾸기

`CopyLM` 이 지켜야 하는 최소 인터페이스:

- `forward(x, ty, ch, st, en) -> loss`
- `generate(inp: str, max_steps) -> (text, trace)`

이 둘만 맞으면 `train.py` / `demo.py` 를 그대로 쓸 수 있습니다. 토큰 단위 트랜스포머, 다른 포인터 구조 등을 실험해 보세요.
