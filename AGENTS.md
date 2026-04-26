# AGENTS.md

이 파일은 이 작업 폴더에서 Codex가 항상 먼저 따라야 하는 최상위 규칙이다.

## 1. 파일 수정 규칙

- 기존 파일을 수정해야 할 때는 먼저 원본을 백업 복사한 뒤, 기존 파일을 직접 수정한다.
- 백업 파일은 항상 루트의 `9_BACKUPS` 폴더 안에 저장한다.
- 백업 파일명은 항상 `OLD_백업번호_원본파일명` 형식으로 만든다.
- 같은 파일을 여러 번 수정할 수 있으므로, 백업번호를 붙여 구분한다.
- 번호는 `01`, `02`, `03`처럼 두 자리 이상으로 관리한다.
- 파일을 수정한 뒤에는 vsc 의 '선택항목비교' 기능을 통해 수정후 원본파일 - 백업본 비교 창을 띄워.
- 수정 후에는 아래 형식으로 VS Code diff를 보여주기만 하지 말고 항상 바로 실행해.
- `code.cmd --diff 기존파일 9_BACKUPS/OLD_번호_기존파일`
- 예:
  - 원본: `PYLLM_03_tool_node.py`
  - 1차 백업: `9_BACKUPS/OLD_01_PYLLM_03_tool_node.py`
  - 2차 백업: `9_BACKUPS/OLD_02_PYLLM_03_tool_node.py`
- 이 규칙은 특히 `PYLLM_*` 형식의 학습용 파일에 우선 적용한다.
- 원본 직접 수정 전에는 반드시 백업 파일이 먼저 있어야 한다.

## 2. 학습용 코드 작성 규칙

- 이 저장소의 목적은 "동작하는 코드 작성"만이 아니라 "코드 구조 학습"이다.
- 새 클래스, 새 함수, 새 구조, 새 라이브러리 사용이 나오면 쉽게 설명하는 주석을 충분히 붙인다.
- 사용자가 이미 익숙해진 패턴과 구조를 우선한다.
- 더 현대적인 방식이 있더라도 바로 강제 교체하지 말고, 먼저 유지 가능한 형태로 제안한다.

## 3. 아래 문서들의 역할

Codex는 아래 파일들을 함께 참고해야 한다.

- `0_INSTRUCTIONS/LEARNING_PROFILE.md`
  - 사용자의 공부 방식, 선호 설명 방식, 누적된 학습 성향을 기록한다.
- `0_INSTRUCTIONS/MASTERED_FILES.md`
  - 사용자가 공부 완료한 `PYLLM_*` 파일 목록을 기록한다.
- `0_INSTRUCTIONS/STYLE_GUIDE.md`
  - 사용자가 익숙한 변수명, 함수명, 구조, 주석 스타일을 기록한다.
- `0_INSTRUCTIONS/REVIEW_POLICY.md`
  - 기존 패턴을 바로 뜯어고치지 않고, 먼저 제안 중심으로 접근해야 하는 규칙을 기록한다.

## 4. 문서 갱신 규칙

- Codex는 작업 도중 아래 문서들의 변경이 필요하다고 판단되면 갱신할 수 있다.
- 다만 갱신 시에도 사용자의 학습 목적을 우선한다.
- 변경 이유가 분명해야 하며, 사용자의 학습 흐름을 깨지 않는 방향으로만 수정한다.

대상 문서:
- `0_INSTRUCTIONS/LEARNING_PROFILE.md`
- `0_INSTRUCTIONS/MASTERED_FILES.md`
- `0_INSTRUCTIONS/STYLE_GUIDE.md`
- `0_INSTRUCTIONS/REVIEW_POLICY.md`

## 5. 우선순위

이 폴더에서 작업할 때 우선순위는 아래와 같다.

1. `AGENTS.md`
2. `0_INSTRUCTIONS/REVIEW_POLICY.md`
3. `0_INSTRUCTIONS/LEARNING_PROFILE.md`
4. `0_INSTRUCTIONS/STYLE_GUIDE.md`
5. `0_INSTRUCTIONS/MASTERED_FILES.md`
