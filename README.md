# MCP RAG 스타터 Kit

이 프로젝트는 MCP(Model Context Protocol)를 사용하여 문서 검색 및 처리를 위한 RAG(Retrieval-Augmented Generation) 서버를 구현한 스타터 템플릿입니다.

## 기능

- 문서 벡터 검색 (RAG)
- 텍스트 처리 프롬프트 템플릿 (요약, 키워드 추출, 번역, 질문-답변)
- 다국어 임베딩 지원 (한국어 포함)

## 설치 방법

### 필수 요구사항

- Python 3.11 이상
- 문서 파일을 저장할 docs 디렉토리
- OpenAI API 키

### 설치 단계

1. **저장소 클론**:

   ```bash
   git clone https://github.com/tsdata/mcp-rag-starter.git
   cd mcp-rag-starter
   ```

2. **가상 환경 생성 및 활성화**:

   ```bash
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # 또는
   .venv\Scripts\activate  # Windows
   ```

3. **의존성 설치**:
   ```bash
   uv pip install -e .
   ```
   또는 pip를 사용하는 경우:
   ```bash
   pip install -e .
   ```

## 주요 의존성 패키지

이 프로젝트는 다음 주요 패키지에 의존합니다:

- **langchain**: LLM 애플리케이션을 위한 프레임워크
- **langchain-chroma**: 벡터 데이터베이스 연동
- **langchain-huggingface**: 허깅페이스 모델 통합
- **langchain-openai**: OpenAI 모델 통합
- **mcp**: Model Context Protocol 구현

## 서버 설정 및 실행

### 서버 실행

```bash
uv python main.py
```

- 실행하면 기본적으로 `http://127.0.0.1:8000`에서 MCP 서버가 시작됩니다.

### 환경 설정

`.env` 파일에 필요한 환경 변수를 설정합니다:

```.env
OPENAI_API_KEY=your_openai_api_key
```

## Claude Desktop 설정

1.  **Claude Desktop 설치**: [Claude Desktop 다운로드](<https://claude.ai/download)
2.  **설정 파일 편집**:

    - **MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

3.  **설정 파일에 다음 내용 추가**:

    ```json
    {
      "mcpServers": {
        "mcp-rag-starter": {
          "command": "uv",
          "args": [
            "--directory",
            "/절대/경로/mcp-rag-starter",
            "run",
            "main.py"
          ]
        }
      }
    }
    ```

    **중요**:

    - `/절대/경로/mcp-rag-starter`를 실제 `main.py` 파일이 위치하는 폴더의 절대 경로로 변경하세요.

4.  **Claude Desktop 재시작**
    - 재시작 후, Claude Desktop 하단에 MCP 도구 아이콘이 나타납니다.

## 문서 추가하기

- `docs` 디렉토리에 텍스트 파일(`.txt`)을 추가합니다.
- 서버를 시작하면 자동으로 문서가 로드되고 벡터 저장소에 인덱싱됩니다.
- 문서가 변경되면 다음 실행 시 자동으로 재인덱싱됩니다.

## Claude Desktop에서 도구 사용하기

Claude Desktop에서 다음 기능을 사용할 수 있습니다:

- **문서 검색**: `"내 문서에서 [검색어]에 대한 정보를 찾아줘"` 형식으로 사용
- **텍스트 처리**: Claude에 문서 요약, 키워드 추출, 번역 등을 요청

### 예시 프롬프트:

- `"내 문서에서 MCP 프로토콜에 관한 내용을 찾아줘"`
- `"찾은 내용을 요약해줘"`
- `"이 내용에서 중요한 키워드를 추출해줘"`
