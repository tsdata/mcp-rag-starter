# main.py - MCP(Model Context Protocol) 문서 서버 구현 파일
import os  # 운영 체제 관련 기능 사용
import hashlib  # 해시 함수 사용을 위한 모듈
from typing import Dict, Any, List  # 타입 힌트 사용을 위한 모듈

from pathlib import Path  # 파일 경로 처리를 위한 클래스
from mcp.server.fastmcp import FastMCP  # MCP 서버 구현을 위한 클래스

# 문서 로딩 및 처리를 위한 LangChain 라이브러리 임포트
from langchain_community.document_loaders import TextLoader, DirectoryLoader  # 텍스트 및 디렉토리 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 문서 분할 도구
from langchain_chroma import Chroma  # 벡터 데이터베이스
from langchain_huggingface import HuggingFaceEmbeddings  # 허깅페이스 임베딩 모델


# FastMCP 서버 인스턴스 생성 - 서버 이름 지정
mcp = FastMCP("MCP(model context protocol) Documents Server")

# 문서 파일이 저장된 기본 리소스 디렉토리 경로 설정
RESOURCES_PATH = Path("docs")

@mcp.resource("file:///llms-full.txt")
def get_llms_full_content() -> str:
    """
    llms-full.txt 파일의 내용을 제공하는 리소스 함수입니다.
    
    Returns:
        str: 파일 내용 또는 오류 메시지
    """
    try:
        file_path = RESOURCES_PATH / "llms-full.txt"  # 파일 경로 생성
        with open(file_path, "r", encoding="utf-8") as f:  # UTF-8 인코딩으로 파일 열기
            return f.read()  # 파일 전체 내용 반환
    except FileNotFoundError:  # 파일이 없는 경우
        return "File not found: llms-full.txt"
    except Exception as e:  # 기타 예외 발생 시
        return f"Error reading file: {str(e)}"

@mcp.tool()
def search_rag_docs(query: str) -> str:
    """
    모든 문서에서 특정 쿼리를 벡터 검색하는 도구 함수입니다.
    
    Args:
        query: 검색할 텍스트 쿼리
        
    Returns:
        str: 마크다운 형식의 검색 결과 또는 오류 메시지
    """
    try:
        # 리소스 디렉토리 존재 여부 확인
        if not RESOURCES_PATH.exists() or not RESOURCES_PATH.is_dir():
            return "리소스 디렉토리를 찾을 수 없거나 디렉토리가 아닙니다"
        
        # 벡터 저장소를 위한 영구 디렉토리 경로 설정 및 생성
        persist_directory = Path("vector_store")
        persist_directory.mkdir(exist_ok=True)  # 디렉토리가 없으면 생성
        
        # 문서 변경 감지를 위한 해시 파일 경로 설정
        hash_file = persist_directory / "docs_hash.txt"
        
        # 현재 문서들의 해시값 계산 - 파일 변경 감지용
        current_hash = ""
        for file_path in RESOURCES_PATH.glob("*.*"):  # 모든 파일 순회
            if file_path.is_file():
                file_stat = os.stat(file_path)  # 파일 상태 정보 가져오기
                # 파일명, 수정시간, 크기를 조합하여 해시 생성
                file_info = f"{file_path.name}:{file_stat.st_mtime}:{file_stat.st_size}"
                current_hash += file_info
        
        # 최종 해시값 생성 (MD5)
        current_hash = hashlib.md5(current_hash.encode()).hexdigest()
        
        # 다국어 지원 임베딩 모델 설정 - 한국어 포함 다양한 언어 지원
        embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-multilingual-base", model_kwargs={'trust_remote_code': True})
        
        # 문서 변경 확인 및 벡터 저장소 생성/로드 로직
        vector_store = None
        try:
            if hash_file.exists():  # 이전 해시 파일이 존재하는 경우
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()  # 저장된 해시값 읽기
                
                # 해시값이 동일하고 벡터 DB 파일이 존재하면 기존 DB 사용
                if current_hash == stored_hash and os.path.exists(persist_directory / "chroma.sqlite3"):
                    # 문서 변경 없음, 기존 벡터 저장소 로드
                    vector_store = Chroma(
                        persist_directory=str(persist_directory),
                        embedding_function=embeddings
                    )
                else:
                    # 문서 변경 있음, 벡터 저장소 재생성
                    vector_store = create_vector_store(embeddings, persist_directory)
                    # 새 해시값 저장
                    with open(hash_file, "w") as f:
                        f.write(current_hash)
            else:
                # 처음 실행 시 벡터 저장소 생성
                vector_store = create_vector_store(embeddings, persist_directory)
                # 해시값 저장
                with open(hash_file, "w") as f:
                    f.write(current_hash)
        except Exception as e:
            return f"벡터 저장소 초기화 중 오류 발생: {str(e)}"
            
        if vector_store is None:
            return "벡터 저장소를 초기화할 수 없습니다."
            
        # 유사도 검색 수행 - 상위 10개 결과 반환
        results = vector_store.similarity_search_with_score(query, k=10)

        # score가 0.1 이상인 결과만 반환
        results = [result for result in results if float(result[1]) >= 0.1]
        
        # 검색 결과가 없는 경우
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        # 결과 마크다운 형식으로 포맷팅
        formatted_results = ["# 검색 결과\n"]
        
        # 각 검색 결과 처리
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "알 수 없는 소스")  # 문서 출처
            source_name = Path(source).name  # 파일명만 추출
            # 결과 헤더 추가 (결과 번호, 파일명, 유사도 점수)
            formatted_results.append(f"## 결과 {i} - {source_name} (유사도: {score:.4f})\n")
            # 코드 블록으로 문서 내용 표시
            formatted_results.append(f"```\n{doc.page_content}\n```\n")
        
        # 모든 결과를 하나의 문자열로 결합
        return "\n".join(formatted_results)
    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return f"벡터 검색 중 오류 발생: {str(e)}"

def create_vector_store(embeddings, persist_directory):
    """
    문서를 로드하고 벡터 저장소를 생성하는 헬퍼 함수입니다.
    
    Args:
        embeddings: 사용할 임베딩 모델
        persist_directory: 벡터 저장소 저장 경로
        
    Returns:
        Chroma: 생성된 벡터 저장소 객체
    """
    
    # 문서 로더 설정 - 지정된 디렉토리의 모든 파일 로드
    loader = DirectoryLoader(
        str(RESOURCES_PATH),
        glob="*.*",  # 모든 파일 확장자 포함
        loader_cls=TextLoader,  # 텍스트 파일 로더 사용
        loader_kwargs={"encoding": "utf-8"}  # UTF-8 인코딩 지정
    )
    
    # 문서 로드
    documents = loader.load()
    if not documents:
        raise ValueError("문서를 찾을 수 없습니다.")  # 문서가 없으면 오류 발생
    
    # 문서 분할 - 대용량 문서를 적절한 크기로 나누기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # 청크 크기 (문자 수)
        chunk_overlap=600,  # 청크 간 중복 영역 (문맥 유지용)
        length_function=len  # 길이 측정 함수
    )
    chunks = text_splitter.split_documents(documents)  # 문서 분할 실행
    
    # 벡터 저장소 생성 및 저장
    vector_store = Chroma.from_documents(
        documents=chunks,  # 분할된 문서 청크
        embedding=embeddings,  # 임베딩 모델
        persist_directory=str(persist_directory)  # 저장 경로
    )
    
    return vector_store

@mcp.prompt()
def text_processing_prompt(task: str, text: str = "") -> str:
    """
    다양한 텍스트 처리 작업을 위한 프롬프트 템플릿을 제공하는 함수입니다.
    
    Args:
        task: 수행할 작업 유형 코드 (s: 요약, k: 키워드 추출, t: 번역, a: 질문-답변)
        
    Returns:
        str: 선택된 작업에 맞는 프롬프트 템플릿
    """
    
    if task.lower() in ["s", "요약"]:  # 요약(Summarize) 작업
        # 요약 작업 지침
        summary_guidelines = """
1. 중요한 정보만 포함하세요
2. 원문의 의미를 유지하세요
3. 불필요한 세부 사항은 생략하세요
4. 컨텍스트에 없는 정보는 추가하지 마세요
"""
        
        if len(text) > 0:
            return f"""
## 입력 텍스트
{text}

## 요약 작업
위 문서의 핵심 내용을 간결하게 요약해 주세요. 다음 사항을 지켜주세요:
{summary_guidelines}"""
        else:
            return f"""
위 문서의 핵심 내용을 간결하게 요약해 주세요. 다음 사항을 지켜주세요:
{summary_guidelines}"""
        
    elif task.lower() == "k" or task.lower() == "키워드":  # 키워드(Keywords) 추출 작업
        # 키워드 추출 작업 지침
        keyword_guidelines = """
1. 핵심 개념과 용어를 식별하세요
2. 각 키워드가 왜 중요한지 간략히 설명하세요
3. 키워드는 중요도 순으로 나열하세요
4. 10개 이내의 키워드를 추출하세요
"""
        
        if len(text) > 0:
            return f"""
## 입력 텍스트
{text}

## 키워드 추출 작업
위 문서의 가장 중요한 키워드를 추출해 주세요. 다음 사항을 지켜주세요:
{keyword_guidelines}"""
        else:
            return f"""
위 문서의 가장 중요한 키워드를 추출해 주세요. 다음 사항을 지켜주세요:
{keyword_guidelines}"""
    
    elif task.lower() == "t" or task.lower() == "번역":  # 번역(Translate) 작업
        # 번역 작업 지침
        translation_guidelines = """
1. 원문의 의미를 정확하게 전달하세요
2. 자연스러운 표현을 사용하세요
3. 전문 용어는 컨텍스트를 참고하여 적절히 번역하세요
4. 번역할 수 없는 부분은 원문 그대로 두고 표시하세요
"""

        if len(text) > 0:
            return f"""
## 입력 텍스트
{text}

## 번역 작업
위 문서의 텍스트를 한국어로 번역해 주세요. 다음 사항을 지켜주세요:
{translation_guidelines}"""
        else:
            return f"""
위 문서의 텍스트를 한국어로 번역해 주세요. 다음 사항을 지켜주세요:
{translation_guidelines}"""

    elif task.lower() == "q" or task.lower() == "질문":  # 질문-답변(Question-Answer) 작업
        # 질문-답변 작업 지침
        qa_guidelines = """
1. 컨텍스트에 있는 정보만 사용하세요
2. 단계별로 명확하게 설명해 주세요
3. 필요한 경우 예시를 들어 설명해 주세요
4. 불확실한 내용은 추측하지 말고 솔직하게 모른다고 말해주세요
"""
        
        if len(text) > 0:
            return f"""
## 입력 텍스트
{text}

## 질문-답변 작업
위 문서를 기반으로 다음 질문에 답변해 주세요. 컨텍스트에 관련 정보가 없는 경우, 
"제공된 정보만으로는 답변할 수 없습니다"라고 솔직하게 답변해 주세요.

답변 시 다음 사항을 지켜주세요:
{qa_guidelines}"""
        
        else:
            return f"""
위 문서를 기반으로 다음 질문에 답변해 주세요. 컨텍스트에 관련 정보가 없는 경우, 
"제공된 정보만으로는 답변할 수 없습니다"라고 솔직하게 답변해 주세요.

답변 시 다음 사항을 지켜주세요:
{qa_guidelines}"""
        

    else:
        # 지원하지 않는 작업 코드인 경우
        return "유효하지 않은 작업 유형입니다. 다시 시도해주세요."


# 서버가 직접 실행될 때 (모듈로 임포트되지 않은 경우)
if __name__ == "__main__":
    # MCP 서버 시작
    mcp.run()