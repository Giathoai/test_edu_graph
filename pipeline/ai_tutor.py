import os
import json
import uuid
from neo4j import GraphDatabase
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class AnswerAnalysis(BaseModel):
    is_correct: bool = Field(description="True nếu trả lời đúng, False nếu sai.")
    is_meaningful: bool = Field(description="True nếu lỗi sai là do hiểu lầm kiến thức. False nếu là đùa cợt, vô nghĩa, lạc đề.")
    logical_break: str = Field(description="Logic sai cốt lõi.")
    suggested_name: str = Field(description="Đề xuất tên ngắn gọn cho lỗi này (vd: Nhầm lẫn khái niệm X).")
    suggested_remedy: str = Field(description="Đề xuất cách giải thích để khắc phục lỗi này.")

class AITutor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.gen_model = genai.GenerativeModel('gemini-2.5-flash')
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def extract_keywords(self, question: str) -> str:
        prompt = f"Trích xuất 1 từ khóa (danh từ/cụm danh từ) quan trọng nhất từ câu hỏi này: '{question}'. Chỉ trả về 1 cụm."
        res = self.gen_model.generate_content(prompt)
        return res.text.strip()

    def get_ground_truth(self, keyword: str) -> dict:
        query = """
        MATCH (c:Concept)-[:APPEARS_IN]->(ch:Chunk)
        WHERE toLower(c.name) CONTAINS toLower($kw) OR toLower($kw) CONTAINS toLower(c.name)
        RETURN c.name AS concept, ch.text AS content LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, kw=keyword)
            record = result.single()
            return record.data() if record else None

    def verify_and_analyze(self, question: str, student_answer: str, truth_content: str) -> dict:
        prompt = f"""
        Kiến thức chuẩn: {truth_content}
        Câu hỏi: {question}
        Học sinh: {student_answer}

        Hãy phân tích câu trả lời. Nếu sai nhưng là hiểu lầm thật sự, hãy đánh dấu is_meaningful = true và đề xuất name, remedy.
        """
        response = self.gen_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AnswerAnalysis,
                temperature=0.1
            )
        )
        return json.loads(response.text)

    def retrieve_misconception(self, logical_break: str, top_k: int = 1) -> dict:
        if not logical_break: return None
        vector = self.embed_model.encode(logical_break).tolist()
        
        query = """
        CALL db.index.vector.queryNodes('misconception_idx', $k, $vector)
        YIELD node AS m, score
        WHERE score > 0.65
        MATCH (c:Concept)-[:HAS_MISCONCEPTION]->(m)
        RETURN m.name AS error_name, m.logic AS error_logic, m.remedy AS remedy, score
        ORDER BY score DESC LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, vector=vector, k=top_k)
            record = result.single()
            return record.data() if record else None

    def learn_new_misconception(self, concept_name: str, analysis: dict):
        m_id = f"err_{uuid.uuid4().hex[:8]}"
        vector = self.embed_model.encode(analysis['logical_break']).tolist()
        
        query = """
        MATCH (c:Concept {name: $c_name})
        MERGE (m:Misconception {m_id: $m_id})
        SET m.name = $name, m.logic = $logic, m.remedy = $remedy, m.embedding = $emb
        MERGE (c)-[:HAS_MISCONCEPTION]->(m)
        """
        with self.driver.session() as session:
            session.run(query, c_name=concept_name, m_id=m_id, 
                        name=analysis['suggested_name'], logic=analysis['logical_break'], 
                        remedy=analysis['suggested_remedy'], emb=vector)

    def generate_feedback(self, question: str, student_answer: str, analysis: dict, truth: dict, misconcept: dict) -> str:
        if analysis['is_correct']:
            prompt = f"Học sinh trả lời ĐÚNG câu '{question}' với đáp án '{student_answer}'. Hãy khen ngợi."
        elif not analysis['is_meaningful']:
            prompt = f"Học sinh trả lời vô nghĩa hoặc đùa cợt: '{student_answer}'. Hãy nhắc nhở nhẹ nhàng quay lại bài học."
        else:
            err_info = f"Lỗi: {misconcept['error_name']} - Sửa: {misconcept['remedy']}" if misconcept else f"Lỗi: {analysis['suggested_name']} - Sửa: {analysis['suggested_remedy']}"
            prompt = f"Câu hỏi: {question}\nHọc sinh sai: {student_answer}\nChuẩn: {truth['content']}\nSửa lỗi: {err_info}\nHãy đóng vai gia sư giải thích lại nhẹ nhàng."
            
        return self.gen_model.generate_content(prompt).text

    def close(self):
        self.driver.close()