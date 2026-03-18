import os
import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ReasoningPath(BaseModel):
    logical_break: str = Field(description="Đoạn giải thích logic bị gãy của học sinh. Dùng làm query vector.")
    target_concept: str = Field(description="Tên khái niệm liên quan.")

class GeminiRAGClient:
    def __init__(self):
        self.gen_model = genai.GenerativeModel('gemini-2.5-flash')
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def extract_reasoning(self, dialogue: str) -> dict:
        """Phân tích hội thoại để tìm 'điểm gãy' logic"""
        prompt = f"Phân tích hội thoại của học sinh và tìm điểm logic bị sai: '{dialogue}'"
        response = self.gen_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ReasoningPath,
                temperature=0.1
            )
        )
        return json.loads(response.text)

    def get_embedding(self, text: str) -> list[float]:
        """Tạo vector từ điểm gãy để search trong Neo4j (Kết quả trả về 384 chiều)"""
        return self.embed_model.encode(text).tolist()
        
    def final_diagnosis(self, dialogue: str, graph_context: str) -> str:
        """Kết hợp câu hỏi của học sinh + dữ liệu lấy từ Knowledge Graph để trả lời"""
        prompt = f"""
        Hội thoại học sinh: "{dialogue}"
        Dữ liệu chuyên môn từ Knowledge Graph:
        {graph_context}
        
        Nhiệm vụ: Dựa trên dữ liệu chuyên môn, hãy chỉ rõ học sinh đang hiểu sai chỗ nào 
        và giải thích lại một cách dễ hiểu (Remedy).
        """
        response = self.gen_model.generate_content(prompt)
        return response.text