import os
import glob
import json
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import google.generativeai as genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class MisconceptionSchema(BaseModel):
    id: str = Field(description="Unique error ID")
    name: str = Field(description="Error name")
    logic: str = Field(description="Detailed incorrect logic")
    remedy: str = Field(description="Fix/Explanation")

class ConceptSchema(BaseModel):
    name: str = Field(description="Concept name")
    description: str = Field(description="Definition")
    misconceptions: list[MisconceptionSchema] = Field(default_factory=list)

class RelationSchema(BaseModel):
    source: str
    target: str
    relation_type: str

class LessonExtractionSchema(BaseModel):
    lesson_name: str
    concepts: list[ConceptSchema]
    relations: list[RelationSchema]

class AutoLessonBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.gen_model = genai.GenerativeModel("gemini-2.5-flash")
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self._setup_vector_index()

    def _setup_vector_index(self):
        query = """
        CREATE VECTOR INDEX misconception_idx IF NOT EXISTS
        FOR (m:Misconception) ON (m.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """
        with self.driver.session() as session:
            session.run(query)

    def _get_embedding(self, text: str) -> list[float]:
        return self.embed_model.encode(text).tolist()

    def _extract_knowledge(self, text: str) -> dict:
        prompt = f"""
        Extract knowledge from the lesson below. 
        IMPORTANT: Your response MUST be a valid JSON matching the schema. 
        If no misconceptions are found for a concept, return an empty list [] for the 'misconceptions' field.
        
        Lesson Content:
        {text}
        """
        response = self.gen_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LessonExtractionSchema,
                temperature=0.1
            )
        )
        return json.loads(response.text)

    def ingest_single_file(self, filepath: str):
        filename = Path(filepath).stem
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data = self._extract_knowledge(content)
        lesson_name = data.get('lesson_name', 'Unknown Lesson')
        chunk_id = f"chunk_{filename}"

        with self.driver.session() as session:
            session.run("""
                MERGE (ch:Chunk {chunk_id: $chunk_id})
                SET ch.text = $text, ch.lesson_name = $ln, ch.source = $fn
            """, chunk_id=chunk_id, text=content, ln=lesson_name, fn=filename)

            for concept in data.get('concepts', []):
                session.run("""
                    MERGE (c:Concept {name: $name})
                    SET c.description = $desc
                    WITH c
                    MATCH (ch:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:APPEARS_IN]->(ch)
                """, name=concept.get('name'), desc=concept.get('description'), chunk_id=chunk_id)

                for err in concept.get('misconceptions', []):
                    emb = self._get_embedding(err.get('logic', ''))
                    session.run("""
                        MERGE (m:Misconception {m_id: $m_id})
                        SET m.name = $name, m.logic = $logic, m.remedy = $remedy, m.embedding = $emb
                        WITH m
                        MATCH (c:Concept {name: $c_name})
                        MERGE (c)-[:HAS_MISCONCEPTION]->(m)
                    """, m_id=err.get('id'), name=err.get('name'), logic=err.get('logic'), 
                         remedy=err.get('remedy'), emb=emb, c_name=concept.get('name'))

            for rel in data.get('relations', []):
                session.run("""
                    MATCH (src:Concept {name: $src})
                    MATCH (tgt:Concept {name: $tgt})
                    MERGE (src)-[r:RELATED_TO {lesson: $ln}]->(tgt)
                    SET r.type = $rel_type
                """, src=rel.get('source'), tgt=rel.get('target'), rel_type=rel.get('relation_type'), ln=lesson_name)

    def process_folder(self, path: str):
        files = glob.glob(os.path.join(path, "*.txt"))
        for f in files:
            try:
                self.ingest_single_file(f)
                print(f"✅ Đã xử lý xong: {f}")
            except Exception as e:
                print(f"❌ Lỗi tại file {f}: {e}")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    builder = AutoLessonBuilder()
    try:
        builder.process_folder("./data/lessons")
    finally:
        builder.close()