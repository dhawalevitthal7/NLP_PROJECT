from typing import Any

import chromadb

from app.config import settings


class ChromaQuestionRetriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)

    def index_scheme_units(
        self,
        collection_name: str,
        scheme_units: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        collection = self.client.get_or_create_collection(name=collection_name)
        if collection.count() > 0:
            self.client.delete_collection(name=collection_name)
            collection = self.client.get_or_create_collection(name=collection_name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for unit in scheme_units:
            ids.append(unit["question_id"])
            documents.append(unit["answer_text"])
            metadatas.append(
                {
                    "question_id": unit["question_id"],
                    "question_no": int(unit["question_no"]),
                    "sub_question_no": unit.get("sub_question_no") or "",
                    "mcq_option": unit.get("mcq_option") or "",
                    "total_marks": float(unit.get("max_marks") or 0.0),
                }
            )

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def index_scheme_questions(
        self,
        collection_name: str,
        scheme_questions: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        collection = self.client.get_or_create_collection(name=collection_name)
        if collection.count() > 0:
            self.client.delete_collection(name=collection_name)
            collection = self.client.get_or_create_collection(name=collection_name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for question in scheme_questions:
            ids.append(question["question_id"])
            documents.append(question["answer_text"])
            metadatas.append(
                {
                    "question_id": question["question_id"],
                    "question_no": int(question["question_no"]),
                    "mcq_option": question.get("mcq_option") or "",
                    "total_marks": float(question.get("total_marks") or 0.0),
                }
            )

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def get_question_by_number(self, collection_name: str, question_no: int) -> dict[str, Any] | None:
        collection = self.client.get_collection(name=collection_name)
        result = collection.get(where={"question_no": int(question_no)})
        documents = result.get("documents", [])
        if not documents:
            return None

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        return {
            "question_id": ids[0] if ids else f"q{question_no}",
            "document": documents[0],
            "metadata": metadatas[0] if metadatas else {},
        }

    def retrieve_for_student_unit(
        self,
        collection_name: str,
        query_embedding: list[float],
        question_no: int,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        collection = self.client.get_collection(name=collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"question_no": int(question_no)},
        )
        result_ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]

        matches: list[dict[str, Any]] = []
        for idx, item_id in enumerate(result_ids):
            matches.append(
                {
                    "question_id": item_id,
                    "distance": float(distances[idx]) if idx < len(distances) else None,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "document": documents[idx] if idx < len(documents) else "",
                }
            )
        return matches
