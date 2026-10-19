"""Core RAG + memory logic (UI-independent so it can be tested/reused)."""
import json, os, re
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
NOT_FOUND = ("I'm sorry, I couldn't find that in the University Helpdesk knowledge base. "
             "Please contact the helpdesk at helpdesk@university.edu or 0183-555-0100 (Mon-Fri, 9-5).")
FOLLOWUP_WORDS = {"it", "that", "this", "they", "them", "those", "these", "there", "its", "same", "one", "ones"}
FOLLOWUP_STARTS = ("and ", "also ", "what about", "how about", "what else", "how long", "how much", "why")


class Retriever:
    def __init__(self, path=KB_PATH, threshold=0.15):
        with open(path, encoding="utf-8") as f:
            self.docs = json.load(f)
        self.threshold = threshold
        corpus = [f"{d['question']} {d['answer']}" for d in self.docs]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)

    def search(self, query, k=2):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        ranked = sorted(zip(scores, self.docs), key=lambda x: -x[0])[:k]
        return [(float(s), d) for s, d in ranked if s >= self.threshold]


@dataclass
class Memory:
    turns: list = field(default_factory=list)      # [(role, text)]
    facts: dict = field(default_factory=dict)       # e.g. {"name": "Asha"}
    last_topic_docs: list = field(default_factory=list)

    def add(self, role, text): self.turns.append((role, text))
    def last_user_questions(self, n=2):
        return [t for r, t in self.turns if r == "user"][-n:]

    def learn_facts(self, text):
        m = re.search(r"\bmy name is ([A-Za-z]+)", text, re.I)
        if m: self.facts["name"] = m.group(1).capitalize()
        m = re.search(r"\bi(?:'m| am) (?:an? )?(undergraduate|postgraduate)", text, re.I)
        if m: self.facts["level"] = m.group(1).lower()


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class SupportBot:
    def __init__(self, api_key=None, model=None):
        self.retriever = Retriever()
        self.memory = Memory()
        self.client, self.last_error = None, None
        api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL") or GROQ_MODEL
        if api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                self._resolve_model()
            except Exception as e:
                self.last_error = str(e)

    def _resolve_model(self):
        """Ensure the chosen model exists or fallback to available chat models."""
        if not self.client:
            return
        try:
            available = {m.id for m in self.client.models.list().data}
            if self.model not in available:
                candidates = [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.8-27b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant"
                ]
                for c in candidates:
                    if c in available:
                        self.model = c
                        break
        except Exception:
            pass

    def _contextual_query(self, q):
        """Short / pronoun-heavy questions are follow-ups: enrich with earlier user turns."""
        words = re.findall(r"[a-z']+", q.lower())
        is_followup = any(w in FOLLOWUP_WORDS for w in words) or q.lower().strip().startswith(FOLLOWUP_STARTS)
        prev = self.memory.last_user_questions(2)
        extra = " ".join(prev) if (is_followup and prev) else ""
        # also reuse the topic of the previous retrieval
        topic = " ".join(d["question"] for d in self.memory.last_topic_docs[:1]) if is_followup else ""
        return f"{q} {extra} {topic}".strip()

    def ask(self, question):
        self.memory.learn_facts(question)
        results = self.retriever.search(question)                       # 1) try the question alone
        if not results:
            results = self.retriever.search(self._contextual_query(question))  # 2) use memory
        self.memory.add("user", question)
        if not results:
            answer = self._greet(NOT_FOUND)
            self.memory.add("assistant", answer)
            return answer, []
        self.memory.last_topic_docs = [d for _, d in results]
        answer = self._generate(question, results)
        self.memory.add("assistant", answer)
        return answer, results

    def _greet(self, text):
        name = self.memory.facts.get("name")
        return f"{name}, {text}" if name else text

    def _generate(self, question, results):
        context = "\n\n".join(f"[FAQ {d['id']}] Q: {d['question']}\nA: {d['answer']}" for _, d in results)
        if self.client:
            history = "\n".join(f"{r}: {t}" for r, t in self.memory.turns[-8:-1])
            facts = ", ".join(f"{k}={v}" for k, v in self.memory.facts.items()) or "none"
            system = ("You are a University Helpdesk assistant. Answer ONLY using the CONTEXT. "
                      "If the context does not contain the answer, reply exactly: " + NOT_FOUND +
                      " Use the conversation history to resolve follow-up references. "
                      "Be concise, and cite FAQ ids like [FAQ 4].")
            msg = f"CONTEXT:\n{context}\n\nKNOWN USER FACTS: {facts}\n\nHISTORY:\n{history}\n\nQUESTION: {question}"
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=400,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
                self.last_error = None
                return r.choices[0].message.content.strip()
            except Exception as e:
                self.last_error = str(e)  # shown in sidebar; falls back to extractive answer
        # Offline extractive fallback: answer is the best-matching FAQ text verbatim
        best = results[0][1]
        text = f"{best['answer']} [FAQ {best['id']}]"
        return self._greet(text) if self.memory.facts.get("name") and len(self.memory.turns) <= 1 else text

    def reset(self):
        self.memory = Memory()
