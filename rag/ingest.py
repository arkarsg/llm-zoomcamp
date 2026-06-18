import requests
from minsearch import Index
from pydantic import BaseModel

COURSE_URL = "https://datatalks.club/faq/json/courses.json"
FAQ_URL_PREFIX = "https://datatalks.club/faq"

class CourseModel(BaseModel):
    course: str
    course_name: str
    path: str
    questions_count: int

def get(url: str) -> dict:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def get_faq(course: CourseModel):
    faq_url = f"{FAQ_URL_PREFIX}/{course.path}"
    return get(faq_url)

def load_courses() -> list[CourseModel]:
    courses = get(COURSE_URL)
    return [CourseModel(**course) for course in courses]

def load_all_faqs() -> list[dict]:
    courses = load_courses()
    documents = [
        doc
        for course in courses
        for doc in get_faq(course)
    ]
    return documents

def build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index