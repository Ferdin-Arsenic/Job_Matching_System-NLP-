"""Write the demo inputs: examples/job{1,2}.txt and examples/cvs/cv{1..5}.pdf.

ALL TEXT HERE IS SYNTHETIC: written for the demo, not taken from any dataset or real person, and not an evaluation
set. The "expected" notes are the author's judgement, not ground truth.
Run from the repo root:  python scripts/make_examples.py      (needs fpdf2)
"""
from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "examples"

JOBS = {
    "job1.txt": """Backend Software Engineer

We are looking for a Backend Software Engineer to build and maintain the services behind our logistics platform.

Responsibilities
- Design, build and maintain REST APIs and background workers in Python.
- Model data and write efficient queries in PostgreSQL.
- Package and deploy services with Docker on AWS; maintain CI/CD pipelines.
- Write unit and integration tests, review code, and take part in an Agile team.

Requirements
- 3+ years of professional experience with Python (Django or Flask).
- Strong knowledge of REST API design and relational databases (PostgreSQL).
- Experience with Docker, Git and AWS, plus CI/CD tools.
- Bachelor's degree in Computer Science or a related field.
- Good communication skills and a habit of writing clear documentation.
""",
    "job2.txt": """Staff Accountant

A growing distribution company is hiring a Staff Accountant to support the monthly close and daily accounting operations.

Responsibilities
- Process accounts payable and accounts receivable, and prepare bank and account reconciliations.
- Prepare journal entries and support the month-end close in line with GAAP.
- Assist with budgeting, financial reporting and audit schedules.
- Maintain the general ledger in QuickBooks and prepare reports in Excel.

Requirements
- 2+ years of accounting experience, including month-end close and reconciliations.
- Bachelor's degree in Accounting or Finance; CPA candidate or CPA is a plus.
- Working knowledge of GAAP, QuickBooks or SAP, and advanced Excel.
- Attention to detail and the ability to meet reporting deadlines.
""",
}

CVS = {  # author's expectation vs job1 / job2 in the comment
    "cv1.pdf": """Candidate 1 - Backend Developer

Summary
Backend developer with 5 years of experience building REST APIs and data services in Python.

Experience
Backend Developer, 2021 - Present
- Built and maintained REST APIs with Django and Flask serving more than 2 million requests per day.
- Designed PostgreSQL schemas and optimised slow queries.
- Containerised services with Docker, deployed on AWS, and set up CI/CD pipelines with GitHub Actions.
- Wrote unit and integration tests with pytest; worked in an Agile team and wrote technical documentation.

Software Developer, 2019 - 2021
- Developed internal tools in Python and SQL; used Git for code review and version control.

Education
Bachelor of Science in Computer Science

Skills
Python, Django, Flask, REST API, PostgreSQL, Docker, AWS, Git, CI/CD, pytest, Agile
""",  # expected: good fit for job1
    "cv2.pdf": """Candidate 2 - Junior Web Designer

Summary
Web designer with 1 year of experience creating websites for small businesses.

Experience
Web Designer, 2024 - Present
- Built responsive websites with HTML, CSS and JavaScript and customised WordPress themes.
- Designed page layouts in Figma and optimised images for fast loading.
- Wrote small Python scripts to automate content updates.

Education
Diploma in Information Technology

Skills
HTML, CSS, JavaScript, WordPress, Figma, basic Python
""",  # expected: same field (IT) as job1 but a weaker, different fit
    "cv3.pdf": """Candidate 3 - Mechanical Engineer

Summary
Mechanical engineer with 6 years of experience in product design and manufacturing.

Experience
Mechanical Design Engineer, 2020 - Present
- Designed machine components in SolidWorks and AutoCAD and produced technical drawings.
- Supported CNC machining, tolerance analysis and quality inspection on the production floor.
- Led root-cause analysis for equipment failures and improved maintenance schedules.

Production Engineer, 2018 - 2020
- Monitored production lines, reduced scrap rate and supported lean manufacturing projects.

Education
Bachelor of Engineering in Mechanical Engineering

Skills
SolidWorks, AutoCAD, CNC machining, lean manufacturing, GD&T, root-cause analysis
""",  # expected: different field from both jobs
    "cv4.pdf": """Candidate 4 - Accountant

Summary
Accountant with 4 years of experience in accounts payable, accounts receivable and month-end close.

Experience
Accountant, 2022 - Present
- Processed accounts payable and receivable and prepared bank reconciliations every month.
- Prepared journal entries and supported the month-end close under GAAP.
- Maintained the general ledger in QuickBooks and built financial reports in Excel.
- Supported the annual audit and budgeting process.

Junior Accountant, 2020 - 2022
- Handled invoices, payments and expense reports.

Education
Bachelor of Accounting; CPA candidate

Skills
GAAP, QuickBooks, Excel, reconciliations, accounts payable, accounts receivable, month-end close
""",  # expected: good fit for job2
    "cv5.pdf": """Candidate 5 - Primary School Teacher

Summary
Primary school teacher with 7 years of experience in classroom teaching and curriculum planning.

Experience
Class Teacher, 2018 - Present
- Planned and delivered lessons in mathematics and science for grades 3 to 5.
- Assessed student progress, communicated with parents and organised learning activities.
- Mentored new teachers and helped update the school curriculum.

Education
Bachelor of Education

Skills
Lesson planning, classroom management, assessment, curriculum development, communication
""",  # expected: different field from both jobs
}


def write_pdf(path: Path, text: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1") or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


if __name__ == "__main__":
    (OUT / "cvs").mkdir(parents=True, exist_ok=True)
    for name, text in JOBS.items():
        (OUT / name).write_text(text, encoding="utf-8")
    for name, text in CVS.items():
        write_pdf(OUT / "cvs" / name, text)
    print("written:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))
