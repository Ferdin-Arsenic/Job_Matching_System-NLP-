FIELD_GROUPS = {
    "IT": (
        "Java Developer", "Python Developer", "DotNet Developer", "SQL Developer",
        "SAP Developer", "ETL Developer", "React Developer", "Web Designing",
        "Database", "DevOps", "Testing", "Network Security Engineer", "Data Science",
        "Blockchain", "Information Technology",
    ),
    "Engineering & Construction": (
        "Electrical Engineering", "Mechanical Engineer", "Civil Engineer",
        "Building and Construction", "Architecture", "Automobile",
    ),
    "Finance": ("Accountant", "Finance", "Banking"),
    "Business & Management": (
        "Business Analyst", "Consultant", "Management", "Operations Manager", "PMO",
        "Sales", "Human Resources", "BPO", "Advocate",
    ),
    "Creative & Media": ("Designing", "Digital Media", "Arts", "Apparel", "Public Relations"),
    "Education & Health": ("Education", "Health and Fitness"),
    "Services & Industry": ("Food and Beverages", "Aviation", "Agriculture"),
}
FIELD_MAP = {sub: field for field, subs in FIELD_GROUPS.items() for sub in subs}
FIELD_LABELS = tuple(FIELD_GROUPS)
SUB_FIELD_LABELS = tuple(sorted(FIELD_MAP))
LEVEL_LABELS = ("Entry", "Mid", "Senior")
PROFILE_LEVELS = (*LEVEL_LABELS, "Unknown")
