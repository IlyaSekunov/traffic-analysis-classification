"""
Configuration settings for the IT Developer Level Classification project.
"""

# IT-related keywords for filtering resumes
IT_KEYWORDS = [
    'разработчик', 'developer', 'программист', 'engineer',
    'data scientist', 'data science', 'ml engineer', 'ai engineer',
    'web developer', 'backend', 'frontend', 'fullstack', 'full stack',
    'системный администратор', 'sysadmin', 'devops',
    'тестировщик', 'qa engineer', 'qa', 'quality assurance',
    'аналитик данных', 'data analyst', 'бизнес-аналитик',
    'android developer', 'ios developer', 'mobile developer',
    'game developer', 'геймдев'
]

# Keywords for level determination
SENIOR_KEYWORDS = {'senior', 'сеньор', 'ведущий', 'старший', 'руководитель'}
MIDDLE_KEYWORDS = {'middle', 'миддл', 'опытный'}
JUNIOR_KEYWORDS = {'junior', 'джуниор', 'младший', 'начальный'}

# Experience thresholds (in months)
JUNIOR_EXPERIENCE_MAX = 12  # Up to 1 year
MIDDLE_EXPERIENCE_MAX = 60  # 1 to 5 years
SENIOR_EXPERIENCE_MIN = 60  # 5+ years

# Salary validation thresholds
MIN_SALARY = 10000
MAX_SALARY = 5000000

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Number of top cities to keep
TOP_CITIES_COUNT = 20