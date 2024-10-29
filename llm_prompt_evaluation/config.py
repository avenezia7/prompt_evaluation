# STANDARD PARAMETERS

# MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"   # per Claude
# MODEL = "incept5/llama3.1-claude"                     # per Ollama in locale se scaricato
MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_TYPE = "BEDROCK"
TEMPERATURE = 0.1
TOP_K = 20
TOP_P = 0.3
MAX_TOKENS = 4096
USE_HISTORY = True
SIMILARITY_THRESHOLD = 0.7
SIMILARITY_EMBEDDING_THRESHOLD = 0.7
BLUE_THRESHOLD = 0.7
ROUGE_THRESHOLD = 0.7
# INPUT_COST_PER_TOKEN = 0.003          # Costo per 1000 token di input claude
# OUTPUT_COST_PER_TOKEN = 0.015         # Costo per 1000 token di output claude
INPUT_COST_PER_TOKEN = 0.00025          # Costo per 1000 token di input haiku
OUTPUT_COST_PER_TOKEN = 0.00125         # Costo per 1000 token di output haiku

# MODEL PARAMETERS
AWS_REGION = "us-west-2"            # per AWS necessario specificare la regione
AWS_CLIENT_CONNECT_TIMEOUT = 120    # per client AWS timeout di connessione
AWS_CLIENT_READ_TIMEOUT = 120       # per client AWS timeout di lettura
AWS_MAX_ATTEMPTS = 1                 # per client AWS numero di tentativi massimi
URL_MODEL = None    # per Ollama necessario per specificare l'URL del modello http://localhost:11434
NUM_GPU = 1         # per Ollama numero di GPU
NUM_THREAD = 8      # per Ollama numero di thread
