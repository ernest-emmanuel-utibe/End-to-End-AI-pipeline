.
├── .github/
│   └── workflows/
│       └── cicd.yaml        # GitHub Actions Automation
├── pipeline/
│   ├── Dockerfile.pipeline  # Training pipeline container instructions
│   └── pipeline.py          # Your python training script from earlier
├── server/
│   ├── Dockerfile.server    # FastAPI Inference API container instructions
│   └── main.py              # Your python FastAPI app script from earlier
├── docker-compose.yml       # Infrastructure container mesh orchestration
└── requirements.txt         # Unified environment dependencies
