# coaction_agent_platform/app/main.py
"""FastAPI application entry point for the Coaction Agent Platform.

Wires all layers per HLD:
- Boto3SessionFactory (centralized AWS client creation)
- RuntimeOrchestrator (standard execution pipeline)
- Control plane (agent registry, execution profiles)
- Services (authorization, guardrails, memory, model gateway, tool gateway, telemetry, audit)
- Middleware (correlation ID, error handling)
- Routers (auth, sessions, knowledge bases, agent invoke)
"""

import os
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# AWS Adapters
from coaction_agent_platform.adapters.aws.boto3_factory import Boto3SessionFactory
from coaction_agent_platform.adapters.aws.cognito import CognitoAdapter, CognitoConfig
from coaction_agent_platform.adapters.aws.dynamodb import DynamoDBAdapter
from coaction_agent_platform.adapters.aws.bedrock_kb_manager import BedrockKBManager

# Control Plane
from coaction_agent_platform.control_plane.agent_registry import AgentRegistryRepository
from coaction_agent_platform.control_plane.execution_profile_repository import ExecutionProfileRepository

# Services
from coaction_agent_platform.services.authorization import AuthorizationService
from coaction_agent_platform.services.guardrails import GuardrailService
from coaction_agent_platform.services.memory import AgentCoreMemoryProvider
from coaction_agent_platform.services.model_gateway import BedrockModelGateway
from coaction_agent_platform.services.tool_gateway import AgentCoreReadOnlyToolGateway
from coaction_agent_platform.services.telemetry import CloudWatchTelemetryEmitter
from coaction_agent_platform.services.audit import MetadataOnlyAuditLogger
from coaction_agent_platform.services.agent_service import AgentService

# Runtime
from coaction_agent_platform.runtime.orchestrator import RuntimeOrchestrator
from coaction_agent_platform.runtime.response_composer import ResponseComposer

# Identity
from coaction_agent_platform.app.dependencies.identity import init_jwt_verifier

# Middleware
from coaction_agent_platform.app.middleware.correlation import CorrelationIdMiddleware
from coaction_agent_platform.app.middleware.errors import ErrorHandlerMiddleware

# Routers
from coaction_agent_platform.app.routers.auth_router import router as auth_router, init_auth_router
from coaction_agent_platform.app.routers.session_router import router as session_router, init_session_router
from coaction_agent_platform.app.routers.kb_router import router as kb_router, init_kb_router
from coaction_agent_platform.app.routers.agent_router import router as agent_router, init_agent_router

logger = structlog.get_logger(__name__)


def _env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize all services at startup per HLD build sequence."""
    logger.info("app_starting")

    # ── Configuration from environment ──
    region = _env("AWS_REGION", "us-east-1")
    cognito_user_pool_id = _env("COGNITO_USER_POOL_ID")
    cognito_app_client_id = _env("COGNITO_APP_CLIENT_ID")
    dynamodb_table = _env("DYNAMODB_TABLE_NAME", "CoactionPlatform")
    kb_role_arn = _env("BEDROCK_KB_ROLE_ARN")
    embedding_model_arn = _env(
        "EMBEDDING_MODEL_ARN",
        f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0",
    )
    rds_resource_arn = _env("RDS_RESOURCE_ARN")
    rds_credentials_secret_arn = _env("RDS_CREDENTIALS_SECRET_ARN")
    config_dir = _env("CONFIG_DIR", "config/execution_profiles")

    # ── Step 1: Boto3 Client Factory (HLD §15) ──
    boto3_factory = Boto3SessionFactory(region_name=region)

    # ── Step 2: Cognito Auth ──
    cognito_adapter = None
    if cognito_user_pool_id and cognito_app_client_id:
        cognito_config = CognitoConfig(
            region=region,
            user_pool_id=cognito_user_pool_id,
            app_client_id=cognito_app_client_id,
        )
        cognito_adapter = CognitoAdapter(cognito_config)
        init_jwt_verifier(region, cognito_user_pool_id, cognito_app_client_id)
        logger.info("cognito_initialized")
    else:
        logger.warning("cognito_not_configured")

    # ── Step 3: DynamoDB ──
    dynamodb_adapter = DynamoDBAdapter(table_name=dynamodb_table, region=region)

    # ── Step 4: Control Plane Repositories (HLD §5) ──
    agent_registry = AgentRegistryRepository(dynamodb_adapter=dynamodb_adapter)
    profile_repo = ExecutionProfileRepository(
        dynamodb_adapter=dynamodb_adapter,
        config_dir=config_dir,
    )

    # Register the default underwriting agent
    agent_registry.register("coaction-underwriting", "latest")

    # ── Step 5: Services (HLD §8, §9, §12) ──
    authorization = AuthorizationService()
    guardrails = GuardrailService(boto3_factory=boto3_factory)
    memory = AgentCoreMemoryProvider(dynamodb_adapter=dynamodb_adapter, boto3_factory=boto3_factory)
    model_gateway = BedrockModelGateway(region=region)
    tool_gateway = AgentCoreReadOnlyToolGateway(boto3_factory=boto3_factory)
    response_composer = ResponseComposer()
    telemetry = CloudWatchTelemetryEmitter(boto3_factory=boto3_factory)
    audit = MetadataOnlyAuditLogger()

    # ── Step 6: Runtime Orchestrator (HLD §8) ──
    orchestrator = RuntimeOrchestrator(
        profile_repo=profile_repo,
        authorization=authorization,
        guardrails=guardrails,
        retriever=None,  # Retrieval happens inside Strands agent tools
        memory=memory,
        model_gateway=model_gateway,
        tool_gateway=tool_gateway,
        response_composer=response_composer,
        telemetry=telemetry,
        audit=audit,
    )

    # ── Step 7: Agent Service ──
    agent_service = AgentService(dynamodb=dynamodb_adapter, region=region)

    # ── Step 8: Bedrock KB Manager ──
    kb_manager = BedrockKBManager(
        region=region,
        role_arn=kb_role_arn,
        embedding_model_arn=embedding_model_arn,
    )
    kb_manager._rds_resource_arn = rds_resource_arn
    kb_manager._rds_credentials_secret_arn = rds_credentials_secret_arn

    # ── Step 9: Wire Routers ──
    init_auth_router(cognito_adapter, dynamodb_adapter)
    init_session_router(dynamodb_adapter)
    init_kb_router(kb_manager, dynamodb_adapter)
    init_agent_router(agent_service)

    logger.info(
        "app_ready",
        region=region,
        dynamodb_table=dynamodb_table,
        config_dir=config_dir,
    )

    yield  # Application runs

    logger.info("app_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application per HLD §10."""
    app = FastAPI(
        title="Coaction Agent Platform",
        description=(
            "Project Vega — Standard Agent Runtime. "
            "Configuration-driven agent platform with Strands orchestration, "
            "Bedrock KB retrieval, AgentCore Memory, Cognito auth, and DynamoDB storage."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── Middleware (HLD §12) ──
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers (HLD §10) — prefixed with /v1 ──
    app.include_router(auth_router, prefix="/v1")
    app.include_router(session_router, prefix="/v1")
    app.include_router(kb_router, prefix="/v1")
    app.include_router(agent_router, prefix="/v1")

    # ── Health & Readiness (HLD §10) ──
    @app.get("/health")
    async def health():
        """Liveness check."""
        return {"status": "healthy", "service": "coaction-agent-platform"}

    @app.get("/ready")
    async def ready():
        """Dependency readiness check."""
        return {"status": "ready", "service": "coaction-agent-platform"}

    return app


# Create the app instance
app = create_app()
