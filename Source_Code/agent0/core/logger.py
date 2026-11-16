import logging
import sys
from pathlib import Path
from typing import Optional

logger: logging.Logger = logging.getLogger("agent0")
LOG_LLM_IO: bool = False
LLM_IO_MAX_CHARS: int = 20000

def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    quiet: bool = False,
    log_llm_io: bool = False,
    llm_io_max_chars: int = 20000,
) -> logging.Logger:
    global LOG_LLM_IO
    global LLM_IO_MAX_CHARS
    
    LOG_LLM_IO = bool(log_llm_io)
    LLM_IO_MAX_CHARS = max(0, int(llm_io_max_chars))
    
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(resolved_level)
    logger.propagate = False
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.WARNING if quiet else resolved_level
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized to {log_file}")
        
    logger.info("LLM I/O logging: %s (max_chars=%s)", "enabled" if LOG_LLM_IO else "disabled", LLM_IO_MAX_CHARS)
    return logger
