#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LogUtilityModule
ProvidesLogFunctionality
"""

import os
import sys
import logging
import datetime
import traceback
from typing import Optional, Dict, Any, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class Logger:
    """
    LogClass
    Log、Fileconvert to、outputFunctionality
    """
    
    LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL
    }
    
    # LogConfig
    COLORS = {
        "DEBUG": "\033[36m",  # 
        "INFO": "\033[32m",  # 
        "WARNING": "\033[33m",  # 
        "ERROR": "\033[31m",  # 
        "CRITICAL": "\033[35m",  # 
        "RESET": "\033[0m"  # 
    }
    
    def __init__(self, name: str = "mmrag", **kwargs):
        """
        InitializeLog
        
        Args:
            name: Log
            log_level: Log
            log_file: LogFilePath
            console_output: output
            file_output: outputFile
            max_bytes: LogFileMax(convert to)
            backup_count: File
            use_color: output
            log_format: Log
            date_format: 
            **kwargs: Parameter
        """
        self.name = name
        self.log_level = kwargs.get("log_level", "INFO")
        self.log_file = kwargs.get("log_file", None)
        self.console_output = kwargs.get("console_output", True)
        self.file_output = kwargs.get("file_output", False)
        self.max_bytes = kwargs.get("max_bytes", 10 * 1024 * 1024)  # 10MB
        self.backup_count = kwargs.get("backup_count", 5)
        self.use_color = kwargs.get("use_color", True)
        
        default_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_format = kwargs.get("log_format", default_log_format)
        self.date_format = kwargs.get("date_format", "%Y-%m-%d %H:%M:%S")
        
        # Configlogger
        self.logger = self._create_logger()
        
        # LogDirectory
        if self.log_file and not os.path.exists(os.path.dirname(os.path.abspath(self.log_file))):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_file)))
            except Exception as e:
                print(f"Failed to create Log directory: {e}")
    
    def _create_logger(self) -> logging.Logger:
        """
        Configlogger
        
        Returns:
            Configlogger
        """
        # logger
        logger = logging.getLogger(self.name)
        
        # settingLog
        logger.setLevel(self._get_level(self.log_level))
        
        # LogLog, output
        logger.propagate = False
        
        # alreadyhandler
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Configoutput
        if self.console_output:
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)
        
        # ConfigFileoutput
        if self.file_output and self.log_file:
            file_handler = self._create_file_handler()
            logger.addHandler(file_handler)
        
        return logger
    
    def _create_console_handler(self) -> logging.Handler:
        """
        handler
        
        Returns:
            handler
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._get_level(self.log_level))
        
        # formatter
        if self.use_color:
            formatter = ColoredFormatter(
                self.log_format,
                datefmt=self.date_format,
                colors=self.COLORS
            )
        else:
            formatter = logging.Formatter(
                self.log_format,
                datefmt=self.date_format
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> logging.Handler:
        """
        Filehandler
        
        Returns:
            Filehandler
        """
        rotate_type = self.log_file.split('.')[-1] if '.' in self.log_file else None
        
        if rotate_type == "rotating":
            handler = RotatingFileHandler(
                self.log_file[:-9],  # remove.rotating
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
        elif rotate_type == "timed":
            handler = TimedRotatingFileHandler(
                self.log_file[:-6],  # remove.timed
                when="midnight",  # 
                interval=1,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
        else:
            # File
            handler = logging.FileHandler(self.log_file, encoding="utf-8")
        
        handler.setLevel(self._get_level(self.log_level))
        
        # formatter(FileLog)
        formatter = logging.Formatter(
            self.log_format,
            datefmt=self.date_format
        )
        
        handler.setFormatter(formatter)
        return handler
    
    def _get_level(self, level_str: str) -> int:
        """
        Log
        
        Args:
            level_str: Logcharacter
            
        Returns:
            Log
        """
        return self.LEVEL_MAP.get(level_str.upper(), logging.INFO)
    
    def debug(self, message: Union[str, Exception], **kwargs):
        """
        debugLog
        
        Args:
            message: Log
            **kwargs: Parameter
        """
        if isinstance(message, Exception):
            message = self._format_exception(message)
        self.logger.debug(message, **kwargs)
    
    def info(self, message: Union[str, Exception], **kwargs):
        """
        infoLog
        
        Args:
            message: Log
            **kwargs: Parameter
        """
        if isinstance(message, Exception):
            message = self._format_exception(message)
        self.logger.info(message, **kwargs)
    
    def warning(self, message: Union[str, Exception], **kwargs):
        """
        warningLog
        
        Args:
            message: Log
            **kwargs: Parameter
        """
        if isinstance(message, Exception):
            message = self._format_exception(message)
        self.logger.warning(message, **kwargs)
    
    def warn(self, message: Union[str, Exception], **kwargs):
        """
        warning
        """
        self.warning(message, **kwargs)
    
    def error(self, message: Union[str, Exception], **kwargs):
        """
        errorLog
        
        Args:
            message: Log
            **kwargs: Parameter
        """
        if isinstance(message, Exception):
            message = self._format_exception(message)
        self.logger.error(message, **kwargs)
    
    def critical(self, message: Union[str, Exception], **kwargs):
        """
        criticalLog
        
        Args:
            message: Log
            **kwargs: Parameter
        """
        if isinstance(message, Exception):
            message = self._format_exception(message)
        self.logger.critical(message, **kwargs)
    
    def fatal(self, message: Union[str, Exception], **kwargs):
        """
        critical
        """
        self.critical(message, **kwargs)
    
    def exception(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """
        RaisesLog
        
        Args:
            message: Error
            exc_info: Raises
            **kwargs: Parameter
        """
        if exc_info:
            full_message = f"{message}\n{self._format_exception(exc_info)}"
            self.logger.error(full_message, exc_info=False, **kwargs)
        else:
            self.logger.error(message, exc_info=True, **kwargs)
    
    def log(self, level: Union[str, int], message: Union[str, Exception], **kwargs):
        """
        Log
        
        Args:
            level: Log
            message: Log
            **kwargs: Parameter
        """
        if isinstance(level, str):
            level = self._get_level(level)
        
        if isinstance(message, Exception):
            message = self._format_exception(message)
        
        self.logger.log(level, message, **kwargs)
    
    def set_level(self, level: Union[str, int]):
        """
        settingLog
        
        Args:
            level: Log
        """
        if isinstance(level, str):
            level = self._get_level(level)
        
        self.log_level = level
        self.logger.setLevel(level)
        
        # handler
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def set_log_file(self, file_path: str):
        """
        settingLogFilePath
        
        Args:
            file_path: LogFilePath
        """
        self.log_file = file_path
        self.file_output = True
        
        # logger
        self.logger = self._create_logger()
    
    def disable_console_output(self):
        """
        output
        """
        self.console_output = False
        self.logger = self._create_logger()
    
    def enable_console_output(self):
        """
        output
        """
        self.console_output = True
        self.logger = self._create_logger()
    
    def disable_file_output(self):
        """
        Fileoutput
        """
        self.file_output = False
        self.logger = self._create_logger()
    
    def enable_file_output(self):
        """
        Fileoutput
        """
        if self.log_file:
            self.file_output = True
            self.logger = self._create_logger()
        else:
            print("Error: LogFile path not set")
    
    def _format_exception(self, exception: Exception) -> str:
        """
        RaisesInfo
        
        Args:
            exception: Raises
            
        Returns:
            RaisesInfocharacter
        """
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        formatted = f"Exception: {error_type}\nMessage: {error_message}\nStack Trace:\n{stack_trace}"
        return formatted
    
    def log_dict(self, level: Union[str, int], data: Dict[str, Any], prefix: str = ""):
        """
        Data
        
        Args:
            level: Log
            data: Data
            prefix: 
        """
        if isinstance(level, str):
            level = self._get_level(level)
        
        formatted = self._format_dict(data, prefix)
        self.logger.log(level, formatted)
    
    def _format_dict(self, data: Dict[str, Any], prefix: str = "", indent: int = 2) -> str:
        """
        character
        
        Args:
            data: Data
            prefix: 
            indent: 
            
        Returns:
            character
        """
        lines = []
        current_indent = prefix + " " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, current_indent))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: [")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_dict(item, current_indent))
                    else:
                        lines.append(f"{current_indent}{str(item)}")
                lines.append(f"{prefix}]")
            else:
                lines.append(f"{prefix}{key}: {str(value)}")
        
        return "\n".join(lines)
    
    def get_logger(self) -> logging.Logger:
        """
        logger
        
        Returns:
            logger
        """
        return self.logger

class ColoredFormatter(logging.Formatter):
    """
    Log
    """
    
    def __init__(self, fmt: str, datefmt: str = None, colors: Dict[str, str] = None):
        """
        Initialize
        
        Args:
            fmt: Log
            datefmt: 
            colors: Config
        """
        super().__init__(fmt, datefmt)
        self.colors = colors or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Log, 
        
        Args:
            record: Log
            
        Returns:
            Logcharacter
        """
        # SaveResult
        original_fmt = super().format(record)
        
        level_name = record.levelname
        if level_name in self.colors and "RESET" in self.colors:
            colored_fmt = f"{self.colors[level_name]}{original_fmt}{self.colors['RESET']}"
            return colored_fmt
        
        return original_fmt

# logger
_global_logger = None

_logger_cache = {}

def get_logger(name: str = "mmrag", **kwargs) -> Logger:
    """
    Log
    Log
    
    Args:
        name: Log
        **kwargs: InitializeParameter
        
    Returns:
        Logger
    """
    global _global_logger
    
    # Logalready
    if name in _logger_cache:
        return _logger_cache[name]
    
    logger = Logger(name, **kwargs)
    
    _logger_cache[name] = logger
    
    # Log, Save_global_logger
    if name == "mmrag":
        _global_logger = logger
    
    return logger

def setup_global_logger(**kwargs):
    """
    settingLog
    
    Args:
        **kwargs: InitializeParameter
    """
    global _global_logger
    _global_logger = Logger("mmrag", **kwargs)
    return _global_logger

__all__ = ["Logger", "ColoredFormatter", "get_logger", "setup_global_logger"]