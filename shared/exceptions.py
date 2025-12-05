"""
Custom exception classes for ComfyUI DonutNodes.

This module provides a hierarchy of custom exceptions for handling various
error conditions that can occur during model merging and processing operations.
"""


class DonutNodeError(Exception):
    """Base exception class for all DonutNodes-related errors.
    
    This serves as the base class for all custom exceptions in the DonutNodes
    package. It allows for catching all DonutNodes-specific errors with a
    single except clause.
    """
    pass


class ModelError(DonutNodeError):
    """Base class for model-related errors.
    
    Raised when there are issues with model loading, processing, or validation.
    """
    pass


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded.
    
    This exception is raised when there are issues loading a model file,
    such as file not found, corrupted file, or unsupported format.
    """
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails.
    
    This exception is raised when a loaded model fails validation checks,
    such as incompatible architecture, missing required components, or
    invalid parameter shapes.
    """
    pass


class MergeError(DonutNodeError):
    """Base class for merge operation errors.
    
    Raised when there are issues during model merging operations.
    """
    pass


class IncompatibleModelsError(MergeError):
    """Raised when models are incompatible for merging.
    
    This exception is raised when attempting to merge models that have
    incompatible architectures, different parameter shapes, or other
    structural differences that prevent successful merging.
    """
    pass


class MergeConfigurationError(MergeError):
    """Raised when merge configuration is invalid.
    
    This exception is raised when merge parameters are invalid or
    contradictory, such as invalid merge strengths, unsupported merge
    methods, or conflicting configuration options.
    """
    pass


class MemoryExhaustionError(DonutNodeError):
    """Raised when system runs out of memory during processing.
    
    This exception is raised when memory usage exceeds available system
    memory during model loading, merging, or other memory-intensive
    operations. It provides information about the operation that failed
    and suggestions for reducing memory usage.
    """
    
    def __init__(self, message="Memory exhausted during operation", operation=None, suggestion=None):
        """Initialize MemoryExhaustionError.
        
        Args:
            message (str): Error message describing the memory exhaustion
            operation (str, optional): The operation that caused memory exhaustion
            suggestion (str, optional): Suggested action to resolve the issue
        """
        super().__init__(message)
        self.operation = operation
        self.suggestion = suggestion
        
    def __str__(self):
        """Return formatted error message."""
        parts = [str(self.args[0])]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class ComputationError(DonutNodeError):
    """Base class for computation-related errors.
    
    Raised when there are issues during mathematical computations or
    tensor operations.
    """
    pass


class TensorShapeError(ComputationError):
    """Raised when tensor shapes are incompatible for an operation.
    
    This exception is raised when tensor operations fail due to shape
    mismatches, such as matrix multiplication with incompatible dimensions
    or broadcasting errors.
    """
    pass


class NumericalInstabilityError(ComputationError):
    """Raised when numerical computations become unstable.
    
    This exception is raised when computations result in NaN values,
    infinite values, or other numerical instabilities that would
    compromise the integrity of the results.
    """
    pass


class ConfigurationError(DonutNodeError):
    """Raised when node configuration is invalid.
    
    This exception is raised when node input parameters are invalid,
    missing required parameters, or contain contradictory settings.
    """
    pass


class UnsupportedOperationError(DonutNodeError):
    """Raised when an unsupported operation is requested.
    
    This exception is raised when attempting to perform operations
    that are not supported by the current implementation or hardware
    configuration.
    """
    pass


class ResourceError(DonutNodeError):
    """Base class for resource-related errors.
    
    Raised when there are issues with system resources such as disk space,
    file permissions, or hardware capabilities.
    """
    pass


class InsufficientDiskSpaceError(ResourceError):
    """Raised when there is insufficient disk space for an operation.
    
    This exception is raised when operations require more disk space
    than is available, such as when saving large merged models.
    """
    pass


class FilePermissionError(ResourceError):
    """Raised when file operations fail due to permission issues.
    
    This exception is raised when the application lacks necessary
    permissions to read from or write to required files or directories.
    """
    pass


class TimeoutError(DonutNodeError):
    """Raised when an operation times out.
    
    This exception is raised when operations take longer than the
    configured timeout period, which may indicate system overload
    or infinite loops.
    """
    pass


# Exception hierarchy summary:
# DonutNodeError (base)
# ├── ModelError
# │   ├── ModelLoadError
# │   └── ModelValidationError
# ├── MergeError
# │   ├── IncompatibleModelsError
# │   └── MergeConfigurationError
# ├── ComputationError
# │   ├── TensorShapeError
# │   └── NumericalInstabilityError
# ├── ConfigurationError
# ├── UnsupportedOperationError
# ├── ResourceError
# │   ├── InsufficientDiskSpaceError
# │   └── FilePermissionError
# ├── MemoryExhaustionError
# └── TimeoutError