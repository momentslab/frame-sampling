"""
Video Backend Manager for customizing video reading backends.

This module provides a clean interface for registering and managing
custom video reading backends across different models.
"""

import logging
from typing import Dict, Callable, Any

logger = logging.getLogger(__name__)


class VideoBackendManager:
    """
    Manages video reading backends with clean registration and context management.
    
    This class provides a centralized way to register custom video readers
    and temporarily override backends for specific models.
    """
    
    def __init__(self):
        self._original_backends: Dict[str, Callable] = {}
        self._custom_backends: Dict[str, Callable] = {}
        self._active_patches: Dict[str, str] = {}
    
    def register_backend(self, name: str, backend_func: Callable, description: str = ""):
        """
        Register a custom video backend.
        
        Args:
            name: Backend name (e.g., 'torchcodec_custom')
            backend_func: The video reading function
            description: Optional description of the backend
        """
        self._custom_backends[name] = backend_func
        logger.info(f"Registered custom video backend '{name}': {description}")
    
    def patch_backend(self, target_module, backend_name: str, custom_backend_name: str):
        """
        Patch a backend in the target module's VIDEO_READER_BACKENDS.

        Args:
            target_module: Module containing VIDEO_READER_BACKENDS
            backend_name: Name of backend to replace (e.g., 'torchcodec')
            custom_backend_name: Name of registered custom backend
        """
        if not hasattr(target_module, 'VIDEO_READER_BACKENDS'):
            raise ValueError(f"Module {target_module} doesn't have VIDEO_READER_BACKENDS")

        if custom_backend_name not in self._custom_backends:
            raise ValueError(f"Custom backend '{custom_backend_name}' not registered")

        if backend_name not in self._original_backends:
            self._original_backends[backend_name] = target_module.VIDEO_READER_BACKENDS.get(backend_name)

        target_module.VIDEO_READER_BACKENDS[backend_name] = self._custom_backends[custom_backend_name]
        self._active_patches[backend_name] = custom_backend_name

        logger.info(f"✅ Patched VIDEO_READER_BACKENDS['{backend_name}'] with '{custom_backend_name}'")

    def patch_module_function(self, target_module, function_name: str, custom_backend_name: str):
        """
        Patch a function in a module (for SmolVLM-style patching).

        Args:
            target_module: Module containing the function to patch
            function_name: Name of function to replace (e.g., 'load_video')
            custom_backend_name: Name of registered custom backend
        """
        if not hasattr(target_module, function_name):
            raise ValueError(f"Module {target_module} doesn't have function '{function_name}'")

        if custom_backend_name not in self._custom_backends:
            raise ValueError(f"Custom backend '{custom_backend_name}' not registered")

        module_name = getattr(target_module, '__name__', str(target_module))
        patch_key = f"{module_name}.{function_name}"

        if patch_key not in self._original_backends:
            self._original_backends[patch_key] = getattr(target_module, function_name)

        setattr(target_module, function_name, self._custom_backends[custom_backend_name])
        self._active_patches[patch_key] = custom_backend_name

        logger.info(f"✅ Patched {module_name}.{function_name} with '{custom_backend_name}'")

    def register_and_patch_backend(self, target_module, backend_name: str, custom_backend_name: str, backend_func, description: str = ""):
        """
        Register a custom backend and patch it in one call (for VIDEO_READER_BACKENDS).

        Args:
            target_module: Module containing VIDEO_READER_BACKENDS
            backend_name: Name of backend to replace (e.g., 'torchcodec')
            custom_backend_name: Name for the custom backend
            backend_func: The custom backend function
            description: Description of the backend
        """
        self.register_backend(custom_backend_name, backend_func, description)
        self.patch_backend(target_module, backend_name, custom_backend_name)

    def register_and_patch_module_function(self, target_module, function_name: str, custom_backend_name: str, backend_func, description: str = ""):
        """
        Register a custom backend and patch a module function in one call.

        Args:
            target_module: Module containing the function to patch
            function_name: Name of function to replace (e.g., 'load_video')
            custom_backend_name: Name for the custom backend
            backend_func: The custom backend function
            description: Description of the backend
        """
        self.register_backend(custom_backend_name, backend_func, description)
        self.patch_module_function(target_module, function_name, custom_backend_name)


    

    def restore_backend(self, target_module, backend_name: str):
        """
        Restore original backend.

        Args:
            target_module: Module containing VIDEO_READER_BACKENDS
            backend_name: Name of backend to restore
        """
        if backend_name in self._original_backends:
            original = self._original_backends[backend_name]
            if original is not None:
                target_module.VIDEO_READER_BACKENDS[backend_name] = original
            else:
                target_module.VIDEO_READER_BACKENDS.pop(backend_name, None)

            self._active_patches.pop(backend_name, None)
            logger.info(f"🔄 Restored original VIDEO_READER_BACKENDS['{backend_name}']")

    def restore_module_function(self, target_module, function_name: str):
        """
        Restore original function in a module.

        Args:
            target_module: Module containing the function
            function_name: Name of function to restore
        """
        module_name = getattr(target_module, '__name__', str(target_module))
        patch_key = f"{module_name}.{function_name}"

        if patch_key in self._original_backends:
            original = self._original_backends[patch_key]
            setattr(target_module, function_name, original)
            self._active_patches.pop(patch_key, None)
            logger.info(f"🔄 Restored original {module_name}.{function_name}")


    
    def list_backends(self) -> Dict[str, Any]:
        """List all registered custom backends and active patches."""
        return {
            'custom_backends': list(self._custom_backends.keys()),
            'active_patches': self._active_patches.copy(),
            'original_backends': list(self._original_backends.keys())
        }

video_backend_manager = VideoBackendManager()
