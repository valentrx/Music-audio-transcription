import json
import time
import threading
from typing import Dict, Optional

class ProgressTracker:
    def __init__(self):
        self._progress_data: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def create_session(self, session_id: str, total_steps: int = 100):
        """Create a new progress tracking session"""
        with self._lock:
            self._progress_data[session_id] = {
                'progress': 0,
                'total': total_steps,
                'current_step': '',
                'status': 'starting',
                'last_updated': time.time()
            }
    
    def update_progress(self, session_id: str, progress: int, step_name: str = ''):
        """Update the progress for a given session"""
        with self._lock:
            if session_id in self._progress_data:
                self._progress_data[session_id].update({
                    'progress': min(progress, self._progress_data[session_id]['total']),
                    'current_step': step_name,
                    'status': 'processing',
                    'last_updated': time.time()
                })
    
    def complete_session(self, session_id: str):
        """Mark a session as completed"""
        with self._lock:
            if session_id in self._progress_data:
                self._progress_data[session_id].update({
                    'progress': self._progress_data[session_id]['total'],
                    'current_step': 'Completed',
                    'status': 'completed',
                    'last_updated': time.time()
                })
    
    def error_session(self, session_id: str, error_message: str):
        """Mark a session as having an error"""
        with self._lock:
            if session_id in self._progress_data:
                self._progress_data[session_id].update({
                    'current_step': f'Error: {error_message}',
                    'status': 'error',
                    'last_updated': time.time()
                })
    
    def get_progress(self, session_id: str) -> Optional[Dict]:
        """Get the current progress for a session"""
        with self._lock:
            return self._progress_data.get(session_id, None)
    
    def cleanup_session(self, session_id: str):
        """Remove a session from tracking"""
        with self._lock:
            self._progress_data.pop(session_id, None)

# Global instance
progress_tracker = ProgressTracker()