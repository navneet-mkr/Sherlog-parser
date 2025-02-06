import time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ProgressStats:
    """Class to store progress statistics."""
    completed: int
    total: int
    speed: float
    eta: str
    progress_pct: float

class DownloadStats:
    """Class to track download statistics with moving average speed calculation."""
    def __init__(self, window_size: int = 5):
        self.speeds: List[float] = []
        self.window_size = window_size
        self.last_update = time.time()
        self.last_completed = 0
        self.start_time = time.time()

    def update(self, completed: int) -> float:
        """Update stats and return current speed in bytes/second."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        if elapsed > 0:
            speed = (completed - self.last_completed) / elapsed
            self.speeds.append(speed)
            # Keep only the last window_size speeds
            if len(self.speeds) > self.window_size:
                self.speeds.pop(0)
            self.last_update = current_time
            self.last_completed = completed
            return sum(self.speeds) / len(self.speeds)
        return 0

    def get_eta(self, completed: int, total: int) -> str:
        """Calculate ETA based on moving average speed."""
        if not self.speeds or completed == 0:
            return "calculating..."
        
        avg_speed = sum(self.speeds) / len(self.speeds)
        remaining_bytes = total - completed
        eta_seconds = remaining_bytes / avg_speed if avg_speed > 0 else 0
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f} seconds"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.0f} minutes"
        else:
            return f"{eta_seconds/3600:.1f} hours"

    def get_stats(self, completed: int, total: int) -> ProgressStats:
        """Get current progress statistics."""
        speed = self.update(completed)
        progress_pct = (completed / total) * 100 if total > 0 else 0
        return ProgressStats(
            completed=completed,
            total=total,
            speed=speed,
            eta=self.get_eta(completed, total),
            progress_pct=progress_pct
        )

def format_size(size_bytes: float) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB" 