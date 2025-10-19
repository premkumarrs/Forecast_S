"""
Rate limiting for GDELT API calls.
"""

import threading
import time
import random
import logging

logger = logging.getLogger(__name__)


class GDELTRateLimiter:
    """Thread-safe rate limiter for GDELT API with random delays."""
    
    def __init__(self, min_seconds: float = 5.0, max_seconds: float = 6.0):
        """
        Initialize rate limiter.
        
        Args:
            min_seconds: Minimum seconds between API calls
            max_seconds: Maximum seconds between API calls
        """
        self.min_interval = min_seconds
        self.max_interval = max_seconds
        self.last_call = 0
        self.lock = threading.Lock()
        self.call_count = 0
    
    def acquire(self):
        """Wait if necessary to respect rate limit with random interval."""
        with self.lock:
            # Determine a random interval for this specific call
            target_interval = random.uniform(self.min_interval, self.max_interval)
            
            elapsed = time.time() - self.last_call
            if elapsed < target_interval:
                wait_time = target_interval - elapsed
                logger.debug(
                    f"Rate limiting - waiting {wait_time:.1f} seconds "
                    f"(randomized interval)"
                )
                time.sleep(wait_time)
            
            self.last_call = time.time()
            self.call_count += 1
    
    def reset(self):
        """Reset the rate limiter."""
        with self.lock:
            self.last_call = 0
            self.call_count = 0
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                'call_count': self.call_count,
                'last_call': self.last_call,
                'min_interval': self.min_interval,
                'max_interval': self.max_interval
            }
    
    def adjust_rate(self, min_seconds: float, max_seconds: float):
        """
        Adjust rate limiting parameters.
        
        Args:
            min_seconds: New minimum seconds between calls
            max_seconds: New maximum seconds between calls
        """
        with self.lock:
            self.min_interval = min_seconds
            self.max_interval = max_seconds
            logger.info(
                f"Rate limiter adjusted: {min_seconds:.1f}s - {max_seconds:.1f}s"
            )