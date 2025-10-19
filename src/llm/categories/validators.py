"""
Category validation utilities.
"""

from typing import Set, List, Dict, Any


class CategoryValidator:
    """Validates and normalizes categories."""
    
    def __init__(self, valid_categories: List[str] = None):
        """
        Initialize validator.
        
        Args:
            valid_categories: List of valid category names
        """
        self.valid_categories = set(valid_categories) if valid_categories else set()
        self.default_category = 'Neutral/Noise'
    
    def set_valid_categories(self, categories: List[str]):
        """
        Set valid categories.
        
        Args:
            categories: List of valid category names
        """
        self.valid_categories = set(categories)
    
    def validate(self, category: str) -> str:
        """
        Validate and normalize a category name.
        
        Args:
            category: Raw category from LLM
            
        Returns:
            Valid category name or default
        """
        if not category:
            return self.default_category
        
        # Exact match
        if category in self.valid_categories:
            return category
        
        # Case-insensitive match
        category_lower = category.lower()
        for valid_cat in self.valid_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat
        
        # Partial match (contains)
        for valid_cat in self.valid_categories:
            if valid_cat.lower() in category_lower or category_lower in valid_cat.lower():
                return valid_cat
        
        return self.default_category
    
    def validate_batch(self, categories: List[str]) -> List[str]:
        """
        Validate multiple categories.
        
        Args:
            categories: List of raw categories
            
        Returns:
            List of validated categories
        """
        return [self.validate(cat) for cat in categories]
    
    def is_valid(self, category: str) -> bool:
        """
        Check if category is valid.
        
        Args:
            category: Category name
            
        Returns:
            True if valid
        """
        return category in self.valid_categories
    
    def get_valid_categories(self) -> List[str]:
        """
        Get list of valid categories.
        
        Returns:
            Sorted list of valid category names
        """
        return sorted(list(self.valid_categories))
    
    def add_category(self, category: str):
        """
        Add a new valid category.
        
        Args:
            category: Category name to add
        """
        self.valid_categories.add(category)
    
    def remove_category(self, category: str):
        """
        Remove a valid category.
        
        Args:
            category: Category name to remove
        """
        self.valid_categories.discard(category)


# Backward compatibility function
def validate_category(category: str, category_set: set) -> str:
    """
    Backward compatibility wrapper.
    
    Args:
        category: Raw category from LLM
        category_set: Set of valid category names
        
    Returns:
        Valid category name or 'Neutral/Noise'
    """
    validator = CategoryValidator(list(category_set))
    return validator.validate(category)