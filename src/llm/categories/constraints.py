"""
Category constraints for growth rate validation.
"""

from typing import Dict, List, Any


class CategoryConstraints:
    """Manages growth rate constraints for categories."""
    
    def __init__(self, categories: List[Dict[str, Any]] = None):
        """
        Initialize constraints manager.
        
        Args:
            categories: List of category definitions
        """
        self.constraints = {}
        if categories:
            self.build_constraints(categories)
    
    def build_constraints(
        self,
        categories: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a dictionary of category constraints.
        
        Args:
            categories: List of category definitions
            
        Returns:
            Dict mapping category names to their min/max growth constraints
        """
        self.constraints = {}
        for cat in categories:
            self.constraints[cat['name']] = {
                'min': cat.get('growth_constraint_min', -10.0),
                'max': cat.get('growth_constraint_max', 10.0)
            }
        return self.constraints
    
    def apply_constraint(
        self,
        growth_rate: float,
        category: str
    ) -> float:
        """
        Apply category constraints to a growth rate.
        
        Args:
            growth_rate: Raw growth rate
            category: Category name
            
        Returns:
            Constrained growth rate
        """
        if category in self.constraints:
            min_rate = self.constraints[category]['min']
            max_rate = self.constraints[category]['max']
            return max(min_rate, min(max_rate, growth_rate))
        return growth_rate
    
    def get_constraint(self, category: str) -> Dict[str, float]:
        """
        Get constraints for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dict with 'min' and 'max' constraints
        """
        return self.constraints.get(
            category,
            {'min': -10.0, 'max': 10.0}  # Default constraints
        )
    
    def validate_growth_rate(
        self,
        growth_rate: float,
        category: str
    ) -> bool:
        """
        Check if growth rate is within category constraints.
        
        Args:
            growth_rate: Growth rate to validate
            category: Category name
            
        Returns:
            True if within constraints
        """
        constraint = self.get_constraint(category)
        return constraint['min'] <= growth_rate <= constraint['max']
    
    def get_all_constraints(self) -> Dict[str, Dict[str, float]]:
        """
        Get all category constraints.
        
        Returns:
            Dictionary of all constraints
        """
        return self.constraints.copy()


# Backward compatibility functions
def build_category_constraints(
    categories: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Backward compatibility wrapper."""
    constraints_manager = CategoryConstraints(categories)
    return constraints_manager.constraints


def apply_category_constraints(
    growth_rate: float,
    category: str,
    constraints: Dict[str, Dict[str, float]]
) -> float:
    """Backward compatibility wrapper."""
    if category in constraints:
        min_rate = constraints[category]['min']
        max_rate = constraints[category]['max']
        return max(min_rate, min(max_rate, growth_rate))
    return growth_rate