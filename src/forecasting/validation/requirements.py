"""
Method requirements definitions.
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class MethodRequirements:
    """Data requirements for a forecasting method."""
    
    requires_global: bool
    requires_countries: bool
    min_countries: int
    description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'requires_global': self.requires_global,
            'requires_countries': self.requires_countries,
            'min_countries': self.min_countries,
            'description': self.description
        }


# Pre-defined requirements for each method
METHOD_REQUIREMENTS = {
    'Global Only': MethodRequirements(
        requires_global=True,
        requires_countries=False,
        min_countries=0,
        description='Requires global market data only'
    ),
    'Top-Down': MethodRequirements(
        requires_global=True,
        requires_countries=True,
        min_countries=1,
        description='Requires global data + country data for distribution'
    ),
    'Bottom-Up': MethodRequirements(
        requires_global=False,
        requires_countries=True,
        min_countries=2,
        description='Requires multiple countries with market data'
    ),
    'Country-Specific': MethodRequirements(
        requires_global=False,
        requires_countries=True,
        min_countries=1,
        description='Requires selected countries to have market data'
    )
}