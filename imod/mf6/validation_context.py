from dataclasses import dataclass


@dataclass
class ValidationContext:
    validate: bool = True
    strict_well_validation: bool = True
