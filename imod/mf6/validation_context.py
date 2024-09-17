from dataclasses import dataclass


@dataclass
class ValidationContext:
    validate: bool = True
    relax_well_validation: bool = False
    