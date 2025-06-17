from dataclasses import dataclass


@dataclass
class ValidationContext:
    validate: bool = True
    strict_well_validation: bool = True
    strict_hfb_validation: bool = True
    ignore_time_no_data: bool = False
