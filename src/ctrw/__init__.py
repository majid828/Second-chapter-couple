"""
Continuous-Time Random Walk (CTRW) analysis module.

Provides:

- power-law tail estimation
- waiting-time distribution analysis
- CTRW-memory comparison


Used for interpreting recovered memory functions
within anomalous transport theory.

"""


from .ctrw_analysis import (

    estimate_power_law_exponent,

    generate_waiting_time_distribution,

    ctrw_memory_comparison,

    tail_statistics

)


__all__=[

    "estimate_power_law_exponent",

    "generate_waiting_time_distribution",

    "ctrw_memory_comparison",

    "tail_statistics"

]
