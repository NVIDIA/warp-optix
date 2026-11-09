# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global warp-optix settings, mirroring ``wp.config``.

Settings are read when a device's OptiX context is first created, i.e. on the
first OptiX operation on that device. Later changes only affect devices whose
context does not exist yet.
"""

validation: bool = False
"""Enable OptiX validation mode (slow; reports API misuse through the log)."""

log_level: int = 4
"""OptiX log level: 0 disables logging, 1 fatal, 2 error, 3 warning, 4 print (forwarded to ``logging``)."""
