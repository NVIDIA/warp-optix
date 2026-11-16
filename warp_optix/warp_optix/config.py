# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global warp-optix settings, mirroring ``wp.config``.

``validation`` and ``log_level`` are read when a device's OptiX context is first
created, i.e. on the first OptiX operation on that device; later changes only
affect devices whose context does not exist yet. ``verify_instances`` is read on
every instance build.
"""

validation: bool = False
"""Enable OptiX validation mode (slow; reports API misuse through the log)."""

log_level: int = 4
"""OptiX log level: 0 disables logging, 1 fatal, 2 error, 3 warning, 4 print (forwarded to ``logging``)."""

verify_instances: bool = False
"""Check that instance builds reference only their listed children (synchronizes the build stream), like ``wp.config.verify_cuda``."""
