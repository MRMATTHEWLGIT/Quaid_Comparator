# Modified Quaid Environment

This directory contains a modified copy of the Quaid environment originally
sourced from:

https://github.com/real-world-drl/esp-dl-quant-icra2026

The original project is licensed under the MIT License.
The original license is retained in LICENSE.

An `UPSTREAM_COMMIT.txt` file is also included to document:
- the original repository URL
- the upstream commit hash
- the date this code was copied/forked

This helps preserve traceability back to the original source version.

Modifications made for this project include:

- Added access to raw (pre-normalised) observations 

These modifications were made to support the Quaid comparator runtime and
real-time policy-switching research project.