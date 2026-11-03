# Retained S4 CPU Evidence

Each retained attempt is immutable and names the exact Git commit it exercised.
The Python-standard-library capture script writes locked command logs, input and
output expectations, failure records, and SHA-256 hashes before returning a
command failure. Attempts are never written into `target/` and the script
refuses to overwrite one.
