
# TODO: install tarpaulin if not installed

COVROOT=".cov"
mkdir -p "$COVROOT"
cargo tarpaulin --engine llvm -o Html -o Lcov --output-dir "$COVROOT" --fail-under 30

open "$COVROOT/tarpaulin-report.html"
