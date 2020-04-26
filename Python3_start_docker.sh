#!/bin/bash

set -e

# PROJECT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/"
exec docker run --rm -it -v "$(pwd)" "ml4qs:base:root/Python3Code"
/bin/bash
