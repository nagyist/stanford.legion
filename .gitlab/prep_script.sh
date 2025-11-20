#!/bin/bash

set -e
set -x

# be sure we're starting in a (really) clean repository
git clean -fxd
rm -rfv language/terra.build gasnet Thrust
git status
