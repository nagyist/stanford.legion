#!/bin/bash

set -e
set -x

umask 002 # Make sure files are group-writable

# job directory
JOB_WORKDIR="${EXTERNAL_WORKDIR}_${CI_JOB_ID:-legion${TEST_LEGION_CXX:-1}_regent${TEST_REGENT:-1}}"
rm -rf $JOB_WORKDIR
cp -r $CI_PROJECT_DIR $JOB_WORKDIR
cd $JOB_WORKDIR
echo "Running tests in $JOB_WORKDIR"

# copy files from shared environment
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    cp -r $EXTERNAL_WORKDIR/gasnet .
fi
ln -s $EXTERNAL_WORKDIR/terra language/terra

# setup environment
if [[ "$LMOD_SYSTEM_NAME" = frontier ]]; then
    cat >>env.sh <<EOF
module load PrgEnv-gnu
module load cray-python
module load rocm/$ROCM_VERSION
export CC=cc
export CXX=CC
if [[ "\$REALM_NETWORKS" != "" ]]; then
    RANKS_PER_NODE=4
    export LAUNCHER="srun -n\$(( RANKS_PER_NODE * SLURM_JOB_NUM_NODES )) --cpus-per-task \$(( 56 / RANKS_PER_NODE )) --gpus-per-task \$(( 8 / RANKS_PER_NODE )) --cpu-bind cores"
    if [[ SLURM_JOB_NUM_NODES -eq 1 ]]; then
        export LAUNCHER+=" --network=single_node_vni"
    fi
fi
# Important: Thrust must be in \$EXTERNAL_WORKDIR or else CMake sees it in-source
export THRUST_PATH=\$EXTERNAL_WORKDIR/Thrust
export REGENT_LLVM_PATH="\$EXTERNAL_WORKDIR/llvm"

# Proxy settings for Frontier: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#proxy-settings
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
EOF
else
    echo "Don't know how to build on this system"
    exit 1
fi

# GASNet environment
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    if [[ "$GASNET_DEBUG" -eq 1 ]]; then
        cat >>env.sh <<EOF
export GASNET_ROOT="\$JOB_WORKDIR/gasnet/debug"
EOF
    else
        cat >>env.sh <<EOF
export GASNET_ROOT="\$JOB_WORKDIR/gasnet/release"
EOF
    fi
fi

cat env.sh
set +x # don't spew all the module goop to the console
source env.sh
set -x

# build GASNet
if [[ "$REALM_NETWORKS" == gasnet* ]]; then
    set +x # makes the build very noisy
    CONDUIT=$GASNET_CONDUIT make -C gasnet -j${THREADS:-16}
    set -x
fi

# required for machine_config test to pin NUMA memory
ulimit -l $(( 1024 * 1024 )) # KB

# get backtraces if necessary
export REALM_BACKTRACE=1

# run test script
./tools/add_github_host_key.sh
grep 'model name' /proc/cpuinfo | uniq -c || true
which $CXX
$CXX --version
free

if [[ -z "$TEST_PYTHON_EXE" ]]; then
    export TEST_PYTHON_EXE=`which python3 python | head -1`
fi

chmod -R g+w $JOB_WORKDIR
set +e
$TEST_PYTHON_EXE ./test.py -j${THREADS:-16}
ok=$?
chmod -R g+w $JOB_WORKDIR
exit $ok
