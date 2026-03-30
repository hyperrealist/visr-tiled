#!/bin/bash

USER_ID=$(id -u)
GROUP_ID=$(id -g)

cat > /tmp/passwd <<EOF
user:x:${USER_ID}:${GROUP_ID}:Dynamic User:/home/user:/bin/bash
EOF

export NSS_WRAPPER_PASSWD=/tmp/passwd
export NSS_WRAPPER_GROUP=/etc/group
export LD_PRELOAD=libnss_wrapper.so
export HOME=/home/user

exec "$@"
