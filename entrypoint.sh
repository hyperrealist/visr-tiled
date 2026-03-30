#!/bin/bash

echo "entrypoint.sh: running as uid=$(id -u) gid=$(id -g)" >&2

USER_ID=$(id -u)
GROUP_ID=$(id -g)

cat > /tmp/passwd <<EOF
user:x:${USER_ID}:${GROUP_ID}:Dynamic User:/home/user:/bin/bash
EOF

export NSS_WRAPPER_PASSWD=/tmp/passwd
export NSS_WRAPPER_GROUP=/etc/group
export HOME=/home/user

NSS_WRAPPER_LIB=$(ldconfig -p | grep libnss_wrapper | awk '{print $NF}' | head -1)
if [[ -z "${NSS_WRAPPER_LIB}" ]]; then
    echo "entrypoint.sh: WARNING libnss_wrapper not found" >&2
else
    echo "entrypoint.sh: using ${NSS_WRAPPER_LIB}" >&2
    export LD_PRELOAD=${NSS_WRAPPER_LIB}
fi

exec "$@"
