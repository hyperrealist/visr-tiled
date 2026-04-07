# The devcontainer should use the developer target and run as root with podman
# or docker with user namespaces.
FROM ghcr.io/diamondlightsource/ubuntu-devcontainer:noble AS developer

# Add any system dependencies for the developer/build environment here
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    graphviz \
    && apt-get dist-clean

# The build stage installs the context into the venv
FROM developer AS build

# Change the working directory to the `app` directory
# and copy in the project
WORKDIR /app
COPY . /app
RUN chmod o+wrX .

# Tell uv sync to install python in a known location so we can copy it out later
ENV UV_PYTHON_INSTALL_DIR=/python

# Sync the project without its dev dependencies
# ----------------------------------------------------------------------------------------------------- debugpy
RUN uv add debugpy
# ----------------------------------------------------------------------------------------------------- /debugpy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev


# The runtime stage copies the built venv into a runtime container
FROM ubuntu:noble AS runtime

# Add apt-get system dependecies for runtime here if needed
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     some-library \
#     && apt-get dist-clean
# ----------------------------------------------------------------------------------------------------- gdb
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gdb libnss-wrapper \
    && apt-get dist-clean
# ----------------------------------------------------------------------------------------------------- /gdb

# Copy the python installation from the build stage
COPY --from=build /python /python

# Copy the environment, but not the source code
# COPY --from=build /app/.venv /app/.venv
# ENV PATH=/app/.venv/bin:$PATH
# ----------------------------------------------------------------------------------------------------- venv
COPY --chown=1000:1000 --from=build /app/.venv /app/.venv
RUN chmod o+wrX /app/.venv
ENV PATH=/app/.venv/bin:$PATH
# ----------------------------------------------------------------------------------------------------- /venv



# ----------------------------------------------------------------------------------------------------- symlink
WORKDIR /app/.venv/lib
RUN ln -s python* python
# ----------------------------------------------------------------------------------------------------- /symlink



# ----------------------------------------------------------------------------------------------------- source code
WORKDIR /workspaces
COPY --chown=1000:1000 . visr-tiled
# ----------------------------------------------------------------------------------------------------- /source code




# ----------------------------------------------------------------------------------------------------- uv
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/
# ----------------------------------------------------------------------------------------------------- /uv




# # change this entrypoint if it is not the same as the repo
# ENTRYPOINT ["visr-tiled"]
# CMD ["--version"]




# Tiled-specifics

RUN mkdir -p /deploy/config
COPY example_configs/single_catalog_single_user.yml /deploy/config
ENV TILED_CONFIG=/deploy/config

EXPOSE 8000


# ----------------------------------------------------------------------------------------------------- user
RUN echo "user:x:37149:37149:Dynamic User:/home/user:/bin/bash" >> /etc/passwd
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# ----------------------------------------------------------------------------------------------------- /user
CMD ["tiled", "serve", "config", "--host", "0.0.0.0", "--port", "8000", "--scalable"]
# CMD ["python", "-Xfrozen_modules=off", "-m", "debugpy", \
#     "--listen", "0.0.0.0:5678", "--wait-for-client", \
#     "-m", "tiled", "serve", "config", \
#     "--host", "0.0.0.0", "--port", "8000", "--scalable"]
