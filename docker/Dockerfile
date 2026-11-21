# Image bac à sable de l'agent : les commandes shell générées par le LLM
# s'exécutent ici (voir agent/sandbox.py), jamais sur l'hôte.
FROM python:3.12-slim

ARG UID=1000
ARG GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sudo curl wget git ca-certificates file jq unzip \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur avec le même UID/GID que l'hôte : les fichiers écrits dans le
# workspace monté restent à l'utilisateur. sudo sans mot de passe pour apt.
RUN groupadd -o -g "${GID}" agent \
    && useradd -o -m -u "${UID}" -g "${GID}" -s /bin/bash agent \
    && echo "agent ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/agent \
    && chmod 0440 /etc/sudoers.d/agent

# Le LLM écrit souvent « apt-get install » sans sudo : ces wrappers l'ajoutent.
RUN for cmd in apt apt-get; do \
        printf '#!/bin/sh\nif [ "$(id -u)" -ne 0 ]; then exec sudo /usr/bin/%s "$@"; fi\nexec /usr/bin/%s "$@"\n' "$cmd" "$cmd" \
            > "/usr/local/bin/$cmd"; \
        chmod +x "/usr/local/bin/$cmd"; \
    done

USER agent
CMD ["sleep", "infinity"]
