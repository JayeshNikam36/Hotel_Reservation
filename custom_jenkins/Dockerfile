FROM jenkins/jenkins:lts

USER root

# Just basic tools, no Docker inside
RUN apt-get update -y && \
    apt-get install -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER jenkins
