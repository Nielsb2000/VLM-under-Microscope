
FROM ghcr.io/agent-infra/sandbox:latest

# Ensure all screenshots subfolders are created and writable
RUN mkdir -p /home/gem/screenshots/browser_steps \
	&& chmod -R 0777 /home/gem/screenshots

# Switch to user gem for all subsequent commands (including entrypoint)
USER gem
