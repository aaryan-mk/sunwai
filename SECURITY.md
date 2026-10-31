# Security Policy

## Supported versions

SunwAI is pre-1.0. Security fixes are made on the latest release and on `main`.

## Reporting a vulnerability

Please do **not** open a public issue. Report privately using GitHub's "Report a vulnerability" button under the repository's Security tab (private vulnerability reporting). Include the affected version or commit, reproduction steps, and the impact you expect.

You can expect an acknowledgement within a few days and a status update as the issue is triaged. Please allow reasonable time for a fix before public disclosure.

## Scope and deployment guidance

- **Secrets**: API keys (Bodhan, OpenRouter or Gemini, Hugging Face) and the control token belong in the environment or a local `.env`, never in Git. `.env`, cookies, job output and weights are git-ignored; keep them that way.
- **Control plane**: set a control token. Without authentication the control plane refuses to start rather than running open. Rate limits and daily spend caps bound abuse of paid LLM and synthesis calls.
- **LLM output is untrusted**: tool calls from the chat planner are validated against schemas and authorised before execution.
- **yt-dlp cookies** grant access to an account. Use a dedicated account, keep the file mode 600, and never commit it.
- **Network exposure**: the server is not a hardened multi-tenant service. Bind to localhost or place it behind a reverse proxy with TLS and your own authentication. Run a single API process.
- **Downloaded media** is untrusted input to ffmpeg and ML models; run workers with least privilege and resource limits.
- Report any suspected credential leak in the history immediately and rotate the key.
