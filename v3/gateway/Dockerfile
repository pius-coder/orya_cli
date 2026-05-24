FROM oven/bun:1.1-alpine AS base
WORKDIR /app/gateway
COPY gateway/package.json gateway/bun.lock* ./
RUN bun install --frozen-lockfile || bun install
COPY gateway/ ./
EXPOSE 4001
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost:4001/health || exit 1
USER bun
CMD ["bun", "src/index.ts"]
