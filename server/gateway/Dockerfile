FROM oven/bun:1.1-alpine
WORKDIR /app
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile || bun install
COPY . .
EXPOSE 4001
CMD ["bun", "src/index.ts"]
