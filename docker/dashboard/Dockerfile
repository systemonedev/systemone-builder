# Unified Web Dashboard (Next.js + Tailwind)
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ARG NEXT_PUBLIC_S1_API_URL=
# Published API port the browser calls (compose passes S1_API_PORT).
ARG NEXT_PUBLIC_S1_API_PORT=8090
ENV NEXT_PUBLIC_S1_API_URL=$NEXT_PUBLIC_S1_API_URL NEXT_PUBLIC_S1_API_PORT=$NEXT_PUBLIC_S1_API_PORT NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
# Static server only: no reason to run as root.
USER node
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD wget -q -O /dev/null http://127.0.0.1:3000/ || exit 1
CMD ["node", "server.js"]
