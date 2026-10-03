# Unified Web Dashboard (Next.js + Tailwind) behind a small gateway.
# The browser only talks to this container: frontend/gateway.mjs forwards
# /api/v1/* to the API and adds S1_API_KEY server-side, so the key never
# reaches the browser, and nothing about the API's address is baked in here.
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/gateway.mjs ./gateway.mjs
# No reason to run as root.
USER node
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD wget -q -O /dev/null http://127.0.0.1:3000/ || exit 1
# The gateway listens on :3000 and runs the Next.js server on a private port.
CMD ["node", "gateway.mjs"]
