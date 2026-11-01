FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The browser talks to the backend directly, so this must be the
# browser-reachable backend URL (default: local backend on :8000).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]
