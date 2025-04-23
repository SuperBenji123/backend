# Use an official Node.js runtime as a base image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy rest of the code
COPY . .

# Expose port
EXPOSE 8000

ENV ELEVENLABS_AGENT_ID="1LCFrqGanZNyW0aDicia"
ENV ELEVENLABS_API_KEY="sk_aff972695f992995e0ba36fe7a6ec51760ae8bacd77d273e"

# Twilio
ENV TWILIO_ACCOUNT_SID="AC372ebb7bb0e95da7ac31c692bae63352"
ENV TWILIO_AUTH_TOKEN="42c4e53f504442ce5f506afb1d9a4369"
ENV TWILIO_PHONE_NUMBER="+44 178 9335858"

# Run the app
CMD ["node", "outbound.js"]