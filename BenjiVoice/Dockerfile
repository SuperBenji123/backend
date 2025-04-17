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

# Run the app
CMD ["node", "outbound.js"]