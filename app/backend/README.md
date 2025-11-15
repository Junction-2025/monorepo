# Backend API

Express.js backend server for the application.

## Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn

## Installation

1. Navigate to the backend directory:
```bash
cd app/backend
```

2. Install dependencies:
```bash
npm install
```

## Running the Server

### Development Mode

Start the development server:
```bash
node app.js
```

The server will start on `http://localhost:8000`

### Add a Start Script (Optional)

You can add a start script to `package.json`:

```json
"scripts": {
  "start": "node app.js",
  "dev": "node app.js"
}
```

Then run:
```bash
npm start
```

## API Endpoints

- `GET /` - Health check endpoint
- More endpoints can be added in `routes/` directory

## Project Structure

```
backend/
├── app.js          # Main server file
├── routes/         # API routes (create as needed)
├── controllers/    # Business logic (create as needed)
└── package.json
```

## Environment Variables

Create a `.env` file for environment-specific configuration (optional):

```
PORT=8000
NODE_ENV=development
```

